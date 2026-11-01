"""A trained student plus its routing settings (calibration + confidence threshold).

Two ways to run the model:
- LoadedStudent: weights loaded into the gateway process (encoder or LLM, CPU or GPU).
- RemoteStudent: an LLM student served by vLLM (or any OpenAI-compatible server that
  returns token logprobs). The gateway only keeps the calibration and threshold.
"""

import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from distiller.eval.calibration import Calibrator
from distiller.spec import TaskSpec, TaskType
from distiller.training.base import META_FILE, Prediction, Student


@dataclass
class Answer:
    output: Any
    confidence: float  # calibrated
    confident: bool  # confidence >= threshold and output is valid


class _Routed:
    """Calibration + threshold from routing.json, shared by local and remote students."""

    def _load_routing(self, model_dir: Path) -> None:
        routing = json.loads((model_dir / "routing.json").read_text())
        self.calibrator = Calibrator(**routing["calibration"])
        t = routing["threshold"]["threshold"]
        self.threshold = math.inf if t is None else float(t)

    def _route(self, preds: list[Prediction]) -> list[Answer]:
        logits = [p.logits for p in preds] if preds and preds[0].logits else None
        conf = self.calibrator.apply([p.confidence for p in preds], logits)
        return [
            Answer(p.output, c, p.output is not None and c >= self.threshold)
            for p, c in zip(preds, conf)
        ]


class LoadedStudent(_Routed):
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.student = Student.load(self.model_dir)
        self._load_routing(self.model_dir)
        self._lock = threading.Lock()  # models are not safe to call from many threads at once

    @property
    def spec(self) -> TaskSpec:
        return self.student.spec

    def answer(self, inputs: list[str]) -> list[Answer]:
        with self._lock:
            preds = self.student.predict(inputs)
        return self._route(preds)


class RemoteStudent(_Routed):
    """An llm-lora student served by vLLM's OpenAI-compatible API.

    Start the server with `distiller serve-vllm <model_dir>`. Confidence is the geometric
    mean of the returned token probabilities, as for local generation, then calibrated
    with the same routing.json. For extraction tasks the schema is sent as a
    json_schema response_format, so vLLM constrains decoding to it.
    """

    def __init__(
        self,
        model_dir: str | Path,
        url: str,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 30,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model_dir = Path(model_dir)
        meta = json.loads((self.model_dir / META_FILE).read_text())
        if meta["backend"] != "llm-lora":
            raise ValueError("remote serving supports llm-lora students only")
        self._spec = TaskSpec.model_validate(meta["spec"])
        self.hparams = meta["hparams"]
        self.model = model or self._spec.name
        self._load_routing(self.model_dir)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(
            base_url=url.rstrip("/"), timeout=timeout, headers=headers, transport=transport
        )

    @property
    def spec(self) -> TaskSpec:
        return self._spec

    def _body(self, text: str) -> dict[str, Any]:
        messages = (
            [{"role": "system", "content": self.spec.instruction}] if self.spec.instruction else []
        )
        messages.append({"role": "user", "content": text})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.hparams.get("max_new_tokens", 256),
            "logprobs": True,
        }
        if (
            self.spec.type == TaskType.extraction
            and self.spec.output_schema
            and self.hparams.get("constrained_decoding", True)
        ):
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": self.spec.name, "schema": self.spec.output_schema},
            }
        return body

    def _predict_one(self, text: str) -> Prediction:
        from distiller.training.lora_llm import parse_output

        r = self.client.post("/chat/completions", json=self._body(text))
        r.raise_for_status()
        choice = r.json()["choices"][0]
        content = choice["message"].get("content") or ""
        logprobs = [
            t["logprob"]
            for t in ((choice.get("logprobs") or {}).get("content") or [])
            if t.get("logprob") is not None and math.isfinite(t["logprob"])
        ]
        conf = math.exp(sum(logprobs) / len(logprobs)) if logprobs else 0.0
        return Prediction(parse_output(self.spec, content), conf)

    def answer(self, inputs: list[str]) -> list[Answer]:
        return self._route([self._predict_one(t) for t in inputs])


def load_student(model_dir: str | Path, serving=None):
    """Build the right student for a task's serving settings (gateway.config.ServingSettings)."""
    if serving is not None and serving.backend == "vllm":
        return RemoteStudent(
            model_dir,
            serving.url,
            model=serving.model,
            api_key=os.environ.get(serving.api_key_env, "") if serving.api_key_env else None,
            timeout=serving.timeout_seconds,
        )
    return LoadedStudent(model_dir)
