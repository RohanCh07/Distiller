"""Talk to the big "teacher" model (any OpenAI-compatible API).

Two jobs:
- label:    get teacher outputs for raw inputs (when a user has inputs but no outputs).
- augment:  write new, varied inputs (e.g. for rare labels), then have the teacher label
            them; for classification, keep only those the teacher labels as intended.

Config via env: DISTILLER_TEACHER_BASE_URL (default OpenAI), DISTILLER_TEACHER_API_KEY
(or OPENAI_API_KEY), DISTILLER_TEACHER_MODEL.
"""

import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx

from distiller.data.io import Example
from distiller.data.validate import make_validator
from distiller.spec import TaskSpec, TaskType

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


@dataclass
class TeacherConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls, model: str | None = None) -> "TeacherConfig":
        model = model or os.environ.get("DISTILLER_TEACHER_MODEL")
        if not model:
            raise ValueError("Set DISTILLER_TEACHER_MODEL or pass --model")
        return cls(
            base_url=os.environ.get("DISTILLER_TEACHER_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("DISTILLER_TEACHER_API_KEY")
            or os.environ.get("OPENAI_API_KEY", ""),
            model=model,
        )


class Teacher:
    def __init__(self, config: TeacherConfig, concurrency: int = 8, timeout: float = 60):
        self.config = config
        self.concurrency = concurrency
        self.client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=timeout,
        )

    def chat(self, messages: list[dict], temperature: float = 0.0, retries: int = 4) -> str:
        body = {"model": self.config.model, "messages": messages, "temperature": temperature}
        for attempt in range(retries + 1):
            try:
                r = self.client.post("/chat/completions", json=body)
                if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(2**attempt + random.random())
                    continue
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except httpx.TransportError:
                if attempt == retries:
                    raise
                time.sleep(2**attempt + random.random())
        raise RuntimeError("unreachable")

    # ---------- labelling ----------

    def _system_prompt(self, spec: TaskSpec, labels: list[str] | None) -> str:
        parts = [spec.instruction.strip()] if spec.instruction else []
        if spec.type == TaskType.classification:
            if labels:
                parts.append("Allowed labels: " + ", ".join(labels))
            parts.append("Reply with exactly one label and nothing else.")
        elif spec.type == TaskType.extraction:
            if spec.output_schema:
                parts.append("JSON Schema for the answer:\n" + json.dumps(spec.output_schema))
            parts.append("Reply with only a JSON object, no explanation.")
        return "\n\n".join(parts)

    def _parse(self, spec: TaskSpec, text: str) -> Any:
        text = text.strip()
        if spec.type == TaskType.extraction:
            m = _JSON_OBJECT.search(text)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return text.strip().strip('"').strip()

    def label(
        self, spec: TaskSpec, inputs: list[str], labels: list[str] | None = None
    ) -> list[Any]:
        """Teacher outputs for each input; None where the answer failed validation."""
        labels = labels or spec.labels
        system = self._system_prompt(spec, labels)
        check = make_validator(spec)

        def one(text: str) -> Any:
            try:
                raw = self.chat(
                    [{"role": "system", "content": system}, {"role": "user", "content": text}]
                )
            except httpx.HTTPError:
                return None
            out = self._parse(spec, raw)
            if labels and spec.type == TaskType.classification and out not in labels:
                return None
            return out if check(out) is None else None

        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(one, inputs))

    # ---------- synthetic data ----------

    def _write_inputs(
        self, spec: TaskSpec, seeds: list[str], n: int, target: str | None
    ) -> list[str]:
        shown = "\n".join(f"- {s}" for s in random.sample(seeds, min(8, len(seeds))))
        goal = f' whose correct answer is "{target}"' if target else ""
        prompt = (
            f"Task: {spec.instruction or spec.name}\n\nReal example inputs:\n{shown}\n\n"
            f"Write {n} new, realistic inputs{goal}. Make them varied: different lengths, tone, "
            "typos, and language mix (English, Hindi, Hinglish) where natural for this task. "
            "Do not copy the examples. Reply with only a JSON array of strings."
        )
        raw = self.chat([{"role": "user", "content": prompt}], temperature=0.9)
        m = _JSON_ARRAY.search(raw)
        if not m:
            return []
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
        return [str(x).strip() for x in items if str(x).strip()]

    def augment(
        self,
        spec: TaskSpec,
        examples: list[Example],
        per_label: int | None = None,
        total: int | None = None,
        batch: int = 10,
    ) -> list[Example]:
        """New teacher-labelled examples, marked synthetic."""
        out: list[Example] = []
        if spec.type == TaskType.classification:
            counts = Counter(ex.output for ex in examples)
            labels = spec.labels or sorted(counts)
            want = per_label or spec.data.min_examples_per_label
            for label in labels:
                missing = want - counts.get(label, 0)
                if missing <= 0:
                    continue
                seeds = [ex.input for ex in examples if ex.output == label] or [
                    ex.input for ex in examples
                ]
                texts: list[str] = []
                while len(texts) < missing:
                    new = self._write_inputs(spec, seeds, min(batch, missing - len(texts)), label)
                    if not new:
                        break
                    texts += new
                texts = texts[:missing]
                # Consistency filter: keep only inputs the teacher itself labels as intended.
                for text, got in zip(texts, self.label(spec, texts, labels)):
                    if got == label:
                        out.append(Example(id="", input=text, output=got, meta={"synthetic": True}))
        else:
            want = total or 200
            seeds = [ex.input for ex in examples]
            texts = []
            while len(texts) < want:
                new = self._write_inputs(spec, seeds, min(batch, want - len(texts)), None)
                if not new:
                    break
                texts += new
            texts = texts[:want]
            for text, got in zip(texts, self.label(spec, texts)):
                if got is not None:
                    out.append(Example(id="", input=text, output=got, meta={"synthetic": True}))
        return out


def to_row(ex: Example) -> str:
    row = {"input": ex.input, "output": ex.output}
    if ex.meta.get("synthetic"):
        row["synthetic"] = True
    return json.dumps(row, ensure_ascii=False)
