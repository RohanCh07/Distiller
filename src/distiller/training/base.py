"""Student interface and backend registry.

A backend wraps one family of small model. Heavy imports (torch, transformers, peft)
happen inside the backend modules, so data prep and evaluation work without them.
"""

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distiller.data.io import Example
from distiller.spec import TaskSpec, TaskType

META_FILE = "distiller.json"


@dataclass
class Prediction:
    output: Any  # None when the student produced something unparseable
    confidence: float  # raw score in [0, 1]; calibrated later
    logits: list[float] | None = None  # classification backends only, for temperature scaling


class Student(ABC):
    name: str
    supports: tuple[TaskType, ...]

    def __init__(self, spec: TaskSpec, labels: list[str] | None, hparams: dict[str, Any]):
        self.spec = spec
        self.labels = labels
        self.hparams = hparams

    @property
    def model_id(self) -> str:
        return self.hparams["model"]

    @abstractmethod
    def fit(self, train: list[Example], val: list[Example], out_dir: Path) -> dict[str, Any]:
        """Train; return a small dict of training facts (time, steps, params)."""

    @abstractmethod
    def predict(self, inputs: list[str]) -> list[Prediction]: ...

    @abstractmethod
    def save(self, out_dir: Path) -> None: ...

    @abstractmethod
    def _load_weights(self, model_dir: Path) -> None: ...

    def free(self) -> None:
        """Release model memory (used between sweep trials)."""
        import gc

        for attr in ("model", "tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _write_meta(self, out_dir: Path) -> None:
        (out_dir / META_FILE).write_text(
            json.dumps(
                {
                    "backend": self.name,
                    "labels": self.labels,
                    "hparams": self.hparams,
                    "spec": self.spec.model_dump(mode="json"),
                },
                indent=2,
            )
        )

    @staticmethod
    def load(model_dir: str | Path) -> "Student":
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / META_FILE).read_text())
        backend = get_backend(meta["backend"])
        student = backend(TaskSpec.model_validate(meta["spec"]), meta["labels"], meta["hparams"])
        student._load_weights(model_dir)
        return student


def warmup_steps(n_train: int, batch_size: int, grad_accum: int, hp: dict[str, Any]) -> int:
    """warmup_ratio -> steps (newer transformers only accept steps)."""
    steps_per_epoch = math.ceil(n_train / (batch_size * grad_accum))
    return int(hp["warmup_ratio"] * steps_per_epoch * hp["epochs"])


def get_backend(name: str) -> type[Student]:
    if name == "encoder":
        from distiller.training.encoder import EncoderStudent

        return EncoderStudent
    if name == "llm-lora":
        from distiller.training.lora_llm import LoraLLMStudent

        return LoraLLMStudent
    raise ValueError(f"unknown backend {name!r}; choose from: encoder, llm-lora")


def default_candidates(task_type: TaskType) -> list[str]:
    # Classification: compare a tiny encoder against a small LLM; others need generation.
    if task_type == TaskType.classification:
        return ["encoder", "llm-lora"]
    return ["llm-lora"]
