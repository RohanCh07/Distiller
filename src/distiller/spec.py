"""Task spec: the single file (task.yaml) that describes one distillable task."""

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class TaskType(str, Enum):
    classification = "classification"  # output is one label from a fixed set
    extraction = "extraction"  # output is a JSON object matching a schema
    generation = "generation"  # output is short free text (e.g. translation)


class DataSpec(BaseModel):
    path: Path
    # Optional teacher-generated examples (`distiller teacher augment`). Used for training
    # only: never in val/test, and dropped if near-identical to a val/test example.
    synthetic_path: Path | None = None
    format: Literal["auto", "jsonl", "csv"] = "auto"
    input_field: str = "input"
    output_field: str = "output"
    val_fraction: float = Field(0.1, gt=0, lt=0.5)
    test_fraction: float = Field(0.1, gt=0, lt=0.5)
    seed: int = 42
    near_dup_threshold: float = Field(0.85, gt=0, le=1)
    min_examples_per_label: int = 10


class StudentSpec(BaseModel):
    # Backends to train and compare. None picks defaults for the task type.
    candidates: list[str] | None = None
    encoder_model: str = "answerdotai/ModernBERT-base"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_length: int = 256
    # Try a small hyperparameter grid per backend and keep the best on validation.
    sweep: bool = False
    # Per-backend hyperparameter overrides, e.g. {"encoder": {"epochs": 5}}.
    overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)


class TargetSpec(BaseModel):
    # Accuracy the student must hold on the requests it keeps (the rest fall back).
    min_accuracy: float = Field(0.95, gt=0, le=1)
    # A generation answer counts as correct when token F1 with the reference is at least this.
    generation_f1_threshold: float = Field(0.8, gt=0, le=1)


class TaskSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9_\-]+$")
    type: TaskType
    instruction: str = ""
    labels: list[str] | None = None
    output_schema: dict[str, Any] | None = None
    data: DataSpec
    student: StudentSpec = Field(default_factory=StudentSpec)
    target: TargetSpec = Field(default_factory=TargetSpec)
    output_dir: Path = Path("runs")

    @model_validator(mode="after")
    def _check_type_fields(self) -> "TaskSpec":
        if self.type != TaskType.classification and self.labels:
            raise ValueError("labels only apply to classification tasks")
        if self.type != TaskType.extraction and self.output_schema:
            raise ValueError("output_schema only applies to extraction tasks")
        if self.labels and len(set(self.labels)) != len(self.labels):
            raise ValueError("labels contain duplicates")
        return self

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.name

    @classmethod
    def load(cls, path: str | Path) -> "TaskSpec":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        spec = cls.model_validate(raw)
        # Relative paths in task.yaml are relative to the file itself.
        base = path.resolve().parent
        if not spec.data.path.is_absolute():
            spec.data.path = base / spec.data.path
        if spec.data.synthetic_path and not spec.data.synthetic_path.is_absolute():
            spec.data.synthetic_path = base / spec.data.synthetic_path
        if not spec.output_dir.is_absolute():
            spec.output_dir = base / spec.output_dir
        return spec
