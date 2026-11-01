"""A trained student plus its routing settings (calibration + confidence threshold)."""

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distiller.eval.calibration import Calibrator
from distiller.spec import TaskSpec
from distiller.training.base import Student


@dataclass
class Answer:
    output: Any
    confidence: float  # calibrated
    confident: bool  # confidence >= threshold and output is valid


class LoadedStudent:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self.student = Student.load(self.model_dir)
        routing = json.loads((self.model_dir / "routing.json").read_text())
        self.calibrator = Calibrator(**routing["calibration"])
        t = routing["threshold"]["threshold"]
        self.threshold = math.inf if t is None else float(t)
        self._lock = threading.Lock()  # models are not safe to call from many threads at once

    @property
    def spec(self) -> TaskSpec:
        return self.student.spec

    def answer(self, inputs: list[str]) -> list[Answer]:
        with self._lock:
            preds = self.student.predict(inputs)
        logits = [p.logits for p in preds] if preds and preds[0].logits else None
        conf = self.calibrator.apply([p.confidence for p in preds], logits)
        return [
            Answer(p.output, c, p.output is not None and c >= self.threshold)
            for p, c in zip(preds, conf)
        ]
