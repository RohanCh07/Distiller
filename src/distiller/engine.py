"""The training engine: task.yaml in, trained + evaluated + calibrated students out.

prepare data -> for each candidate backend: train (optionally a small sweep, picked on
val) -> predict val/test -> calibrate on val -> pick a confidence threshold on val ->
measure it on test -> save -> report. The test split is used once, at the very end.
"""

import json
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from distiller.data.pipeline import PreparedData, prepare
from distiller.eval.calibration import expected_calibration_error, fit_calibrator
from distiller.eval.coverage import apply_threshold, choose_threshold, curve
from distiller.eval.metrics import is_correct, task_metrics
from distiller.eval.report import CandidateResult, write_report
from distiller.spec import TaskSpec, TaskType
from distiller.training.base import Prediction, Student, default_candidates, get_backend
from distiller.training.presets import hyperparams, sweep_space

MIN_TO_TRAIN = 50


@dataclass
class EngineResult:
    results: list[CandidateResult]
    best: CandidateResult | None
    report_path: Path


@dataclass
class _Trial:
    student: Student
    train_info: dict
    val_preds: list[Prediction]
    val_ok: list[bool]

    @property
    def val_score(self) -> float:
        return sum(self.val_ok) / max(len(self.val_ok), 1)


def _train_trial(spec, cls, hp, prepared: PreparedData, work_dir: Path, log) -> _Trial:
    student = cls(spec, prepared.labels, hp)
    train_info = student.fit(prepared.train, prepared.val, work_dir)
    val_preds = student.predict([ex.input for ex in prepared.val])
    val_ok = [is_correct(spec, p.output, ex.output) for p, ex in zip(val_preds, prepared.val)]
    trial = _Trial(student, train_info, val_preds, val_ok)
    log(f"    val score {trial.val_score:.4f}")
    return trial


def _run_candidate(spec, backend, prepared: PreparedData, sweep: bool, log) -> CandidateResult:
    cls = get_backend(backend)
    base_hp = hyperparams(backend, spec, len(prepared.train))
    result = CandidateResult(backend=backend, model=base_hp["model"])
    if spec.type not in cls.supports:
        result.status, result.error = "failed", f"{backend} does not support {spec.type.value}"
        return result
    model_dir = spec.run_dir / "models" / backend

    # 1. Train: one preset run, or a small grid with the best kept by validation score.
    settings = sweep_space(backend) if sweep else [{}]
    best: _Trial | None = None
    for i, override in enumerate(settings):
        hp = {**base_hp, **override}
        log(f"[{backend}] trial {i + 1}/{len(settings)} {override or '(preset)'} "
            f"on {len(prepared.train)} examples")
        trial = _train_trial(spec, cls, hp, prepared, model_dir / f"trial{i}", log)
        if sweep:
            result.sweep.append({"params": override or "preset", "val_score": trial.val_score})
        if best is None or trial.val_score > best.val_score:
            if best is not None:
                best.student.free()
            best = trial
        else:
            trial.student.free()
    student, val_preds, val_ok = best.student, best.val_preds, best.val_ok
    result.train = best.train_info
    for d in model_dir.glob("trial*"):
        shutil.rmtree(d, ignore_errors=True)  # checkpoints only; final model saved below

    # 2. Test predictions (timed) and task metrics.
    log(f"[{backend}] evaluating on test")
    start = time.perf_counter()
    test_preds = student.predict([ex.input for ex in prepared.test])
    result.ms_per_example = 1000 * (time.perf_counter() - start) / max(len(prepared.test), 1)
    test_ok = [is_correct(spec, p.output, ex.output) for p, ex in zip(test_preds, prepared.test)]
    result.test_metrics = task_metrics(
        spec, [p.output for p in test_preds], [ex.output for ex in prepared.test], prepared.labels
    )

    # 3. Calibrate on val, choose threshold on val, measure on test.
    has_logits = spec.type == TaskType.classification and val_preds and val_preds[0].logits
    label_idx = {lbl: i for i, lbl in enumerate(prepared.labels or [])}
    calibrator = fit_calibrator(
        [p.confidence for p in val_preds],
        val_ok,
        logits=[p.logits for p in val_preds] if has_logits else None,
        gold_idx=[label_idx[ex.output] for ex in prepared.val] if has_logits else None,
    )
    val_conf = calibrator.apply(
        [p.confidence for p in val_preds], [p.logits for p in val_preds] if has_logits else None
    )
    test_raw = [p.confidence for p in test_preds]
    test_conf = calibrator.apply(test_raw, [p.logits for p in test_preds] if has_logits else None)
    result.calibration = calibrator.to_dict()
    result.ece_raw = expected_calibration_error(test_raw, test_ok)
    result.ece_calibrated = expected_calibration_error(test_conf, test_ok)
    choice = choose_threshold(val_conf, val_ok, spec.target.min_accuracy)
    result.threshold = apply_threshold(choice, test_conf, test_ok).to_dict()

    # 4. Save the model with everything the gateway needs to route.
    student.save(model_dir)
    (model_dir / "routing.json").write_text(
        json.dumps(
            {
                "calibration": result.calibration,
                "threshold": result.threshold,
                "test_curve": curve(test_conf, test_ok),
            },
            indent=2,
        )
    )
    result.model_dir = str(model_dir)
    student.free()
    return result


def run(
    spec: TaskSpec, candidates: list[str] | None = None, sweep: bool | None = None, log=print
) -> EngineResult:
    log(f"Preparing data for {spec.name}")
    prepared = prepare(spec)
    for w in prepared.report.warnings:
        log(f"  warning: {w}")
    if len(prepared.train) < MIN_TO_TRAIN:
        raise ValueError(
            f"Only {len(prepared.train)} training examples after cleaning; need {MIN_TO_TRAIN}+. "
            f"See {spec.run_dir / 'data' / 'data_report.md'}"
        )

    names = candidates or spec.student.candidates or default_candidates(spec.type)
    do_sweep = spec.student.sweep if sweep is None else sweep
    results = []
    for backend in names:
        try:
            results.append(_run_candidate(spec, backend, prepared, do_sweep, log))
        except Exception as e:  # one failing backend should not lose the others' results
            log(traceback.format_exc())
            try:
                model = hyperparams(backend, spec, len(prepared.train))["model"]
            except ValueError:
                model = "?"
            results.append(
                CandidateResult(backend=backend, model=model, status="failed", error=str(e)[:200])
            )

    ok = [r for r in results if r.status == "ok"]
    best = max(ok, key=CandidateResult.score) if ok else None
    report_path = write_report(spec, prepared.report, results, best)
    if best:
        (spec.run_dir / "best.json").write_text(
            json.dumps({"backend": best.backend, "model_dir": best.model_dir}, indent=2)
        )
    log(f"Report: {report_path}")
    return EngineResult(results, best, report_path)
