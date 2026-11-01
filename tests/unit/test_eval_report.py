import json

from distiller.data.report import DataReport
from distiller.eval.report import CandidateResult, write_report
from distiller.spec import TaskSpec


def _spec(tmp_path, type_="classification"):
    return TaskSpec.model_validate(
        {"name": "rep", "type": type_, "data": {"path": "x"}, "output_dir": str(tmp_path)}
    )


def _ok(backend, coverage, ms, threshold=0.8):
    return CandidateResult(
        backend=backend,
        model=f"{backend}-model",
        train={"train_seconds": 12.0},
        test_metrics={
            "accuracy": 0.9,
            "macro_f1": 0.85,
            "top_confusions": [],
            "worst_labels": [("a", 0.1)],
        },
        ece_raw=0.1,
        ece_calibrated=0.02,
        calibration={"method": "temperature"},
        threshold={"threshold": threshold, "test_coverage": coverage, "test_accuracy": 0.97},
        ms_per_example=ms,
        model_dir=f"/runs/{backend}",
    )


def test_ranking_prefers_promotable_then_coverage_then_speed():
    not_promotable = _ok("x", 0.9, 1, threshold=None)
    low_cov = _ok("y", 0.5, 1)
    high_cov_slow = _ok("z", 0.8, 50)
    high_cov_fast = _ok("w", 0.8, 5)
    failed = CandidateResult(backend="f", model="m", status="failed")
    assert not not_promotable.promotable and not failed.promotable
    ranked = sorted(
        [not_promotable, low_cov, high_cov_slow, high_cov_fast],
        key=CandidateResult.score,
        reverse=True,
    )
    assert [r.backend for r in ranked] == ["w", "z", "y", "x"]


def test_write_report_promotable(tmp_path):
    spec = _spec(tmp_path)
    best = _ok("encoder", 0.75, 3.2)
    failed = CandidateResult(backend="llm-lora", model="qwen", status="failed", error="CUDA OOM")
    dr = DataReport(task="rep", split_sizes={"train": 80, "val": 10, "test": 10})
    path = write_report(spec, dr, [best, failed], best)
    md = path.read_text()
    assert "# Training report: rep" in md
    assert "80 train / 10 val / 10 test" in md
    assert "| llm-lora | qwen | failed: CUDA OOM |" in md
    assert (
        "Use **encoder**" in md
        and "75.0% of test requests" in md
        and "25.0% go to the teacher" in md
    )
    assert "0.100 → 0.020" in md
    assert "## encoder details" in md and "- accuracy: 0.9000" in md
    results = json.loads((tmp_path / "rep" / "results.json").read_text())
    assert results["best"] == "encoder" and len(results["candidates"]) == 2


def test_write_report_not_promotable_and_all_failed(tmp_path):
    spec = _spec(tmp_path, "extraction")
    r = _ok("llm-lora", 0.0, 10, threshold=None)
    r.test_metrics = {"exact_match": 0.5, "field_f1": 0.7}
    r.sweep = [{"params": {"learning_rate": 1e-4}, "val_score": 0.5}]
    md = write_report(spec, DataReport(task="rep"), [r], r).read_text()
    assert "0% (never reaches target)" in md
    assert "No student reaches 95%" in md
    assert "0.500 / 0.700" in md  # extraction headline metrics
    assert "{'learning_rate': 0.0001}: 0.5000" in md

    f = CandidateResult(backend="encoder", model="m", status="failed", error="boom")
    md = write_report(spec, DataReport(task="rep"), [f], None).read_text()
    assert "Every candidate failed" in md
