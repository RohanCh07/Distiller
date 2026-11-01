"""Per-run results file and the human-readable comparison report."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from distiller.data.report import DataReport
from distiller.spec import TaskSpec


@dataclass
class CandidateResult:
    backend: str
    model: str
    status: str = "ok"  # "ok" | "failed"
    error: str | None = None
    train: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    ece_raw: float | None = None
    ece_calibrated: float | None = None
    calibration: dict[str, Any] = field(default_factory=dict)
    threshold: dict[str, Any] = field(default_factory=dict)
    ms_per_example: float | None = None  # batched throughput on the test split
    model_dir: str | None = None
    sweep: list[dict[str, Any]] = field(default_factory=list)  # [{params, val_score}]

    @property
    def promotable(self) -> bool:
        return self.status == "ok" and self.threshold.get("threshold") is not None

    def score(self) -> tuple:
        """Rank: promotable first, then test coverage at target, then speed."""
        return (
            self.promotable,
            self.threshold.get("test_coverage", 0.0),
            -(self.ms_per_example or math.inf),
        )


def _headline(spec: TaskSpec, metrics: dict[str, Any]) -> str:
    keys = {
        "classification": ("accuracy", "macro_f1"),
        "extraction": ("exact_match", "field_f1"),
        "generation": ("exact_match", "token_f1"),
    }[spec.type.value]
    return " / ".join(f"{metrics.get(k, 0):.3f}" for k in keys), " / ".join(keys)


def write_report(
    spec: TaskSpec, data_report: DataReport, results: list[CandidateResult], best: CandidateResult | None
) -> Path:
    run_dir = spec.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "task": spec.name,
                "target_accuracy": spec.target.min_accuracy,
                "best": best.backend if best else None,
                "candidates": [asdict(r) for r in results],
            },
            indent=2,
            default=str,
        )
    )

    target = spec.target.min_accuracy
    _, metric_names = _headline(spec, {})
    lines = [
        f"# Training report: {spec.name}",
        "",
        f"Task type: **{spec.type.value}**. Target: keep only requests the student answers at "
        f"**≥ {target:.0%}** accuracy; everything else falls back to the teacher.",
        "",
        f"Data: {data_report.split_sizes.get('train', 0)} train / "
        f"{data_report.split_sizes.get('val', 0)} val / {data_report.split_sizes.get('test', 0)} test "
        "(see data/data_report.md).",
        "",
        f"| Student | Model | Test {metric_names} | Traffic kept at target (test) | "
        "Accuracy on kept (test) | ECE raw → calibrated | ms / example | Train time |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.status != "ok":
            lines.append(f"| {r.backend} | {r.model} | failed: {r.error} | | | | | |")
            continue
        headline, _ = _headline(spec, r.test_metrics)
        t = r.threshold
        kept = f"{t.get('test_coverage', 0):.1%}" if r.promotable else "0% (never reaches target)"
        lines.append(
            f"| {r.backend} | {r.model} | {headline} | {kept} | {t.get('test_accuracy', 0):.1%} | "
            f"{r.ece_raw:.3f} → {r.ece_calibrated:.3f} | {r.ms_per_example:.1f} | "
            f"{r.train.get('train_seconds', 0):.0f}s |"
        )

    lines += ["", "## Recommendation", ""]
    if best and best.promotable:
        t = best.threshold
        lines.append(
            f"Use **{best.backend}** ({best.model}). At confidence ≥ {t['threshold']:.3f} it answers "
            f"{t['test_coverage']:.1%} of test requests with {t['test_accuracy']:.1%} accuracy; "
            f"the other {1 - t['test_coverage']:.1%} go to the teacher."
        )
    elif best:
        lines.append(
            f"No student reaches {target:.0%} accuracy on any confident slice yet. Best so far: "
            f"**{best.backend}**. Next steps: more or cleaner data, a larger base model, or a "
            "lower target."
        )
    else:
        lines.append("Every candidate failed to train; see errors above.")

    for r in results:
        if r.status != "ok":
            continue
        lines += ["", f"## {r.backend} details", ""]
        for k, v in r.test_metrics.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
            elif v:
                lines.append(f"- {k}: {v}")
        lines.append(f"- calibration: {r.calibration.get('method')}")
        if r.sweep:
            lines.append("- sweep (val score per setting):")
            lines += [f"  - {t['params']}: {t['val_score']:.4f}" for t in r.sweep]
        lines.append(f"- saved to: `{r.model_dir}`")

    path = run_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
