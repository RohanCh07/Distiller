"""Data health report: what was dropped, what looks risky, and how the split came out."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class DataReport:
    task: str
    total_rows: int = 0
    unreadable_rows: int = 0
    invalid_outputs: int = 0
    exact_duplicates: int = 0
    conflicting_inputs: int = 0
    near_duplicate_groups: int = 0  # groups with 2+ members
    examples_in_near_duplicate_groups: int = 0
    usable_examples: int = 0
    synthetic_used: int = 0  # teacher-generated, added to train only
    synthetic_dropped: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    rare_labels: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)  # in spec, absent from data
    input_chars: dict[str, int] = field(default_factory=dict)  # p50 / p95 / max
    split_sizes: dict[str, int] = field(default_factory=dict)
    samples: dict[str, list[str]] = field(default_factory=dict)  # example problems per check
    warnings: list[str] = field(default_factory=list)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "data_report.json").write_text(json.dumps(asdict(self), indent=2))
        (directory / "data_report.md").write_text(self.to_markdown(), encoding="utf-8")

    def to_markdown(self) -> str:
        lines = [
            f"# Data report: {self.task}",
            "",
            "| Check | Count |",
            "|---|---|",
            f"| Rows read | {self.total_rows} |",
            f"| Unreadable rows | {self.unreadable_rows} |",
            f"| Invalid outputs | {self.invalid_outputs} |",
            f"| Exact duplicates removed | {self.exact_duplicates} |",
            f"| Inputs with conflicting outputs | {self.conflicting_inputs} |",
            f"| Near-duplicate groups | {self.near_duplicate_groups} "
            f"({self.examples_in_near_duplicate_groups} examples) |",
            f"| **Usable examples** | **{self.usable_examples}** |",
            f"| Synthetic examples added to train | {self.synthetic_used} "
            f"({self.synthetic_dropped} dropped) |",
            "",
            "Split: " + ", ".join(f"{k} {v}" for k, v in self.split_sizes.items()),
            "",
            "Input length (chars): " + ", ".join(f"{k} {v}" for k, v in self.input_chars.items()),
        ]
        if self.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in self.warnings]
        if self.label_counts:
            lines += ["", "## Labels", "", "| Label | Examples |", "|---|---|"]
            lines += [
                f"| {k} | {v} |" for k, v in sorted(self.label_counts.items(), key=lambda kv: kv[1])
            ]
        for check, items in self.samples.items():
            if items:
                lines += ["", f"### Samples: {check}", ""] + [f"- {s}" for s in items]
        return "\n".join(lines) + "\n"
