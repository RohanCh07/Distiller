"""prepare(spec): raw file -> clean, split dataset on disk + data report."""

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from distiller.data.dedup import near_duplicate_groups, normalize, remove_exact_duplicates
from distiller.data.io import Example, load_examples, read_jsonl, write_jsonl
from distiller.data.report import DataReport
from distiller.data.split import split_examples
from distiller.data.validate import make_validator
from distiller.spec import TaskSpec, TaskType

MIN_USABLE = 200  # below this, training a student is not worth it


@dataclass
class PreparedData:
    train: list[Example]
    val: list[Example]
    test: list[Example]
    labels: list[str] | None
    report: DataReport


def data_dir(spec: TaskSpec) -> Path:
    return spec.run_dir / "data"


def prepare(spec: TaskSpec) -> PreparedData:
    report = DataReport(task=spec.name)
    loaded = load_examples(spec)
    report.total_rows = loaded.total_rows
    report.unreadable_rows = len(loaded.errors)
    report.samples["unreadable"] = loaded.errors[:5]

    check = make_validator(spec)
    valid, invalid_samples = [], []
    for ex in loaded.examples:
        err = check(ex.output)
        if err is None and ex.input:
            valid.append(ex)
        elif len(invalid_samples) < 5:
            invalid_samples.append(f"{ex.input[:60]!r}: {err or 'empty input'}")
    report.invalid_outputs = len(loaded.examples) - len(valid)
    report.samples["invalid"] = invalid_samples

    dedup = remove_exact_duplicates(valid)
    examples = dedup.examples
    report.exact_duplicates = dedup.exact_duplicates
    report.conflicting_inputs = dedup.conflicts
    report.samples["conflicts"] = dedup.conflict_samples

    groups = near_duplicate_groups(examples, spec.data.near_dup_threshold, seed=spec.data.seed)
    sizes = Counter(groups)
    report.near_duplicate_groups = sum(1 for s in sizes.values() if s > 1)
    report.examples_in_near_duplicate_groups = sum(s for s in sizes.values() if s > 1)
    report.usable_examples = len(examples)

    if examples:
        lengths = np.array([len(ex.input) for ex in examples])
        report.input_chars = {
            "p50": int(np.percentile(lengths, 50)),
            "p95": int(np.percentile(lengths, 95)),
            "max": int(lengths.max()),
        }

    labels = None
    if spec.type == TaskType.classification:
        counts = Counter(ex.output for ex in examples)
        report.label_counts = dict(counts)
        labels = spec.labels or sorted(counts)
        report.missing_labels = [lbl for lbl in labels if lbl not in counts]
        report.rare_labels = sorted(
            lbl for lbl, c in counts.items() if c < spec.data.min_examples_per_label
        )
        if not spec.labels:
            report.warnings.append(
                f"No labels in task.yaml; using the {len(labels)} labels found in the data."
            )
        if report.rare_labels:
            report.warnings.append(
                f"{len(report.rare_labels)} labels have fewer than "
                f"{spec.data.min_examples_per_label} examples; expect weak accuracy on them."
            )
        if report.missing_labels:
            report.warnings.append(
                f"{len(report.missing_labels)} labels in task.yaml never appear in the data."
            )
        if counts and max(counts.values()) > 0.5 * len(examples):
            report.warnings.append(
                "One label covers over half the data; watch macro-F1, not just accuracy."
            )

    if report.conflicting_inputs > 0.02 * max(len(examples), 1):
        report.warnings.append(
            "Over 2% of inputs had conflicting outputs: the teacher or labels may be inconsistent."
        )
    if len(examples) < MIN_USABLE:
        report.warnings.append(
            f"Only {len(examples)} usable examples; at least {MIN_USABLE} are needed, "
            "ideally a few thousand."
        )

    train, val, test = split_examples(
        examples,
        groups,
        spec.data.val_fraction,
        spec.data.test_fraction,
        spec.data.seed,
        stratify=spec.type == TaskType.classification,
    )
    if spec.data.synthetic_path:
        train = train + _synthetic_for_training(spec, check, labels, examples, val + test, report)
        random.Random(spec.data.seed).shuffle(train)
    report.split_sizes = {"train": len(train), "val": len(val), "test": len(test)}

    out = data_dir(spec)
    for name, part in (("train", train), ("val", val), ("test", test)):
        write_jsonl(out / f"{name}.jsonl", part)
    report.save(out)
    return PreparedData(train, val, test, labels, report)


def _synthetic_for_training(spec, check, labels, real, held_out, report) -> list[Example]:
    """Teacher-generated examples go to train only, never near-copies of val/test rows."""
    path = spec.data.synthetic_path
    if not path.exists():
        report.warnings.append(f"synthetic_path {path} does not exist; skipped.")
        return []
    loaded = load_examples(spec, path, synthetic=True)
    allowed = set(labels) if labels else None
    candidates = [
        ex
        for ex in loaded.examples
        if ex.input and check(ex.output) is None and (allowed is None or ex.output in allowed)
    ]
    real_inputs = {normalize(ex.input) for ex in real}
    candidates = [
        ex
        for ex in remove_exact_duplicates(candidates).examples
        if normalize(ex.input) not in real_inputs
    ]
    kept = candidates
    if candidates and held_out:
        groups = near_duplicate_groups(
            held_out + candidates, spec.data.near_dup_threshold, seed=spec.data.seed
        )
        held_groups = set(groups[: len(held_out)])
        kept = [ex for ex, g in zip(candidates, groups[len(held_out) :]) if g not in held_groups]
    report.synthetic_used = len(kept)
    report.synthetic_dropped = loaded.total_rows - len(kept)
    return kept


def load_prepared(spec: TaskSpec) -> tuple[list[Example], list[Example], list[Example]]:
    d = data_dir(spec)
    return tuple(read_jsonl(d / f"{n}.jsonl") for n in ("train", "val", "test"))
