"""Task-type-aware quality metrics, plus a per-example "was it right?" signal.

The per-example signal is what calibration and the coverage curve are built on, so it
must match what "good enough to replace the teacher" means for each task type.
"""

import re
from collections import Counter
from typing import Any

from distiller.spec import TaskSpec, TaskType

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_f1(pred: str, gold: str) -> float:
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """{"a": {"b": 1}, "c": [2, 3]} -> {"a.b": 1, "c": "[2, 3]"}; lists compared as a whole."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
        return out
    key = prefix.rstrip(".")
    if isinstance(obj, list):
        return {key: repr(sorted(map(repr, obj)))}
    if isinstance(obj, str):
        return {key: obj.strip().lower()}
    return {key: obj}


def is_correct(spec: TaskSpec, pred: Any, gold: Any) -> bool:
    if pred is None:
        return False
    if spec.type == TaskType.classification:
        return pred == gold
    if spec.type == TaskType.extraction:
        return isinstance(pred, dict) and flatten(pred) == flatten(gold)
    return token_f1(str(pred), str(gold)) >= spec.target.generation_f1_threshold


def classification_metrics(preds: list[Any], golds: list[str], labels: list[str]) -> dict[str, Any]:
    per_label = {}
    for lbl in labels:
        tp = sum(p == lbl and g == lbl for p, g in zip(preds, golds))
        fp = sum(p == lbl and g != lbl for p, g in zip(preds, golds))
        fn = sum(p != lbl and g == lbl for p, g in zip(preds, golds))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[lbl] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    present = [v for v in per_label.values() if v["support"] > 0]
    confusions = Counter((g, p) for p, g in zip(preds, golds) if p != g)
    return {
        "accuracy": sum(p == g for p, g in zip(preds, golds)) / max(len(golds), 1),
        "macro_f1": sum(v["f1"] for v in present) / max(len(present), 1),
        "invalid_rate": sum(p not in per_label for p in preds) / max(len(preds), 1),
        "worst_labels": sorted(
            ((k, round(v["f1"], 3)) for k, v in per_label.items() if v["support"]),
            key=lambda kv: kv[1],
        )[:5],
        "top_confusions": [
            {"gold": g, "predicted": p, "count": c} for (g, p), c in confusions.most_common(5)
        ],
    }


def extraction_metrics(preds: list[Any], golds: list[dict]) -> dict[str, Any]:
    tp = fp = fn = 0
    field_errors: Counter = Counter()
    for pred, gold in zip(preds, golds):
        g = flatten(gold)
        p = flatten(pred) if isinstance(pred, dict) else {}
        for k, v in p.items():
            if g.get(k, object()) == v:
                tp += 1
            else:
                fp += 1
                field_errors[k] += 1
        missing = [k for k in g if k not in p]
        fn += len(missing)
        field_errors.update(missing)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "exact_match": sum(
            isinstance(p, dict) and flatten(p) == flatten(g) for p, g in zip(preds, golds)
        )
        / max(len(golds), 1),
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "invalid_json_rate": sum(not isinstance(p, dict) for p in preds) / max(len(preds), 1),
        "worst_fields": field_errors.most_common(5),
    }


def generation_metrics(preds: list[Any], golds: list[str]) -> dict[str, Any]:
    f1s = [token_f1(str(p or ""), g) for p, g in zip(preds, golds)]
    return {
        "exact_match": sum(
            (p or "").strip().lower() == g.strip().lower() for p, g in zip(preds, golds)
        )
        / max(len(golds), 1),
        "token_f1": sum(f1s) / max(len(f1s), 1),
    }


def task_metrics(spec: TaskSpec, preds: list[Any], golds: list[Any], labels: list[str] | None):
    if spec.type == TaskType.classification:
        return classification_metrics(preds, golds, labels or sorted(set(golds)))
    if spec.type == TaskType.extraction:
        return extraction_metrics(preds, golds)
    return generation_metrics(preds, golds)
