"""Coverage vs accuracy: how much traffic can the student keep at the target accuracy?

The threshold is picked on validation using the Wilson lower bound of accuracy, so a
lucky streak on a few examples cannot pass. It is then applied unchanged to the test
split, which gives an honest estimate of what will happen on real traffic.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

MIN_KEPT = 30  # never trust a threshold that keeps fewer validation examples than this


def wilson_lower_bound(correct: int, n: int, z: float = 1.645) -> float:
    """One-sided 95% lower bound on accuracy."""
    if n == 0:
        return 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


def curve(conf: list[float], correct: list[bool], points: int = 20) -> list[dict[str, float]]:
    order = np.argsort(conf)[::-1]
    c = np.asarray(correct, dtype=float)[order]
    s = np.asarray(conf, dtype=float)[order]
    n = len(c)
    out = []
    for frac in np.linspace(1 / points, 1, points):
        k = max(1, int(round(frac * n)))
        out.append(
            {"coverage": k / n, "accuracy": float(c[:k].mean()), "threshold": float(s[k - 1])}
        )
    return out


@dataclass
class ThresholdChoice:
    threshold: float  # inf means "never serve from the student"
    val_coverage: float
    val_accuracy: float
    val_accuracy_lower_bound: float
    test_coverage: float = 0.0
    test_accuracy: float = 0.0
    test_examples_kept: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        if math.isinf(self.threshold):
            d["threshold"] = None
        return d


def choose_threshold(
    conf: list[float], correct: list[bool], min_accuracy: float
) -> ThresholdChoice:
    order = np.argsort(conf)[::-1]
    s = np.asarray(conf, dtype=float)[order]
    c = np.asarray(correct, dtype=int)[order]
    cum = np.cumsum(c)
    best = ThresholdChoice(math.inf, 0.0, 0.0, 0.0)
    n = len(c)
    for k in range(MIN_KEPT, n + 1):
        # Only cut between distinct scores, otherwise ties would be split arbitrarily.
        if k < n and s[k] == s[k - 1]:
            continue
        lb = wilson_lower_bound(int(cum[k - 1]), k)
        if lb >= min_accuracy:
            best = ThresholdChoice(float(s[k - 1]), k / n, cum[k - 1] / k, lb)
    return best


def apply_threshold(
    choice: ThresholdChoice, conf: list[float], correct: list[bool]
) -> ThresholdChoice:
    kept = [ok for s, ok in zip(conf, correct) if s >= choice.threshold]
    choice.test_examples_kept = len(kept)
    choice.test_coverage = len(kept) / max(len(conf), 1)
    choice.test_accuracy = sum(kept) / len(kept) if kept else 0.0
    return choice
