"""Make student confidence mean what it says: 0.9 confidence should be right ~90% of the time.

Encoder students expose logits, so we fit a single temperature (keeps the ranking,
fixes over-confidence). LLM students only give a sequence score, so we fit isotonic
regression from score to observed accuracy. Both are fit on the validation split only.
"""

from dataclasses import dataclass, field

import numpy as np


def expected_calibration_error(conf: list[float], correct: list[bool], bins: int = 15) -> float:
    conf_a, corr_a = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    if len(conf_a) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf_a > lo) & (conf_a <= hi)
        if mask.any():
            ece += mask.mean() * abs(conf_a[mask].mean() - corr_a[mask].mean())
    return float(ece)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: list[list[float]], gold_idx: list[int]) -> float:
    """Grid search the temperature that minimises validation negative log-likelihood."""
    x, y = np.asarray(logits, dtype=float), np.asarray(gold_idx)
    best_t, best_nll = 1.0, float("inf")
    for t in np.exp(np.linspace(np.log(0.05), np.log(10), 200)):
        p = _softmax(x, t)[np.arange(len(y)), y]
        nll = -np.log(np.clip(p, 1e-12, 1)).mean()
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def fit_isotonic(conf: list[float], correct: list[bool]) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators: monotone map from raw score to accuracy."""
    order = np.argsort(conf)
    xs = np.asarray(conf, dtype=float)[order]
    ys = np.asarray(correct, dtype=float)[order]
    blocks = [[y, 1.0, x, x] for x, y in zip(xs, ys)]  # mean, weight, x_min, x_max
    merged: list[list[float]] = []
    for b in blocks:
        merged.append(b)
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            m2, m1 = merged.pop(), merged.pop()
            w = m1[1] + m2[1]
            merged.append([(m1[0] * m1[1] + m2[0] * m2[1]) / w, w, m1[2], m2[3]])
    knots_x = [(b[2] + b[3]) / 2 for b in merged]
    knots_y = [b[0] for b in merged]
    return knots_x, knots_y


@dataclass
class Calibrator:
    method: str  # "temperature" | "isotonic" | "none"
    temperature: float = 1.0
    knots_x: list[float] = field(default_factory=list)
    knots_y: list[float] = field(default_factory=list)

    def apply(self, conf: list[float], logits: list[list[float]] | None = None) -> list[float]:
        if self.method == "temperature" and logits is not None:
            return _softmax(np.asarray(logits, dtype=float), self.temperature).max(axis=1).tolist()
        if self.method == "isotonic" and self.knots_x:
            return np.interp(conf, self.knots_x, self.knots_y).tolist()
        return list(conf)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "temperature": self.temperature,
            "knots_x": self.knots_x,
            "knots_y": self.knots_y,
        }


def fit_calibrator(
    conf: list[float],
    correct: list[bool],
    logits: list[list[float]] | None = None,
    gold_idx: list[int] | None = None,
) -> Calibrator:
    if len(conf) < 30:
        return Calibrator("none")  # too little validation data to fit anything honestly
    if logits is not None and gold_idx is not None:
        return Calibrator("temperature", temperature=fit_temperature(logits, gold_idx))
    xs, ys = fit_isotonic(conf, correct)
    return Calibrator("isotonic", knots_x=xs, knots_y=ys)
