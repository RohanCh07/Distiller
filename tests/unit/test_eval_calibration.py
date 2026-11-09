import numpy as np
import pytest

from distiller.eval.calibration import (
    Calibrator,
    _softmax,
    expected_calibration_error,
    fit_calibrator,
    fit_isotonic,
    fit_temperature,
)


def test_ece_zero_when_calibrated():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.05, 1, 20000)
    correct = rng.uniform(size=conf.size) < conf
    assert expected_calibration_error(conf.tolist(), correct.tolist()) < 0.02


def test_ece_overconfident():
    conf = [0.99] * 100
    correct = [True] * 60 + [False] * 40
    assert expected_calibration_error(conf, correct) == pytest.approx(0.39)


def test_ece_empty():
    assert expected_calibration_error([], []) == 0.0


def test_ece_hand_computed_two_bins():
    # bin (0.2,0.267]: conf .25 acc 0 ; bin (0.933,1]: conf .95 acc 1  -> 0.5*0.25 + 0.5*0.05
    assert expected_calibration_error([0.25, 0.95], [False, True]) == pytest.approx(0.15)


def _synthetic_logits(true_t: float, n=3000, k=5, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 3, size=(n, k))
    probs = _softmax(logits, true_t)
    gold = np.array([rng.choice(k, p=p) for p in probs])
    return logits, gold


@pytest.mark.parametrize("true_t", [0.5, 2.0])
def test_fit_temperature_recovers_true_temperature(true_t):
    logits, gold = _synthetic_logits(true_t)
    t = fit_temperature(logits.tolist(), gold.tolist())
    assert t == pytest.approx(true_t, rel=0.15)


def test_temperature_scaling_reduces_ece():
    logits, gold = _synthetic_logits(3.0, seed=1)  # model is over-confident at T=1
    raw = _softmax(logits, 1.0)
    correct = (raw.argmax(1) == gold).tolist()
    cal = fit_calibrator(raw.max(1).tolist(), correct, logits.tolist(), gold.tolist())
    assert cal.method == "temperature"
    after = cal.apply(raw.max(1).tolist(), logits.tolist())
    assert (
        expected_calibration_error(after, correct)
        < expected_calibration_error(raw.max(1).tolist(), correct) / 2
    )
    # Temperature keeps the ranking.
    assert (
        np.argsort(after).tolist() == np.argsort(_softmax(logits, cal.temperature).max(1)).tolist()
    )


def test_isotonic_is_monotone_and_fits():
    rng = np.random.default_rng(0)
    raw = rng.uniform(size=2000)
    correct = rng.uniform(size=2000) < raw**2  # true accuracy is raw^2
    xs, ys = fit_isotonic(raw.tolist(), correct.tolist())
    assert all(a <= b for a, b in zip(ys, ys[1:]))
    assert all(a <= b for a, b in zip(xs, xs[1:]))
    mapped = np.interp([0.3, 0.9], xs, ys)
    assert mapped[0] == pytest.approx(0.09, abs=0.08)
    assert mapped[1] == pytest.approx(0.81, abs=0.08)


def test_isotonic_pools_violators():
    xs, ys = fit_isotonic([0.1, 0.2, 0.3], [True, False, True])
    assert ys == pytest.approx([0.5, 1.0])


def test_fit_calibrator_needs_30_examples():
    assert fit_calibrator([0.9] * 29, [True] * 29).method == "none"
    cal = fit_calibrator([0.1 * (i % 10) for i in range(40)], [i % 2 == 0 for i in range(40)])
    assert cal.method == "isotonic"


def test_calibrator_apply_and_roundtrip():
    assert Calibrator("none").apply([0.3]) == [0.3]
    iso = Calibrator("isotonic", knots_x=[0.0, 1.0], knots_y=[0.2, 0.8])
    assert iso.apply([0.5]) == pytest.approx([0.5])
    temp = Calibrator("temperature", temperature=2.0)
    assert temp.apply([0.9], None) == [0.9]  # no logits -> unchanged
    assert Calibrator(**temp.to_dict()) == temp
