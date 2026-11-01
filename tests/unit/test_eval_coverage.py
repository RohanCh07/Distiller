import math

import pytest

from distiller.eval.coverage import (
    MIN_KEPT,
    ThresholdChoice,
    apply_threshold,
    choose_threshold,
    curve,
    wilson_lower_bound,
)

Z = 1.645


def test_wilson_known_values():
    assert wilson_lower_bound(0, 0) == 0.0
    # p = 1: lower bound is n / (n + z^2)
    assert wilson_lower_bound(100, 100) == pytest.approx(100 / (100 + Z * Z))
    assert wilson_lower_bound(0, 10) == pytest.approx(0.0, abs=1e-12)
    # 95/100: textbook one-sided 95% Wilson bound ~ 0.9001
    assert wilson_lower_bound(95, 100) == pytest.approx(0.9001, abs=1e-3)


def test_wilson_grows_with_n_and_below_p():
    small, big = wilson_lower_bound(19, 20), wilson_lower_bound(950, 1000)
    assert small < big < 0.95


def test_curve_points():
    conf = [0.9, 0.8, 0.7, 0.6]
    ok = [True, True, False, True]
    pts = curve(conf, ok, points=4)
    assert [p["coverage"] for p in pts] == [0.25, 0.5, 0.75, 1.0]
    assert [p["accuracy"] for p in pts] == pytest.approx([1, 1, 2 / 3, 0.75])
    assert [p["threshold"] for p in pts] == [0.9, 0.8, 0.7, 0.6]


def _data(n_good=200, n_bad=100):
    """High confidences are always right, low ones always wrong."""
    conf = [1 - i / 1000 for i in range(n_good)] + [0.3 - i / 1000 for i in range(n_bad)]
    ok = [True] * n_good + [False] * n_bad
    return conf, ok


def test_choose_threshold_picks_lowest_safe_cut():
    conf, ok = _data()  # already sorted by confidence, no ties
    choice = choose_threshold(conf, ok, 0.95)
    expected_k = max(
        k for k in range(MIN_KEPT, len(conf) + 1) if wilson_lower_bound(sum(ok[:k]), k) >= 0.95
    )
    assert expected_k > 200  # a few wrong answers are tolerated on top of the 200 right ones
    assert choice.val_coverage == expected_k / len(conf)
    assert choice.threshold == conf[expected_k - 1]
    assert choice.val_accuracy == pytest.approx(200 / expected_k)
    assert choice.val_accuracy_lower_bound >= 0.95


def test_choose_threshold_never_trusts_fewer_than_min_kept():
    conf = [0.99] * (MIN_KEPT - 1) + [0.1] * 100
    ok = [True] * (MIN_KEPT - 1) + [False] * 100
    assert math.isinf(choose_threshold(conf, ok, 0.5).threshold)


def test_choose_threshold_unreachable_target():
    conf, ok = [0.9] * 100, [i % 2 == 0 for i in range(100)]
    choice = choose_threshold(conf, ok, 0.95)
    assert math.isinf(choice.threshold) and choice.val_coverage == 0
    assert choice.to_dict()["threshold"] is None


def test_ties_are_not_split():
    # 60 at 0.9 (all right) then 60 tied at 0.8 (half wrong). Any cut inside the tie is illegal.
    conf = [0.9] * 60 + [0.8] * 60
    ok = [True] * 60 + [True, False] * 30
    choice = choose_threshold(conf, ok, 0.9)
    assert choice.threshold == 0.9 and choice.val_coverage == 0.5


def test_apply_threshold():
    choice = ThresholdChoice(0.5, 0.0, 0.0, 0.0)
    out = apply_threshold(choice, [0.9, 0.6, 0.5, 0.4], [True, False, True, True])
    assert out.test_examples_kept == 3
    assert out.test_coverage == 0.75
    assert out.test_accuracy == pytest.approx(2 / 3)
    never = apply_threshold(ThresholdChoice(math.inf, 0, 0, 0), [0.99], [True])
    assert never.test_coverage == 0 and never.test_accuracy == 0
