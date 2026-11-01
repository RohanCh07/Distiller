import pytest

from distiller.eval.metrics import (
    classification_metrics,
    extraction_metrics,
    flatten,
    generation_metrics,
    is_correct,
    task_metrics,
    token_f1,
)
from distiller.spec import TaskSpec


def spec(type_, **kw):
    return TaskSpec.model_validate({"name": "t", "type": type_, "data": {"path": "x"}, **kw})


def test_token_f1():
    assert token_f1("the cat sat", "the cat sat") == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("dog", "cat") == 0.0
    # pred has 2 tokens, gold 4, 2 shared: P=1, R=0.5, F1=2/3
    assert token_f1("The Cat!", "the cat sat down") == pytest.approx(2 / 3)


def test_flatten():
    assert flatten({"a": {"b": 1}, "c": [3, 2], "d": " Hi "}) == {
        "a.b": 1,
        "c": repr(sorted(["3", "2"])),
        "d": "hi",
    }
    assert flatten({"c": [1, 2]}) == flatten({"c": [2, 1]})  # lists compared as multisets


def test_is_correct_by_type():
    assert is_correct(spec("classification"), "a", "a")
    assert not is_correct(spec("classification"), "a", "b")
    assert not is_correct(spec("classification"), None, "a")
    ext = spec("extraction")
    assert is_correct(ext, {"m": "Swiggy ", "n": 1}, {"n": 1, "m": "swiggy"})
    assert not is_correct(ext, {"m": "x"}, {"m": "x", "n": None})
    assert not is_correct(ext, "not a dict", {"m": "x"})
    gen = spec("generation", target={"generation_f1_threshold": 0.6})
    assert is_correct(gen, "the cat sat", "the cat sat down")  # F1 0.857
    assert not is_correct(gen, "a dog", "the cat sat down")


def test_classification_metrics_hand_computed():
    golds = ["a", "a", "a", "b", "b", "c"]
    preds = ["a", "a", "b", "b", "b", "zzz"]
    m = classification_metrics(preds, golds, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(4 / 6)
    # a: P=1 R=2/3 F1=0.8 ; b: P=2/3 R=1 F1=0.8 ; c: P=0 R=0 F1=0
    assert m["macro_f1"] == pytest.approx((0.8 + 0.8 + 0) / 3)
    assert m["invalid_rate"] == pytest.approx(1 / 6)
    assert m["worst_labels"][0] == ("c", 0.0)
    assert {"gold": "a", "predicted": "b", "count": 1} in m["top_confusions"]


def test_classification_macro_f1_ignores_labels_without_support():
    m = classification_metrics(["a", "a"], ["a", "a"], ["a", "never"])
    assert m["macro_f1"] == 1.0


def test_extraction_metrics():
    golds = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
    preds = [{"a": 1, "b": "X"}, {"a": 2, "b": "wrong"}, None]
    m = extraction_metrics(preds, golds)
    assert m["exact_match"] == pytest.approx(1 / 3)
    # tp=3 (a,b ; a), fp=1 (b), fn=2 (both fields of the missing prediction)
    assert m["field_precision"] == pytest.approx(3 / 4)
    assert m["field_recall"] == pytest.approx(3 / 5)
    assert m["invalid_json_rate"] == pytest.approx(1 / 3)
    assert dict(m["worst_fields"])["b"] == 2


def test_generation_metrics():
    m = generation_metrics(["Hello there", None], ["hello there", "bye"])
    assert m["exact_match"] == 0.5
    assert m["token_f1"] == pytest.approx(0.5)


def test_task_metrics_dispatch():
    assert "macro_f1" in task_metrics(spec("classification"), ["a"], ["a"], None)
    assert "field_f1" in task_metrics(spec("extraction"), [{}], [{}], None)
    assert "token_f1" in task_metrics(spec("generation"), ["x"], ["x"], None)


def test_empty_inputs_do_not_divide_by_zero():
    assert classification_metrics([], [], ["a"])["accuracy"] == 0
    assert extraction_metrics([], [])["field_f1"] == 0
    assert generation_metrics([], [])["token_f1"] == 0
