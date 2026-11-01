import pytest

from distiller.data.validate import make_validator
from distiller.spec import TaskSpec

SCHEMA = {
    "type": "object",
    "required": ["amount"],
    "additionalProperties": False,
    "properties": {"amount": {"type": ["number", "null"]}, "dir": {"enum": ["in", "out"]}},
}


def spec(**kw):
    return TaskSpec.model_validate({"name": "t", "data": {"path": "x"}, **kw})


def test_classification_with_labels():
    check = make_validator(spec(type="classification", labels=["a", "b"]))
    assert check("a") is None
    assert "unknown label" in check("c")
    assert check("") is not None
    assert check(3) is not None


def test_classification_without_labels_accepts_any_string():
    check = make_validator(spec(type="classification"))
    assert check("anything") is None
    assert check(None) is not None


@pytest.mark.parametrize(
    "output,ok",
    [
        ({"amount": 1.5}, True),
        ({"amount": None, "dir": "in"}, True),
        ({"dir": "in"}, False),  # missing required
        ({"amount": "12"}, False),  # wrong type
        ({"amount": 1, "dir": "sideways"}, False),
        ({"amount": 1, "extra": 1}, False),
        ("{}", False),
        ([1], False),
    ],
)
def test_extraction_schema(output, ok):
    check = make_validator(spec(type="extraction", output_schema=SCHEMA))
    assert (check(output) is None) == ok


def test_extraction_without_schema_needs_object():
    check = make_validator(spec(type="extraction"))
    assert check({"any": 1}) is None
    assert check("text") is not None


def test_generation():
    check = make_validator(spec(type="generation"))
    assert check("hello") is None
    assert check("") is not None
    assert check({"a": 1}) is not None
