import pytest
from pydantic import ValidationError

from distiller.spec import TaskSpec, TaskType
from tests.conftest import write_task


def _spec(**kw):
    base = {"name": "t", "type": "classification", "data": {"path": "d.jsonl"}}
    base.update(kw)
    return TaskSpec.model_validate(base)


def test_defaults():
    spec = _spec()
    assert spec.type == TaskType.classification
    assert spec.data.val_fraction == 0.1 and spec.data.test_fraction == 0.1
    assert spec.target.min_accuracy == 0.95
    assert spec.student.candidates is None
    assert spec.run_dir.name == "t"


def test_load_resolves_paths_relative_to_task_file(tmp_path):
    path = write_task(
        tmp_path / "proj",
        [{"input": "a", "output": "x"}],
        data={"path": "data.jsonl", "synthetic_path": "syn.jsonl"},
        output_dir="../runs",
    )
    spec = TaskSpec.load(path)
    base = (tmp_path / "proj").resolve()
    assert spec.data.path == base / "data.jsonl"
    assert spec.data.synthetic_path == base / "syn.jsonl"
    assert spec.output_dir == base / ".." / "runs"
    assert spec.run_dir == base / ".." / "runs" / "demo"


def test_load_keeps_absolute_paths(tmp_path):
    abs_data = tmp_path / "elsewhere.jsonl"
    path = write_task(tmp_path / "p", [], data={"path": str(abs_data)}, output_dir=str(tmp_path))
    spec = TaskSpec.load(path)
    assert spec.data.path == abs_data
    assert spec.output_dir == tmp_path


@pytest.mark.parametrize("name", ["Upper", "has space", "dots.x", ""])
def test_bad_names_rejected(name):
    with pytest.raises(ValidationError):
        _spec(name=name)


def test_labels_only_for_classification():
    with pytest.raises(ValidationError, match="labels only apply"):
        _spec(type="extraction", labels=["a"])


def test_schema_only_for_extraction():
    with pytest.raises(ValidationError, match="output_schema only applies"):
        _spec(type="classification", output_schema={"type": "object"})


def test_duplicate_labels_rejected():
    with pytest.raises(ValidationError, match="duplicates"):
        _spec(labels=["a", "b", "a"])


@pytest.mark.parametrize(
    "field,value", [("val_fraction", 0.5), ("test_fraction", 0), ("near_dup_threshold", 1.5)]
)
def test_data_bounds(field, value):
    with pytest.raises(ValidationError):
        _spec(data={"path": "d.jsonl", field: value})


def test_unknown_type_rejected():
    with pytest.raises(ValidationError):
        _spec(type="regression")


def test_example_task_files_load():
    from pathlib import Path

    root = Path(__file__).parents[2] / "examples"
    b = TaskSpec.load(root / "banking77" / "task.yaml")
    t = TaskSpec.load(root / "transactions" / "task.yaml")
    assert b.type == TaskType.classification and b.student.candidates == ["encoder", "llm-lora"]
    assert t.type == TaskType.extraction and "direction" in t.output_schema["properties"]
    assert t.data.synthetic_path.name == "synthetic.jsonl"
