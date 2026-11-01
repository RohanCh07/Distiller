import json

from distiller.data.dedup import normalize
from distiller.data.pipeline import load_prepared, prepare
from distiller.data.report import DataReport
from tests.conftest import classification_rows, write_jsonl


def test_prepare_cleans_and_reports(make_task):
    rows = classification_rows(80)
    rows += [rows[0], rows[1]]  # exact duplicates
    rows += [
        {
            "input": rows[2]["input"],
            "output": "top_up" if rows[2]["output"] != "top_up" else "exchange_rate",
        }
    ]
    rows += [{"input": "bad label row", "output": "nope"}, {"input": "", "output": "top_up"}]
    spec = make_task(rows, labels=["card_arrival", "top_up", "exchange_rate", "unused"])
    prepared = prepare(spec)
    r = prepared.report

    assert r.total_rows == len(rows)
    assert r.invalid_outputs == 2  # unknown label + empty input
    assert r.exact_duplicates == 3  # two copies + the conflicting row collapse away
    assert r.conflicting_inputs == 1
    assert r.usable_examples == 240
    assert r.missing_labels == ["unused"]
    assert sum(r.split_sizes.values()) == r.usable_examples
    assert prepared.labels == ["card_arrival", "top_up", "exchange_rate", "unused"]
    assert any("never appear" in w for w in r.warnings)

    data = spec.run_dir / "data"
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "data_report.md", "data_report.json"):
        assert (data / name).exists()
    train, val, test = load_prepared(spec)
    assert len(train) == len(prepared.train) and len(test) == len(prepared.test)
    assert json.loads((data / "data_report.json").read_text())["usable_examples"] == 240


def test_no_leak_of_inputs_between_splits(make_task):
    spec = make_task(classification_rows(80))
    p = prepare(spec)
    train = {normalize(e.input) for e in p.train}
    assert not train & {normalize(e.input) for e in p.val + p.test}


def test_labels_taken_from_data_and_warnings(make_task):
    rows = classification_rows(5)  # 15 rows: too few, all labels rare
    rows += [{"input": f"dominant {i}", "output": "top_up"} for i in range(40)]
    p = prepare(make_task(rows))
    assert p.labels == sorted({r["output"] for r in rows})
    joined = " ".join(p.report.warnings)
    assert "No labels in task.yaml" in joined
    assert "fewer than 10 examples" in joined
    assert "over half the data" in joined
    assert "usable examples" in joined


def test_synthetic_rows_train_only_and_filtered(make_task, tmp_path):
    rows = classification_rows(60)
    spec = make_task(rows, data={"path": "data.jsonl", "synthetic_path": "syn.jsonl"})
    # Peek at the split first, without synthetic data, to craft synthetic rows.
    spec.data.synthetic_path = None
    first = prepare(spec)
    test_text = first.test[0].input
    synthetic = [
        {"input": "brand new synthetic card question", "output": "card_arrival"},
        {"input": "another synthetic exchange question", "output": "exchange_rate"},
        {"input": rows[0]["input"], "output": rows[0]["output"]},  # duplicates a real row
        {"input": test_text + "!", "output": first.test[0].output},  # near-copy of a test row
        {"input": "synthetic with an unknown label", "output": "not_a_label"},
    ]
    syn_path = write_jsonl(spec.data.path.parent / "syn.jsonl", synthetic)
    spec.data.synthetic_path = syn_path
    p = prepare(spec)

    syn_in_train = [e for e in p.train if e.meta.get("synthetic")]
    assert {e.input for e in syn_in_train} == {
        "brand new synthetic card question",
        "another synthetic exchange question",
    }
    assert not any(e.meta.get("synthetic") for e in p.val + p.test)
    assert [e.id for e in p.test] == [
        e.id for e in first.test
    ]  # synthetic data never moves the split
    assert p.report.synthetic_used == 2 and p.report.synthetic_dropped == 3


def test_missing_synthetic_file_is_a_warning(make_task):
    spec = make_task(
        classification_rows(30), data={"path": "data.jsonl", "synthetic_path": "missing.jsonl"}
    )
    p = prepare(spec)
    assert any("does not exist" in w for w in p.report.warnings)


def test_extraction_pipeline(make_task):
    schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}
    rows = [{"input": f"number {i}", "output": {"n": i}} for i in range(50)]
    rows.append({"input": "broken", "output": {"n": "x"}})
    rows.append({"input": "string json", "output": '{"n": 99}'})
    p = prepare(make_task(rows, type="extraction", output_schema=schema))
    assert p.labels is None
    assert p.report.invalid_outputs == 1
    assert p.report.usable_examples == 51
    assert p.report.label_counts == {}


def test_report_markdown():
    r = DataReport(
        task="x", total_rows=10, usable_examples=8, split_sizes={"train": 6, "val": 1, "test": 1}
    )
    r.label_counts = {"b": 5, "a": 3}
    r.warnings = ["careful"]
    r.samples = {"invalid": ["row 1: bad"], "conflicts": []}
    md = r.to_markdown()
    assert "# Data report: x" in md
    assert "| **Usable examples** | **8** |" in md
    assert "Split: train 6, val 1, test 1" in md
    assert "- careful" in md
    assert md.index("| a | 3 |") < md.index("| b | 5 |")  # sorted by count
    assert "### Samples: invalid" in md and "### Samples: conflicts" not in md


def test_report_save(tmp_path):
    DataReport(task="x").save(tmp_path / "out")
    assert json.loads((tmp_path / "out" / "data_report.json").read_text())["task"] == "x"
    assert (tmp_path / "out" / "data_report.md").read_text().startswith("# Data report: x")
