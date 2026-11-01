import json

from distiller.data.io import (
    Example,
    input_from_messages,
    load_examples,
    read_jsonl,
    serialize_output,
    write_jsonl,
)
from distiller.spec import TaskSpec


def _spec(tmp_path, type_="classification", **data):
    return TaskSpec.model_validate(
        {"name": "t", "type": type_, "data": {"path": str(tmp_path / "d.jsonl"), **data}}
    )


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_jsonl_input_output(tmp_path):
    _write(
        tmp_path / "d.jsonl",
        [
            json.dumps({"input": "  hi  ", "output": " a "}),
            "",
            json.dumps({"input": "yo", "output": "b"}),
        ],
    )
    res = load_examples(_spec(tmp_path))
    assert res.total_rows == 2  # blank lines are not rows
    assert [(e.input, e.output) for e in res.examples] == [("hi", "a"), ("yo", "b")]
    assert res.errors == []
    assert len({e.id for e in res.examples}) == 2


def test_custom_field_names(tmp_path):
    _write(tmp_path / "d.jsonl", [json.dumps({"text": "x", "label": "y"})])
    res = load_examples(_spec(tmp_path, input_field="text", output_field="label"))
    assert res.examples[0].input == "x" and res.examples[0].output == "y"


def test_bad_rows_are_counted_not_fatal(tmp_path):
    _write(
        tmp_path / "d.jsonl",
        ["{not json", json.dumps({"input": "x"}), json.dumps({"input": "ok", "output": "a"})],
    )
    res = load_examples(_spec(tmp_path))
    assert res.total_rows == 3
    assert len(res.examples) == 1
    assert any("invalid JSON" in e for e in res.errors)
    assert any("output" in e for e in res.errors)


def test_csv(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text('input,output\n"hello, world",greet\nbye,leave\n', encoding="utf-8")
    spec = _spec(tmp_path)
    spec.data.path = path
    res = load_examples(spec)
    assert [(e.input, e.output) for e in res.examples] == [
        ("hello, world", "greet"),
        ("bye", "leave"),
    ]


def test_chat_messages_rows(tmp_path):
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "label_x"},
    ]
    _write(
        tmp_path / "d.jsonl",
        [
            json.dumps({"messages": msgs}),
            json.dumps({"messages": msgs[:2]}),  # no final assistant turn
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": "s"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ),
        ],
    )
    res = load_examples(_spec(tmp_path))
    assert len(res.examples) == 1
    assert res.examples[0].input == "first\n\nsecond"
    assert res.examples[0].output == "label_x"
    assert any("assistant" in e for e in res.errors)
    assert any("no user turn" in e for e in res.errors)


def test_extraction_output_strings_are_parsed(tmp_path):
    _write(
        tmp_path / "d.jsonl",
        [
            json.dumps({"input": "a", "output": '{"x": 1}'}),
            json.dumps({"input": "b", "output": {"x": 2}}),
            json.dumps({"input": "c", "output": "not json"}),
        ],
    )
    res = load_examples(_spec(tmp_path, "extraction"))
    assert [e.output for e in res.examples] == [{"x": 1}, {"x": 2}]
    assert any("not valid JSON" in e for e in res.errors)


def test_synthetic_marking(tmp_path):
    _write(
        tmp_path / "d.jsonl",
        [
            json.dumps({"input": "a", "output": "x", "synthetic": True}),
            json.dumps({"input": "b", "output": "y"}),
        ],
    )
    res = load_examples(_spec(tmp_path))
    assert res.examples[0].meta == {"synthetic": True} and res.examples[0].id.startswith("syn")
    assert res.examples[1].meta == {}
    forced = load_examples(_spec(tmp_path), synthetic=True)
    assert all(e.meta["synthetic"] for e in forced.examples)


def test_input_from_messages_only_user_turns():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": 5},
    ]
    assert input_from_messages(msgs) == "a\n\n5"
    assert input_from_messages([]) == ""


def test_to_messages_and_serialize():
    ex = Example("1", "hi", {"a": "é"})
    assert ex.to_messages("inst") == [
        {"role": "system", "content": "inst"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": '{"a": "é"}'},
    ]
    assert ex.to_messages("")[0]["role"] == "user"
    assert serialize_output("plain") == "plain"


def test_jsonl_roundtrip(tmp_path):
    exs = [Example("1", "a", "x"), Example("2", "b", {"k": [1, 2]}, {"synthetic": True})]
    write_jsonl(tmp_path / "sub" / "out.jsonl", exs)
    assert read_jsonl(tmp_path / "sub" / "out.jsonl") == exs
