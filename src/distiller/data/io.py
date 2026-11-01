"""Load examples from JSONL/CSV into one internal shape."""

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from distiller.spec import TaskSpec, TaskType


@dataclass
class Example:
    id: str
    input: str
    output: Any  # str for classification/generation, dict for extraction
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Example":
        return cls(id=d["id"], input=d["input"], output=d["output"], meta=d.get("meta", {}))

    def to_messages(self, instruction: str) -> list[dict[str, str]]:
        """Chat format, the same shape the gateway will log in M6."""
        messages = [{"role": "system", "content": instruction}] if instruction else []
        messages.append({"role": "user", "content": self.input})
        messages.append({"role": "assistant", "content": serialize_output(self.output)})
        return messages


@dataclass
class LoadResult:
    examples: list[Example]
    errors: list[str]  # one line per row that could not be read
    total_rows: int


def serialize_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


def _row_id(input_text: str, index: int) -> str:
    return hashlib.blake2b(f"{index}:{input_text}".encode(), digest_size=6).hexdigest()


def _iter_rows(path: Path, fmt: str):
    if fmt == "auto":
        fmt = "csv" if path.suffix.lower() == ".csv" else "jsonl"
    with path.open(encoding="utf-8", newline="") as f:
        if fmt == "csv":
            for i, row in enumerate(csv.DictReader(f)):
                yield i, row, None
        else:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    yield i, json.loads(line), None
                except json.JSONDecodeError as e:
                    yield i, None, f"row {i}: invalid JSON ({e.msg})"


def input_from_messages(messages: list[dict]) -> str:
    """The student's input for a chat request: all user turns, joined.

    Used both when reading logged traffic and when the gateway serves a request, so
    training and serving always see the same input text.
    """
    return "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")


def _from_messages(messages: list[dict]) -> tuple[str, str]:
    """Chat-format row: last assistant turn is the output, everything before is the input."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("messages must end with an assistant turn")
    input_text = input_from_messages(messages[:-1])
    if not input_text:
        raise ValueError("messages have no user turn")
    return input_text, messages[-1]["content"]


def load_examples(spec: TaskSpec, path: Path | None = None, synthetic: bool = False) -> LoadResult:
    examples, errors, total = [], [], 0
    path = path or spec.data.path
    for i, row, err in _iter_rows(path, spec.data.format):
        total += 1
        if err:
            errors.append(err)
            continue
        try:
            if "messages" in row:
                input_text, output = _from_messages(row["messages"])
            else:
                input_text, output = row[spec.data.input_field], row[spec.data.output_field]
        except (KeyError, ValueError, TypeError) as e:
            errors.append(f"row {i}: {e}")
            continue
        if spec.type == TaskType.extraction and isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                errors.append(f"row {i}: output is not valid JSON")
                continue
        if isinstance(output, str):
            output = output.strip()
        input_text = str(input_text).strip()
        meta = {"synthetic": True} if synthetic or row.get("synthetic") else {}
        prefix = "syn" if meta else ""
        examples.append(
            Example(id=prefix + _row_id(input_text, i), input=input_text, output=output, meta=meta)
        )
    return LoadResult(examples=examples, errors=errors, total_rows=total)


def write_jsonl(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Example]:
    with path.open(encoding="utf-8") as f:
        return [Example.from_dict(json.loads(line)) for line in f if line.strip()]
