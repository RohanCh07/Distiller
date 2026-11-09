"""Find distillable tasks in untagged traffic.

Untagged calls are logged as `unassigned:<hash of system prompt>`, so each group is one
repeated prompt. For every group with enough teacher answers we look at the answers:

- mostly JSON objects with the same keys       -> extraction (schema inferred from them)
- few distinct short answers, each repeated     -> classification (labels = those answers)
- short free text                              -> generation
- long free-form answers                       -> not a good fit; reported, not proposed

The result is a proposed task.yaml plus its training data (the logged calls). `adopt`
moves the group's calls to the new task name, so they show up under it everywhere.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from distiller.data.io import input_from_messages
from distiller.storage.db import Store

_STOP = {
    "the", "a", "an", "of", "to", "and", "or", "into", "in", "for", "from", "this", "that",
    "is", "are", "be", "with", "on", "as", "by", "only", "you", "your", "please", "reply",
    "answer", "given", "following", "text", "message", "exactly", "one", "it", "its", "at",
}  # fmt: skip


@dataclass
class Candidate:
    group: str  # unassigned:<hash>
    calls: int
    system_prompt: str
    type: str | None  # classification | extraction | generation | None (not a fit)
    reason: str
    name: str
    labels: list[str] | None = None
    output_schema: dict[str, Any] | None = None
    samples: list[str] = field(default_factory=list)

    @property
    def proposed(self) -> bool:
        return self.type is not None

    def task_yaml(self, data_file: str = "data.jsonl") -> dict[str, Any]:
        spec: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "instruction": self.system_prompt,
            "data": {"path": data_file},
        }
        if self.labels:
            spec["labels"] = self.labels
        if self.output_schema:
            spec["output_schema"] = self.output_schema
        return spec

    @property
    def match_system_prompt(self) -> str:
        """A distinctive substring for distiller.yaml (first line, trimmed)."""
        return self.system_prompt.strip().splitlines()[0][:80] if self.system_prompt.strip() else ""


def suggest_name(prompt: str, taken: set[str]) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", prompt.lower()) if w not in _STOP and len(w) > 2]
    base = "_".join(words[:3]) or "task"
    name, i = base, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    return name


def _json_type(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    return "object"


def infer_schema(objects: list[dict], max_enum: int = 20) -> dict[str, Any]:
    """A JSON Schema that every object in the sample satisfies."""
    n = len(objects)
    keys = Counter(k for o in objects for k in o)
    props: dict[str, Any] = {}
    for key in keys:
        values = [o[key] for o in objects if key in o]
        types = sorted({_json_type(v) for v in values})
        prop: dict[str, Any] = {"type": types[0] if len(types) == 1 else types}
        strings = [v for v in values if isinstance(v, str)]
        distinct = set(strings)
        # Low-cardinality, repeated strings are categories (e.g. direction, category).
        if strings and len(distinct) <= max_enum and len(strings) >= 3 * len(distinct):
            prop = {"enum": sorted(distinct) + ([None] if "null" in types else [])}
        props[key] = prop
    required = sorted(k for k, c in keys.items() if c == n)
    return {"type": "object", "required": required, "properties": props}


def classify_group(outputs: list[str], max_labels: int = 50) -> tuple[str | None, str, dict]:
    n = len(outputs)
    parsed = []
    for o in outputs:
        try:
            v = json.loads(o)
        except (ValueError, TypeError):
            continue
        if isinstance(v, dict):
            parsed.append(v)
    if len(parsed) >= 0.9 * n:
        return (
            "extraction",
            f"{len(parsed)}/{n} answers are JSON objects",
            {"output_schema": infer_schema(parsed)},
        )
    counts = Counter(o.strip() for o in outputs)
    short = all(len(o) <= 60 for o in counts)
    if short and len(counts) <= max_labels and len(counts) <= 0.2 * n:
        return (
            "classification",
            f"{len(counts)} distinct short answers over {n} calls",
            {"labels": sorted(counts)},
        )
    lengths = sorted(len(o) for o in outputs)
    median = lengths[len(lengths) // 2]
    if median <= 300:
        return "generation", f"short free-text answers (median {median} chars)", {}
    return None, f"long free-form answers (median {median} chars): not a good student task", {}


def discover(store: Store, min_calls: int = 100, taken: set[str] | None = None) -> list[Candidate]:
    groups: dict[str, list[dict]] = {}
    for row in store.iter_calls(task_prefix="unassigned:"):
        if row["teacher_output"] is not None:
            groups.setdefault(row["task"], []).append(row)
    taken = set(taken or ())
    out = []
    for group, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(rows) < min_calls:
            continue
        messages = json.loads(rows[0]["request"] or "[]")
        prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        type_, reason, extra = classify_group([r["teacher_output"] for r in rows])
        name = suggest_name(prompt, taken)
        taken.add(name)
        samples = [input_from_messages(json.loads(r["request"] or "[]"))[:120] for r in rows[:3]]
        out.append(
            Candidate(group, len(rows), prompt, type_, reason, name, samples=samples, **extra)
        )
    return out


def write_candidate(store: Store, cand: Candidate, out_dir: Path) -> Path:
    """Write <out_dir>/<name>/task.yaml and data.jsonl (the group's logged calls)."""
    d = out_dir / cand.name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "data.jsonl").open("w", encoding="utf-8") as f:
        for row in store.export_training_rows(cand.group):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    header = (
        f"# Discovered from {cand.calls} calls in {cand.group}: {cand.reason}.\n"
        "# Check the instruction, labels/schema and name, then: distiller train task.yaml\n"
    )
    (d / "task.yaml").write_text(
        header + yaml.safe_dump(cand.task_yaml(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return d / "task.yaml"


def adopt(store: Store, cand: Candidate) -> int:
    """Move the group's logged calls to the new task name."""
    moved = store.rename_task(cand.group, cand.name)
    store.add_event(cand.name, "discovery", f"adopted {moved} calls from {cand.group}")
    return moved
