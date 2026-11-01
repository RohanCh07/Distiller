"""Check each example's output against the task spec."""

from typing import Any

import jsonschema

from distiller.spec import TaskSpec, TaskType


def make_validator(spec: TaskSpec):
    """Return a function output -> error message (or None when valid)."""
    if spec.type == TaskType.classification:
        allowed = set(spec.labels) if spec.labels else None

        def check(output: Any) -> str | None:
            if not isinstance(output, str) or not output:
                return "label must be a non-empty string"
            if allowed is not None and output not in allowed:
                return f"unknown label {output!r}"
            return None

    elif spec.type == TaskType.extraction:
        schema_validator = (
            jsonschema.Draft202012Validator(spec.output_schema) if spec.output_schema else None
        )

        def check(output: Any) -> str | None:
            if not isinstance(output, dict):
                return "output must be a JSON object"
            if schema_validator is not None:
                err = next(iter(schema_validator.iter_errors(output)), None)
                if err is not None:
                    return f"schema: {err.message}"
            return None

    else:

        def check(output: Any) -> str | None:
            if not isinstance(output, str) or not output:
                return "output must be non-empty text"
            return None

    return check
