"""Default hyperparameters by backend, task type and dataset size.

Small datasets get more epochs; large ones fewer, to avoid overfitting to teacher quirks.
Anything here can be overridden per backend in task.yaml under student.overrides.
"""

from typing import Any

from distiller.spec import TaskSpec, TaskType


def _epochs(n_train: int, small: int, medium: int, large: int) -> int:
    if n_train < 1_000:
        return small
    if n_train < 10_000:
        return medium
    return large


def hyperparams(backend: str, spec: TaskSpec, n_train: int) -> dict[str, Any]:
    if backend == "encoder":
        params = {
            "model": spec.student.encoder_model,
            "epochs": _epochs(n_train, 10, 5, 3),
            "learning_rate": 5e-5,
            "batch_size": 32,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_length": spec.student.max_length,
        }
    elif backend == "llm-lora":
        params = {
            "model": spec.student.llm_model,
            "epochs": _epochs(n_train, 5, 3, 2),
            "learning_rate": 2e-4,
            "batch_size": 8,
            "grad_accum": 2,
            "warmup_ratio": 0.05,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "max_length": spec.student.max_length,
            "load_in_4bit": False,
            # Labels are short; JSON needs room. Recomputed from data in fit().
            "max_new_tokens": 16 if spec.type == TaskType.classification else 256,
        }
    else:
        raise ValueError(f"no presets for backend {backend!r}")
    params.update(spec.student.overrides.get(backend, {}))
    return params


def sweep_space(backend: str) -> list[dict[str, Any]]:
    """Small grids around the presets. Kept tiny on purpose: free GPU hours are limited,
    and data quality moves results far more than fine hyperparameter tuning."""
    if backend == "encoder":
        return [{"learning_rate": lr} for lr in (3e-5, 5e-5, 8e-5)]
    if backend == "llm-lora":
        return [
            {"learning_rate": lr, "lora_r": r, "lora_alpha": 2 * r}
            for lr in (1e-4, 2e-4)
            for r in (8, 16)
        ]
    raise ValueError(f"no sweep space for backend {backend!r}")
