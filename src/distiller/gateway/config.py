"""Gateway settings, read from distiller.yaml."""

import os
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Mode(str, Enum):
    off = "off"  # plain passthrough, nothing logged
    observe = "observe"  # teacher answers; call is logged (training data)
    shadow = "shadow"  # teacher answers; student answers in the background, agreement logged
    serve = "serve"  # student answers when confident; otherwise teacher (fallback, logged)


class TeacherSettings(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    # Used only when the client sends no Authorization header of its own.
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 120

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


class DemoteSettings(BaseModel):
    min_agreement: float = 0.92  # audited agreement below this -> back to shadow
    window: int = 200  # look at the last N audited requests
    min_samples: int = 50  # don't judge on fewer than this


class TaskSettings(BaseModel):
    mode: Mode = Mode.observe
    # A trained model folder, or "best" to use runs/<task>/best.json from `distiller train`.
    model_dir: Path | None = None
    runs_dir: Path = Path("runs")
    # Requests without an X-Distiller-Task header are matched if the system prompt contains this.
    match_system_prompt: str | None = None


class GatewayConfig(BaseModel):
    teacher: TeacherSettings = Field(default_factory=TeacherSettings)
    database: Path = Path("distiller.db")
    audit_rate: float = Field(0.02, ge=0, le=1)
    demote: DemoteSettings = Field(default_factory=DemoteSettings)
    tasks: dict[str, TaskSettings] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "GatewayConfig":
        path = Path(path)
        cfg = cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        base = path.resolve().parent
        if not cfg.database.is_absolute():
            cfg.database = base / cfg.database
        for t in cfg.tasks.values():
            if not t.runs_dir.is_absolute():
                t.runs_dir = base / t.runs_dir
            if t.model_dir and str(t.model_dir) != "best" and not t.model_dir.is_absolute():
                t.model_dir = base / t.model_dir
        return cfg
