"""Gateway settings, read from distiller.yaml."""

import os
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


def is_url(database: Path | str) -> bool:
    return "://" in str(database)


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


class RetrainSettings(BaseModel):
    # Queue a retrain job when a task is auto-demoted (drift). `distiller retrain-worker`
    # runs the jobs; each task needs task_file so the worker knows its spec.
    enabled: bool = False
    min_new_examples: int = 100  # logged teacher answers needed before retraining
    reload_seconds: float = 30  # how often gateways check runs/<task>/best.json for a new model


class PromoteSettings(BaseModel):
    # Move shadow -> serve by itself once shadow agreement is high enough.
    enabled: bool = False
    min_agreement: float = 0.97
    window: int = 300
    min_samples: int = 200


class HostedSettings(BaseModel):
    # Cloud tier: every /v1 call needs a key attached to an account (`distiller accounts
    # create`, `distiller keys create --account`). Calls are tagged with the account;
    # account admins see only their own stats and bill.
    enabled: bool = False


class SecuritySettings(BaseModel):
    # Every /v1 request must carry a valid gateway key (dk-...). Keys live in the database
    # (`distiller keys create`); app_keys_env may also hold comma-separated bootstrap keys.
    require_api_key: bool = False
    app_keys_env: str = "DISTILLER_API_KEYS"
    # Admin endpoints (/admin, /stats, /dashboard) need an admin key whenever one exists
    # (in the database or in this env var), and always when require_api_key is on.
    admin_key_env: str = "DISTILLER_ADMIN_KEY"
    # Replace emails, phone numbers, card numbers, Aadhaar, PAN, IBAN and IPs with
    # placeholders before anything is stored. The student sees the same scrubbed input.
    scrub_pii: bool = False
    pii_kinds: list[str] = Field(
        default_factory=lambda: ["email", "card", "aadhaar", "phone", "pan", "iban", "ip"]
    )
    # Encrypt stored content with Fernet. The env var holds comma-separated keys (first one
    # encrypts; all decrypt, for rotation). Create one with `distiller keys encryption-key`.
    encrypt_logs: bool = False
    encryption_key_env: str = "DISTILLER_ENCRYPTION_KEY"

    @staticmethod
    def _env_list(name: str) -> list[str]:
        return [k.strip() for k in os.environ.get(name, "").split(",") if k.strip()]

    @property
    def env_app_keys(self) -> list[str]:
        return self._env_list(self.app_keys_env)

    @property
    def env_admin_keys(self) -> list[str]:
        return self._env_list(self.admin_key_env)

    @property
    def encryption_keys(self) -> str:
        return os.environ.get(self.encryption_key_env, "")


class ServingSettings(BaseModel):
    # local: load the model into the gateway. vllm: call a vLLM server (llm-lora students,
    # started with `distiller serve-vllm`); the gateway keeps only calibration + threshold.
    backend: Literal["local", "vllm"] = "local"
    url: str = "http://localhost:8001/v1"
    model: str | None = None  # served model name; defaults to the task name
    api_key_env: str | None = None
    timeout_seconds: float = 30

    @model_validator(mode="after")
    def _check(self) -> "ServingSettings":
        if self.backend == "vllm" and not self.url:
            raise ValueError("serving.url is required for the vllm backend")
        return self


class TaskSettings(BaseModel):
    mode: Mode = Mode.observe
    # A trained model folder, or "best" to use runs/<task>/best.json from `distiller train`.
    model_dir: Path | None = None
    runs_dir: Path = Path("runs")
    # Requests without an X-Distiller-Task header are matched if the system prompt contains this.
    match_system_prompt: str | None = None
    serving: ServingSettings = Field(default_factory=ServingSettings)
    # The task's task.yaml, needed for automatic retraining.
    task_file: Path | None = None


class GatewayConfig(BaseModel):
    teacher: TeacherSettings = Field(default_factory=TeacherSettings)
    # SQLite file path, or a postgresql://... URL.
    database: str | Path = Path("distiller.db")
    audit_rate: float = Field(0.02, ge=0, le=1)
    demote: DemoteSettings = Field(default_factory=DemoteSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    retrain: RetrainSettings = Field(default_factory=RetrainSettings)
    promote: PromoteSettings = Field(default_factory=PromoteSettings)
    hosted: HostedSettings = Field(default_factory=HostedSettings)
    tasks: dict[str, TaskSettings] = Field(default_factory=dict)

    def open_store(self):
        """The Store for this config, with PII scrubbing and encryption as configured."""
        from distiller.security.crypto import FieldCipher
        from distiller.security.pii import Scrubber
        from distiller.storage.db import Store

        sec = self.security
        cipher = None
        if sec.encrypt_logs:
            if not sec.encryption_keys:
                raise ValueError(
                    f"security.encrypt_logs is on but ${sec.encryption_key_env} is empty"
                )
            cipher = FieldCipher.from_env_value(sec.encryption_keys)
        scrubber = Scrubber(sec.pii_kinds) if sec.scrub_pii else None
        return Store(self.database, cipher=cipher, scrubber=scrubber)

    @classmethod
    def load(cls, path: str | Path) -> "GatewayConfig":
        path = Path(path)
        cfg = cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        base = path.resolve().parent
        if not is_url(cfg.database) and not Path(cfg.database).is_absolute():
            cfg.database = base / cfg.database
        for t in cfg.tasks.values():
            if not t.runs_dir.is_absolute():
                t.runs_dir = base / t.runs_dir
            if t.model_dir and str(t.model_dir) != "best" and not t.model_dir.is_absolute():
                t.model_dir = base / t.model_dir
            if t.task_file and not t.task_file.is_absolute():
                t.task_file = base / t.task_file
        return cfg
