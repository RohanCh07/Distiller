"""`distiller init` (write a working setup) and `distiller doctor` (check one)."""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TYPES = ("classification", "extraction", "generation")


def task_template(
    name: str, type_: str, instruction: str, labels: list[str] | None
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": name,
        "type": type_,
        "instruction": instruction,
        "data": {"path": "data.jsonl"},
        "target": {"min_accuracy": 0.95},
        "output_dir": "../../runs",
    }
    if type_ == "classification" and labels:
        spec["labels"] = labels
    if type_ == "extraction":
        spec["output_schema"] = {
            "type": "object",
            "required": ["field"],
            "properties": {"field": {"type": ["string", "null"]}},
        }
    return spec


def gateway_template(
    teacher_url: str, api_key_env: str, database: str, task: str, instruction: str
) -> dict[str, Any]:
    return {
        "teacher": {"base_url": teacher_url, "api_key_env": api_key_env},
        "database": database,
        "audit_rate": 0.02,
        "security": {"require_api_key": False, "scrub_pii": True, "encrypt_logs": False},
        "tasks": {
            task: {
                "mode": "observe",
                "model_dir": "best",
                "runs_dir": "runs",
                "task_file": f"tasks/{task}/task.yaml",
                "match_system_prompt": instruction.strip().splitlines()[0][:80]
                if instruction
                else None,
            }
        },
    }


def write_setup(
    root: Path,
    task: str,
    type_: str,
    instruction: str,
    labels: list[str] | None,
    teacher_url: str,
    api_key_env: str,
    database: str,
    overwrite: bool = False,
) -> list[Path]:
    """Write distiller.yaml and tasks/<task>/task.yaml (+ an empty data.jsonl)."""
    if type_ not in TYPES:
        raise ValueError(f"type must be one of {TYPES}")
    cfg_path = root / "distiller.yaml"
    task_dir = root / "tasks" / task
    if cfg_path.exists() and not overwrite:
        raise FileExistsError(f"{cfg_path} exists (use --force to overwrite)")
    task_dir.mkdir(parents=True, exist_ok=True)
    cfg = gateway_template(teacher_url, api_key_env, database, task, instruction)
    cfg_path.write_text(
        "# Written by `distiller init`. Docs: docs/ARCHITECTURE.md, `distiller doctor` to check.\n"
        + yaml.safe_dump(cfg, sort_keys=False)
    )
    spec_path = task_dir / "task.yaml"
    spec_path.write_text(
        '# Add examples to data.jsonl ({"input": ..., "output": ...} per line), or let the\n'
        "# gateway collect them in observe mode and run `distiller logs export`.\n"
        + yaml.safe_dump(task_template(task, type_, instruction, labels), sort_keys=False)
    )
    data = task_dir / "data.jsonl"
    if not data.exists():
        data.write_text("")
    return [cfg_path, spec_path, data]


# ---------- doctor ----------


@dataclass
class Check:
    status: str  # ok | warn | fail
    message: str

    def line(self) -> str:
        return {"ok": "  ok  ", "warn": " warn ", "fail": " FAIL "}[self.status] + self.message


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def doctor(config_path: Path, online: bool = True) -> list[Check]:
    from distiller.gateway.config import GatewayConfig, is_url

    out: list[Check] = []
    try:
        cfg = GatewayConfig.load(config_path)
        out.append(Check("ok", f"config {config_path} is valid ({len(cfg.tasks)} tasks)"))
    except Exception as e:
        return [Check("fail", f"config {config_path}: {e}")]

    for module, why in (("fastapi", "gateway"), ("uvicorn", "gateway")):
        if not _has(module):
            out.append(Check("fail", f"{module} missing ({why}): pip install 'distiller[gateway]'"))
    if is_url(cfg.database) and not _has("psycopg"):
        out.append(
            Check(
                "fail", "Postgres database but psycopg missing: pip install 'distiller[postgres]'"
            )
        )
    sec = cfg.security
    if sec.encrypt_logs and not _has("cryptography"):
        out.append(
            Check(
                "fail",
                "encrypt_logs on but cryptography missing: pip install 'distiller[security]'",
            )
        )
    if sec.encrypt_logs and not sec.encryption_keys:
        out.append(Check("fail", f"encrypt_logs on but ${sec.encryption_key_env} is empty"))

    store = None
    try:
        store = cfg.open_store()
        n = store.query("SELECT COUNT(*) AS n FROM calls")[0]["n"]
        out.append(Check("ok", f"database {cfg.database} opens ({n} logged calls)"))
    except Exception as e:
        out.append(Check("fail", f"database {cfg.database}: {e}"))

    if not sec.require_api_key:
        out.append(
            Check(
                "warn",
                "security.require_api_key is off: anyone who can reach the gateway can use it",
            )
        )
    has_admin = bool(sec.env_admin_keys) or (store is not None and store.has_admin_key())
    if not has_admin:
        out.append(
            Check(
                "warn",
                "no admin key: /admin, /stats and /dashboard are open (distiller keys create ops --role admin)",
            )
        )
    if not sec.scrub_pii:
        out.append(Check("warn", "security.scrub_pii is off: logs keep personal data as sent"))
    if not cfg.teacher.api_key:
        out.append(
            Check(
                "warn",
                f"${cfg.teacher.api_key_env} is empty: apps must send their own Authorization",
            )
        )

    if online:
        import httpx

        try:
            headers = (
                {"Authorization": f"Bearer {cfg.teacher.api_key}"} if cfg.teacher.api_key else {}
            )
            r = httpx.get(cfg.teacher.base_url.rstrip("/") + "/models", headers=headers, timeout=10)
            status = "ok" if r.status_code == 200 else "warn"
            out.append(
                Check(
                    status, f"teacher {cfg.teacher.base_url} answered /models with {r.status_code}"
                )
            )
        except httpx.HTTPError as e:
            out.append(Check("fail", f"teacher {cfg.teacher.base_url} unreachable: {e}"))

    for name, t in cfg.tasks.items():
        if t.task_file:
            try:
                from distiller.spec import TaskSpec

                TaskSpec.load(t.task_file)
                out.append(Check("ok", f"task {name}: task_file loads"))
            except Exception as e:
                out.append(Check("fail", f"task {name}: task_file {t.task_file}: {e}"))
        elif cfg.retrain.enabled:
            out.append(
                Check("warn", f"task {name}: no task_file, so it cannot be retrained automatically")
            )
        if t.model_dir is None:
            out.append(Check("ok", f"task {name}: no student yet (mode {t.mode.value})"))
            continue
        if str(t.model_dir) == "best":
            best = t.runs_dir / name / "best.json"
            if not best.exists():
                level = "warn" if t.mode.value in ("observe", "off") else "fail"
                out.append(Check(level, f"task {name}: {best} not found (run distiller train)"))
                continue
            model_dir = Path(json.loads(best.read_text())["model_dir"])
        else:
            model_dir = t.model_dir
        missing = [f for f in ("distiller.json", "routing.json") if not (model_dir / f).exists()]
        if missing:
            out.append(Check("fail", f"task {name}: {model_dir} lacks {', '.join(missing)}"))
            continue
        routing = json.loads((model_dir / "routing.json").read_text())
        th = routing["threshold"]
        if th.get("threshold") is None:
            out.append(
                Check(
                    "warn", f"task {name}: student never reaches target; it will always fall back"
                )
            )
        else:
            out.append(
                Check(
                    "ok",
                    f"task {name}: student ready, keeps {th.get('test_coverage', 0):.1%} of test traffic",
                )
            )
        if t.serving.backend == "local" and not _has("torch"):
            out.append(
                Check(
                    "fail",
                    f"task {name}: local student needs torch: pip install 'distiller[train]'",
                )
            )
    return out
