import json

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from distiller.cli import app as cli
from distiller.gateway.config import GatewayConfig
from distiller.onboarding import doctor
from distiller.security.keys import hash_key
from distiller.spec import TaskSpec


def test_init_non_interactive(tmp_path):
    res = CliRunner().invoke(
        cli,
        [
            "init", str(tmp_path), "--task", "tickets", "--type", "classification",
            "--instruction", "Route the support ticket", "--labels", "billing, tech,other", "-y",
        ],
    )  # fmt: skip
    assert res.exit_code == 0, res.output
    cfg = GatewayConfig.load(tmp_path / "distiller.yaml")
    t = cfg.tasks["tickets"]
    assert t.mode.value == "observe" and t.match_system_prompt == "Route the support ticket"
    spec = TaskSpec.load(t.task_file)
    assert spec.labels == ["billing", "tech", "other"]
    assert spec.run_dir.resolve() == (t.runs_dir / "tickets").resolve()  # where the gateway looks
    key = next(line.strip() for line in res.output.splitlines() if line.strip().startswith("dk-"))
    assert cfg.open_store().find_api_key(hash_key(key))["role"] == "admin"
    again = CliRunner().invoke(cli, ["init", str(tmp_path), "-y"])
    assert again.exit_code != 0 and "exists" in again.output


def test_init_interactive_extraction(tmp_path):
    answers = (
        "\n".join(["receipts", "extraction", "Extract the total", "http://localhost:9/v1"]) + "\n"
    )
    res = CliRunner().invoke(cli, ["init", str(tmp_path)], input=answers)
    assert res.exit_code == 0, res.output
    spec = yaml.safe_load((tmp_path / "tasks/receipts/task.yaml").read_text())
    assert spec["type"] == "extraction" and "output_schema" in spec
    cfg = GatewayConfig.load(tmp_path / "distiller.yaml")
    assert cfg.teacher.base_url == "http://localhost:9/v1"


def test_init_rejects_bad_type(tmp_path):
    res = CliRunner().invoke(cli, ["init", str(tmp_path), "--type", "regression", "-y"])
    assert res.exit_code != 0


def _statuses(checks):
    return {c.status for c in checks}, "\n".join(c.line() for c in checks)


def test_doctor_on_fresh_init(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    CliRunner().invoke(cli, ["init", str(tmp_path), "--task", "t", "-y"])
    statuses, text = _statuses(doctor(tmp_path / "distiller.yaml", online=False))
    assert "fail" not in statuses, text
    assert "require_api_key is off" in text and "best.json not found" in text
    assert "no admin key" not in text  # init created one


def test_doctor_finds_problems(tmp_path, monkeypatch):
    monkeypatch.delenv("DISTILLER_ENCRYPTION_KEY", raising=False)
    model = tmp_path / "model"
    model.mkdir()
    (model / "distiller.json").write_text("{}")
    (tmp_path / "distiller.yaml").write_text(
        yaml.safe_dump(
            {
                "database": "d.db",
                "security": {"encrypt_logs": True},
                "tasks": {
                    "a": {"mode": "serve", "model_dir": "best"},  # no best.json while serving
                    "b": {"mode": "serve", "model_dir": str(model)},  # no routing.json
                    "c": {"task_file": "missing.yaml"},
                },
            }
        )
    )
    checks = doctor(tmp_path / "distiller.yaml", online=False)
    text = "\n".join(c.line() for c in checks)
    assert "DISTILLER_ENCRYPTION_KEY is empty" in text
    assert "FAIL task a:" in text and "lacks routing.json" in text and "FAIL task c:" in text
    res = CliRunner().invoke(cli, ["doctor", "-c", str(tmp_path / "distiller.yaml"), "--offline"])
    assert res.exit_code == 1


def test_doctor_ready_student_and_teacher_check(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "distiller.json").write_text("{}")
    (model / "routing.json").write_text(
        json.dumps({"threshold": {"threshold": 0.8, "test_coverage": 0.81}})
    )
    (tmp_path / "distiller.yaml").write_text(
        yaml.safe_dump(
            {"database": "d.db", "tasks": {"a": {"mode": "serve", "model_dir": str(model)}}}
        )
    )
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(200, json={"data": []}))
    text = "\n".join(c.line() for c in doctor(tmp_path / "distiller.yaml", online=True))
    assert "keeps 81.0% of test traffic" in text
    assert "answered /models with 200" in text

    def down(url, **kw):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(httpx, "get", down)
    text = "\n".join(c.line() for c in doctor(tmp_path / "distiller.yaml", online=True))
    assert "unreachable" in text


def test_doctor_bad_config(tmp_path):
    (tmp_path / "distiller.yaml").write_text("tasks: {a: {mode: sometimes}}")
    [check] = doctor(tmp_path / "distiller.yaml", online=False)
    assert check.status == "fail"


@pytest.mark.parametrize("missing", [False])
def test_cli_docs_are_current(missing):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).parents[2]
    r = subprocess.run(
        [sys.executable, str(root / "scripts/gen_cli_docs.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
