"""Remote (vLLM) students, the serve-vllm command, and several gateways on one database."""

import json
import math

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from distiller.gateway.app import create_app
from distiller.gateway.config import GatewayConfig, ServingSettings
from distiller.serving.registry import RemoteStudent, load_student
from distiller.spec import TaskSpec
from tests.integration.test_gateway import FakeStudent, FakeUpstream

SCHEMA = {
    "type": "object",
    "required": ["amount"],
    "additionalProperties": False,
    "properties": {"amount": {"type": ["number", "null"]}},
}


def _model_dir(tmp_path, *, threshold=0.5, adapter=True, calibration=None):
    """A saved llm-lora student folder without weights (enough for remote serving)."""
    d = tmp_path / "models" / "llm-lora"
    d.mkdir(parents=True)
    spec = TaskSpec.model_validate(
        {
            "name": "tx",
            "type": "extraction",
            "instruction": "Extract the amount.",
            "output_schema": SCHEMA,
            "data": {"path": "x"},
        }
    )
    meta = {
        "backend": "llm-lora",
        "labels": None,
        "hparams": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "max_new_tokens": 40, "lora_r": 8},
        "spec": spec.model_dump(mode="json"),
    }
    (d / "distiller.json").write_text(json.dumps(meta))
    (d / "routing.json").write_text(
        json.dumps(
            {
                "calibration": calibration or {"method": "none"},
                "threshold": {"threshold": threshold},
            }
        )
    )
    if adapter:
        (d / "adapter_config.json").write_text("{}")
    return d


class FakeVLLM:
    def __init__(self, content='{"amount": 450.0}', logprobs=(-0.01, -0.02, -0.05)):
        self.content, self.logprobs, self.bodies = content, logprobs, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": self.content},
                        "logprobs": {
                            "content": [{"token": "t", "logprob": lp} for lp in self.logprobs]
                        },
                    }
                ]
            },
        )


def test_remote_student_request_and_confidence(tmp_path):
    vllm = FakeVLLM()
    s = RemoteStudent(
        _model_dir(tmp_path),
        "http://vllm:8001/v1",
        api_key="sk-v",
        transport=httpx.MockTransport(vllm),
    )
    [ans] = s.answer(["Rs 450 debited"])
    assert ans.output == {"amount": 450.0}
    assert ans.confidence == pytest.approx(math.exp((-0.01 - 0.02 - 0.05) / 3))
    assert ans.confident
    body = vllm.bodies[0]
    assert body["model"] == "tx" and body["temperature"] == 0 and body["logprobs"] is True
    assert body["max_tokens"] == 40
    assert body["messages"][0] == {"role": "system", "content": "Extract the amount."}
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA


def test_remote_student_invalid_or_unsure_output_is_not_confident(tmp_path):
    bad = RemoteStudent(
        _model_dir(tmp_path),
        "http://v/v1",
        transport=httpx.MockTransport(FakeVLLM('{"amount": "x"}')),
    )
    assert bad.answer(["x"])[0].output is None and not bad.answer(["x"])[0].confident
    unsure = RemoteStudent(
        _model_dir(tmp_path / "b"),
        "http://v/v1",
        transport=httpx.MockTransport(FakeVLLM(logprobs=(-3.0, -2.0))),
    )
    assert not unsure.answer(["x"])[0].confident


def test_remote_student_applies_calibration(tmp_path):
    cal = {"method": "isotonic", "knots_x": [0.0, 1.0], "knots_y": [0.0, 0.5]}
    s = RemoteStudent(
        _model_dir(tmp_path, threshold=0.6, calibration=cal),
        "http://v/v1",
        transport=httpx.MockTransport(FakeVLLM(logprobs=(0.0,))),
    )
    [ans] = s.answer(["x"])
    assert ans.confidence == pytest.approx(0.5) and not ans.confident


def test_remote_only_for_llm_students(tmp_path):
    d = _model_dir(tmp_path)
    meta = json.loads((d / "distiller.json").read_text())
    meta["backend"] = "encoder"
    (d / "distiller.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        RemoteStudent(d, "http://v/v1")


def test_load_student_picks_backend(tmp_path):
    d = _model_dir(tmp_path)
    s = load_student(d, ServingSettings(backend="vllm", url="http://v/v1", model="m"))
    assert isinstance(s, RemoteStudent) and s.model == "m"


def test_gateway_serves_from_remote_student(tmp_path):
    d = _model_dir(tmp_path)
    cfg = GatewayConfig.model_validate(
        {
            "database": str(tmp_path / "gw.db"),
            "audit_rate": 0,
            "teacher": {"base_url": "https://up.test/v1"},
            "tasks": {
                "tx": {
                    "mode": "serve",
                    "model_dir": str(d),
                    "serving": {"backend": "vllm", "url": "http://v/v1"},
                }
            },
        }
    )
    up = FakeUpstream()
    app = create_app(cfg, transport=httpx.MockTransport(up))
    with TestClient(app) as client:
        assert client.get("/health").json()["students"] == ["tx"]
        app.state.students["tx"].client = httpx.Client(
            base_url="http://v/v1", transport=httpx.MockTransport(FakeVLLM())
        )
        r = client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "Rs 450 debited"}]},
            headers={"X-Distiller-Task": "tx"},
        )
        assert r.headers["x-distiller-route"] == "student"
        assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"amount": 450.0}
        assert up.chat_calls() == []


def test_serve_vllm_command(tmp_path):
    from typer.testing import CliRunner

    from distiller.cli import app as cli

    runner = CliRunner()
    adapter = _model_dir(tmp_path / "a")
    out = runner.invoke(cli, ["serve-vllm", str(adapter), "--print", "--port", "9000"]).output
    assert "vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-lora --lora-modules tx=" in out
    assert "--max-lora-rank 8 --port 9000" in out
    merged = _model_dir(tmp_path / "m", adapter=False)
    out = runner.invoke(cli, ["serve-vllm", str(merged), "--print"]).output
    assert f"vllm serve {merged.resolve()} --served-model-name tx --port 8001" in out


# ---------- several gateway instances on one database ----------


def _two_gateways(tmp_path, db, **cfg):
    apps = []
    for _ in range(2):
        config = GatewayConfig.model_validate(
            {
                "database": db,
                "teacher": {"base_url": "https://up.test/v1"},
                "tasks": {"banking77": {"mode": "shadow"}},
                **cfg,
            }
        )
        apps.append(create_app(config, transport=httpx.MockTransport(FakeUpstream())))
    return apps


def _ask(client):
    return client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "card?"}]},
        headers={"X-Distiller-Task": "banking77"},
    )


def test_mode_change_on_one_instance_applies_to_all(tmp_path, db_url):
    a, b = _two_gateways(tmp_path, db_url, audit_rate=0)
    with TestClient(a) as ca, TestClient(b) as cb:
        for app in (a, b):
            app.state.students["banking77"] = FakeStudent()
        assert _ask(cb).headers["x-distiller-route"] == "teacher"
        ca.post("/admin/tasks/banking77/mode", json={"mode": "serve"})
        assert _ask(cb).headers["x-distiller-route"] == "student"
        assert cb.get("/stats").json()["tasks"][0]["calls"] == 2  # both see all traffic


def test_keys_created_once_work_everywhere(tmp_path, db_url):
    from distiller.security.keys import generate_key, hash_key

    a, b = _two_gateways(tmp_path, db_url, security={"require_api_key": True})
    with TestClient(a), TestClient(b) as cb:
        key = generate_key()
        a.state.store.add_api_key("web", hash_key(key), "app")
        r = cb.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
            headers={"X-Distiller-Key": key},
        )
        assert r.status_code == 200
        a.state.store.revoke_api_key("web")
        assert _ask(cb).status_code == 401


def test_app_factory_reads_config_env(tmp_path, monkeypatch):
    from distiller.gateway.app import app_from_env

    cfg = tmp_path / "distiller.yaml"
    cfg.write_text("database: gw.db\ntasks: {t: {mode: serve}}\n")
    monkeypatch.setenv("DISTILLER_CONFIG", str(cfg))
    with TestClient(app_from_env()) as c:
        assert c.get("/health").json()["ok"] is True
    assert (tmp_path / "gw.db").exists()
