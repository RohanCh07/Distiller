"""API-key auth, admin protection, PII scrubbing and encrypted logs through the gateway."""

import json

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cryptography")

from fastapi.testclient import TestClient

from distiller.gateway.app import create_app
from distiller.gateway.config import GatewayConfig
from distiller.security.crypto import FieldCipher
from distiller.security.keys import generate_key, hash_key
from distiller.storage.db import Store
from tests.integration.test_gateway import FakeStudent, FakeUpstream


@pytest.fixture
def secure(tmp_path, monkeypatch):
    monkeypatch.setenv("UPSTREAM_KEY", "sk-server")
    monkeypatch.delenv("DISTILLER_ADMIN_KEY", raising=False)
    monkeypatch.delenv("DISTILLER_API_KEYS", raising=False)
    opened = []

    def _make(security=None, mode="observe", student=None):
        cfg = GatewayConfig.model_validate(
            {
                "database": str(tmp_path / "gw.db"),
                "audit_rate": 0,
                "teacher": {"base_url": "https://upstream.test/v1", "api_key_env": "UPSTREAM_KEY"},
                "security": security or {},
                "tasks": {"bank": {"mode": mode}},
            }
        )
        up = FakeUpstream()
        app = create_app(cfg, transport=httpx.MockTransport(up))
        client = TestClient(app)
        client.__enter__()
        opened.append(client)
        if student:
            app.state.students["bank"] = student
        return client, app, up

    yield _make
    for c in opened:
        c.__exit__(None, None, None)


def post(client, headers=None, text="hello"):
    return client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": text}]},
        headers={"X-Distiller-Task": "bank", **(headers or {})},
    )


def add_key(app, name, role="app"):
    key = generate_key()
    app.state.store.add_api_key(name, hash_key(key), role)
    return key


def test_required_key_rejects_missing_and_wrong(secure):
    client, app, up = secure({"require_api_key": True})
    r = post(client)
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "invalid_api_key"  # OpenAI-style error body
    assert post(client, {"X-Distiller-Key": "dk-wrong"}).status_code == 401
    assert up.chat_calls() == []


def test_valid_key_header_forwards_app_authorization(secure):
    client, app, up = secure({"require_api_key": True})
    key = add_key(app, "web")
    r = post(client, {"X-Distiller-Key": key, "Authorization": "Bearer sk-app-own"})
    assert r.status_code == 200
    assert up.chat_calls()[0]["headers"]["authorization"] == "Bearer sk-app-own"
    assert app.state.store.query("SELECT api_key FROM calls")[0]["api_key"] == "web"


def test_key_in_authorization_is_never_sent_upstream(secure):
    """OPENAI_API_KEY=dk-... in an app: the gateway swaps in its own teacher key."""
    client, app, up = secure({"require_api_key": True})
    key = add_key(app, "sdk")
    assert post(client, {"Authorization": f"Bearer {key}"}).status_code == 200
    sent = up.chat_calls()[0]["headers"]["authorization"]
    assert sent == "Bearer sk-server" and key not in json.dumps(up.calls)


def test_revoked_key_stops_working(secure):
    client, app, _ = secure({"require_api_key": True})
    key = add_key(app, "temp")
    assert post(client, {"X-Distiller-Key": key}).status_code == 200
    app.state.store.revoke_api_key("temp")
    assert post(client, {"X-Distiller-Key": key}).status_code == 401


def test_env_bootstrap_keys(secure, monkeypatch):
    monkeypatch.setenv("DISTILLER_API_KEYS", "dk-envapp1,dk-envapp2")
    client, app, _ = secure({"require_api_key": True})
    assert post(client, {"X-Distiller-Key": "dk-envapp2"}).status_code == 200


def test_optional_auth_still_rejects_bad_keys(secure):
    client, _, _ = secure()
    assert post(client).status_code == 200  # no key needed
    assert post(client, {"X-Distiller-Key": "dk-nope"}).status_code == 401


def test_admin_endpoints_open_until_an_admin_key_exists(secure):
    client, app, _ = secure()
    assert client.get("/stats").status_code == 200
    admin = add_key(app, "ops", "admin")
    app_key = add_key(app, "web", "app")
    assert client.get("/stats").status_code == 401
    assert client.get("/stats", headers={"X-Distiller-Key": app_key}).status_code == 403
    assert client.get("/stats", headers={"X-Distiller-Key": admin}).status_code == 200
    assert client.get(f"/stats?key={admin}").status_code == 200
    assert client.get("/stats/timeseries").status_code == 401
    assert client.get("/dashboard").status_code == 200  # empty shell; its data calls need the key
    r = client.post("/admin/tasks/bank/mode", json={"mode": "serve"})
    assert r.status_code == 401 and app.state.store.get_mode("bank") is None
    r = client.post(
        "/admin/tasks/bank/mode", json={"mode": "serve"}, headers={"X-Distiller-Key": admin}
    )
    assert r.status_code == 200
    assert client.get("/health").status_code == 200  # health stays open for load balancers


def test_env_admin_key(secure, monkeypatch):
    monkeypatch.setenv("DISTILLER_ADMIN_KEY", "dk-rootadmin")
    client, _, _ = secure()
    assert client.get("/stats").status_code == 401
    assert client.get("/stats", headers={"X-Distiller-Key": "dk-rootadmin"}).status_code == 200


def test_pii_scrubbed_in_logs_and_student_input(secure):
    student = FakeStudent(confidence=0.1)  # always falls back, so we see what it was given
    client, app, up = secure({"scrub_pii": True}, mode="serve", student=student)
    r = post(client, text="card 4111 1111 1111 1111 lost, call 9876543210")
    assert r.status_code == 200
    assert (
        "4111" in up.chat_calls()[0]["body"]["messages"][0]["content"]
    )  # teacher sees the real text
    assert student.seen == ["card <CARD> lost, call <PHONE>"]
    row = app.state.store.query("SELECT input, request FROM calls")[0]
    assert row["input"] == "card <CARD> lost, call <PHONE>"
    assert "4111" not in row["request"] and "9876543210" not in row["request"]


def test_encrypted_logs_end_to_end(secure, monkeypatch, tmp_path):
    key = FieldCipher.generate_key()
    monkeypatch.setenv("DISTILLER_ENCRYPTION_KEY", key)
    client, app, _ = secure({"encrypt_logs": True})
    post(client, text="private message")
    raw = Store(tmp_path / "gw.db").query("SELECT input, teacher_output FROM calls")[0]
    assert raw["input"].startswith("enc:v1:") and raw["teacher_output"].startswith("enc:v1:")
    rows = list(app.state.store.export_training_rows("bank"))
    assert rows[0]["messages"][0]["content"] == "private message"
    assert client.get("/stats").json()["tasks"][0]["calls"] == 1  # stats need no content


def test_encrypt_logs_without_key_fails_fast(secure, monkeypatch):
    monkeypatch.delenv("DISTILLER_ENCRYPTION_KEY", raising=False)
    with pytest.raises(ValueError, match="DISTILLER_ENCRYPTION_KEY"):
        secure({"encrypt_logs": True})


def test_keys_cli(tmp_path):
    from typer.testing import CliRunner

    from distiller.cli import app as cli

    cfg = tmp_path / "distiller.yaml"
    cfg.write_text("database: gw.db\n")
    runner = CliRunner()
    res = runner.invoke(cli, ["keys", "create", "web", "--role", "admin", "-c", str(cfg)])
    assert res.exit_code == 0, res.output
    key = res.output.splitlines()[0]
    store = Store(tmp_path / "gw.db")
    assert store.find_api_key(hash_key(key))["role"] == "admin"
    assert '"name": "web"' in runner.invoke(cli, ["keys", "list", "-c", str(cfg)]).output
    assert runner.invoke(cli, ["keys", "revoke", "web", "-c", str(cfg)]).exit_code == 0
    assert runner.invoke(cli, ["keys", "revoke", "web", "-c", str(cfg)]).exit_code != 0
    assert (
        runner.invoke(cli, ["keys", "create", "x", "--role", "god", "-c", str(cfg)]).exit_code != 0
    )
    enc = runner.invoke(cli, ["keys", "encryption-key"]).output.strip()
    assert FieldCipher([enc]).decrypt(FieldCipher([enc]).encrypt("ok")) == "ok"
