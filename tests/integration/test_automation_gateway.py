"""Gateway side of automation: retrain on drift, auto-promotion, model hot reload."""

import json
import os
import time

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from distiller.gateway.app import create_app
from distiller.gateway.config import GatewayConfig
from distiller.serving import registry
from tests.integration.test_gateway import FakeStudent, FakeUpstream, ask, wait_background


def _app(tmp_path, mode, **cfg):
    config = GatewayConfig.model_validate(
        {
            "database": str(tmp_path / "gw.db"),
            "teacher": {"base_url": "https://up.test/v1"},
            "tasks": {
                "banking77": {"mode": mode, "model_dir": "best", "runs_dir": str(tmp_path / "runs")}
            },
            **cfg,
        }
    )
    return create_app(config, transport=httpx.MockTransport(FakeUpstream()))


def test_drift_queues_retrain(tmp_path):
    app = _app(
        tmp_path,
        "serve",
        audit_rate=1,
        demote={"min_samples": 3, "min_agreement": 0.9},
        retrain={"enabled": True},
    )
    with TestClient(app) as client:
        app.state.students["banking77"] = FakeStudent(output="top_up")
        for _ in range(4):
            ask(client)
            wait_background(app)
        jobs = client.get("/stats").json()["retrain_jobs"]
        assert (
            len(jobs) == 1
            and jobs[0]["status"] == "queued"
            and jobs[0]["reason"] == "drift demotion"
        )


def test_auto_promotion_after_good_shadow(tmp_path):
    app = _app(
        tmp_path, "shadow", promote={"enabled": True, "min_samples": 5, "min_agreement": 0.9}
    )
    with TestClient(app) as client:
        app.state.students["banking77"] = FakeStudent()
        routes = []
        for _ in range(7):
            routes.append(ask(client).headers["x-distiller-route"])
            wait_background(app)
        assert routes[:5] == ["teacher"] * 5 and routes[-1] == "student"
        assert "auto-promoted" in client.get("/stats").json()["events"][0]["detail"]


def test_no_promotion_when_disabled_or_disagreeing(tmp_path):
    app = _app(
        tmp_path, "shadow", promote={"enabled": True, "min_samples": 5, "min_agreement": 0.9}
    )
    with TestClient(app) as client:
        app.state.students["banking77"] = FakeStudent(output="wrong")
        for _ in range(8):
            assert ask(client).headers["x-distiller-route"] == "teacher"
            wait_background(app)


@pytest.fixture
def fake_loader(monkeypatch):
    loaded = []

    def load_student(model_dir, serving=None):
        loaded.append(str(model_dir))
        return FakeStudent(output=os.path.basename(str(model_dir)))

    monkeypatch.setattr(registry, "load_student", load_student)
    return loaded


def _publish(tmp_path, name):
    best = tmp_path / "runs" / "banking77" / "best.json"
    best.parent.mkdir(parents=True, exist_ok=True)
    best.write_text(json.dumps({"backend": "encoder", "model_dir": str(tmp_path / name)}))


def test_hot_reload_by_admin_and_watcher(tmp_path, fake_loader):
    _publish(tmp_path, "model_v1")
    app = _app(tmp_path, "serve", audit_rate=0, retrain={"reload_seconds": 0.05})
    with TestClient(app) as client:
        assert ask(client).json()["choices"][0]["message"]["content"] == "model_v1"
        r = client.post("/admin/tasks/banking77/reload")
        assert r.json()["reloaded"] is True  # forced: weights may have changed in place
        time.sleep(0.02)
        _publish(tmp_path, "model_v2")
        deadline = time.time() + 5
        while ask(client).json()["choices"][0]["message"]["content"] != "model_v2":
            assert time.time() < deadline, "watcher did not reload"
            time.sleep(0.05)
        v1, v2 = str(tmp_path / "model_v1"), str(tmp_path / "model_v2")
        assert fake_loader == [v1, v1, v2]  # start, forced reload, watcher (once)
        events = [e["detail"] for e in client.get("/stats").json()["events"]]
        assert any("new model" in e for e in events)  # fresh drift/promotion window
        assert client.post("/admin/tasks/nope/reload").status_code == 404


def test_broken_new_model_keeps_old_one(tmp_path, monkeypatch):
    calls = {"n": 0}

    def load_student(model_dir, serving=None):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("corrupt weights")
        return FakeStudent(output="old")

    monkeypatch.setattr(registry, "load_student", load_student)
    _publish(tmp_path, "v1")
    app = _app(tmp_path, "serve", audit_rate=0)
    with TestClient(app) as client:
        _publish(tmp_path, "v2")
        assert client.post("/admin/tasks/banking77/reload").json()["reloaded"] is False
        assert ask(client).json()["choices"][0]["message"]["content"] == "old"


def test_admin_retrain_and_discover_endpoints(tmp_path):
    app = _app(tmp_path, "observe")
    with TestClient(app) as client:
        assert client.post("/admin/tasks/banking77/retrain").json()["queued"] is True
        assert client.post("/admin/tasks/banking77/retrain").json()["queued"] is False
        for i in range(12):
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [
                        {"role": "system", "content": "Sort the email into a folder"},
                        {"role": "user", "content": f"email {i}"},
                    ],
                },
            )
        found = client.get("/admin/discover?min_calls=10").json()["candidates"]
        assert found[0]["calls"] == 12 and found[0]["type"] == "classification"
        assert found[0]["name"] == "sort_email_folder"
