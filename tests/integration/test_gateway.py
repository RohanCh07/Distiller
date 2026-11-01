"""Gateway against a mocked upstream LLM: every mode, fallback, drift, dashboard, export.

The upstream ("teacher") is an httpx.MockTransport; students are small fakes with the
same interface as serving.registry.LoadedStudent, so these tests need no model weights.
"""

import json
import threading
import time

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from distiller.data.io import load_examples
from distiller.gateway.app import create_app
from distiller.gateway.config import GatewayConfig
from distiller.serving.registry import Answer
from distiller.spec import TaskSpec

SYSTEM = "Classify the bank customer's message"


class _Chunks(httpx.AsyncByteStream):
    """A real (not pre-read) streaming body, like a network response."""

    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    async def __aiter__(self):
        for c in self.chunks:
            yield c


class FakeUpstream:
    """OpenAI-style chat endpoint. Answers with `reply(user_text)`; records every request."""

    def __init__(self, reply=lambda text: "card_arrival"):
        self.reply = reply
        self.calls: list[dict] = []
        self.status = 200
        self.fail = False
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.fail:
            raise httpx.ConnectError("upstream down")
        with self._lock:
            self.calls.append(
                {
                    "path": request.url.path,
                    "headers": dict(request.headers),
                    "body": json.loads(request.content) if request.content else None,
                }
            )
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "big-model"}]})
        body = json.loads(request.content)
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "nope"}})
        user = [m["content"] for m in body["messages"] if m["role"] == "user"][-1]
        if body.get("stream"):
            chunks = [
                f"data: {json.dumps({'choices': [{'delta': {'content': part}}]})}\n\n"
                for part in ("card_", "arrival")
            ] + ["data: [DONE]\n\n"]
            return httpx.Response(
                200,
                stream=_Chunks([c.encode() for c in chunks]),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "id": "up-1",
                "object": "chat.completion",
                "model": "big-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": self.reply(user)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
            },
        )

    def chat_calls(self):
        return [c for c in self.calls if c["path"].endswith("/chat/completions")]


class FakeStudent:
    """Same interface as LoadedStudent: .spec and .answer(list[str]) -> list[Answer]."""

    def __init__(self, output="card_arrival", confidence=0.99, threshold=0.9, error=None):
        self.spec = TaskSpec.model_validate(
            {"name": "banking77", "type": "classification", "data": {"path": "x"}}
        )
        self.output, self.confidence, self.threshold, self.error = (
            output,
            confidence,
            threshold,
            error,
        )
        self.seen: list[str] = []

    def answer(self, inputs):
        if self.error:
            raise self.error
        self.seen += inputs
        out = self.output(inputs[0]) if callable(self.output) else self.output
        return [Answer(out, self.confidence, out is not None and self.confidence >= self.threshold)]


def _config(tmp_path, mode="observe", **kw):
    return GatewayConfig.model_validate(
        {
            "database": str(tmp_path / "gw.db"),
            "teacher": {
                "base_url": "https://upstream.test/v1",
                "api_key_env": "DISTILLER_TEST_KEY",
            },
            "tasks": {"banking77": {"mode": mode, "match_system_prompt": SYSTEM}},
            **kw,
        }
    )


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """gateway(mode=..., student=..., **config) -> (client, app, upstream). Closed after the test."""
    monkeypatch.setenv("DISTILLER_TEST_KEY", "sk-server")
    opened = []

    def _make(mode="observe", student=None, upstream=None, **kw):
        upstream = upstream or FakeUpstream()
        app = create_app(_config(tmp_path, mode, **kw), transport=httpx.MockTransport(upstream))
        client = TestClient(app)
        client.__enter__()
        opened.append(client)
        if student is not None:
            app.state.students["banking77"] = student
        return client, app, upstream

    yield _make
    for c in opened:
        c.__exit__(None, None, None)


def ask(client, text="my card has not arrived", tagged=True, **extra):
    headers = {"X-Distiller-Task": "banking77"} if tagged else {}
    headers.update(extra.pop("headers", {}))
    body = {
        "model": "big-model",
        "messages": [
            {"role": "system", "content": SYSTEM + " into one intent."},
            {"role": "user", "content": text},
        ],
        **extra,
    }
    return client.post("/v1/chat/completions", json=body, headers=headers)


def calls(app):
    return app.state.store.query("SELECT * FROM calls ORDER BY id")


def wait_background(app, timeout=5.0):
    deadline = time.time() + timeout
    while app.state.background and time.time() < deadline:
        time.sleep(0.01)
    assert not app.state.background, "background work did not finish"


# ---------- modes ----------


def test_off_mode_passthrough_logs_nothing(gateway):
    client, app, up = gateway("off")
    r = ask(client)
    assert r.status_code == 200 and r.json()["id"] == "up-1"
    assert calls(app) == []
    assert len(up.chat_calls()) == 1


def test_observe_mode_logs_teacher_answer(gateway):
    client, app, up = gateway("observe", student=FakeStudent())
    r = ask(client, "my card has not arrived")
    assert r.headers["x-distiller-route"] == "teacher"
    assert r.json()["choices"][0]["message"]["content"] == "card_arrival"
    [row] = calls(app)
    assert row["task"] == "banking77" and row["mode"] == "observe" and row["route"] == "teacher"
    assert row["input"] == "my card has not arrived"
    assert row["teacher_output"] == "card_arrival"
    assert (
        row["prompt_tokens"] == 20 and row["completion_tokens"] == 3 and row["model"] == "big-model"
    )
    assert row["student_output"] is None  # student is not consulted in observe mode


def test_shadow_mode_teacher_answers_student_compared(gateway):
    student = FakeStudent(output=lambda t: "card_arrival" if "card" in t else "top_up")
    client, app, up = gateway("shadow", student=student)
    assert ask(client, "my card is late").headers["x-distiller-route"] == "teacher"
    assert (
        ask(client, "what is the rate").json()["choices"][0]["message"]["content"] == "card_arrival"
    )
    wait_background(app)
    rows = calls(app)
    assert [r["route"] for r in rows] == ["teacher", "teacher"]
    assert [r["agree"] for r in rows] == [1, 0]
    assert rows[1]["student_output"] == "top_up" and rows[1]["student_confidence"] == 0.99
    assert student.seen == ["my card is late", "what is the rate"]
    stats = client.get("/stats").json()["tasks"][0]
    assert stats["shadow_agreement"] == 0.5 and stats["shadow_samples"] == 2


def test_shadow_without_student_just_logs(gateway):
    client, app, _ = gateway("shadow")
    ask(client)
    wait_background(app)
    assert calls(app)[0]["agree"] is None


def test_serve_confident_student_answers_without_upstream(gateway):
    client, app, up = gateway("serve", student=FakeStudent(confidence=0.97), audit_rate=0)
    r = ask(client)
    assert r.status_code == 200
    assert r.headers["x-distiller-route"] == "student"
    body = r.json()
    assert body["object"] == "chat.completion" and body["model"] == "distiller/banking77"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "card_arrival"}
    assert body["distiller"] == {"route": "student", "confidence": 0.97}
    assert up.chat_calls() == []
    [row] = calls(app)
    assert row["route"] == "student" and row["audited"] == 0 and row["teacher_output"] is None


def test_serve_low_confidence_falls_back_to_teacher(gateway):
    client, app, up = gateway(
        "serve", student=FakeStudent(output="top_up", confidence=0.4), audit_rate=0
    )
    r = ask(client)
    assert r.headers["x-distiller-route"] == "fallback"
    assert r.json()["choices"][0]["message"]["content"] == "card_arrival"  # teacher's answer
    [row] = calls(app)
    assert row["route"] == "fallback"
    assert row["student_output"] == "top_up" and row["student_confidence"] == 0.4
    assert row["teacher_output"] == "card_arrival"
    assert len(up.chat_calls()) == 1


def test_serve_invalid_student_output_falls_back(gateway):
    client, app, _ = gateway(
        "serve", student=FakeStudent(output=None, confidence=0.99), audit_rate=0
    )
    assert ask(client).headers["x-distiller-route"] == "fallback"
    assert calls(app)[0]["student_output"] is None


def test_serve_student_crash_fails_open(gateway):
    client, app, _ = gateway("serve", student=FakeStudent(error=RuntimeError("model exploded")))
    r = ask(client)
    assert r.status_code == 200 and r.headers["x-distiller-route"] == "fallback"
    assert r.json()["choices"][0]["message"]["content"] == "card_arrival"


def test_serve_without_student_uses_teacher(gateway):
    client, app, _ = gateway("serve")
    assert ask(client).headers["x-distiller-route"] == "teacher"


def test_serve_audit_sends_copy_to_teacher(gateway):
    client, app, up = gateway("serve", student=FakeStudent(output="top_up"), audit_rate=1)
    r = ask(client)
    assert r.headers["x-distiller-route"] == "student"
    wait_background(app)
    [row] = calls(app)
    assert row["audited"] == 1 and row["agree"] == 0 and row["teacher_output"] == "card_arrival"
    assert len(up.chat_calls()) == 1


# ---------- drift ----------


def test_drift_auto_demotes_serve_to_shadow(gateway):
    client, app, up = gateway(
        "serve",
        student=FakeStudent(output="top_up"),
        audit_rate=1,
        demote={"min_agreement": 0.9, "window": 50, "min_samples": 5},
    )
    routes = []
    for i in range(8):
        routes.append(ask(client, f"card question {i}").headers["x-distiller-route"])
        wait_background(app)
    # 5 audited disagreements trigger the demotion; afterwards the teacher answers.
    assert routes[:5] == ["student"] * 5
    assert set(routes[5:]) == {"teacher"}
    assert app.state.store.get_mode("banking77") == "shadow"
    events = client.get("/stats").json()["events"]
    assert "auto-demoted" in events[0]["detail"]
    stats = client.get("/stats").json()["tasks"][0]
    assert stats["mode"] == "shadow" and stats["audit_samples"] == 5


def test_good_student_is_not_demoted(gateway):
    client, app, _ = gateway(
        "serve",
        student=FakeStudent(),
        audit_rate=1,
        demote={"min_agreement": 0.9, "min_samples": 5},
    )
    for _ in range(10):
        assert ask(client).headers["x-distiller-route"] == "student"
        wait_background(app)
    assert app.state.store.get_mode("banking77") is None


# ---------- upstream behaviour, auth, identification ----------


def test_upstream_unreachable_returns_502(gateway):
    up = FakeUpstream()
    up.fail = True
    client, app, _ = gateway("observe", upstream=up)
    r = ask(client)
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "bad_gateway"


def test_upstream_error_status_is_passed_through(gateway):
    up = FakeUpstream()
    up.status = 429
    client, app, _ = gateway("observe", upstream=up)
    r = ask(client)
    assert r.status_code == 429 and r.json()["error"]["message"] == "nope"
    assert calls(app)[0]["teacher_output"] is None  # errors are never training data


def test_app_authorization_forwarded_else_server_key(gateway):
    client, app, up = gateway("observe")
    ask(client, headers={"Authorization": "Bearer sk-app"})
    ask(client)
    assert up.chat_calls()[0]["headers"]["authorization"] == "Bearer sk-app"
    assert up.chat_calls()[1]["headers"]["authorization"] == "Bearer sk-server"


def test_untagged_requests_identified_by_system_prompt_or_hash(gateway):
    client, app, _ = gateway("observe")
    ask(client, tagged=False)
    client.post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "system", "content": "Translate to French"},
                {"role": "user", "content": "hi"},
            ],
        },
    )
    tasks = [r["task"] for r in calls(app)]
    assert tasks[0] == "banking77"
    assert tasks[1].startswith("unassigned:")


def test_streaming_is_passed_through(gateway):
    client, app, up = gateway("serve", student=FakeStudent())
    r = ask(client, stream=True)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "card_" in r.text and "[DONE]" in r.text
    assert calls(app) == []  # streaming bypasses routing for now


def test_models_endpoint_proxied(gateway):
    client, _, _ = gateway()
    assert client.get("/v1/models").json()["data"][0]["id"] == "big-model"


# ---------- admin, stats, dashboard, export ----------


def test_admin_mode_switch_takes_effect_immediately(gateway):
    client, app, _ = gateway("shadow", student=FakeStudent(), audit_rate=0)
    assert ask(client).headers["x-distiller-route"] == "teacher"
    r = client.post("/admin/tasks/banking77/mode", json={"mode": "serve"})
    assert r.json() == {"task": "banking77", "mode": "serve"}
    assert ask(client).headers["x-distiller-route"] == "student"


def test_admin_rejects_unknown_mode(gateway):
    client, app, _ = gateway()
    r = client.post("/admin/tasks/banking77/mode", json={"mode": "bogus"})
    assert r.status_code == 422
    assert app.state.store.get_mode("banking77") is None


def test_dashboard_renders_numbers(gateway):
    client, app, _ = gateway("serve", student=FakeStudent(), audit_rate=0)
    for _ in range(3):
        ask(client)
    client.post("/admin/tasks/banking77/mode", json={"mode": "shadow"})
    ask(client)
    wait_background(app)
    html = client.get("/dashboard").text
    assert html.startswith("<!doctype html>")
    assert "<td>banking77</td>" in html and "shadow ✓" in html
    assert "<td>4</td>" in html  # calls
    assert "75.0%" in html  # served by student
    assert "shadow (admin api)" in html  # event list
    assert client.get("/health").json() == {"ok": True, "students": ["banking77"]}


def test_dashboard_with_no_traffic(gateway):
    client, _, _ = gateway()
    html = client.get("/dashboard").text
    assert "<li>none</li>" in html


def test_exported_logs_are_valid_training_data(gateway, tmp_path):
    client, app, _ = gateway(
        "observe",
        upstream=FakeUpstream(reply=lambda t: "top_up" if "money" in t else "card_arrival"),
    )
    texts = [f"add money {i}" for i in range(5)] + [f"card late {i}" for i in range(5)]
    for t in texts:
        ask(client, t)
    out = tmp_path / "exported.jsonl"
    with out.open("w") as f:
        for row in app.state.store.export_training_rows("banking77"):
            f.write(json.dumps(row) + "\n")
    spec = TaskSpec.model_validate(
        {"name": "banking77", "type": "classification", "data": {"path": str(out)}}
    )
    loaded = load_examples(spec)
    assert loaded.errors == []
    assert [(e.input, e.output) for e in loaded.examples] == [
        (t, "top_up" if "money" in t else "card_arrival") for t in texts
    ]


def test_cli_logs_export_and_stats(gateway, tmp_path):
    from typer.testing import CliRunner

    from distiller.cli import app as cli

    client, app, _ = gateway("observe")
    ask(client, "one")
    ask(client, "two")
    cfg = tmp_path / "distiller.yaml"
    cfg.write_text(f"database: {tmp_path / 'gw.db'}\n")
    out = tmp_path / "data.jsonl"
    runner = CliRunner()
    res = runner.invoke(cli, ["logs", "export", "banking77", "--out", str(out), "-c", str(cfg)])
    assert res.exit_code == 0, res.output
    assert "Exported 2 examples" in res.output
    assert len(out.read_text().splitlines()) == 2
    res = runner.invoke(cli, ["logs", "stats", "-c", str(cfg)])
    assert json.loads(res.output)["tasks"][0]["calls"] == 2
    res = runner.invoke(cli, ["task", "mode", "banking77", "serve", "-c", str(cfg)])
    assert res.exit_code == 0 and app.state.store.get_mode("banking77") == "serve"
