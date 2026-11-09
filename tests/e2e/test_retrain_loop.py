"""The whole automatic loop with a real (tiny) model:

train -> gateway serves -> traffic drifts -> auto-demotion queues a retrain ->
`distiller retrain-worker --once` trains on original + logged data -> new best.json ->
the gateway reloads it.
"""

import json

import httpx
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("transformers")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from distiller.cli import app as cli  # noqa: E402
from distiller.gateway.app import create_app  # noqa: E402
from distiller.gateway.config import GatewayConfig  # noqa: E402
from tests.conftest import classification_rows, write_task  # noqa: E402
from tests.integration.test_gateway import wait_background  # noqa: E402

pytestmark = pytest.mark.slow

LOST = [
    "i lost my card",
    "my card was stolen yesterday",
    "card missing from my wallet",
    "someone took my card",
    "cannot find my card anywhere",
    "please block my lost card",
]
FILL = ["", " help", " urgent", " what now", " today", " please"]


def drifted(i: int) -> str:
    return f"{LOST[i % len(LOST)]}{FILL[(i // len(LOST)) % len(FILL)]} ({i})"


def test_drift_retrain_reload(tmp_path, tiny_encoder):
    task = write_task(
        tmp_path / "task",
        classification_rows(40),
        name="intents",
        data={"path": "data.jsonl", "val_fraction": 0.25, "test_fraction": 0.25},
        student={
            "candidates": ["encoder"],
            "encoder_model": tiny_encoder,
            "max_length": 32,
            "overrides": {"encoder": {"epochs": 6, "learning_rate": 1e-3, "batch_size": 16}},
        },
        target={"min_accuracy": 0.05},  # tiny model: any confident slice counts as promotable
        output_dir=str(tmp_path / "runs"),
    )
    runner = CliRunner()
    assert runner.invoke(cli, ["train", str(task)], catch_exceptions=False).exit_code == 0
    first = json.loads((tmp_path / "runs" / "intents" / "best.json").read_text())["model_dir"]

    cfg_path = tmp_path / "distiller.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "database": "gw.db",
                "audit_rate": 1,
                "demote": {"min_samples": 5, "min_agreement": 0.99, "window": 50},
                "retrain": {"enabled": True, "min_new_examples": 20, "reload_seconds": 3600},
                "teacher": {"base_url": "https://up.test/v1"},
                "tasks": {
                    "intents": {
                        "mode": "serve",
                        "model_dir": "best",
                        "runs_dir": str(tmp_path / "runs"),
                        "task_file": str(task),
                    }
                },
            }
        )
    )

    # Traffic has drifted: a new intent appears that the student never learned.
    upstream = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "card_lost"}}],
                "usage": {},
            },
        )
    )
    app = create_app(GatewayConfig.load(cfg_path), transport=upstream)
    with TestClient(app) as client:
        # Force every answer to be "confident" so audits happen on each request.
        app.state.students["intents"].threshold = 0.0
        for i in range(30):
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": drifted(i)}],
                },
                headers={"X-Distiller-Task": "intents"},
            )
            wait_background(app)
        assert app.state.store.get_mode("intents") == "shadow"  # demoted on drift
        # In shadow mode the teacher keeps answering and logging: that is the new training data.
        for i in range(30):
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "m",
                    "messages": [{"role": "user", "content": drifted(30 + i)}],
                },
                headers={"X-Distiller-Task": "intents"},
            )
        jobs = app.state.store.retrain_jobs()
        assert [j["status"] for j in jobs] == ["queued"]

        res = runner.invoke(
            cli, ["retrain-worker", "--once", "-c", str(cfg_path)], catch_exceptions=False
        )
        assert res.exit_code == 0, res.output
        assert "done: published encoder" in res.output, res.output
        second = json.loads((tmp_path / "runs" / "intents" / "best.json").read_text())["model_dir"]
        assert second != first and "/retrain/" in second

        r = client.post("/admin/tasks/intents/reload").json()
        assert r["reloaded"] and r["model_dir"] == second
        # The retrained student knows the new intent.
        assert "card_lost" in app.state.students["intents"].student.labels
        assert app.state.store.retrain_jobs()[0]["status"] == "done"
