"""End-to-end CLI runs on CPU with tiny random models (built offline in conftest).

These prove every command runs and writes the right files; the scores are meaningless.
Run only these with:  pytest -m slow
"""

import json
import shutil
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from typer.testing import CliRunner  # noqa: E402

from distiller.cli import app as cli  # noqa: E402
from tests.conftest import classification_rows, write_task  # noqa: E402
from tests.e2e.fake_teacher import FakeTeacherServer  # noqa: E402

pytestmark = pytest.mark.slow
ROOT = Path(__file__).parents[2]


def run(*args):
    res = CliRunner().invoke(cli, [str(a) for a in args], catch_exceptions=False)
    assert res.exit_code == 0, res.output
    return res.output


@pytest.fixture
def teacher_env(monkeypatch):
    with FakeTeacherServer() as server:
        monkeypatch.setenv("DISTILLER_TEACHER_BASE_URL", server.base_url)
        monkeypatch.setenv("DISTILLER_TEACHER_API_KEY", "sk-fake")
        monkeypatch.setenv("DISTILLER_TEACHER_MODEL", "fake-teacher")
        yield server


# ---------- classification: data check -> train (encoder) -> report -> predict -> gateway ----------


@pytest.fixture(scope="module")
def trained_classifier(tmp_path_factory, tiny_encoder):
    """Train the tiny encoder once on a keyword-separable 3-intent task via the CLI."""
    d = tmp_path_factory.mktemp("cls")
    task = write_task(
        d,
        classification_rows(100),
        name="intents",
        data={"path": "data.jsonl", "val_fraction": 0.2, "test_fraction": 0.2},
        student={
            "candidates": ["encoder"],
            "encoder_model": tiny_encoder,
            "max_length": 32,
            "overrides": {"encoder": {"epochs": 12, "learning_rate": 1e-3, "batch_size": 16}},
        },
        target={"min_accuracy": 0.8},
    )
    data_out = run("data", "check", task)
    train_out = run("train", task)
    return task, d / "runs" / "intents", data_out, train_out


def test_data_check_writes_report(trained_classifier):
    _, run_dir, out, _ = trained_classifier
    assert "| **Usable examples** | **300** |" in out
    assert (run_dir / "data" / "data_report.md").exists()


def test_train_writes_report_model_and_routing(trained_classifier):
    _, run_dir, _, out = trained_classifier
    assert "# Training report: intents" in out
    model_dir = run_dir / "models" / "encoder"
    for name in ("distiller.json", "routing.json", "config.json", "tokenizer.json"):
        assert (model_dir / name).exists(), name
    assert not list(model_dir.glob("trial*"))  # sweep/checkpoint folders cleaned up
    routing = json.loads((model_dir / "routing.json").read_text())
    assert routing["calibration"]["method"] == "temperature"  # 60 val examples >= 30
    assert len(routing["test_curve"]) == 20
    best = json.loads((run_dir / "best.json").read_text())
    assert best["backend"] == "encoder" and Path(best["model_dir"]) == model_dir
    results = json.loads((run_dir / "results.json").read_text())
    [cand] = results["candidates"]
    assert cand["status"] == "ok"
    assert 0 <= cand["test_metrics"]["accuracy"] <= 1
    # The keyword task is easy: even a tiny random-init encoder should learn it.
    assert cand["test_metrics"]["accuracy"] > 0.8, cand["test_metrics"]


def test_predict_command(trained_classifier):
    _, run_dir, _, _ = trained_classifier
    out = run("predict", run_dir / "models" / "encoder", "my card has not arrived yet", "what exchange rate do you use")
    assert out.count("conf=") == 2 and "route=" in out


def test_gateway_serves_trained_student(trained_classifier, tmp_path):
    import httpx
    from fastapi.testclient import TestClient

    from distiller.gateway.app import create_app
    from distiller.gateway.config import GatewayConfig

    _, run_dir, _, _ = trained_classifier
    cfg = GatewayConfig.model_validate(
        {
            "database": str(tmp_path / "gw.db"),
            "audit_rate": 0,
            "teacher": {"base_url": "https://upstream.test/v1"},
            "tasks": {"intents": {"mode": "serve", "model_dir": "best", "runs_dir": str(run_dir.parent)}},
        }
    )
    upstream = httpx.MockTransport(
        lambda r: httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "top_up"}}], "usage": {}},
        )
    )
    with TestClient(create_app(cfg, transport=upstream)) as client:
        assert client.get("/health").json()["students"] == ["intents"]
        routes = []
        for text in ["my card has not arrived yet please", "how do i top up my account", "hello?"]:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": text}]},
                headers={"X-Distiller-Task": "intents"},
            )
            assert r.status_code == 200
            routes.append(r.headers["x-distiller-route"])
        assert set(routes) <= {"student", "fallback"}


def test_sweep_trains_grid_and_keeps_best(tmp_path, tiny_encoder):
    task = write_task(
        tmp_path,
        classification_rows(30),
        name="sweep",
        student={"candidates": ["encoder"], "encoder_model": tiny_encoder, "max_length": 32,
                 "overrides": {"encoder": {"epochs": 1}}},
    )
    run("train", task, "--sweep")
    results = json.loads((tmp_path / "runs" / "sweep" / "results.json").read_text())
    assert [s["params"]["learning_rate"] for s in results["candidates"][0]["sweep"]] == [3e-5, 5e-5, 8e-5]


# ---------- transactions example: teacher label -> augment -> train (llm-lora) -> report ----------


@pytest.fixture
def transactions_task(tmp_path, tiny_llm):
    d = tmp_path / "transactions"
    shutil.copytree(ROOT / "examples" / "transactions", d)
    spec = yaml.safe_load((d / "task.yaml").read_text())
    spec["data"]["path"] = "labelled.jsonl"
    spec["student"].update({"llm_model": tiny_llm, "max_length": 160})
    spec["student"]["overrides"]["llm-lora"].update({"epochs": 1, "batch_size": 8})
    spec["output_dir"] = "runs"
    (d / "task.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return d


def _raw_messages(n: int) -> list[str]:
    seeds = [json.loads(line)["input"] for line in (ROOT / "examples/transactions/seed.jsonl").read_text().splitlines()]
    shops = ["Swiggy", "Zomato", "Amazon", "Uber", "Jio", "BigBasket", "Local Kirana"]
    made = [
        f"INR {50 + 13 * i}.00 debited from A/c XX{2000 + i} {'UPI' if i % 2 else 'card'} {shops[i % len(shops)]}"
        for i in range(n)
    ]
    return seeds + made


def test_transactions_full_cli_flow(transactions_task, teacher_env):
    d = transactions_task
    task = d / "task.yaml"
    raw = d / "raw.txt"
    raw.write_text("\n".join(_raw_messages(120)) + "\n")

    out = run("teacher", "label", task, "--inputs", raw, "--out", d / "labelled.jsonl")
    assert "Labelled 144/144" in out
    first = json.loads((d / "labelled.jsonl").read_text().splitlines()[0])
    assert set(first["output"]) == {"direction", "amount", "merchant", "category", "method"}

    out = run("teacher", "augment", task, "--total", 60, "--out", d / "synthetic.jsonl")
    assert "Added 60 synthetic examples" in out
    assert all(json.loads(line)["synthetic"] for line in (d / "synthetic.jsonl").read_text().splitlines())

    out = run("data", "check", task)
    assert "| **Usable examples** | **144** |" in out
    assert "Synthetic examples added to train | 60" in out

    out = run("train", task)
    assert "# Training report: transactions" in out
    run_dir = d / "runs" / "transactions"
    results = json.loads((run_dir / "results.json").read_text())
    [cand] = results["candidates"]
    assert cand["backend"] == "llm-lora" and cand["status"] == "ok", cand.get("error")
    for key in ("exact_match", "field_f1", "invalid_json_rate"):
        assert key in cand["test_metrics"]
    model_dir = run_dir / "models" / "llm-lora"
    assert (model_dir / "adapter_config.json").exists() and (model_dir / "routing.json").exists()

    out = run("predict", model_dir, "Rs 450 debited via UPI to swiggy@ybl")
    assert "conf=" in out

    merged = d / "merged"
    run("export", model_dir, "--out", merged)
    assert (merged / "routing.json").exists() and (merged / "config.json").exists()
    assert not (merged / "adapter_config.json").exists()


def test_encoder_on_extraction_fails_cleanly(transactions_task, tiny_encoder):
    """The encoder only does classification: asking for it on transactions must not crash."""
    d = transactions_task
    rows = [json.loads(line) for line in (d / "seed.jsonl").read_text().splitlines()]
    (d / "labelled.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows * 1))
    # Enough training rows: vary the seeds a little.
    extra = [{"input": f"{r['input']} (copy {i})", "output": r["output"]} for i in range(4) for r in rows]
    with (d / "labelled.jsonl").open("a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in extra))
    out = run("train", d / "task.yaml", "--candidates", "encoder")
    assert "failed: encoder does not support extraction" in out
    assert "Every candidate failed" in out
