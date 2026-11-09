from pathlib import Path

import pytest
from pydantic import ValidationError

from distiller.gateway.config import GatewayConfig, Mode
from distiller.tasks.identify import identify, system_prompt


def test_defaults():
    cfg = GatewayConfig()
    assert cfg.teacher.base_url == "https://api.openai.com/v1"
    assert cfg.audit_rate == 0.02
    assert (
        cfg.demote.min_agreement == 0.92
        and cfg.demote.window == 200
        and cfg.demote.min_samples == 50
    )
    assert cfg.tasks == {}


def test_load_resolves_relative_paths(tmp_path):
    (tmp_path / "conf").mkdir()
    path = tmp_path / "conf" / "distiller.yaml"
    path.write_text(
        """
database: data/d.db
audit_rate: 0.5
tasks:
  a: {mode: serve, model_dir: best, runs_dir: runs}
  b: {mode: shadow, model_dir: models/b}
  c: {model_dir: /abs/model}
"""
    )
    cfg = GatewayConfig.load(path)
    base = path.resolve().parent
    assert cfg.database == base / "data" / "d.db"
    assert cfg.audit_rate == 0.5
    assert cfg.tasks["a"].mode == Mode.serve
    assert str(cfg.tasks["a"].model_dir) == "best"  # kept as the keyword
    assert cfg.tasks["a"].runs_dir == base / "runs"
    assert cfg.tasks["b"].model_dir == base / "models" / "b"
    assert cfg.tasks["c"].model_dir == Path("/abs/model")
    assert cfg.tasks["c"].mode == Mode.observe


def test_empty_file(tmp_path):
    (tmp_path / "e.yaml").write_text("")
    assert GatewayConfig.load(tmp_path / "e.yaml").tasks == {}


def test_example_config_loads():
    cfg = GatewayConfig.load(Path(__file__).parents[2] / "distiller.example.yaml")
    assert cfg.tasks["banking77"].mode == Mode.shadow
    assert cfg.tasks["transactions"].mode == Mode.observe


def test_invalid_values():
    with pytest.raises(ValidationError):
        GatewayConfig.model_validate({"tasks": {"a": {"mode": "sometimes"}}})
    with pytest.raises(ValidationError):
        GatewayConfig(audit_rate=1.5)


def test_api_key_from_env(monkeypatch):
    cfg = GatewayConfig.model_validate({"teacher": {"api_key_env": "MY_KEY"}})
    monkeypatch.delenv("MY_KEY", raising=False)
    assert cfg.teacher.api_key == ""
    monkeypatch.setenv("MY_KEY", "sk-test")
    assert cfg.teacher.api_key == "sk-test"


# ---------- task identification ----------


def test_identify_header_then_prompt_then_hash():
    cfg = GatewayConfig.model_validate(
        {"tasks": {"bank": {"match_system_prompt": "bank customer"}}}
    )
    msgs = [
        {"role": "system", "content": "Classify the bank customer message"},
        {"role": "user", "content": "x"},
    ]
    assert identify({"x-distiller-task": " tagged "}, msgs, cfg) == "tagged"
    assert identify({}, msgs, cfg) == "bank"
    other = [{"role": "system", "content": "Translate to French"}]
    h = identify({}, other, cfg)
    assert h.startswith("unassigned:") and len(h) == len("unassigned:") + 8
    assert identify({}, other, cfg) == h  # stable
    assert identify({}, [{"role": "user", "content": "x"}], cfg) == "unassigned"
    assert (
        system_prompt([{"role": "system", "content": "a"}, {"role": "system", "content": "b"}])
        == "a\nb"
    )
