"""Hosted tier: accounts, tenant isolation, metering and share-of-savings invoices."""

import json
from datetime import UTC, datetime

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from distiller.cli import app as cli
from distiller.gateway.app import create_app
from distiller.gateway.config import GatewayConfig
from distiller.hosted.billing import compute_invoice, invoice_csv, month_bounds
from distiller.security.keys import generate_key, hash_key
from distiller.storage.db import Store
from tests.integration.test_gateway import FakeStudent, FakeUpstream

OCT = datetime(2026, 10, 15, tzinfo=UTC).timestamp()
SEP = datetime(2026, 9, 15, tzinfo=UTC).timestamp()
TERMS = dict(share=0.3, min_monthly_usd=0, teacher_usd_per_1m_in=2.0, teacher_usd_per_1m_out=8.0)


def test_month_bounds():
    start, end = month_bounds("2026-12")
    assert datetime.fromtimestamp(start, UTC) == datetime(2026, 12, 1, tzinfo=UTC)
    assert datetime.fromtimestamp(end, UTC) == datetime(2027, 1, 1, tzinfo=UTC)


def _log(store, account, task, route, n, ts, prompt=None, completion=None):
    for _ in range(n):
        store.log_call(task=task, mode="serve", route=route, account=account, ts=ts,
                       prompt_tokens=prompt, completion_tokens=completion)  # fmt: skip


def test_invoice_math(db_url):
    store = Store(db_url)
    store.add_account("acme", student_usd_per_1k=0.5, **TERMS)
    store.add_account("other", **TERMS)
    # acme / tickets in October: 100 teacher-answered calls at 1000 in + 10 out tokens,
    # 900 student calls. Teacher cost per call = (1000*2 + 10*8)/1e6 = 0.00208 USD.
    _log(store, "acme", "tickets", "teacher", 60, OCT, 1000, 10)
    _log(store, "acme", "tickets", "fallback", 40, OCT, 1000, 10)
    _log(store, "acme", "tickets", "student", 900, OCT)
    # acme / receipts: only student calls in October -> all-time averages (September).
    _log(store, "acme", "receipts", "teacher", 10, SEP, 500, 50)
    _log(store, "acme", "receipts", "student", 1000, OCT)
    _log(store, "other", "tickets", "student", 50, OCT)  # another tenant: not billed to acme
    _log(store, "acme", "tickets", "student", 7, SEP)  # another month

    inv = compute_invoice(store, "acme", "2026-10")
    lines = {line["task"]: line for line in inv["lines"]}
    t = lines["tickets"]
    assert (t["calls"], t["student_calls"], t["teacher_calls"]) == (1000, 900, 100)
    assert t["teacher_usd_per_call"] == pytest.approx(0.00208)
    assert t["avoided_usd"] == pytest.approx(900 * 0.00208)  # 1.872
    assert t["serving_usd"] == pytest.approx(0.45)  # 900 * 0.5 / 1000
    assert t["savings_usd"] == pytest.approx(1.872 - 0.45)
    r = lines["receipts"]
    per_call = (500 * 2 + 50 * 8) / 1e6  # 0.0014 from September averages
    assert r["teacher_usd_per_call"] == pytest.approx(per_call)
    assert r["savings_usd"] == pytest.approx(1000 * per_call - 0.5)
    savings = (1.872 - 0.45) + (1.4 - 0.5)
    assert inv["savings_usd"] == pytest.approx(savings)
    assert inv["amount_usd"] == pytest.approx(round(0.3 * savings, 2))
    assert inv["customer_net_saving_usd"] == pytest.approx(savings - inv["amount_usd"], abs=0.01)
    assert not inv["minimum_applied"]


def test_minimum_fee_and_no_savings(db_url):
    store = Store(db_url)
    store.add_account("small", **{**TERMS, "min_monthly_usd": 49.0, "student_usd_per_1k": 100.0})
    _log(store, "small", "t", "teacher", 10, OCT, 100, 1)
    _log(store, "small", "t", "student", 10, OCT)  # serving costs more than it saves
    inv = compute_invoice(store, "small", "2026-10")
    assert inv["lines"][0]["savings_usd"] == 0
    assert inv["amount_usd"] == 49.0 and inv["minimum_applied"]
    with pytest.raises(ValueError):
        compute_invoice(store, "ghost", "2026-10")


def test_keys_need_existing_account(db_url):
    store = Store(db_url)
    with pytest.raises(ValueError):
        store.add_api_key("k", hash_key("dk-x"), "app", account="ghost")


# ---------- gateway in hosted mode ----------


@pytest.fixture
def hosted(tmp_path):
    cfg = GatewayConfig.model_validate(
        {
            "database": str(tmp_path / "gw.db"),
            "audit_rate": 0,
            "hosted": {"enabled": True},
            "teacher": {"base_url": "https://up.test/v1"},
            "tasks": {"banking77": {"mode": "serve"}},
        }
    )
    app = create_app(cfg, transport=httpx.MockTransport(FakeUpstream()))
    store = app.state.store
    for name in ("acme", "globex"):
        store.add_account(name, **TERMS)
    keys = {}
    for name, role, account in [
        ("acme-app", "app", "acme"), ("acme-admin", "admin", "acme"),
        ("globex-app", "app", "globex"), ("platform", "admin", None), ("legacy", "app", None),
    ]:  # fmt: skip
        keys[name] = generate_key()
        store.add_api_key(name, hash_key(keys[name]), role, account)
    with TestClient(app) as client:
        app.state.students["banking77"] = FakeStudent()
        yield client, app, keys


def call(client, key):
    return client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "card?"}]},
        headers={"X-Distiller-Task": "banking77", "X-Distiller-Key": key},
    )


def test_hosted_requires_account_keys(hosted):
    client, app, keys = hosted
    assert call(client, keys["acme-app"]).status_code == 200
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": []})
    assert r.status_code == 401
    assert call(client, keys["legacy"]).status_code == 403  # key without an account


def test_calls_are_tagged_and_stats_isolated(hosted):
    client, app, keys = hosted
    for _ in range(3):
        call(client, keys["acme-app"])
    call(client, keys["globex-app"])
    rows = app.state.store.query("SELECT account, api_key FROM calls ORDER BY id")
    assert [r["account"] for r in rows] == ["acme", "acme", "acme", "globex"]
    acme = client.get("/stats", headers={"X-Distiller-Key": keys["acme-admin"]}).json()
    assert acme["tasks"][0]["calls"] == 3 and acme["retrain_jobs"] == []
    platform = client.get("/stats", headers={"X-Distiller-Key": keys["platform"]}).json()
    assert platform["tasks"][0]["calls"] == 4
    ts = client.get("/stats/timeseries", headers={"X-Distiller-Key": keys["acme-admin"]}).json()
    assert sum(p["calls"] for p in ts["points"]) == 3


def test_account_admins_cannot_change_global_state(hosted):
    client, app, keys = hosted
    h = {"X-Distiller-Key": keys["acme-admin"]}
    assert (
        client.post("/admin/tasks/banking77/mode", json={"mode": "off"}, headers=h).status_code
        == 403
    )
    assert client.post("/admin/tasks/banking77/retrain", headers=h).status_code == 403
    assert client.get("/admin/discover", headers=h).status_code == 403
    p = {"X-Distiller-Key": keys["platform"]}
    assert (
        client.post("/admin/tasks/banking77/mode", json={"mode": "shadow"}, headers=p).status_code
        == 200
    )


def test_usage_endpoint(hosted):
    client, app, keys = hosted
    call(client, keys["acme-app"])
    own = client.get("/account/usage", headers={"X-Distiller-Key": keys["acme-admin"]}).json()
    assert own["account"] == "acme" and own["lines"][0]["student_calls"] == 1
    # An account admin cannot read another account's bill.
    other = client.get(
        "/account/usage?account=globex", headers={"X-Distiller-Key": keys["acme-admin"]}
    )
    assert other.json()["account"] == "acme"
    p = {"X-Distiller-Key": keys["platform"]}
    assert client.get("/account/usage?account=globex", headers=p).json()["lines"] == []
    assert client.get("/account/usage", headers=p).status_code == 400
    assert client.get("/account/usage?account=ghost", headers=p).status_code == 404


def test_billing_cli(tmp_path):
    cfg = tmp_path / "distiller.yaml"
    cfg.write_text("database: gw.db\n")
    r = CliRunner()
    args = [
        "--share",
        "0.25",
        "--price-in",
        "200",
        "--price-out",
        "8",
        "--min-monthly",
        "10",
        "-c",
        str(cfg),
    ]
    assert r.invoke(cli, ["accounts", "create", "acme", *args]).exit_code == 0
    assert (
        r.invoke(
            cli,
            [
                "accounts",
                "create",
                "bad",
                "--share",
                "2",
                "--price-in",
                "1",
                "--price-out",
                "1",
                "-c",
                str(cfg),
            ],
        ).exit_code
        != 0
    )
    res = r.invoke(cli, ["keys", "create", "acme-app", "--account", "acme", "-c", str(cfg)])
    assert res.exit_code == 0
    assert (
        r.invoke(cli, ["keys", "create", "x", "--account", "ghost", "-c", str(cfg)]).exit_code != 0
    )
    store = Store(tmp_path / "gw.db")
    _log(store, "acme", "t", "teacher", 10, OCT, 1000, 0)
    _log(store, "acme", "t", "student", 1000, OCT)
    res = r.invoke(
        cli, ["billing", "invoice", "acme", "--month", "2026-10", "--save", "-c", str(cfg)]
    )
    assert res.exit_code == 0, res.output
    # 1000 calls x (1000 tokens x 200 USD / 1M) = 200 saved; 25% = 50.
    assert "Saved invoice acme 2026-10: $50.00" in res.output
    again = r.invoke(
        cli, ["billing", "invoice", "acme", "--month", "2026-10", "--save", "-c", str(cfg)]
    )
    assert again.exit_code != 0
    out = tmp_path / "inv.csv"
    assert r.invoke(cli, ["billing", "export", "--out", str(out), "-c", str(cfg)]).exit_code == 0
    lines = out.read_text().splitlines()
    assert lines[0].startswith("account,period,task") and lines[1].startswith(
        "acme,2026-10,t,1000,"
    )
    assert (
        json.loads(r.invoke(cli, ["accounts", "list", "-c", str(cfg)]).output)[0]["share"] == 0.25
    )


def test_invoice_csv_empty_month():
    csv = invoice_csv([{"account": "a", "period": "2026-10", "lines": [], "amount_usd": 49.0}])
    assert csv.splitlines()[1] == "a,2026-10,,0,0,0,0,49.0"
