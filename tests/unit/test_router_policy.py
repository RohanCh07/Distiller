from distiller.gateway.config import DemoteSettings, GatewayConfig
from distiller.router import policy
from distiller.router.policy import check_drift, should_audit
from distiller.storage.db import Store


def _config(tmp_path, **demote):
    return GatewayConfig(database=tmp_path / "d.db", demote=DemoteSettings(**demote))


def test_should_audit_rate(tmp_path, monkeypatch):
    assert not should_audit(GatewayConfig(audit_rate=0))
    assert should_audit(GatewayConfig(audit_rate=1))
    monkeypatch.setattr(policy.random, "random", lambda: 0.3)
    assert should_audit(GatewayConfig(audit_rate=0.31))
    assert not should_audit(GatewayConfig(audit_rate=0.3))


def _audits(store, task, n, agree_every):
    for i in range(n):
        store.log_call(
            task=task, mode="serve", route="student", audited=1, agree=int(i % agree_every == 0)
        )


def test_demotes_when_agreement_drops(tmp_path):
    cfg = _config(tmp_path, min_agreement=0.9, window=100, min_samples=20)
    store = Store(cfg.database)
    store.set_mode("t", "serve")
    _audits(store, "t", 30, agree_every=2)  # 50% agreement
    assert check_drift(store, cfg, "t") is True
    assert store.get_mode("t") == "shadow"
    assert "auto-demoted: audit agreement 50.0% over 30 requests" in store.events()[0]["detail"]
    # The mode change resets the window: no repeated demotion on the same evidence.
    assert check_drift(store, cfg, "t") is False


def test_no_demotion_with_too_few_samples(tmp_path):
    cfg = _config(tmp_path, min_agreement=0.9, min_samples=50)
    store = Store(cfg.database)
    _audits(store, "t", 49, agree_every=10)
    assert check_drift(store, cfg, "t") is False
    assert store.get_mode("t") is None


def test_no_demotion_when_agreement_is_good(tmp_path):
    cfg = _config(tmp_path, min_agreement=0.9, min_samples=10)
    store = Store(cfg.database)
    _audits(store, "t", 40, agree_every=1)
    assert check_drift(store, cfg, "t") is False


def test_window_uses_latest_audits(tmp_path):
    cfg = _config(tmp_path, min_agreement=0.9, window=20, min_samples=20)
    store = Store(cfg.database)
    _audits(store, "t", 100, agree_every=1)  # old, all good
    _audits(store, "t", 20, agree_every=4)  # recent, 25%
    assert check_drift(store, cfg, "t") is True
