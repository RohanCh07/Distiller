from distiller.gateway.config import DemoteSettings, GatewayConfig
from distiller.router import policy
from distiller.router.policy import check_drift, should_audit
from distiller.storage.db import Store


def _config(db_url, **demote):
    return GatewayConfig(database=db_url, demote=DemoteSettings(**demote))


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


def test_demotes_when_agreement_drops(db_url):
    cfg = _config(db_url, min_agreement=0.9, window=100, min_samples=20)
    store = Store(cfg.database)
    store.set_mode("t", "serve")
    _audits(store, "t", 30, agree_every=2)  # 50% agreement
    assert check_drift(store, cfg, "t") is True
    assert store.get_mode("t") == "shadow"
    assert "auto-demoted: audit agreement 50.0% over 30 requests" in store.events()[0]["detail"]
    # The mode change resets the window: no repeated demotion on the same evidence.
    assert check_drift(store, cfg, "t") is False


def test_no_demotion_with_too_few_samples(db_url):
    cfg = _config(db_url, min_agreement=0.9, min_samples=50)
    store = Store(cfg.database)
    _audits(store, "t", 49, agree_every=10)
    assert check_drift(store, cfg, "t") is False
    assert store.get_mode("t") is None


def test_no_demotion_when_agreement_is_good(db_url):
    cfg = _config(db_url, min_agreement=0.9, min_samples=10)
    store = Store(cfg.database)
    _audits(store, "t", 40, agree_every=1)
    assert check_drift(store, cfg, "t") is False


def test_window_uses_latest_audits(db_url):
    cfg = _config(db_url, min_agreement=0.9, window=20, min_samples=20)
    store = Store(cfg.database)
    _audits(store, "t", 100, agree_every=1)  # old, all good
    _audits(store, "t", 20, agree_every=4)  # recent, 25%
    assert check_drift(store, cfg, "t") is True


def test_concurrent_gateways_demote_once(db_url):
    """Two gateway processes see the same drift at the same time: one demotion, one event."""
    import threading

    cfg = _config(db_url, min_agreement=0.9, min_samples=10)
    setup = Store(db_url)
    setup.set_mode("t", "serve")
    _audits(setup, "t", 20, agree_every=5)
    stores = [Store(db_url) for _ in range(6)]  # separate connections, like separate processes
    results = []
    barrier = threading.Barrier(len(stores))

    def run(store):
        barrier.wait()
        results.append(check_drift(store, cfg, "t"))

    threads = [threading.Thread(target=run, args=(s,)) for s in stores]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1
    assert setup.get_mode("t") == "shadow"
    assert sum("auto-demoted" in e["detail"] for e in setup.events()) == 1
