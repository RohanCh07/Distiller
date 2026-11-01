import json
import threading

from distiller.storage.db import Store


def test_log_and_update_call(tmp_path):
    store = Store(tmp_path / "sub" / "d.db")
    cid = store.log_call(
        task="t",
        mode="observe",
        route="teacher",
        input="hi",
        request=[{"role": "user", "content": "hi"}],
    )
    store.update_call(cid, student_output="x", agree=1)
    row = store._db.execute("SELECT * FROM calls WHERE id = ?", (cid,)).fetchone()
    assert row["student_output"] == "x" and row["agree"] == 1
    assert json.loads(row["request"]) == [{"role": "user", "content": "hi"}]
    assert row["ts"] > 0


def test_modes_and_events(tmp_path):
    store = Store(tmp_path / "d.db")
    assert store.get_mode("t") is None
    store.set_mode("t", "shadow", reason="cli")
    store.set_mode("t", "serve")
    assert store.get_mode("t") == "serve"
    events = store.events()
    assert [e["detail"] for e in events] == ["serve (manual)", "shadow (cli)"]
    assert all(e["kind"] == "mode" for e in events)
    assert len(store.events(limit=1)) == 1


def test_mode_persists_across_instances(tmp_path):
    Store(tmp_path / "d.db").set_mode("t", "serve")
    assert Store(tmp_path / "d.db").get_mode("t") == "serve"


def test_recent_agreement_window_and_mode_reset(tmp_path):
    store = Store(tmp_path / "d.db")
    assert store.recent_agreement("t", 10) == (0, 1.0)
    for i in range(10):
        store.log_call(
            task="t", mode="serve", route="student", audited=1, agree=int(i < 3), ts=100 + i
        )
    store.log_call(
        task="t", mode="serve", route="student", audited=0, agree=0, ts=200
    )  # not audited
    store.log_call(
        task="t", mode="serve", route="student", audited=1, agree=None, ts=201
    )  # pending
    store.log_call(task="other", mode="serve", route="student", audited=1, agree=0, ts=202)
    assert store.recent_agreement("t", 100) == (10, 0.3)
    assert store.recent_agreement("t", 5) == (5, 0.0)  # newest five are all disagreements
    store.set_mode("t", "serve")  # a mode change starts a fresh window
    assert store.recent_agreement("t", 100) == (0, 1.0)


def test_stats(tmp_path):
    store = Store(tmp_path / "d.db")
    common = {"task": "a", "prompt_tokens": 10, "completion_tokens": 5}
    store.log_call(mode="shadow", route="teacher", agree=1, teacher_ms=100, **common)
    store.log_call(mode="shadow", route="teacher", agree=0, teacher_ms=300, **common)
    store.log_call(mode="serve", route="student", student_ms=4, audited=1, agree=1, task="a")
    store.log_call(mode="serve", route="fallback", teacher_ms=200, **common)
    store.log_call(mode="observe", route="teacher", task="b")
    by_task = {r["task"]: r for r in store.stats()}
    a = by_task["a"]
    assert a["calls"] == 4
    assert a["served_by_student"] == 1 and a["fallbacks"] == 1 and a["teacher_only"] == 2
    assert a["shadow_agreement"] == 0.5 and a["shadow_samples"] == 2
    assert a["audit_agreement"] == 1.0 and a["audit_samples"] == 1
    assert a["teacher_ms"] == 200 and a["student_ms"] == 4
    assert a["teacher_tokens_per_call"] == 15
    assert store.stats()[0]["task"] == "a"  # busiest first
    assert by_task["b"]["shadow_agreement"] is None


def test_export_training_rows(tmp_path):
    store = Store(tmp_path / "d.db")
    req = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "q2"},
    ]
    store.log_call(task="t", mode="observe", route="teacher", request=req, teacher_output="A")
    store.log_call(
        task="t", mode="serve", route="student", request=req, student_output="B"
    )  # no teacher
    store.log_call(task="u", mode="observe", route="teacher", request=req, teacher_output="C")
    rows = list(store.export_training_rows("t"))
    assert rows == [
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "q1"},
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "A"},
            ]
        }
    ]


def test_concurrent_writes(tmp_path):
    store = Store(tmp_path / "d.db")

    def work():
        for _ in range(50):
            cid = store.log_call(task="t", mode="observe", route="teacher")
            store.update_call(cid, agree=1)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert store.stats()[0]["calls"] == 400
