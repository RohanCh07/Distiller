import json
import threading

import yaml

from distiller.data.pipeline import prepare
from distiller.spec import TaskSpec
from distiller.storage.db import Store
from distiller.tasks.discover import (
    adopt,
    classify_group,
    discover,
    infer_schema,
    suggest_name,
    write_candidate,
)


def _log_group(store, group, system, pairs):
    for text, answer in pairs:
        store.log_call(
            task=group,
            mode="observe",
            route="teacher",
            input=text,
            request=[{"role": "system", "content": system}, {"role": "user", "content": text}],
            teacher_output=answer,
        )


def test_classify_group():
    assert classify_group(["a", "b", "a", "b", "a"] * 10)[0] == "classification"
    t, _, extra = classify_group([json.dumps({"x": i}) for i in range(20)])
    assert t == "extraction" and extra["output_schema"]["required"] == ["x"]
    assert classify_group([f"answer number {i}" for i in range(50)])[0] == "generation"
    t, reason, _ = classify_group(["long essay " * 100] * 10)
    assert t is None and "long free-form" in reason


def test_infer_schema():
    objs = [
        {"dir": ["in", "out"][i % 2], "amt": i if i % 3 else None, "note": f"n{i}"}
        for i in range(30)
    ]
    objs[0]["extra"] = 1
    schema = infer_schema(objs)
    assert schema["required"] == ["amt", "dir", "note"]
    assert schema["properties"]["dir"] == {"enum": ["in", "out"]}
    assert schema["properties"]["amt"]["type"] == ["null", "number"]
    assert schema["properties"]["note"]["type"] == "string"  # too many distinct values for an enum
    import jsonschema

    for o in objs:
        jsonschema.validate(o, schema)


def test_suggest_name():
    assert suggest_name("Classify the bank customer's message", set()) == "classify_bank_customer"
    assert suggest_name(
        "Classify the bank customer's message", {"classify_bank_customer"}
    ).endswith("_2")
    assert suggest_name("", set()) == "task"


def test_discover_write_and_adopt(db_url, tmp_path):
    store = Store(db_url)
    labels = ["refund", "shipping", "other"]
    _log_group(
        store,
        "unassigned:aaaa",
        "Route the support ticket to a team",
        [(f"ticket {i}", labels[i % 3]) for i in range(60)],
    )
    _log_group(
        store,
        "unassigned:bbbb",
        "Extract the order id and total",
        [
            (f"order {i} total {i * 2}", json.dumps({"order": i, "total": i * 2.0}))
            for i in range(60)
        ],
    )
    _log_group(
        store, "unassigned:cccc", "Write an essay", [(f"q{i}", "word " * 200) for i in range(60)]
    )
    _log_group(store, "unassigned:dddd", "Rare prompt", [("x", "y")] * 5)
    _log_group(store, "known", "Tagged already", [("x", "y")] * 80)

    found = discover(store, min_calls=50, taken={"route_support_ticket"})
    by_group = {c.group: c for c in found}
    assert set(by_group) == {"unassigned:aaaa", "unassigned:bbbb", "unassigned:cccc"}
    tickets = by_group["unassigned:aaaa"]
    assert tickets.type == "classification" and tickets.labels == sorted(labels)
    assert tickets.name == "route_support_ticket_2"  # name taken in distiller.yaml
    assert by_group["unassigned:bbbb"].type == "extraction"
    assert not by_group["unassigned:cccc"].proposed

    # The written task must be trainable as-is: its data passes the pipeline.
    for cand in (tickets, by_group["unassigned:bbbb"]):
        path = write_candidate(store, cand, tmp_path / "tasks")
        spec = TaskSpec.load(path)
        assert yaml.safe_load(path.read_text())["instruction"] == cand.system_prompt
        prepared = prepare(spec)
        assert prepared.report.usable_examples == 60 and prepared.report.invalid_outputs == 0

    assert adopt(store, tickets) == 60
    assert {r["task"] for r in store.stats()} >= {"route_support_ticket_2"}
    assert not [c for c in discover(store, min_calls=50) if c.group == "unassigned:aaaa"]
    assert "adopted 60 calls" in store.events(kind="discovery")[0]["detail"]


# ---------- retrain jobs ----------


def test_retrain_queue_dedup_and_finish(db_url):
    store = Store(db_url)
    j1 = store.enqueue_retrain("t", "drift")
    assert j1 is not None and store.enqueue_retrain("t", "again") is None
    assert store.enqueue_retrain("u", "x") is not None
    job = store.claim_retrain()
    assert job["id"] == j1 and job["status"] == "running"
    assert store.enqueue_retrain("t", "while running") is None
    store.finish_retrain(j1, "done", "published")
    assert store.enqueue_retrain("t", "after") is not None
    assert "done: published" in [e["detail"] for e in store.events(kind="retrain")]


def test_claim_is_exclusive(db_url):
    store = Store(db_url)
    for i in range(5):
        store.enqueue_retrain(f"t{i}", "x")
    got, lock = [], threading.Lock()

    def worker():
        s = Store(db_url)
        while (job := s.claim_retrain()) is not None:
            with lock:
                got.append(job["id"])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(got) == sorted(set(got)) and len(got) == 5


def test_retrain_task_skips_and_fails_cleanly(tmp_path):
    from distiller.automation.retrain import work_once
    from distiller.gateway.config import GatewayConfig

    cfg = GatewayConfig.model_validate(
        {
            "database": str(tmp_path / "d.db"),
            "retrain": {"min_new_examples": 10},
            "tasks": {"nofile": {}, "few": {"task_file": str(tmp_path / "task.yaml")}},
        }
    )
    (tmp_path / "task.yaml").write_text(
        yaml.safe_dump({"name": "few", "type": "classification", "data": {"path": "data.jsonl"}})
    )
    store = cfg.open_store()
    assert work_once(cfg, store, log=lambda *_: None) is None
    store.enqueue_retrain("nofile", "x")
    assert work_once(cfg, store, log=lambda *_: None)["status"] == "failed"
    store.enqueue_retrain("few", "x")
    job = work_once(cfg, store, log=lambda *_: None)
    assert job["status"] == "skipped" and "only 0 logged" in job["detail"]
