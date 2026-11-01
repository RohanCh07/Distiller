"""Retrain a task on its original data plus the traffic the gateway logged since.

Jobs are rows in the database (queued by the gateway on drift demotion, or by
`distiller retrain <task>`). `distiller retrain-worker` claims them one at a time, so
training never runs inside a gateway process and several workers can share a queue.

A new model is published (runs/<task>/best.json is pointed at it) only when it is
promotable, i.e. it reaches the target accuracy on some confident slice of a test set
that now includes recent traffic. Gateways notice the new best.json and reload it; the
task stays in shadow mode until a person (or `promote.enabled`) moves it to serve.
"""

import json
import time
from pathlib import Path

from distiller.gateway.config import GatewayConfig
from distiller.spec import TaskSpec
from distiller.storage.db import Store


def runs_dir_for(config: GatewayConfig, task: str) -> Path:
    return config.tasks[task].runs_dir / task


def retrain_task(config: GatewayConfig, store: Store, task: str, log=print) -> tuple[str, str]:
    """Run one retrain. Returns (status, detail): done | skipped | failed."""
    from distiller.engine import run

    settings = config.tasks.get(task)
    if settings is None or settings.task_file is None:
        return "failed", f"no task_file configured for {task} in distiller.yaml"
    spec = TaskSpec.load(settings.task_file)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = runs_dir_for(config, task) / "retrain" / stamp
    work.mkdir(parents=True, exist_ok=True)

    logged = list(store.export_training_rows(task))
    if len(logged) < config.retrain.min_new_examples:
        return "skipped", (
            f"only {len(logged)} logged teacher answers; need {config.retrain.min_new_examples}"
        )

    # Original rows + logged chat rows in one file (the loader accepts both shapes per row).
    combined = work / "data.jsonl"
    with combined.open("w", encoding="utf-8") as f:
        if spec.data.path.exists():
            f.write(spec.data.path.read_text(encoding="utf-8").rstrip("\n") + "\n")
        for row in logged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    new_spec = spec.model_copy(deep=True)
    new_spec.data.path = combined
    new_spec.output_dir = work / "runs"
    log(f"[retrain] {task}: {len(logged)} logged + original data -> {combined}")

    result = run(new_spec, log=log)
    best = result.best
    if best is None or not best.promotable:
        return (
            "done",
            f"new model not promotable; kept the current one (report {result.report_path})",
        )

    best_file = runs_dir_for(config, task) / "best.json"
    best_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = best_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"backend": best.backend, "model_dir": best.model_dir}, indent=2))
    tmp.replace(best_file)  # atomic: gateways never read a half-written file
    cov = best.threshold.get("test_coverage", 0)
    return "done", f"published {best.backend} ({cov:.1%} coverage at target) from {work}"


def work_once(config: GatewayConfig, store: Store, log=print) -> dict | None:
    job = store.claim_retrain()
    if job is None:
        return None
    try:
        status, detail = retrain_task(config, store, job["task"], log=log)
    except Exception as e:  # report, keep the worker alive
        status, detail = "failed", f"{type(e).__name__}: {e}"[:500]
    store.finish_retrain(job["id"], status, detail)
    log(f"[retrain] job {job['id']} {job['task']}: {status}: {detail}")
    return {**job, "status": status, "detail": detail}


def worker_loop(config: GatewayConfig, store: Store, poll_seconds: float = 30, log=print) -> None:
    while True:
        if work_once(config, store, log=log) is None:
            time.sleep(poll_seconds)
