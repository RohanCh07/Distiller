import json
from pathlib import Path

import typer

from distiller import __version__

app = typer.Typer(help="Distiller: turn repetitive LLM tasks into small, cheap models.")
data_app = typer.Typer(help="Inspect and prepare task data.")
teacher_app = typer.Typer(help="Use the big model to label inputs or create synthetic data.")
logs_app = typer.Typer(help="Work with traffic logged by the gateway.")
task_app = typer.Typer(help="Change how the gateway handles a task.")
keys_app = typer.Typer(help="Gateway API keys and the log encryption key.")
app.add_typer(data_app, name="data")
app.add_typer(teacher_app, name="teacher")
app.add_typer(logs_app, name="logs")
app.add_typer(task_app, name="task")
app.add_typer(keys_app, name="keys")

TASK = typer.Argument(..., help="Path to task.yaml")
CONFIG = typer.Option("distiller.yaml", "--config", "-c", help="Gateway config file")


@app.command()
def version() -> None:
    print(__version__)


# ---------- engine ----------


@data_app.command("check")
def data_check(task: Path = TASK) -> None:
    """Validate, dedup and split the data; write a data health report."""
    from distiller.data.pipeline import data_dir, prepare
    from distiller.spec import TaskSpec

    spec = TaskSpec.load(task)
    prepared = prepare(spec)
    print(prepared.report.to_markdown())
    print(f"Written to {data_dir(spec)}")


@app.command()
def train(
    task: Path = TASK,
    candidates: str = typer.Option(
        None, help="Comma-separated backends, e.g. encoder,llm-lora (default: by task type)"
    ),
    sweep: bool = typer.Option(None, "--sweep/--no-sweep", help="Try a small hyperparameter grid"),
) -> None:
    """Prepare data, train and compare students, calibrate, and write report.md."""
    from distiller.engine import run
    from distiller.spec import TaskSpec

    spec = TaskSpec.load(task)
    names = [c.strip() for c in candidates.split(",")] if candidates else None
    result = run(spec, names, sweep=sweep)
    print(result.report_path.read_text(encoding="utf-8"))


@app.command()
def predict(
    model_dir: Path = typer.Argument(..., help="runs/<task>/models/<backend>"),
    text: list[str] = typer.Argument(..., help="One or more inputs"),
) -> None:
    """Run a trained student on some inputs (shows calibrated confidence and route)."""
    from distiller.data.io import serialize_output
    from distiller.serving.registry import LoadedStudent

    student = LoadedStudent(model_dir)
    for t, ans in zip(text, student.answer(text)):
        route = "student" if ans.confident else "teacher (low confidence)"
        print(
            f"{t!r}\n  -> {serialize_output(ans.output)}  conf={ans.confidence:.3f}  route={route}"
        )


@app.command()
def export(
    model_dir: Path = typer.Argument(..., help="runs/<task>/models/llm-lora"),
    out: Path = typer.Option(..., help="Folder for the merged model"),
) -> None:
    """Merge a LoRA adapter into its base model (for vLLM, or GGUF conversion via llama.cpp)."""
    import shutil

    from distiller.training.base import Student

    student = Student.load(model_dir)
    if not hasattr(student, "export_merged"):
        raise typer.BadParameter("only llm-lora students have an adapter to merge")
    student.export_merged(out)
    shutil.copy(model_dir / "routing.json", out / "routing.json")
    print(f"Merged model written to {out}")


# ---------- teacher ----------


@teacher_app.command("label")
def teacher_label(
    task: Path = TASK,
    inputs: Path = typer.Option(..., help="Text file, one input per line"),
    out: Path = typer.Option(..., help="JSONL to write (input/output rows)"),
    model: str = typer.Option(None, help="Teacher model (or DISTILLER_TEACHER_MODEL)"),
) -> None:
    """Get teacher outputs for raw inputs."""
    from distiller.spec import TaskSpec
    from distiller.teacher import Teacher, TeacherConfig

    spec = TaskSpec.load(task)
    texts = [ln.strip() for ln in inputs.read_text(encoding="utf-8").splitlines() if ln.strip()]
    outputs = Teacher(TeacherConfig.from_env(model)).label(spec, texts)
    kept = 0
    with out.open("w", encoding="utf-8") as f:
        for t, o in zip(texts, outputs):
            if o is not None:
                f.write(json.dumps({"input": t, "output": o}, ensure_ascii=False) + "\n")
                kept += 1
    print(f"Labelled {kept}/{len(texts)} inputs -> {out} ({len(texts) - kept} failed validation)")


@teacher_app.command("augment")
def teacher_augment(
    task: Path = TASK,
    out: Path = typer.Option(..., help="JSONL to append synthetic rows to"),
    per_label: int = typer.Option(None, help="Classification: top every label up to this many"),
    total: int = typer.Option(None, help="Extraction/generation: how many new examples"),
    model: str = typer.Option(None, help="Teacher model (or DISTILLER_TEACHER_MODEL)"),
) -> None:
    """Create new teacher-labelled examples (training only; never used for val/test)."""
    from distiller.data.io import load_examples
    from distiller.spec import TaskSpec
    from distiller.teacher import Teacher, TeacherConfig, to_row

    spec = TaskSpec.load(task)
    real = load_examples(spec).examples
    new = Teacher(TeacherConfig.from_env(model)).augment(spec, real, per_label, total)
    with out.open("a", encoding="utf-8") as f:
        for ex in new:
            f.write(to_row(ex) + "\n")
    print(f"Added {len(new)} synthetic examples to {out}")
    if spec.data.synthetic_path is None or spec.data.synthetic_path.resolve() != out.resolve():
        print("Set data.synthetic_path in task.yaml to this file to use them.")


# ---------- gateway ----------


@app.command()
def gateway(
    config: Path = CONFIG,
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = typer.Option(1, help="Worker processes (each loads its own students)"),
) -> None:
    """Run the OpenAI-compatible gateway. Point OPENAI_BASE_URL at http://host:port/v1.

    Several workers (or several machines) can share one database: task modes, keys and
    drift demotion live there. Use Postgres when the gateways are on different machines.
    """
    import logging
    import os

    import uvicorn

    from distiller.gateway.app import create_app
    from distiller.gateway.config import GatewayConfig

    logging.basicConfig(level=logging.INFO)
    cfg = GatewayConfig.load(config)  # validate before forking workers
    if workers > 1:
        os.environ["DISTILLER_CONFIG"] = str(config.resolve())
        uvicorn.run(
            "distiller.gateway.app:app_from_env",
            factory=True,
            host=host,
            port=port,
            workers=workers,
        )
    else:
        uvicorn.run(create_app(cfg), host=host, port=port)


@app.command("serve-vllm")
def serve_vllm(
    model_dir: Path = typer.Argument(..., help="runs/<task>/models/llm-lora, or a merged export"),
    port: int = typer.Option(8001),
    print_only: bool = typer.Option(False, "--print", help="Print the command, do not run it"),
    extra: list[str] = typer.Argument(None, help="Extra vLLM flags after --"),
) -> None:
    """Serve an LLM student with vLLM (GPU). Then set the task's serving.backend: vllm.

    An adapter folder is served as a LoRA on its base model; a merged export (from
    `distiller export`) is served directly.
    """
    import os
    import shlex

    meta = json.loads((model_dir / "distiller.json").read_text())
    if meta["backend"] != "llm-lora":
        raise typer.BadParameter("only llm-lora students can be served by vLLM")
    name = meta["spec"]["name"]
    if (model_dir / "adapter_config.json").exists():
        cmd = [
            "vllm", "serve", meta["hparams"]["model"],
            "--enable-lora", "--lora-modules", f"{name}={model_dir.resolve()}",
            "--max-lora-rank", str(meta["hparams"].get("lora_r", 16)),
        ]  # fmt: skip
    else:
        cmd = ["vllm", "serve", str(model_dir.resolve()), "--served-model-name", name]
    cmd += ["--port", str(port), *(extra or [])]
    print(shlex.join(cmd))
    print(f"Gateway config:  tasks.{name}.serving: {{backend: vllm, url: http://<host>:{port}/v1}}")
    if not print_only:
        os.execvp(cmd[0], cmd)


@logs_app.command("export")
def logs_export(
    task: str = typer.Argument(..., help="Task name"),
    out: Path = typer.Option(..., help="JSONL to write (chat format)"),
    config: Path = CONFIG,
) -> None:
    """Turn logged teacher answers into a training file for `distiller train`."""
    from distiller.gateway.config import GatewayConfig

    store = GatewayConfig.load(config).open_store()
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for row in store.export_training_rows(task):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Exported {n} examples for {task} -> {out}")


@logs_app.command("stats")
def logs_stats(config: Path = CONFIG) -> None:
    """Per-task traffic, routing and agreement."""
    from distiller.gateway.config import GatewayConfig

    store = GatewayConfig.load(config).open_store()
    print(json.dumps({"tasks": store.stats(), "events": store.events()}, indent=2, default=str))


@task_app.command("mode")
def task_mode(
    task: str = typer.Argument(...),
    mode: str = typer.Argument(..., help="off | observe | shadow | serve"),
    config: Path = CONFIG,
) -> None:
    """Switch a task's mode (takes effect on the next request; no restart)."""
    from distiller.gateway.config import GatewayConfig, Mode

    GatewayConfig.load(config).open_store().set_mode(task, Mode(mode).value, reason="cli")
    print(f"{task} -> {mode}")


@task_app.command("discover")
def task_discover(
    config: Path = CONFIG,
    min_calls: int = typer.Option(100, help="Ignore prompt groups with fewer logged answers"),
    write: Path = typer.Option(None, help="Write <dir>/<name>/task.yaml + data.jsonl per proposal"),
    adopt: bool = typer.Option(False, help="Also move the group's logged calls to the new name"),
) -> None:
    """Find repeated untagged prompts that could become tasks."""
    from distiller.gateway.config import GatewayConfig
    from distiller.tasks import discover as disc

    cfg = GatewayConfig.load(config)
    store = cfg.open_store()
    found = disc.discover(store, min_calls=min_calls, taken=set(cfg.tasks))
    if not found:
        print(f"No untagged prompt group has {min_calls}+ logged teacher answers yet.")
        return
    for c in found:
        mark = "proposed" if c.proposed else "skipped"
        print(f"{c.group}  {c.calls} calls  -> {mark}: {c.type or '-'} ({c.reason})")
        print(f"    name: {c.name}   prompt: {c.system_prompt[:70]!r}")
        if c.labels:
            print(
                f"    labels ({len(c.labels)}): {', '.join(c.labels[:8])}{' ...' if len(c.labels) > 8 else ''}"
            )
        if not c.proposed:
            continue
        if write:
            path = disc.write_candidate(store, c, write)
            print(f"    wrote {path}")
        if adopt:
            print(f"    moved {disc.adopt(store, c)} calls to task {c.name}")
            print(
                f"    add to distiller.yaml:  tasks: {{{c.name}: "
                f"{{mode: observe, match_system_prompt: {c.match_system_prompt!r}}}}}"
            )


@app.command()
def retrain(task: str = typer.Argument(...), config: Path = CONFIG) -> None:
    """Queue a retrain of a task on its original data plus logged traffic."""
    from distiller.gateway.config import GatewayConfig

    job = GatewayConfig.load(config).open_store().enqueue_retrain(task, "cli")
    print(f"Queued job {job}" if job else f"A retrain for {task} is already queued or running")


@app.command("retrain-worker")
def retrain_worker(
    config: Path = CONFIG,
    once: bool = typer.Option(False, help="Run at most one job, then exit"),
    poll: float = typer.Option(30, help="Seconds between queue checks"),
) -> None:
    """Run queued retrain jobs (on a machine that can train; a GPU for LLM students)."""
    from distiller.automation.retrain import work_once, worker_loop
    from distiller.gateway.config import GatewayConfig

    cfg = GatewayConfig.load(config)
    store = cfg.open_store()
    if once:
        job = work_once(cfg, store)
        print("No queued jobs" if job is None else f"{job['status']}: {job['detail']}")
    else:
        worker_loop(cfg, store, poll_seconds=poll)


# ---------- onboarding ----------


@app.command()
def init(
    directory: Path = typer.Argument(Path("."), help="Where to write distiller.yaml and tasks/"),
    task: str = typer.Option(None, help="Task name, e.g. ticket_router"),
    type_: str = typer.Option(None, "--type", help="classification | extraction | generation"),
    instruction: str = typer.Option(None, help="The system prompt your app sends today"),
    labels: str = typer.Option(None, help="Classification labels, comma-separated"),
    teacher_url: str = typer.Option(None, help="OpenAI-compatible base URL of the big model"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept defaults; no questions"),
    force: bool = typer.Option(False, help="Overwrite an existing distiller.yaml"),
) -> None:
    """Set up a project: distiller.yaml, one task, an admin key. Asks a few questions."""
    from distiller.onboarding import TYPES, write_setup
    from distiller.security.keys import generate_key, hash_key

    def ask(value, prompt, default):
        if value is not None or yes:
            return value if value is not None else default
        return typer.prompt(prompt, default=default)

    task = ask(task, "Task name", "my_task")
    type_ = ask(type_, f"Task type ({' / '.join(TYPES)})", "classification")
    instruction = ask(instruction, "System prompt your app uses for it", "Classify the message.")
    if type_ == "classification":
        labels = ask(labels, "Labels, comma-separated (empty: learn from data)", "")
    teacher_url = ask(teacher_url, "Teacher API base URL", "https://api.openai.com/v1")
    label_list = [x.strip() for x in (labels or "").split(",") if x.strip()] or None
    try:
        paths = write_setup(
            directory, task, type_, instruction, label_list, teacher_url,
            "OPENAI_API_KEY", "distiller.db", overwrite=force,
        )  # fmt: skip
    except (FileExistsError, ValueError) as e:
        raise typer.BadParameter(str(e)) from e
    from distiller.gateway.config import GatewayConfig

    admin = generate_key()
    GatewayConfig.load(paths[0]).open_store().add_api_key("admin", hash_key(admin), "admin")
    for p in paths:
        print(f"wrote {p}")
    print(
        f"""
Admin key (shown once; needed for /dashboard, /stats and /admin):
  {admin}

Next:
  export OPENAI_API_KEY=sk-...             # the teacher key
  distiller doctor -c {paths[0]}
  distiller gateway -c {paths[0]}          # task starts in observe mode
  # point your app at http://localhost:8000/v1 and send  X-Distiller-Task: {task}
  # open http://localhost:8000/dashboard?key=<admin key>
  # after a few hundred calls:
  distiller logs export {task} --out {paths[2]} -c {paths[0]}
  distiller train {paths[1]}
  distiller task mode {task} shadow -c {paths[0]}"""
    )


@app.command()
def doctor(
    config: Path = CONFIG,
    offline: bool = typer.Option(False, help="Skip the teacher connectivity check"),
) -> None:
    """Check a setup: config, database, extras, security, teacher, students."""
    from distiller.onboarding import doctor as run_doctor

    checks = run_doctor(config, online=not offline)
    for c in checks:
        print(c.line())
    fails = sum(c.status == "fail" for c in checks)
    warns = sum(c.status == "warn" for c in checks)
    print(f"\n{fails} problems, {warns} warnings")
    raise typer.Exit(1 if fails else 0)


# ---------- keys ----------


@keys_app.command("create")
def keys_create(
    name: str = typer.Argument(..., help="A label, e.g. the app that will use the key"),
    role: str = typer.Option(
        "app", help="app (call /v1) or admin (also /admin, /stats, /dashboard)"
    ),
    config: Path = CONFIG,
) -> None:
    """Create a gateway API key. It is printed once; only its hash is stored."""
    from distiller.gateway.config import GatewayConfig
    from distiller.security.keys import ROLES, generate_key, hash_key

    if role not in ROLES:
        raise typer.BadParameter(f"role must be one of {ROLES}")
    key = generate_key()
    GatewayConfig.load(config).open_store().add_api_key(name, hash_key(key), role)
    print(key)
    print(f"Saved {role} key {name!r}. Store it now: it cannot be shown again.", flush=True)


@keys_app.command("list")
def keys_list(config: Path = CONFIG) -> None:
    """List keys (names, roles, last use); never the keys themselves."""
    from distiller.gateway.config import GatewayConfig

    rows = GatewayConfig.load(config).open_store().list_api_keys()
    print(json.dumps(rows, indent=2))


@keys_app.command("revoke")
def keys_revoke(name: str = typer.Argument(...), config: Path = CONFIG) -> None:
    """Revoke a key by name (takes effect on the next request)."""
    from distiller.gateway.config import GatewayConfig

    if not GatewayConfig.load(config).open_store().revoke_api_key(name):
        raise typer.BadParameter(f"no active key named {name!r}")
    print(f"Revoked {name}")


@keys_app.command("encryption-key")
def keys_encryption_key() -> None:
    """Print a new key for security.encrypt_logs (put it in DISTILLER_ENCRYPTION_KEY)."""
    from distiller.security.crypto import FieldCipher

    print(FieldCipher.generate_key())


@keys_app.command("rotate-encryption")
def keys_rotate_encryption(config: Path = CONFIG) -> None:
    """Re-encrypt stored logs with the first key in DISTILLER_ENCRYPTION_KEY.

    To rotate: put the new key first and the old one after it (comma-separated), run this,
    then remove the old key.
    """
    from distiller.gateway.config import GatewayConfig

    n = GatewayConfig.load(config).open_store().rotate_encryption()
    print(f"Re-encrypted {n} rows")
