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
def gateway(config: Path = CONFIG, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the OpenAI-compatible gateway. Point OPENAI_BASE_URL at http://host:port/v1."""
    import logging

    import uvicorn

    from distiller.gateway.app import create_app
    from distiller.gateway.config import GatewayConfig

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(GatewayConfig.load(config)), host=host, port=port)


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
