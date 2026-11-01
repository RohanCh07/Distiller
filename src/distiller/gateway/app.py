"""OpenAI-compatible gateway: POST /v1/chat/completions.

Per task mode:
  off      passthrough, nothing logged
  observe  teacher answers, call logged
  shadow   teacher answers; student runs in the background, agreement logged
  serve    student answers if confident, else teacher (logged as fallback = new training
           data); a small audited share also goes to the teacher to watch for drift

Fails open: if anything in Distiller breaks, the request goes straight to the teacher.
"""

import asyncio
import json
from contextlib import asynccontextmanager
import logging
import time
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from distiller.data.io import input_from_messages, serialize_output
from distiller.eval.metrics import is_correct
from distiller.gateway.config import GatewayConfig, Mode, TaskSettings
from distiller.router.policy import check_drift, should_audit
from distiller.spec import TaskSpec, TaskType
from distiller.storage.db import Store
from distiller.tasks.identify import identify

log = logging.getLogger("distiller.gateway")


def _parse_teacher(spec: TaskSpec, content: str) -> Any:
    if spec.type == TaskType.extraction:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    return content.strip()


def _student_response(task: str, output: Any, confidence: float) -> dict:
    return {
        "id": f"distiller-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"distiller/{task}",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": serialize_output(output)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "distiller": {"route": "student", "confidence": round(confidence, 4)},
    }


def _resolve_model_dir(task: str, settings: TaskSettings):
    if settings.model_dir is None:
        return None
    if str(settings.model_dir) == "best":
        best = settings.runs_dir / task / "best.json"
        return json.loads(best.read_text())["model_dir"] if best.exists() else None
    return settings.model_dir


def create_app(config: GatewayConfig) -> FastAPI:
    store = Store(config.database)
    students: dict[str, Any] = {}
    background: set[asyncio.Task] = set()
    teacher = httpx.AsyncClient(
        base_url=config.teacher.base_url.rstrip("/"), timeout=config.teacher.timeout_seconds
    )

    def load_students() -> None:
        from distiller.serving.registry import LoadedStudent

        for name, settings in config.tasks.items():
            model_dir = _resolve_model_dir(name, settings)
            if not model_dir:
                continue
            try:
                students[name] = LoadedStudent(model_dir)
                log.info("loaded student for %s from %s", name, model_dir)
            except Exception:  # a broken model must not stop the gateway
                log.exception("could not load student for %s; task stays teacher-only", name)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await run_in_threadpool(load_students)
        yield
        await teacher.aclose()

    app = FastAPI(title="Distiller gateway", lifespan=lifespan)

    def mode_of(task: str) -> Mode:
        stored = store.get_mode(task)
        if stored:
            return Mode(stored)
        settings = config.tasks.get(task)
        return settings.mode if settings else Mode.observe

    def auth_headers(request: Request) -> dict[str, str]:
        auth = request.headers.get("authorization")
        if not auth and config.teacher.api_key:
            auth = f"Bearer {config.teacher.api_key}"
        return {"Authorization": auth} if auth else {}

    async def call_teacher(body: dict, headers: dict) -> tuple[int, dict, float]:
        start = time.perf_counter()
        try:
            r = await teacher.post("/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as e:
            log.error("teacher unreachable: %s", e)
            return 502, {"error": {"message": f"teacher unreachable: {e}", "type": "bad_gateway"}}, 0.0
        ms = 1000 * (time.perf_counter() - start)
        try:
            data = r.json()
        except ValueError:
            data = {"error": {"message": r.text}}
        return r.status_code, data, ms

    def teacher_fields(status: int, data: dict, ms: float) -> dict[str, Any]:
        if status != 200:
            return {"teacher_ms": ms}
        usage = data.get("usage") or {}
        return {
            "teacher_output": data["choices"][0]["message"].get("content"),
            "teacher_ms": ms,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "model": data.get("model"),
        }

    def spawn(coro) -> None:
        t = asyncio.create_task(coro)
        background.add(t)
        t.add_done_callback(background.discard)

    async def shadow_compare(task: str, call_id: int, text: str, teacher_output: str | None):
        try:
            student = students[task]
            start = time.perf_counter()
            ans = (await run_in_threadpool(student.answer, [text]))[0]
            ms = 1000 * (time.perf_counter() - start)
            gold = _parse_teacher(student.spec, teacher_output) if teacher_output else None
            agree = None if gold is None else int(is_correct(student.spec, ans.output, gold))
            store.update_call(
                call_id,
                student_output=serialize_output(ans.output) if ans.output is not None else None,
                student_confidence=ans.confidence,
                student_ms=ms,
                agree=agree,
            )
        except Exception:
            log.exception("shadow comparison failed for %s", task)

    async def audit(task: str, call_id: int, body: dict, headers: dict, output: Any):
        try:
            status, data, ms = await call_teacher(body, headers)
            fields = teacher_fields(status, data, ms)
            spec = students[task].spec
            if fields.get("teacher_output") is not None:
                gold = _parse_teacher(spec, fields["teacher_output"])
                fields["agree"] = int(is_correct(spec, output, gold)) if gold is not None else None
            store.update_call(call_id, **fields)
            if check_drift(store, config, task):
                log.warning("task %s auto-demoted to shadow", task)
        except Exception:
            log.exception("audit failed for %s", task)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "students": sorted(students)}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        headers = auth_headers(request)
        messages = body.get("messages") or []

        # Streaming is passed straight through (student serving is non-streaming for now).
        if body.get("stream"):
            req = teacher.build_request("POST", "/chat/completions", json=body, headers=headers)
            upstream = await teacher.send(req, stream=True)
            return StreamingResponse(
                upstream.aiter_raw(),
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type"),
                background=BackgroundTask(upstream.aclose),
            )

        try:
            task = identify(request.headers, messages, config)
            mode = mode_of(task)
            text = input_from_messages(messages)
        except Exception:
            log.exception("routing failed; passing through")
            task, mode, text = "unassigned", Mode.off, ""

        if mode == Mode.off:
            status, data, _ = await call_teacher(body, headers)
            return JSONResponse(data, status_code=status)

        has_student = task in students
        if mode == Mode.serve and has_student:
            try:
                start = time.perf_counter()
                ans = (await run_in_threadpool(students[task].answer, [text]))[0]
                ms = 1000 * (time.perf_counter() - start)
                if ans.confident:
                    audited = should_audit(config)
                    call_id = store.log_call(
                        task=task, mode=mode.value, route="student", audited=int(audited),
                        input=text, request=messages,
                        student_output=serialize_output(ans.output),
                        student_confidence=ans.confidence, student_ms=ms,
                    )
                    if audited:
                        spawn(audit(task, call_id, body, headers, ans.output))
                    resp = JSONResponse(_student_response(task, ans.output, ans.confidence))
                    resp.headers["x-distiller-route"] = "student"
                    return resp
                route = "fallback"
                student_fields = {
                    "student_output": serialize_output(ans.output) if ans.output is not None else None,
                    "student_confidence": ans.confidence,
                    "student_ms": ms,
                }
            except Exception:
                log.exception("student failed for %s; falling back to teacher", task)
                route, student_fields = "fallback", {}
        else:
            route, student_fields = "teacher", {}

        status, data, ms = await call_teacher(body, headers)
        fields = teacher_fields(status, data, ms)
        try:
            call_id = store.log_call(
                task=task, mode=mode.value, route=route, input=text, request=messages,
                **fields, **student_fields,
            )
            if mode == Mode.shadow and has_student and status == 200:
                spawn(shadow_compare(task, call_id, text, fields.get("teacher_output")))
        except Exception:
            log.exception("logging failed for %s", task)
        resp = JSONResponse(data, status_code=status)
        resp.headers["x-distiller-route"] = route
        return resp

    @app.get("/v1/models")
    async def models(request: Request):
        r = await teacher.get("/models", headers=auth_headers(request))
        return Response(r.content, status_code=r.status_code, media_type="application/json")

    @app.post("/admin/tasks/{task}/mode")
    async def set_mode(task: str, payload: dict):
        mode = Mode(payload["mode"])
        store.set_mode(task, mode.value, reason="admin api")
        return {"task": task, "mode": mode.value}

    @app.get("/stats")
    def stats() -> dict:
        rows = store.stats()
        for r in rows:
            r["mode"] = mode_of(r["task"]).value
            r["has_student"] = r["task"] in students
        return {"tasks": rows, "events": store.events()}

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard(stats())

    app.state.store = store
    app.state.students = students
    return app


def _fmt(v: Any, pct: bool = False) -> str:
    if v is None:
        return "–"
    if pct:
        return f"{v:.1%}"
    if isinstance(v, float):
        return f"{v:.0f}"
    return str(v)


def render_dashboard(data: dict) -> str:
    rows = "".join(
        f"<tr><td>{t['task']}</td><td>{t['mode']}{' ✓' if t['has_student'] else ''}</td>"
        f"<td>{t['calls']}</td>"
        f"<td>{_fmt((t['served_by_student'] or 0) / t['calls'], pct=True)}</td>"
        f"<td>{_fmt(t['fallbacks'])}</td>"
        f"<td>{_fmt(t['shadow_agreement'], pct=True)} ({t['shadow_samples']})</td>"
        f"<td>{_fmt(t['audit_agreement'], pct=True)} ({t['audit_samples']})</td>"
        f"<td>{_fmt(t['teacher_ms'])} / {_fmt(t['student_ms'])}</td>"
        f"<td>{_fmt((t['served_by_student'] or 0) * (t['teacher_tokens_per_call'] or 0))}</td></tr>"
        for t in data["tasks"]
    )
    events = "".join(
        f"<li>{time.strftime('%Y-%m-%d %H:%M', time.localtime(e['ts']))} · {e['task']} · {e['detail']}</li>"
        for e in data["events"]
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Distiller</title>
<meta http-equiv="refresh" content="10">
<style>body{{font-family:system-ui;margin:24px;color:#222}}table{{border-collapse:collapse}}
td,th{{border-bottom:1px solid #ddd;padding:6px 12px;text-align:left}}th{{background:#f5f5f5}}</style>
</head><body><h1>Distiller</h1>
<table><tr><th>Task</th><th>Mode (✓ student loaded)</th><th>Calls</th><th>Served by student</th>
<th>Fallbacks</th><th>Shadow agreement (n)</th><th>Audit agreement (n)</th>
<th>Latency ms teacher / student</th><th>Teacher tokens avoided (est.)</th></tr>{rows}</table>
<h2>Events</h2><ul>{events or '<li>none</li>'}</ul></body></html>"""
