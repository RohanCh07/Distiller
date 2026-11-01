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
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from distiller.data.io import input_from_messages, serialize_output
from distiller.eval.metrics import is_correct
from distiller.gateway.config import GatewayConfig, Mode, TaskSettings
from distiller.router.policy import check_drift, should_audit
from distiller.security.keys import hash_key, key_from_headers, same
from distiller.spec import TaskSpec, TaskType
from distiller.tasks.identify import identify

log = logging.getLogger("distiller.gateway")


class ModeUpdate(BaseModel):
    mode: Mode


class Caller(BaseModel):
    name: str
    role: str  # app | admin
    via_authorization: bool  # the key came in the Authorization header


def _auth_error(status: int, message: str) -> HTTPException:
    return HTTPException(status, detail={"error": {"message": message, "type": "invalid_api_key"}})


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


def _student_sse(task: str, output: Any, confidence: float, body: dict):
    """A student answer as an OpenAI chat.completion.chunk stream.

    The whole answer is computed first (we must know the confidence before we commit to
    the student), then sent in small pieces so streaming clients work unchanged.
    """
    cid = f"distiller-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    text = serialize_output(output)

    def chunk(delta: dict, finish: str | None = None, **extra) -> str:
        data = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": f"distiller/{task}",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            **extra,
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def gen():
        yield chunk({"role": "assistant", "content": ""})
        for i in range(0, len(text), 16):
            yield chunk({"content": text[i : i + 16]})
        extra = {"distiller": {"route": "student", "confidence": round(confidence, 4)}}
        yield chunk({}, "stop", **extra)
        if (body.get("stream_options") or {}).get("include_usage"):
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': f'distiller/{task}', 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return gen()


def _parse_sse(raw: bytes) -> dict[str, Any]:
    """Rebuild the teacher's answer (and usage, if sent) from a captured SSE stream."""
    parts: list[str] = []
    out: dict[str, Any] = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]" or not payload:
            continue
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        for choice in data.get("choices") or []:
            if choice.get("index", 0) == 0:
                parts.append((choice.get("delta") or {}).get("content") or "")
        if data.get("usage"):
            out["prompt_tokens"] = data["usage"].get("prompt_tokens")
            out["completion_tokens"] = data["usage"].get("completion_tokens")
        if data.get("model"):
            out["model"] = data["model"]
    out["teacher_output"] = "".join(parts) if parts else None
    return out


def _resolve_model_dir(task: str, settings: TaskSettings):
    if settings.model_dir is None:
        return None
    if str(settings.model_dir) == "best":
        best = settings.runs_dir / task / "best.json"
        return json.loads(best.read_text())["model_dir"] if best.exists() else None
    return settings.model_dir


def create_app(config: GatewayConfig, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """transport: optional httpx transport for the upstream API (tests use a mock)."""
    store = config.open_store()
    security = config.security
    last_touch: dict[str, float] = {}
    students: dict[str, Any] = {}
    background: set[asyncio.Task] = set()
    teacher = httpx.AsyncClient(
        base_url=config.teacher.base_url.rstrip("/"),
        timeout=config.teacher.timeout_seconds,
        transport=transport,
    )

    loaded_from: dict[str, str] = {}  # task -> model_dir currently loaded

    def load_one(name: str, force: bool = False) -> bool:
        """(Re)load a task's student if its model_dir changed. True if a new model loaded."""
        from distiller.serving.registry import load_student

        settings = config.tasks[name]
        model_dir = _resolve_model_dir(name, settings)
        if not model_dir or (not force and loaded_from.get(name) == str(model_dir)):
            return False
        try:
            new = load_student(model_dir, settings.serving)
        except Exception:  # a broken model must not stop the gateway or replace a good one
            log.exception("could not load student for %s from %s", name, model_dir)
            return False
        reloading = name in students
        students[name] = new  # swap: requests in flight keep the old object
        loaded_from[name] = str(model_dir)
        log.info("loaded student for %s from %s", name, model_dir)
        if reloading:
            # Start fresh drift/promotion windows for the new model (once across instances).
            best = settings.runs_dir / name / "best.json"
            if store.mode_changed_at(name) < (best.stat().st_mtime if best.exists() else 0):
                store.set_mode(name, mode_of(name).value, reason=f"new model {model_dir}")
        return True

    def load_students() -> None:
        for name in config.tasks:
            load_one(name)

    async def watch_models():
        """Pick up new best.json files written by `distiller train` or the retrain worker."""
        while True:
            await asyncio.sleep(config.retrain.reload_seconds)
            for name, settings in config.tasks.items():
                if str(settings.model_dir) == "best":
                    try:
                        await run_in_threadpool(load_one, name)
                    except Exception:
                        log.exception("model reload check failed for %s", name)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await run_in_threadpool(load_students)
        watcher = asyncio.create_task(watch_models())
        yield
        watcher.cancel()
        await teacher.aclose()

    app = FastAPI(title="Distiller gateway", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def openai_style_errors(_request: Request, exc: HTTPException):
        # OpenAI SDKs expect {"error": {...}}, not FastAPI's {"detail": ...}.
        body = (
            exc.detail
            if isinstance(exc.detail, dict) and "error" in exc.detail
            else {"error": {"message": str(exc.detail), "type": "invalid_request_error"}}
        )
        return JSONResponse(body, status_code=exc.status_code, headers=exc.headers)

    def mode_of(task: str) -> Mode:
        stored = store.get_mode(task)
        if stored:
            return Mode(stored)
        settings = config.tasks.get(task)
        return settings.mode if settings else Mode.observe

    def authenticate(request: Request, allow_query: bool = False) -> Caller | None:
        """The caller's gateway key, if any. A key that is sent but unknown is always 401."""
        key, via_auth = key_from_headers(
            request.headers, request.query_params if allow_query else None
        )
        if not key:
            return None
        for k in security.env_admin_keys:
            if same(k, key):
                return Caller(name="env:admin", role="admin", via_authorization=via_auth)
        for k in security.env_app_keys:
            if same(k, key):
                return Caller(name="env:app", role="app", via_authorization=via_auth)
        row = store.find_api_key(hash_key(key))
        if row is None:
            raise _auth_error(401, "invalid or revoked Distiller API key")
        now = time.time()
        if now - last_touch.get(row["name"], 0) > 60:  # at most one write a minute per key
            last_touch[row["name"]] = now
            store.touch_api_key(row["name"])
        return Caller(name=row["name"], role=row["role"], via_authorization=via_auth)

    def require_app(request: Request) -> Caller | None:
        caller = authenticate(request)
        if caller is None and security.require_api_key:
            raise _auth_error(401, "missing Distiller API key (X-Distiller-Key or Bearer dk-...)")
        return caller

    def require_admin(request: Request) -> Caller | None:
        caller = authenticate(request, allow_query=True)
        needed = security.require_api_key or security.env_admin_keys or store.has_admin_key()
        if needed and (caller is None or caller.role != "admin"):
            raise _auth_error(401 if caller is None else 403, "admin key required")
        return caller

    def auth_headers(request: Request, caller: Caller | None = None) -> dict[str, str]:
        # Never forward a Distiller key upstream: use the configured teacher key instead.
        auth = None if caller and caller.via_authorization else request.headers.get("authorization")
        if not auth and config.teacher.api_key:
            auth = f"Bearer {config.teacher.api_key}"
        return {"Authorization": auth} if auth else {}

    async def call_teacher(body: dict, headers: dict) -> tuple[int, dict, float]:
        start = time.perf_counter()
        try:
            r = await teacher.post("/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as e:
            log.error("teacher unreachable: %s", e)
            return (
                502,
                {"error": {"message": f"teacher unreachable: {e}", "type": "bad_gateway"}},
                0.0,
            )
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
            maybe_promote(task)
        except Exception:
            log.exception("shadow comparison failed for %s", task)

    def maybe_promote(task: str) -> None:
        p = config.promote
        if not p.enabled:
            return
        samples, agreement = store.recent_agreement(task, p.window, audited_only=False)
        if samples >= p.min_samples and agreement >= p.min_agreement:
            reason = f"auto-promoted: shadow agreement {agreement:.1%} over {samples} requests"
            if store.set_mode(task, Mode.serve.value, reason=reason, expect=Mode.shadow.value):
                log.warning("task %s %s", task, reason)

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
                if config.retrain.enabled:
                    store.enqueue_retrain(task, "drift demotion")
        except Exception:
            log.exception("audit failed for %s", task)

    async def relay_stream(body: dict, headers: dict, on_done, route: str = "teacher"):
        """Relay the teacher's SSE stream unchanged; capture it to log when it finishes."""
        start = time.perf_counter()
        try:
            req = teacher.build_request("POST", "/chat/completions", json=body, headers=headers)
            upstream = await teacher.send(req, stream=True)
        except httpx.HTTPError as e:
            log.error("teacher unreachable: %s", e)
            return JSONResponse(
                {"error": {"message": f"teacher unreachable: {e}", "type": "bad_gateway"}},
                status_code=502,
            )

        async def body_iter():
            captured = bytearray()
            try:
                async for chunk in upstream.aiter_raw():
                    captured += chunk
                    yield chunk
            finally:
                await upstream.aclose()
            if on_done is not None:
                ms = 1000 * (time.perf_counter() - start)
                if upstream.status_code == 200:
                    on_done(200, {"teacher_ms": ms, **_parse_sse(bytes(captured))})
                else:
                    on_done(upstream.status_code, {"teacher_ms": ms})

        return StreamingResponse(
            body_iter(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type"),
            headers={"x-distiller-route": route},
        )

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "students": sorted(students)}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        caller = require_app(request)
        body = await request.json()
        headers = auth_headers(request, caller)
        key_name = caller.name if caller else None
        messages = body.get("messages") or []
        stream = bool(body.get("stream"))
        # Audits and comparisons always use a plain (non-streaming) teacher call.
        plain_body = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}

        try:
            task = identify(request.headers, messages, config)
            mode = mode_of(task)
            text = input_from_messages(messages)
            if store.scrubber:  # the student sees what training data will look like
                text = store.scrubber.scrub(text)
        except Exception:
            log.exception("routing failed; passing through")
            task, mode, text = "unassigned", Mode.off, ""

        if mode == Mode.off:
            if stream:
                return await relay_stream(body, headers, on_done=None)
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
                        task=task,
                        mode=mode.value,
                        route="student",
                        api_key=key_name,
                        audited=int(audited),
                        input=text,
                        request=messages,
                        student_output=serialize_output(ans.output),
                        student_confidence=ans.confidence,
                        student_ms=ms,
                    )
                    if audited:
                        spawn(audit(task, call_id, plain_body, headers, ans.output))
                    if stream:
                        return StreamingResponse(
                            _student_sse(task, ans.output, ans.confidence, body),
                            media_type="text/event-stream",
                            headers={"x-distiller-route": "student"},
                        )
                    resp = JSONResponse(_student_response(task, ans.output, ans.confidence))
                    resp.headers["x-distiller-route"] = "student"
                    return resp
                route = "fallback"
                student_fields = {
                    "student_output": serialize_output(ans.output)
                    if ans.output is not None
                    else None,
                    "student_confidence": ans.confidence,
                    "student_ms": ms,
                }
            except Exception:
                log.exception("student failed for %s; falling back to teacher", task)
                route, student_fields = "fallback", {}
        else:
            route, student_fields = "teacher", {}

        def log_teacher_call(status: int, fields: dict[str, Any]) -> None:
            try:
                call_id = store.log_call(
                    task=task,
                    mode=mode.value,
                    route=route,
                    api_key=key_name,
                    input=text,
                    request=messages,
                    streamed=int(stream),
                    **fields,
                    **student_fields,
                )
                if mode == Mode.shadow and has_student and status == 200:
                    spawn(shadow_compare(task, call_id, text, fields.get("teacher_output")))
            except Exception:
                log.exception("logging failed for %s", task)

        if stream:
            return await relay_stream(body, headers, on_done=log_teacher_call, route=route)

        status, data, ms = await call_teacher(body, headers)
        fields = teacher_fields(status, data, ms)
        log_teacher_call(status, fields)
        resp = JSONResponse(data, status_code=status)
        resp.headers["x-distiller-route"] = route
        return resp

    @app.get("/v1/models")
    async def models(request: Request):
        caller = require_app(request)
        r = await teacher.get("/models", headers=auth_headers(request, caller))
        return Response(r.content, status_code=r.status_code, media_type="application/json")

    @app.post("/admin/tasks/{task}/mode")
    async def set_mode(task: str, payload: ModeUpdate, request: Request):
        require_admin(request)
        mode = payload.mode
        store.set_mode(task, mode.value, reason="admin api")
        return {"task": task, "mode": mode.value}

    @app.post("/admin/tasks/{task}/reload")
    async def reload_task(task: str, request: Request):
        require_admin(request)
        if task not in config.tasks:
            raise HTTPException(404, detail={"error": {"message": f"unknown task {task}"}})
        loaded = await run_in_threadpool(load_one, task, True)
        return {"task": task, "reloaded": loaded, "model_dir": loaded_from.get(task)}

    @app.post("/admin/tasks/{task}/retrain")
    async def queue_retrain(task: str, request: Request):
        require_admin(request)
        job = store.enqueue_retrain(task, "admin api")
        return {"task": task, "queued": job is not None, "job_id": job}

    @app.get("/admin/discover")
    def discover_tasks(request: Request, min_calls: int = 50):
        from distiller.tasks.discover import discover

        require_admin(request)
        found = discover(store, min_calls=min_calls, taken=set(config.tasks))
        return {
            "candidates": [
                {
                    "group": c.group,
                    "calls": c.calls,
                    "type": c.type,
                    "reason": c.reason,
                    "name": c.name,
                    "labels": c.labels,
                    "output_schema": c.output_schema,
                    "samples": c.samples,
                }
                for c in found
            ]
        }

    @app.get("/stats")
    def stats(request: Request) -> dict:
        require_admin(request)
        return collect_stats()

    def collect_stats() -> dict:
        rows = store.stats()
        for r in rows:
            r["mode"] = mode_of(r["task"]).value
            r["has_student"] = r["task"] in students
        return {"tasks": rows, "events": store.events(), "retrain_jobs": store.retrain_jobs(10)}

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request) -> str:
        require_admin(request)
        return render_dashboard(collect_stats())

    app.state.store = store
    app.state.students = students
    app.state.background = background
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
<h2>Events</h2><ul>{events or "<li>none</li>"}</ul></body></html>"""


def app_from_env() -> FastAPI:
    """App factory for multi-worker uvicorn: reads the config path from DISTILLER_CONFIG."""
    import os

    return create_app(GatewayConfig.load(os.environ.get("DISTILLER_CONFIG", "distiller.yaml")))
