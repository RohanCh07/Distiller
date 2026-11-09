# Gateway

An OpenAI-compatible proxy (`POST /v1/chat/completions`, `GET /v1/models`). Apps change one URL.

## Modes

| Mode | Who answers | Logged |
|---|---|---|
| `off` | teacher | nothing |
| `observe` | teacher | the call (training data) |
| `shadow` | teacher | the call, plus the student's answer and agreement (background) |
| `serve` | student if confident and valid, else teacher | every call; fallbacks are new training data |

Switch live: `distiller task mode <task> <mode>`, the dashboard, or
`POST /admin/tasks/<task>/mode {"mode": "serve"}`. Modes live in the database, so every gateway
instance sees the change on the next request.

Anything that breaks inside Distiller sends the request to the teacher (fail open). If the
teacher itself is unreachable the app gets a 502 in the OpenAI error shape.

## Task identification

1. `X-Distiller-Task` header.
2. `match_system_prompt`: a substring of the system prompt, per task in `distiller.yaml`.
3. Otherwise `unassigned:<hash of the system prompt>`: still logged, and
   [task discovery](automation.md) can turn it into a task.

## Streaming

`stream: true` works in every mode:

- **Student answers** (serve, confident): the answer is computed first, then sent as
  `chat.completion.chunk` events. The confidence must be known before we commit to the student,
  and student answers are short, so this costs nothing noticeable.
- **Teacher answers**: the stream is relayed byte-for-byte and captured at the same time, so the
  call is logged and shadow-compared exactly like a non-streamed one. Audits always use a
  separate non-streaming teacher call.

The official OpenAI SDK is tested against both paths.

## Students: in-process or vLLM

```yaml
tasks:
  banking77:
    mode: serve
    model_dir: best            # runs/banking77/best.json from distiller train
  transactions:
    mode: serve
    model_dir: runs/transactions/models/llm-lora
    serving: {backend: vllm, url: http://gpu-box:8001/v1}
```

`backend: local` (default) loads the model into the gateway: right for encoders, which answer in
milliseconds on CPU. `backend: vllm` keeps only the calibration and threshold in the gateway and
asks a vLLM server, with `logprobs` for confidence and a `json_schema` response format for
extraction tasks. Start one with:

```bash
distiller serve-vllm runs/transactions/models/llm-lora --port 8001
```

An adapter folder is served as a LoRA on its base model; a merged export
(`distiller export ... --out merged/`) is served directly.

## Dashboard

`/dashboard` shows traffic by route over time, share served by the student, fallbacks, shadow
and audit agreement, latency, estimated teacher spend avoided (enter your teacher's prices), mode
switches, events, retrain jobs and discovered tasks. The page itself holds no data; it reads
`/stats` and `/stats/timeseries` with the admin key (open `/dashboard?key=dk-...` once).

## Endpoints

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /v1/chat/completions` | app key if required | the proxy |
| `GET /v1/models` | app key if required | passthrough |
| `GET /health` | none | liveness, loaded students |
| `GET /dashboard` | none (shell only) | UI |
| `GET /stats`, `GET /stats/timeseries` | admin | numbers |
| `POST /admin/tasks/{task}/mode` | admin | switch mode |
| `POST /admin/tasks/{task}/reload` | admin | reload the student from disk |
| `POST /admin/tasks/{task}/retrain` | admin | queue a retrain |
| `GET /admin/discover` | admin | candidate tasks in untagged traffic |
