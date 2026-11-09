# Quickstart

## 1. Install

```bash
pip install -e ".[gateway,train,security]"
```

Extras: `train` (torch, transformers, peft), `gateway` (FastAPI), `security` (encryption),
`postgres` (psycopg), `constrained` (schema-forced JSON), `quant` (4-bit on CUDA).

## 2. Set up a project

```bash
distiller init
```

It asks for a task name, type, the system prompt your app uses, labels, and the teacher URL,
then writes `distiller.yaml`, `tasks/<name>/task.yaml`, and creates an admin key (printed once).
Non-interactive:

```bash
distiller init . --task tickets --type classification \
  --instruction "Route the support ticket" --labels billing,tech,other -y
```

## 3. Check it

```bash
export OPENAI_API_KEY=sk-...
distiller doctor
```

`doctor` checks the config, database, installed extras, security settings, the teacher's
`/models` endpoint and every student. It exits non-zero if something is broken.

## 4. Run the gateway and send traffic

```bash
distiller gateway
```

Point your app at `http://localhost:8000/v1` (e.g. `OPENAI_BASE_URL`) and send the header
`X-Distiller-Task: tickets` (or let `match_system_prompt` find it). Open
`http://localhost:8000/dashboard?key=<admin key>`.

## 5. Train and switch on

```bash
distiller logs export tickets --out tasks/tickets/data.jsonl
distiller train tasks/tickets/task.yaml          # writes runs/tickets/report.md
distiller task mode tickets shadow               # compare in the background
distiller task mode tickets serve                # when shadow agreement looks right
```

Read `runs/<task>/report.md` before serving: it says how much traffic the student keeps at
your target accuracy and what it gets wrong.
