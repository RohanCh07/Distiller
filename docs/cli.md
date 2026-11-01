# CLI reference

Generated from the code by `python scripts/gen_cli_docs.py` (CI checks it is current).

## accounts create

Create a customer account with its pricing terms.

`distiller accounts create NAME [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `NAME` | argument |  |  |
| `--share` | option | Share of net savings we bill, e.g. 0.3 |  |
| `--price-in` | option | Teacher USD per 1M input tokens (their current model) |  |
| `--price-out` | option | Teacher USD per 1M output tokens |  |
| `--min-monthly` | option | Monthly minimum fee, USD |  |
| `--student-per-1k` | option | Our cost to serve 1k student calls, USD |  |
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## accounts list



`distiller accounts list [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## billing export

Export saved invoices as CSV (one row per task line) for an invoicing system.

`distiller billing export [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `--out` | option | CSV file to write |  |
| `--account` | option |  |  |
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## billing invoice

Compute an account's share-of-savings invoice for a month.

`distiller billing invoice ACCOUNT [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `ACCOUNT` | argument |  |  |
| `--month` | option | YYYY-MM (UTC); default: this month so far |  |
| `--save` / `--no-save` | flag | Store it (refuses to overwrite unless --replace) |  |
| `--replace` / `--no-replace` | flag |  |  |
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## data check

Validate, dedup and split the data; write a data health report.

`distiller data check TASK [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `TASK` | argument | Path to task.yaml |  |

## doctor

Check a setup: config, database, extras, security, teacher, students.

`distiller doctor [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |
| `--offline` / `--no-offline` | flag | Skip the teacher connectivity check |  |

## export

Merge a LoRA adapter into its base model (for vLLM, or GGUF conversion via llama.cpp).

`distiller export MODEL_DIR [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `MODEL_DIR` | argument | runs/<task>/models/llm-lora |  |
| `--out` | option | Folder for the merged model |  |

## gateway

Run the OpenAI-compatible gateway. Point OPENAI_BASE_URL at http://host:port/v1.

Several workers (or several machines) can share one database: task modes, keys and
drift demotion live there. Use Postgres when the gateways are on different machines.

`distiller gateway [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |
| `--host` | option |  | `127.0.0.1` |
| `--port` | option |  | `8000` |
| `--workers` | option | Worker processes (each loads its own students) | `1` |

## init

Set up a project: distiller.yaml, one task, an admin key. Asks a few questions.

`distiller init DIRECTORY [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `DIRECTORY` | argument, optional | Where to write distiller.yaml and tasks/ | `.` |
| `--task` | option | Task name, e.g. ticket_router |  |
| `--type` | option | classification \| extraction \| generation |  |
| `--instruction` | option | The system prompt your app sends today |  |
| `--labels` | option | Classification labels, comma-separated |  |
| `--teacher-url` | option | OpenAI-compatible base URL of the big model |  |
| `--yes` / `-y` | flag | Accept defaults; no questions |  |
| `--force` / `--no-force` | flag | Overwrite an existing distiller.yaml |  |

## keys create

Create a gateway API key. It is printed once; only its hash is stored.

`distiller keys create NAME [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `NAME` | argument | A label, e.g. the app that will use the key |  |
| `--role` | option | app (call /v1) or admin (also /admin, /stats, /dashboard) | `app` |
| `--account` | option | Hosted tier: attach the key to this account |  |
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## keys encryption-key

Print a new key for security.encrypt_logs (put it in DISTILLER_ENCRYPTION_KEY).

`distiller keys encryption-key [OPTIONS]`

## keys list

List keys (names, roles, last use); never the keys themselves.

`distiller keys list [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## keys revoke

Revoke a key by name (takes effect on the next request).

`distiller keys revoke NAME [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `NAME` | argument |  |  |
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## keys rotate-encryption

Re-encrypt stored logs with the first key in DISTILLER_ENCRYPTION_KEY.

To rotate: put the new key first and the old one after it (comma-separated), run this,
then remove the old key.

`distiller keys rotate-encryption [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## logs export

Turn logged teacher answers into a training file for `distiller train`.

`distiller logs export TASK [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `TASK` | argument | Task name |  |
| `--out` | option | JSONL to write (chat format) |  |
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## logs stats

Per-task traffic, routing and agreement.

`distiller logs stats [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## predict

Run a trained student on some inputs (shows calibrated confidence and route).

`distiller predict MODEL_DIR TEXT [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `MODEL_DIR` | argument | runs/<task>/models/<backend> |  |
| `TEXT` | argument | One or more inputs |  |

## retrain

Queue a retrain of a task on its original data plus logged traffic.

`distiller retrain TASK [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `TASK` | argument |  |  |
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## retrain-worker

Run queued retrain jobs (on a machine that can train; a GPU for LLM students).

`distiller retrain-worker [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |
| `--once` / `--no-once` | flag | Run at most one job, then exit |  |
| `--poll` | option | Seconds between queue checks | `30` |

## serve-vllm

Serve an LLM student with vLLM (GPU). Then set the task's serving.backend: vllm.

An adapter folder is served as a LoRA on its base model; a merged export (from
`distiller export`) is served directly.

`distiller serve-vllm MODEL_DIR EXTRA [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `MODEL_DIR` | argument | runs/<task>/models/llm-lora, or a merged export |  |
| `--port` | option |  | `8001` |
| `--print` | flag | Print the command, do not run it |  |
| `EXTRA` | argument, optional | Extra vLLM flags after -- |  |

## task discover

Find repeated untagged prompts that could become tasks.

`distiller task discover [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |
| `--min-calls` | option | Ignore prompt groups with fewer logged answers | `100` |
| `--write` | option | Write <dir>/<name>/task.yaml + data.jsonl per proposal |  |
| `--adopt` / `--no-adopt` | flag | Also move the group's logged calls to the new name |  |

## task mode

Switch a task's mode (takes effect on the next request; no restart).

`distiller task mode TASK MODE [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `TASK` | argument |  |  |
| `MODE` | argument | off \| observe \| shadow \| serve |  |
| `--config` / `-c` | option | Gateway config file | `distiller.yaml` |

## teacher augment

Create new teacher-labelled examples (training only; never used for val/test).

`distiller teacher augment TASK [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `TASK` | argument | Path to task.yaml |  |
| `--out` | option | JSONL to append synthetic rows to |  |
| `--per-label` | option | Classification: top every label up to this many |  |
| `--total` | option | Extraction/generation: how many new examples |  |
| `--model` | option | Teacher model (or DISTILLER_TEACHER_MODEL) |  |

## teacher label

Get teacher outputs for raw inputs.

`distiller teacher label TASK [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `TASK` | argument | Path to task.yaml |  |
| `--inputs` | option | Text file, one input per line |  |
| `--out` | option | JSONL to write (input/output rows) |  |
| `--model` | option | Teacher model (or DISTILLER_TEACHER_MODEL) |  |

## train

Prepare data, train and compare students, calibrate, and write report.md.

`distiller train TASK [OPTIONS]`

| Name | Kind | Description | Default |
|---|---|---|---|
| `TASK` | argument | Path to task.yaml |  |
| `--candidates` | option | Comma-separated backends, e.g. encoder,llm-lora (default: by task type) |  |
| `--sweep` / `--no-sweep` | flag | Try a small hyperparameter grid |  |

## version



`distiller version [OPTIONS]`
