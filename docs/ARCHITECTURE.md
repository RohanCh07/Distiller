# Architecture

Distiller is built engine-first (DECISIONS.md #4). The **training engine** turns a task spec plus examples into a trained, evaluated, calibrated small model. The **gateway** feeds it real traffic and serves its output.

## The full loop

```
             ┌──────────────── distiller.yaml (modes, models) ───────────────┐
customer app ──► gateway ──┬──► teacher (any OpenAI-compatible API)          │
                    │      └──► student (runs/<task>/models/<best>)          │
                    ▼                                                        │
              SQLite logs ── distiller logs export ──► task data (chat JSONL)│
                                                          │                  │
            distiller teacher label / augment ────────────┤                  │
                                                          ▼                  │
                                               distiller train task.yaml     │
                                                          │                  │
                                    report.md + models/<backend>/routing.json┘
```

## Engine flow (`distiller train`)

```
task.yaml + data (+ synthetic, train only)
   │
   ▼
data pipeline ── load ─ validate ─ exact dedup + conflict fix ─ near-dup groups ─ split
   │
   ▼
for each candidate student (encoder, llm-lora):
   train (preset, or a small sweep picked on val)
   predict val + test
   calibrate on val (temperature / isotonic)
   choose confidence threshold on val (Wilson lower bound ≥ target)
   measure coverage + accuracy at that threshold on test
   save model + distiller.json + routing.json
   │
   ▼
report.md + best.json
```

## Gateway request flow (`distiller gateway`)

```
POST /v1/chat/completions
  stream? ─────────────────────────────► passthrough
  task = header | system-prompt match | unassigned:<hash>
  mode = DB override | distiller.yaml | observe
  off     → teacher
  observe → teacher, log
  shadow  → teacher, log; background: student answers, agreement logged
  serve   → student; confident & valid? → answer (2% audited by teacher in background,
                                           auto-demote to shadow on drift)
                     else → teacher, logged as fallback (new training data)
  any internal error → teacher (fail open)
```

## Code map

| Path | Job |
|---|---|
| `spec.py` | `TaskSpec`: loads and validates `task.yaml` |
| `data/io.py` | Read JSONL/CSV/chat rows into `Example`; `input_from_messages` |
| `data/validate.py` | Check outputs against labels or JSON schema |
| `data/dedup.py` | Exact dedup, label-conflict resolution, MinHash near-dup grouping |
| `data/split.py` | Group-aware, stratified train/val/test split |
| `data/report.py`, `data/pipeline.py` | Data health report; `prepare(spec)` incl. synthetic data |
| `training/base.py` | `Student` interface, save/load, backend registry |
| `training/presets.py` | Default hyperparameters and sweep grids |
| `training/encoder.py` | Encoder + classification head student |
| `training/lora_llm.py` | Small LLM + LoRA student; schema validation, constrained decoding, merge export |
| `eval/metrics.py` | Accuracy/macro-F1, per-field F1, token F1; per-example correctness |
| `eval/calibration.py` | ECE, temperature scaling, isotonic regression |
| `eval/coverage.py` | Coverage-vs-accuracy curve, threshold choice |
| `eval/report.py` | `results.json` + `report.md` |
| `engine.py` | `run(spec)`: the engine flow |
| `teacher.py` | Teacher client: `label`, `augment` |
| `serving/registry.py` | `LoadedStudent`: model + calibration + threshold → `answer()` |
| `gateway/config.py` | `distiller.yaml` schema, modes |
| `gateway/app.py` | FastAPI app: proxy, modes, audits, `/stats`, `/dashboard`, admin API |
| `tasks/identify.py` | Which task a request belongs to |
| `router/policy.py` | Audit sampling, drift check / auto-demotion |
| `storage/db.py` | SQLite: calls, task modes, events, export |
| `cli.py` | All `distiller ...` commands |

## Commands

| Command | Does |
|---|---|
| `distiller data check task.yaml` | Data report only (seconds, no GPU) |
| `distiller train task.yaml [--sweep] [--candidates encoder,llm-lora]` | Full engine run |
| `distiller predict <model_dir> "text" ...` | Try a trained student |
| `distiller export <llm model_dir> --out merged/` | Merge LoRA into base weights |
| `distiller teacher label task.yaml --inputs raw.txt --out data.jsonl` | Teacher labels raw inputs |
| `distiller teacher augment task.yaml --out synthetic.jsonl [--per-label N / --total N]` | Synthetic training data |
| `distiller gateway -c distiller.yaml` | Run the gateway |
| `distiller task mode <task> <mode> -c distiller.yaml` | Switch mode live |
| `distiller logs export <task> --out data.jsonl` | Logged traffic → training data |
| `distiller logs stats` | Per-task traffic and agreement |
