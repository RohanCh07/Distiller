# Distiller

Turn repetitive LLM tasks into small, cheap models, and prove they are good enough before they serve traffic.

Give Distiller a task spec (`task.yaml`) and examples of inputs and outputs. It:

1. **Checks the data**: validates outputs, removes duplicates, fixes label conflicts, groups near-duplicates, splits train/val/test.
2. **Trains students**: a tiny encoder classifier and/or a small LLM with LoRA.
3. **Evaluates honestly**: task-specific metrics on a held-out test set.
4. **Calibrates confidence** and picks a threshold: "the student can handle X% of requests at ≥ 95% accuracy; send the rest to the big model".
5. **Writes a report** and saves the winning model with its routing settings.

The gateway connects it to live traffic: apps change one URL, the gateway logs calls, runs the student in shadow, then serves it with fallback and drift checks.

## Quick start (gateway)

```bash
pip install -e ".[gateway,train,security]"
distiller init            # writes distiller.yaml + a task, creates an admin key
distiller doctor          # checks the setup
distiller gateway         # apps use OPENAI_BASE_URL=http://localhost:8000/v1
```

Full docs: `mkdocs serve` (or the published site), starting at [docs/quickstart.md](docs/quickstart.md).

## Quick start (engine only)

```bash
pip install -e .                       # light: data checks, reports, teacher tools
python examples/banking77/prepare.py   # downloads ~13k examples
distiller data check examples/banking77/task.yaml
```

**Train** (needs a GPU): run `notebooks/train_kaggle.ipynb` on a free Kaggle T4. It installs
`.[train]` and runs `distiller train examples/banking77/task.yaml` (add `--sweep` for a small grid).

**Try a model:** `distiller predict runs/banking77/models/encoder "my card hasn't arrived"`

**Grow data with the teacher:** set `DISTILLER_TEACHER_MODEL` / `DISTILLER_TEACHER_API_KEY`, then
`distiller teacher label ...` or `distiller teacher augment ...` (see examples/README.md).

**Run the gateway:**
```bash
pip install -e ".[gateway,train]"
cp distiller.example.yaml distiller.yaml   # set tasks and modes
distiller gateway                          # apps use OPENAI_BASE_URL=http://localhost:8000/v1
```
Open http://localhost:8000/dashboard. Switch modes live with `distiller task mode banking77 serve`.
Export logged traffic for retraining with `distiller logs export banking77 --out data.jsonl`.

Install extras: `[train]` models, `[gateway]` server, `[security]` log encryption, `[postgres]`
shared database, `[constrained]` schema-forced JSON (pins transformers 4.x), `[quant]` 4-bit
training on CUDA.

Production features (see the docs): API keys, PII scrubbing and encryption at rest
([security](docs/security.md)); streaming, Postgres, vLLM students and several gateways
([deployment](docs/deployment.md)); task discovery and retraining on drift
([automation](docs/automation.md)); accounts and share-of-savings billing ([hosted](docs/hosted.md)).

## Tests

```bash
pip install -e ".[gateway,train]" pytest
pytest -m "not slow"     # unit + gateway integration, seconds, no torch needed
pytest -m slow           # CLI end to end with tiny offline models (CPU, ~1-2 min)
```

## Docs

- [docs/GOAL.md](docs/GOAL.md): vision and scope
- [docs/PLAN.md](docs/PLAN.md): milestones and status
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): engine flow and code map
- [docs/DECISIONS.md](docs/DECISIONS.md): every design decision and why
- [docs/NO_GPU.md](docs/NO_GPU.md): working without owning a GPU
- [docs/RESULTS.md](docs/RESULTS.md): test suite and the Banking77 run
- [docs/quickstart.md](docs/quickstart.md), [gateway](docs/gateway.md), [security](docs/security.md),
  [deployment](docs/deployment.md), [automation](docs/automation.md), [hosted](docs/hosted.md),
  [CLI reference](docs/cli.md)

## Layout

```
Distiller/
├── docs/
├── examples/               banking77 (classification), transactions (JSON extraction)
├── notebooks/              Kaggle training notebook
├── src/distiller/
│   ├── spec.py             task.yaml schema
│   ├── data/               load, validate, dedup, split, data report
│   ├── training/           student backends (encoder, llm-lora) + presets
│   ├── eval/               metrics, calibration, coverage/threshold, report
│   ├── engine.py           the full train → evaluate → calibrate flow
│   ├── teacher.py          big-model client: label + augment
│   ├── serving/            load a trained student with its routing settings
│   ├── gateway/            OpenAI-compatible proxy, modes, dashboard
│   ├── router/ tasks/ storage/   routing policy, task identification + discovery, SQLite/Postgres
│   ├── security/           API keys, PII scrubbing, encryption at rest
│   ├── automation/         retrain worker
│   ├── hosted/             accounts metering and billing
│   ├── onboarding.py       distiller init / doctor
│   └── cli.py              `distiller ...`
├── tests/
└── deploy/
```
