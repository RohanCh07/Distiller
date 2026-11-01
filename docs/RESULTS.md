# Test results (2026-11-01)

Everything here ran in a cloud container (4 vCPU, no GPU) and on GitHub Actions.

## 1. Test suite

| Suite | Tests | What it covers | Time |
|---|---|---|---|
| Unit (`tests/unit`) | 132 | data (io, dedup, split, validate, pipeline, report), eval (metrics, calibration, coverage, report), spec, router policy, SQLite store, gateway config, task identification, teacher with a mocked API, presets | ~5 s |
| Integration (`tests/integration`) | 25 | gateway with a mocked upstream LLM: all four modes, fallback (low confidence, invalid output, student crash), audits, drift auto-demotion, streaming passthrough, auth forwarding, upstream errors, admin API, stats, dashboard, log export (store and CLI) | ~3 s |
| End to end (`tests/e2e`, `-m slow`) | 7 | the CLI on CPU with tiny random models built offline (see below) | ~1–2 min |

CI (`.github/workflows/ci.yml`) runs lint (`ruff check`, `ruff format --check`), the fast
tests on Python 3.11, 3.12 and 3.13 (without torch), and the end-to-end tests with CPU torch.

Bugs found and fixed while writing the tests:

- `POST /admin/tasks/{task}/mode` with an unknown mode crashed with a 500; it now returns 422.
- The LLM student's `generate` closure captured a loop variable (lint B023); now bound explicitly.

## 2. End-to-end CLI runs on CPU

Tiny models (2-layer BERT / 2-layer Llama with tokenizers trained in the test) prove every
step runs and writes the right files; their scores mean nothing.

- **Classification** (3 intents, 300 rows): `distiller data check` → `distiller train`
  (encoder) → `report.md`, `results.json`, `best.json`, `routing.json` (temperature calibration,
  threshold) → `distiller predict` → the gateway loads `model_dir: best` and serves. A `--sweep`
  run trains the 3-setting grid.
- **Transactions example** (JSON extraction), against a local fake teacher server:
  `distiller teacher label` (144/144 labelled) → `distiller teacher augment --total 60` →
  `distiller data check` (synthetic rows train-only) → `distiller train` (llm-lora) →
  `distiller predict` → `distiller export` (merged model).
- Asking for the **encoder** on transactions reports
  `failed: encoder does not support extraction` instead of crashing. The encoder is a
  classification-only student, so extraction tasks always use `llm-lora` (DECISIONS #9).

## 3. Real run: Banking77 with a real encoder

**Setup.** 13,083 examples (`examples/banking77/prepare.py`), 77 intents. After cleaning:
13,071 usable (12 exact duplicates), split 9,802 / 1,310 / 1,959 (train/val/test). Gold labels
stand in for teacher answers, as in `prepare.py`, so "agreement with the teacher" below means
agreement with those labels.

**Model.** RoBERTa-base (125M parameters), 3 epochs, max length 64, batch 32, lr 5e-5, CPU only
(70 minutes). The Hugging Face Hub was not reachable from the environment, so the encoder
weights were taken from spaCy's `en_core_web_trf` (a RoBERTa-base, downloadable from GitHub
releases) and converted with `scripts/convert_spacy_roberta.py`. These weights were fine-tuned
by spaCy for tagging/parsing/NER, so this is a real pretrained encoder but not the original
checkpoint, and not the ModernBERT default in `task.yaml`.

**Results (held-out test set, 1,959 requests).**

| Metric | Value |
|---|---|
| Accuracy (all requests) | **91.8%** (macro-F1 0.920) |
| Confidence cut-off (calibrated, chosen on val) | **0.737** |
| Traffic the student keeps at that cut-off | **93.1%** |
| Agreement with the teacher on kept traffic | **95.2%** (target 95%: met) |
| Val lower bound (Wilson 95%) at the cut-off | 95.05% |
| Calibration error (ECE), raw → calibrated | 0.058 → 0.024 (temperature 0.78) |
| Student latency | 27 ms/example batched; ~55 ms p50 per request through the gateway (4 vCPU) |

Weakest intents: `lost_or_stolen_card` (F1 0.75), `pending_transfer` (0.78),
`balance_not_updated_after_bank_transfer` (0.79). Most common confusion:
`direct_debit_payment_not_recognised` → `card_payment_not_recognised`.

**Through the gateway.** `scripts/replay_test_split.py` sends the same 1,959 test requests
through the real gateway, with a mocked teacher that answers with the reference labels:

| Mode | Gateway measured |
|---|---|
| shadow | student agreement 91.78% on all traffic |
| serve | student answered 93.11%, agreement on its answers 95.18%; the rest fell back |

These match the offline report exactly, so calibration and threshold survive the trip from
`routing.json` into the serving path.

**Estimated cost savings.** The teacher prompt is the instruction plus the 77 labels: about 473
input and 5 output tokens per call (characters / 4). The student takes 93.1% of calls:

| Teacher price (USD per 1M tokens, in / out) | Teacher cost per 1M calls | Saved per 1M calls |
|---|---|---|
| 0.15 / 0.60 (small model; example) | $74 | **$69** |
| 2.50 / 10.00 (large model; example) | $1,233 | **$1,148** |

Prices are examples; pass your own with `--price-in/--price-out`. The savings do not subtract the
cost of running the student (one CPU core handles roughly 20–35 requests/s at these latencies).

**Reproduce:**

```bash
python examples/banking77/prepare.py
python scripts/convert_spacy_roberta.py <spacy model dir> models/roberta-base-spacy   # or use ModernBERT on a GPU
# in task.yaml: candidates [encoder], encoder_model: models/roberta-base-spacy,
#               max_length 64, overrides.encoder {epochs: 3, batch_size: 32}
distiller train examples/banking77/task.yaml
python scripts/replay_test_split.py runs/banking77 --price-in 0.15 --price-out 0.60
```

## Not covered yet

- The **llm-lora** student on real models (needs a GPU; the notebook path is unchanged).
- A **real teacher API**: no API key was available, so every teacher in these tests is mocked
  or a local fake server. `distiller teacher label/augment` and the gateway passthrough are
  exercised against those, not against a provider.
