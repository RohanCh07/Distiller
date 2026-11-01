# Plan

Milestones are small on purpose. Each ends with something you can run and see.

## M0: Skeleton (½ day)
- `pyproject.toml` with `uv`, ruff, pytest.
- `distiller` CLI prints version; gateway starts and returns `/health`.

> Reprioritized 2026-10-21: the training engine comes first. The gateway is built last and only feeds data into it.
>
> **Status (2026-10-21):** M1–M6 code written. Checked only with tiny random models and a fake teacher (every command and gateway mode runs; scores are meaningless). Real testing is next: unit tests, the Kaggle run on Banking77, and the gateway against a real API.

## M1: Task spec + data pipeline (3-4 days), no GPU · ✅ built
- `task.yaml`: task type (classify / extract / generate), input schema, output schema (labels or JSON schema), metric.
- Load examples (JSONL in chat format, same shape the gateway will log later).
- Validate against schema, dedup + near-dup, label balance check, stratified train/val/test split.
- **Done when:** `distiller data check task.yaml` prints a data health report.

## M2: Training engine v1 on Banking77 (1 week), Kaggle GPU · ✅ built, first GPU run pending
- Two student families compared on the same split:
  - encoder classifier (ModernBERT/DeBERTa class): tiny, CPU-servable, well calibrated;
  - small LLM + LoRA (0.5-1.5B, Qwen class) via Unsloth/TRL.
- Hyperparameter presets per task type and data size (sweep moves to M5).
- **Done when:** both trained from one command, metrics side by side.

## M3: Evaluation + confidence (1 week), no GPU · ✅ built
- Metrics per task type: accuracy/macro-F1, per-field F1 for JSON, schema-valid rate.
- Calibration (ECE) and temperature scaling.
- Coverage vs accuracy curve: "at threshold t the student handles X% of traffic at Y% accuracy".
- **Done when:** an HTML/Markdown eval report per trained model.

## M4: Structured extraction (1 week) · ✅ built
- Second task type: JSON field extraction (Hinglish transaction categorization).
- Constrained decoding so output always matches the schema.
- Teacher-generated synthetic data when the user has too few examples.

## M5: AutoTrain · ✅ built (sweep + export; automatic task discovery later)
- `distiller train task.yaml` picks the student family, runs the sweep, evaluates, and exports the best model (adapter / merged GGUF) with its report.

## M6: Connect to live systems · ✅ built
- Gateway (passthrough + logging), shadow mode, router with fallback, drift checks.
- Reuses the M1-M5 engine; the gateway only supplies data and serves the exported model.

> **Status (2026-11-01):** test suite (unit, gateway integration, CPU end to end) and CI added;
> real Banking77 run with a RoBERTa-base encoder: 91.8% accuracy, 93.1% of traffic kept at 95.2%
> agreement. See [RESULTS.md](RESULTS.md).

## Next steps (testing phase)
1. ~~Unit tests: data pipeline (dedup, split leakage), eval math (ECE, Wilson bound, threshold), gateway modes with a fake teacher.~~ Done.
2. ~~Banking77 run~~ done on CPU with the encoder (RESULTS.md); the llm-lora student still needs the Kaggle GPU run.
3. Transactions: label a few hundred real messages, augment, train, check JSON validity.
4. Gateway against a real API: observe → shadow → serve on Banking77, watch `/dashboard`.
5. Later: automatic task discovery for `unassigned:*` traffic, streaming from students, Postgres.

## M7: Production (2026-11-01) · ✅ built
- Security: hashed API keys, PII scrubbing, encryption at rest.
- Serving: streaming (student and teacher), Postgres, vLLM students, several gateways.
- Automation: task discovery, retraining on drift, hot reload, auto-promotion.
- Onboarding: `distiller init`, `distiller doctor`, new dashboard, docs site.
- Hosted tier foundations: accounts, isolation, metering, share-of-savings invoices.

## Next
- Payment provider adapter for invoices; self-service signup.
- Per-account task namespaces and models in the hosted tier.
- Real token streaming for long generation tasks (needs a mid-stream fallback policy).

## Open questions
- Which teacher API to standardise on for the demo?
- SQLite only until M3, or Postgres from day one?
- Open-source the gateway from day one?
