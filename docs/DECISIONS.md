# Decisions

A running log. Newest at the bottom. Each entry: what we decided, why, and what we gave up.

## 1. Gateway first, library later (2026-10-20)

**Decision:** The core product is an OpenAI-compatible gateway (proxy). Customers change `OPENAI_BASE_URL`; no code change. A thin optional Python SDK (e.g. `task="ticket_tagger"` tagging) comes later, on top of the gateway.

**Why:**
- Works with any language, since every OpenAI SDK supports a custom base URL.
- One gateway sees all of a company's services; one dashboard.
- We can improve routing or swap models without customers redeploying.
- The same container serves both self-host and cloud tiers.

**Gave up:** One extra network hop (a few ms when self-hosted next to the app). Mitigation: the gateway must **fail open**, so if anything inside Distiller breaks, the request goes straight to the provider.

## 2. One engine, three deployment tiers (2026-10-20)

| Tier | Who runs it | Price |
|---|---|---|
| Open source | Customer, on their servers | Free |
| Cloud | Us (we train and host) | Share of savings or monthly fee |
| Enterprise self-host | Customer, with our support | Premium |

The record → train → test → route code is identical across tiers. Only *where* it runs changes.

**First build:** self-host path only. Add cloud when someone asks us to "just do it for me".

## 3. Narrow scope for v1 (2026-10-20)

- OpenAI-compatible `/v1/chat/completions`, text only, non-streaming first.
- Tasks with checkable outputs: classification, extraction, JSON output.
- Task identity comes from a header (`X-Distiller-Task`) before we attempt automatic clustering.

## 4. Training engine first, gateway last (2026-10-21)

**Decision:** Build the task-specific fine-tuning engine first: a user gives a task spec (input/output structure) plus examples and gets back a trained small model and an eval report. The gateway (M6) only feeds data into it and serves the result.

**Why:** The engine is where the product's value and the hard ML problems live. Getting it right first makes the live connection a thin layer.

**Guardrails:** Training data uses the same chat format the gateway will log, so plugging it in later needs no conversion. Evaluation and confidence calibration count as part of the engine, not extras.

## 5. One task spec file per task (2026-10-21)

**Decision:** Each task is described by a `task.yaml`: name, type (`classification` / `extraction` / `generation`), instruction, labels or JSON schema, data path, student options and the target accuracy. Validated with pydantic (`src/distiller/spec.py`).

**Why:** One file is easy to read, version and later generate from the gateway. Every later step reads only this file.

## 6. Data format: input/output rows or chat messages (2026-10-21)

**Decision:** Data is JSONL or CSV with `input`/`output` columns (names configurable), or JSONL rows with OpenAI-style `messages` (last assistant turn = output). Extraction outputs may be JSON objects or JSON strings.

**Why:** Covers hand-made datasets now and gateway logs later without conversion.

## 7. Data cleaning rules (2026-10-21)

- Rows that fail the spec (unknown label, bad JSON, schema violation) are dropped and counted.
- **Exact duplicates** (after Unicode/case/whitespace normalisation) are removed.
- **Same input, different outputs** = label noise: keep the majority output, count the conflict, warn above 2%.
- **Near-duplicates** (MinHash + LSH on character 5-grams, Jaccard ≥ 0.85) are **kept but grouped**, and each group goes to one split only.

**Why:** Near-duplicates are real traffic variants and help training; the real danger is the same sentence sitting in train and test, which inflates scores. Grouping prevents that leak without throwing data away.

## 8. Split: group-aware and label-stratified (2026-10-21)

**Decision:** Default 80/10/10 train/val/test (configurable), stratified by label for classification, seeded. Each label keeps at least one group in train.

**Why:** Rare labels must appear in val/test to be measured. Val is used for choices (calibration, threshold); test is touched once, for the honest number.

## 9. Two student families, compared on every classification task (2026-10-21)

- **encoder**: BERT-style model + classification head (default ModernBERT-base, ~150M params). Tiny, CPU-servable, calibrates well.
- **llm-lora**: small instruct LLM + LoRA (default Qwen2.5-0.5B-Instruct, rank 16, all linear layers), loss on answer tokens only. Handles every task type, including JSON extraction.

Extraction and generation use only `llm-lora`. The engine trains every candidate and picks the winner from the report, rather than us guessing.

## 10. Plain Hugging Face + PEFT, not Unsloth/TRL, for v1 (2026-10-21)

**Decision:** Training uses `transformers.Trainer` + `peft` directly, with our own answer-only loss masking.

**Why:** One code path that runs on CPU and GPU, fewer moving APIs. Unsloth can be added later as an optional speed-up on CUDA. Requires `transformers>=4.56` (we pass `dtype=` and warmup in steps, which newer versions require).

## 11. Confidence, calibration and the serving threshold (2026-10-21)

- **Raw confidence**: encoder = top softmax probability; LLM = geometric mean of generated-token probabilities.
- **Calibration** (fit on val only): encoder → temperature scaling on logits; LLM → isotonic regression from score to accuracy. Skipped with fewer than 30 val examples. Reported as ECE before/after.
- **Threshold**: on val, the lowest confidence cut whose **Wilson 95% lower bound** of accuracy is ≥ the target (default 95%), keeping at least 30 examples. Then applied unchanged to test, and the report shows test coverage and test accuracy at that cut.

**Why:** "Is the student good?" is the wrong question; "how much traffic can it take at the target accuracy?" is what saves money safely. The lower bound stops small lucky samples from passing.

## 12. Run outputs (2026-10-21)

```
runs/<task>/
  data/  train.jsonl val.jsonl test.jsonl data_report.md data_report.json
  models/<backend>/  weights or LoRA adapter, tokenizer, distiller.json, routing.json
  report.md  results.json
```

`routing.json` holds the calibration and threshold, so the gateway (M6) can serve the model without re-running evaluation. Winner = promotable first, then highest test coverage at target, then fastest.

## 13. Dependencies split into extras (2026-10-21)

**Decision:** Core install is light (pydantic, numpy, jsonschema, pyyaml, typer). `[train]` adds torch/transformers/peft/accelerate, `[quant]` adds bitsandbytes, `[gateway]` adds FastAPI and friends.

**Why:** Data checks and reports run on a laptop in seconds; heavy ML packages are only installed where training happens (Kaggle/Colab).

## 14. Saved students are self-describing (2026-10-21)

**Decision:** Every model folder holds `distiller.json` (backend, labels, hyperparameters and the full task spec) and `routing.json` (calibration, threshold, test coverage curve). `Student.load(dir)` rebuilds a working student from the folder alone.

**Why:** The gateway, `distiller predict` and `distiller export` need nothing but a path. No hidden state.

## 15. JSON extraction: validate always, constrain when possible (2026-10-21)

**Decision:** LLM outputs for extraction tasks are parsed and checked against `output_schema`; anything invalid counts as "no answer" (wrong in eval, fallback in the gateway). With the optional `[constrained]` extra, decoding is forced to follow the schema via `lm-format-enforcer`. If the constraint cannot be built or is unsatisfiable, we fall back to unconstrained decoding plus validation.

**Why:** A broken JSON reply reaching a customer's code is worse than a slower teacher reply. Constraining removes most invalid outputs; validation guarantees none get through.

**Caveat:** `lm-format-enforcer` does not support transformers 5 yet, so `[constrained]` pins transformers 4.x. Without it, everything still works with validation only.

## 16. Teacher client: label and augment (2026-10-21)

- `distiller teacher label`: teacher answers for raw inputs (any OpenAI-compatible API, configured by env vars). Answers that fail the spec are dropped.
- `distiller teacher augment`: the teacher writes new varied inputs (English/Hindi/Hinglish, typos, lengths) from real seeds, then labels them. For classification it tops rare labels up to `min_examples_per_label` and keeps only inputs the teacher labels as the intended label (consistency filter).

## 17. Synthetic data is train-only (2026-10-21)

**Decision:** Synthetic rows (`data.synthetic_path`, or rows with `"synthetic": true`) never enter val/test. They are dropped if they duplicate a real input or are near-duplicates of any val/test example. Counts are in the data report.

**Why:** Scores must reflect real traffic. Synthetic text that echoes test rows would inflate them.

## 18. AutoTrain: small sweeps, chosen on validation (2026-10-21)

**Decision:** `--sweep` (or `student.sweep: true`) trains a small grid per backend: encoder learning rate × 3; LLM learning rate × LoRA rank (4 runs). The best run by validation correctness is kept; others are freed. Test is still used once, on the winner only.

**Why:** Free GPU hours are limited, and data quality matters more than fine tuning. Choosing on val keeps the test number honest.

## 19. Gateway modes and fail-open behaviour (2026-10-21)

| Mode | Who answers | What's logged |
|---|---|---|
| `off` | teacher | nothing |
| `observe` | teacher | the call (training data) |
| `shadow` | teacher | the call + student's answer in the background + agreement |
| `serve` | student if calibrated confidence ≥ threshold and output valid; else teacher | every call; fallbacks are new training data |

Any error in Distiller (routing, student, logging) sends the request to the teacher. Teacher unreachable → HTTP 502 with an OpenAI-style error. Streaming requests are passed through (students serve non-streaming only, for now). The app's own `Authorization` header is forwarded; the configured key is used only when none is sent.

## 20. Drift protection: audits and auto-demotion (2026-10-21)

**Decision:** In `serve` mode, `audit_rate` (default 2%) of student answers are also sent to the teacher in the background. If audited agreement since the last mode change drops below `demote.min_agreement` (default 92%, over the last 200, minimum 50 samples), the task is switched back to `shadow` automatically and an event is recorded.

**Why:** Customer traffic changes. The student should lose its job automatically before quality drops for long.

## 21. Task identification (2026-10-21)

**Decision:** `X-Distiller-Task` header first; else a configured `match_system_prompt` substring; else `unassigned:<hash of system prompt>`. Unassigned calls are still logged.

**Why:** Explicit tags are reliable. Hash-grouped unassigned traffic shows which untagged prompts repeat, the starting point for automatic task discovery later.

## 22. Storage: SQLite (2026-10-21)

**Decision:** One SQLite file (WAL mode) holds calls, task modes and events. Mode changes (`distiller task mode`, admin API, auto-demotion) are stored there and override `distiller.yaml` without restarts.

**Why:** Self-hosting stays a single process and a single file. Postgres can replace it when one gateway process is no longer enough.

## 23. Same input text for training and serving (2026-10-21)

**Decision:** The student's input for a chat request is all user turns joined (`input_from_messages`), used both when reading logged traffic and when serving. The system prompt is replaced by the task's `instruction`.

**Why:** Any mismatch between training and serving inputs silently lowers accuracy.

## 24. Tests run offline, on SQLite and Postgres (2026-11-01)

**Decision:** Unit, integration and end-to-end tests need no network: the teacher is a mocked
HTTP transport or a local fake server, and end-to-end runs train tiny random models whose
tokenizers are built in the test. Store tests run on SQLite and on a real Postgres started by
`pgserver`.

**Why:** CI must be fast and deterministic, and "works on SQLite" says nothing about Postgres.

## 25. Gateway API keys: hashed, two ways to send them (2026-11-01)

**Decision:** Keys (`dk-...`) are stored as SHA-256 hashes with a role (app/admin). Apps send them
as `X-Distiller-Key`, or as `Authorization: Bearer dk-...`, in which case the gateway uses its own
teacher key upstream. Admin endpoints require an admin key as soon as one exists.

**Why:** `OPENAI_API_KEY=dk-...` makes adoption a one-line change, and a Distiller key must never
reach the provider. Leaving admin endpoints open until a key exists keeps local trials simple.

## 26. PII is scrubbed before storage, and the student sees the scrubbed text (2026-11-01)

**Decision:** Emails, phones, cards (Luhn), Aadhaar (Verhoeff), PAN, IBAN (mod 97) and IPs are
replaced by placeholders. The teacher gets the original text; the student gets the scrubbed text.

**Why:** Logs are training data and must not hold personal data. The student must see the same
input form at serving as in training (#23). Checksums avoid scrubbing reference numbers.

## 27. Encryption at rest for content columns only (2026-11-01)

**Decision:** Fernet-encrypt input, request and output columns; keep numbers in clear. Keys come from
an env var, comma-separated for rotation.

**Why:** Stats, drift checks and the dashboard work without decrypting; only export and retraining
read content.

## 28. Streaming: student answers are computed, then streamed (2026-11-01)

**Decision:** For `stream: true`, the student's full answer is computed first and then sent as
chunks; teacher streams are relayed unchanged and captured for logging.

**Why:** The fallback decision needs the confidence of the whole answer. Student answers are short,
so the user-visible cost is negligible, and streamed traffic still becomes training data.

## 29. One store for SQLite and Postgres; coordination through the database (2026-11-01)

**Decision:** The same SQL runs on both. Modes, keys, demotion and retrain jobs use conditional
updates, so any number of gateways and workers can share one database.

**Why:** Several gateways must agree on modes and must not demote or retrain twice. The database is
already shared, so it is the simplest coordinator; no extra queue or lock service.

## 30. vLLM as a remote student (2026-11-01)

**Decision:** `serving.backend: vllm` calls a vLLM OpenAI-compatible server with `logprobs`; the
gateway keeps calibration and threshold. Extraction uses `json_schema` response format.

**Why:** LLM students need a GPU and batching; the gateway should stay a light CPU process.

## 31. Retraining is a queued job; publishing needs a promotable model (2026-11-01)

**Decision:** Drift demotion queues a job; `distiller retrain-worker` trains on original plus logged
data and replaces `best.json` atomically only if the new model reaches the target on some slice.
Gateways reload changed `best.json` files; tasks stay in shadow unless auto-promotion is enabled.

**Why:** Training never runs inside a gateway, a bad retrain can't replace a good model, and a
person (or an explicit promotion rule) decides when a new model serves.

## 32. Hosted tier billing: share of net savings, computed from logs (2026-11-01)

**Decision:** Invoice = max(monthly minimum, share × (avoided teacher cost − student serving cost)),
per account and month, from token averages of teacher-answered calls. No payment provider yet.

**Why:** Customers pay only when Distiller saves them money, which matches the product promise.
Computing from the call log keeps one source of truth.
