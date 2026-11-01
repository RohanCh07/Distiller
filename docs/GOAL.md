# Distiller: Project Goal

> Working name. We can rename later.

## One-line vision

**An autopilot that finds repetitive AI work in a company's LLM traffic and safely moves it to small, cheap models, so their AI bill keeps dropping without quality dropping.**

---

## The problem

- Companies call big models (GPT, Claude, Gemini) for **everything**, including narrow, repetitive tasks like classification, extraction, tagging and structured JSON output.
- Those tasks don't need a frontier model. They pay 10-100x more than necessary and get slower responses.
- Doing distillation themselves needs ML engineers, training infra, eval pipelines and serving. Most startups don't have that.
- Existing options either lock you into one provider (OpenAI / Bedrock distillation) or are generic fine-tuning platforms that still need ML know-how.

## The solution

A drop-in gateway. The customer changes **one line** (their API base URL). Then:

```
Watch → Identify → Learn → Test → Replace → Keep watching
```

1. **Watch:** log every prompt and response flowing to the big model.
2. **Identify:** cluster traffic into tasks and flag the ones that are repetitive, narrow and high-volume.
3. **Learn:** fine-tune a small open model (student) on the big model's outputs (teacher).
4. **Test (shadow mode):** student answers silently next to the teacher, and we measure agreement.
5. **Replace:** when quality passes the threshold, the student serves real traffic.
6. **Keep watching:** a router sends low-confidence requests back to the big model; those become new training data; we retrain continuously and detect drift.

## What makes us different

| | Us | Generic distillation tools |
|---|---|---|
| Routing | Per-request smart router with confidence-based fallback | All-or-nothing model swap |
| Learning | Continuous: fallbacks become training data | One-time fine-tune |
| Providers | Neutral: learns from GPT, Claude, Gemini | Often locked to one provider |
| Ownership | Customer can own and self-host the student model | Model lives on the vendor's platform |
| Market wedge | India-first: Indic languages and Hinglish, India data residency, INR pricing | US-first |
| Pricing | Can charge a share of savings | Pay per training / token |

## Target users (first customers)

- Startups spending roughly **$500–$20k/month** on LLM APIs.
- With **narrow, high-volume tasks**: ticket tagging, intent detection, transaction categorization, document/field extraction, structured JSON output, translation, simple tool calls.
- Bonus fit: Indian fintech, health and SaaS companies needing Indic-language support or data residency.

## Not in scope (for now)

- Replacing open-ended reasoning, creative writing or complex multi-step agents.
- Training large models from scratch.
- Supporting every provider and modality on day one (start with OpenAI-compatible chat completions, text only).

## Success metrics

**Product**
- Student vs teacher agreement on promoted tasks: **≥ 95%**
- Cost reduction on promoted tasks: **≥ 70%**
- Latency: student p50 faster than the teacher
- Share of traffic served by students grows month over month

**Learning (personal)**
- Built and understand: an LLM gateway, data pipelines, LoRA fine-tuning, eval design, confidence-based routing, multi-LoRA serving.

## Key technical challenges

1. **Evaluation:** proving "as good as the big model" for each task (exact match, LLM-as-judge, embeddings, task-specific checks).
2. **Confidence estimation:** knowing when the student is likely wrong (log-probs, self-consistency, a learned "will it be right?" classifier).
3. **Task detection:** splitting mixed traffic into distinct, distillable tasks.
4. **Data quality:** filtering wrong teacher outputs, deduplication, PII scrubbing.
5. **Cheap serving:** many small adapters on shared GPUs (vLLM / LoRAX multi-LoRA).
6. **Trust:** security, encryption, clear data policy.

## High-level phases

| Phase | Goal | Outcome |
|---|---|---|
| 1 | **Gateway** | OpenAI-compatible proxy that logs all calls, plus a cost-per-task dashboard |
| 2 | **Manual distillation** | One task distilled end to end on a public dataset; agreement and cost measured |
| 3 | **Automation** | Task clustering, data pipeline, training queue, shadow mode |
| 4 | **Router** | Confidence-based routing, fallback, continuous retraining loop |
| 5 | **Launch** | Open-source gateway, public case study, hosted paid product |

Detailed plan to be written next in `PLAN.md`.

## Initial tech stack (draft)

- **Gateway / API:** Python, FastAPI, LiteLLM
- **Storage:** Postgres (logs, tasks, evals), Redis (queues, cache)
- **Training:** Hugging Face Transformers, TRL, Unsloth, LoRA/QLoRA
- **Base students:** Qwen / Llama / Gemma, 1–8B
- **Serving:** vLLM with multi-LoRA
- **Compute:** Colab/Kaggle for learning; Modal, RunPod or Indian GPU clouds later

## Open questions

- Product name?
- First demo task and dataset?
- Open-source the gateway from day one, or later?
- Hosting region and data-retention policy?
