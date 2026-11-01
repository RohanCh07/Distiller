# Building without a GPU

We don't own a GPU. That's fine: most of the product doesn't need one, and the parts that do can use free or cheap rented compute.

| Part | Needs GPU? | How we do it |
|---|---|---|
| Gateway, logging, dashboard, router logic | No | Plain laptop |
| Teacher calls | No | Cheap API models (a "mini"/"flash" tier) or free tiers; keep datasets to a few thousand rows |
| Running the student (dev/testing) | No | Small quantized model (0.5–1.5B, e.g. Qwen class) on CPU via **Ollama** or llama.cpp. Fine for classification/short JSON |
| Training the student | Yes | **Kaggle** (free GPU hours weekly) or **Colab** free tier, using Unsloth + QLoRA. A 0.5–1.5B LoRA on ~5k short examples is minutes-to-an-hour on a T4 |
| Production serving at scale | Yes | Later: Modal / RunPod / Indian GPU cloud, vLLM multi-LoRA |

## Dev loop

1. On the laptop: `distiller data check task.yaml` (seconds, no GPU) and fix data problems it reports.
2. On Kaggle: run `notebooks/train_kaggle.ipynb` (installs `.[train]`, runs `distiller train`).
3. Download `run_<task>.zip`, read `report.md`.
4. Small encoder students run on CPU for inference; LLM students can be converted to GGUF for Ollama later.

## Testing without any model

- **Fake teacher**: a stub provider that returns canned answers, so gateway tests are free and fast.
- **Fake student**: a stub that agrees with the teacher X% of the time, so router/promotion logic can be tested deterministically.
- **Tiny random models**: a 2-layer BERT and a 2-layer Llama built locally run the whole `distiller train` flow on CPU in seconds. Scores are meaningless but every code path runs.
