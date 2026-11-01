# Testing Distiller, step by step

You need: a Kaggle account (free, **phone-verified**, or GPU and internet stay off), an OpenAI
API key (or any OpenAI-compatible provider), and Python 3.11+ on your laptop.

---

## Part 1: Banking77 on Kaggle (classification)

1. **Zip the project.** In `D:\Projects\Distiller`, select everything *except* `runs`, `.venv`
   and any `*.db` file, right-click → *Compress to ZIP*. Name it `distiller.zip`.
2. **Upload it as a dataset.** kaggle.com → *Create* → *New Dataset* → drop `distiller.zip` →
   title it **distiller** → *Create*. (Kaggle unzips it for you.)
3. **Open the notebook.** *Create* → *New Notebook* → *File* → *Import Notebook* → upload
   `notebooks/train_kaggle.ipynb`.
4. **Attach the data.** Right panel → *Add Input* → *Your Datasets* → **distiller**.
5. **Turn on GPU and internet.** Right panel → *Session options* → *Accelerator* = **GPU T4 x2**
   (or P100), *Internet* = **On**.
6. **Check the path.** Add a cell at the top and run `!ls /kaggle/input/distiller`. You should
   see `pyproject.toml`. If you see a single folder instead (e.g. `Distiller`), change the first
   code cell to `!cp -r /kaggle/input/distiller/Distiller /kaggle/working/distiller`.
7. **Run all.** *Run* → *Run all*. Expect roughly 20–40 minutes. The last cell writes
   `run_banking77.zip`.
8. **Download.** Right panel → *Output* → download `run_banking77.zip`. Unzip it into
   `D:\Projects\Distiller` so you get `D:\Projects\Distiller\runs\banking77\...`.

Optional: add `--sweep` to the train cell (`!distiller train examples/banking77/task.yaml --sweep`)
for the auto-tuning test. It takes ~3–4× longer.

**Send me:** `runs/banking77/report.md`, `runs/banking77/data/data_report.md`, and, if any
cell fails, the full red error text.

---

## Part 2: Transactions (JSON extraction, Hinglish)

Do the data step on your laptop (no GPU needed), then train on Kaggle.

1. **Install on the laptop** (PowerShell, in `D:\Projects\Distiller`):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e ".[gateway,train]"
   ```
2. **Set the teacher** (same PowerShell window):
   ```powershell
   $env:DISTILLER_TEACHER_API_KEY = "sk-..."      # your OpenAI key
   $env:DISTILLER_TEACHER_MODEL   = "gpt-4o-mini" # or any cheap model you have
   ```
3. **Make synthetic training data** (costs a few cents with a cheap model):
   ```powershell
   distiller teacher augment examples/transactions/task.yaml --total 1500 --out examples/transactions/synthetic.jsonl
   distiller data check examples/transactions/task.yaml
   ```
   Note: the seed file has only 24 real rows, so the test score will be on ~5 examples and the
   report will say "never reaches target". That is expected. For a real score, put a few hundred
   real SMS lines in `raw.txt` and run
   `distiller teacher label examples/transactions/task.yaml --inputs raw.txt --out examples/transactions/seed.jsonl`.
4. **Train on Kaggle.** Re-zip the folder (now with `synthetic.jsonl`), upload it as a *new
   version* of the **distiller** dataset, then in the notebook change two cells:
   - install: `!pip install -q -e ".[train,constrained]"`
   - train: `!distiller train examples/transactions/task.yaml`
   - zip: replace `banking77` with `transactions`

**Send me:** `runs/transactions/report.md`, the data report, and any errors.

---

## Part 3: Gateway on your laptop

Uses the Banking77 model from Part 1 (the small encoder runs fine on a laptop CPU).

1. **Config.** Copy `distiller.example.yaml` to `distiller.yaml` (leave it as is: banking77 starts
   in `shadow` mode with `model_dir: best`).
2. **Start it** (PowerShell window 1, venv active):
   ```powershell
   $env:OPENAI_API_KEY = "sk-..."
   distiller gateway
   ```
   Check http://localhost:8000/health shows `"students": ["banking77"]`.
3. **Send traffic** (PowerShell window 2, venv active):
   ```powershell
   pip install openai
   $env:OPENAI_API_KEY = "sk-..."
   python scripts/try_gateway.py
   ```
   Every line should say `teacher` (shadow mode). Open http://localhost:8000/dashboard and see
   shadow agreement appear.
4. **Switch to serve:**
   ```powershell
   distiller task mode banking77 serve
   python scripts/try_gateway.py
   ```
   Now lines should say `student` (confident) or `fallback` (unsure).
5. **Export the logged traffic** as training data:
   ```powershell
   distiller logs export banking77 --out exported.jsonl
   distiller logs stats
   ```

**Send me:** the output of `try_gateway.py` in both modes, `distiller logs stats`, a screenshot
of `/dashboard`, and any error from window 1.

---

## If something breaks

Copy the **whole** error (from the first `Traceback` line to the end), say which part and step,
and paste it here. Don't fix it by hand first, so we see the original problem.
