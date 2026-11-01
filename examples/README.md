Each folder is one task: a `task.yaml` plus its data.

**banking77** (classification, 77 intents, public data)
```
python examples/banking77/prepare.py
distiller data check examples/banking77/task.yaml
distiller train examples/banking77/task.yaml        # GPU (Kaggle/Colab)
```

**transactions** (JSON extraction from Indian bank SMS / UPI text, Hinglish)
Starts from 24 hand-written seeds; the teacher grows it.
```
export DISTILLER_TEACHER_MODEL=<cheap model>  DISTILLER_TEACHER_API_KEY=<key>
distiller teacher augment examples/transactions/task.yaml --total 1500 --out examples/transactions/synthetic.jsonl
distiller train examples/transactions/task.yaml     # GPU
```
Note: with only seeds as real data, val/test are tiny. For an honest score, label a few
hundred real messages (`distiller teacher label ... --inputs raw.txt`) and use them as `data.path`.
