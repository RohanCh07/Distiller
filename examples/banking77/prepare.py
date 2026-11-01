"""Download Banking77 (PolyAI) and write data.jsonl next to task.yaml.

Gold labels stand in for teacher outputs here. With real traffic the outputs would
come from the teacher model's responses.
"""

import csv
import io
import json
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
OUT = Path(__file__).parent / "data.jsonl"


def main() -> None:
    rows = 0
    with OUT.open("w", encoding="utf-8") as out:
        for split in ("train", "test"):
            text = urllib.request.urlopen(f"{BASE}/{split}.csv", timeout=60).read().decode()
            for row in csv.DictReader(io.StringIO(text)):
                out.write(json.dumps({"input": row["text"], "output": row["category"]}) + "\n")
                rows += 1
    print(f"Wrote {rows} examples to {OUT}")


if __name__ == "__main__":
    main()
