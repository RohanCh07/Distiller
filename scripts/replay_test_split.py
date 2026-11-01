"""Replay a trained task's held-out test split through the gateway and estimate savings.

The teacher is mocked with the reference outputs from the test split (for Banking77 those
are the gold labels, which stand in for teacher answers). The script runs the real gateway
in shadow mode, then in serve mode, and prints what the gateway itself measured:
shadow agreement, share served by the student, accuracy of student answers, and an
estimated cost saving.

Usage:
  python scripts/replay_test_split.py runs/banking77 \
      --price-in 0.15 --price-out 0.60      # teacher USD per 1M input / output tokens

Token counts are estimated as characters / 4 (system prompt + input; output = answer).
Pass the prices of the teacher model you would actually use.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from distiller.data.io import read_jsonl, serialize_output
from distiller.gateway.app import create_app
from distiller.gateway.config import GatewayConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="runs/<task> written by `distiller train`")
    ap.add_argument("--limit", type=int, default=None, help="Use only the first N test rows")
    ap.add_argument("--price-in", type=float, required=True, help="Teacher USD per 1M input tokens")
    ap.add_argument(
        "--price-out", type=float, required=True, help="Teacher USD per 1M output tokens"
    )
    ap.add_argument("--calls-per-month", type=int, default=1_000_000)
    args = ap.parse_args()

    task = args.run_dir.name
    best = json.loads((args.run_dir / "best.json").read_text())
    meta = json.loads((Path(best["model_dir"]) / "distiller.json").read_text())
    instruction = meta["spec"]["instruction"]
    labels = meta["labels"] or []
    # What a real app would send the teacher: the instruction plus the allowed labels.
    system = instruction + ("\nAllowed labels: " + ", ".join(labels) if labels else "")
    test = read_jsonl(args.run_dir / "data" / "test.jsonl")[: args.limit]
    gold = {ex.input: serialize_output(ex.output) for ex in test}

    def teacher(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        text = "\n\n".join(m["content"] for m in body["messages"] if m["role"] == "user")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": gold[text]}}],
                "usage": {
                    "prompt_tokens": (len(system) + len(text)) // 4,
                    "completion_tokens": max(1, len(gold[text]) // 4),
                },
                "model": "teacher",
            },
        )

    with tempfile.TemporaryDirectory() as tmp:
        cfg = GatewayConfig.model_validate(
            {
                "database": f"{tmp}/replay.db",
                "audit_rate": 0,
                "teacher": {"base_url": "https://teacher.invalid/v1"},
                "tasks": {
                    task: {
                        "mode": "shadow",
                        "model_dir": "best",
                        "runs_dir": str(args.run_dir.parent),
                    }
                },
            }
        )
        app = create_app(cfg, transport=httpx.MockTransport(teacher))
        with TestClient(app) as client:
            if task not in client.get("/health").json()["students"]:
                raise SystemExit(f"student for {task} did not load")

            def send(text: str) -> httpx.Response:
                return client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "teacher",
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": text},
                        ],
                    },
                    headers={"X-Distiller-Task": task},
                )

            start = time.time()
            for ex in test:
                send(ex.input)
            while app.state.background:
                time.sleep(0.05)
            shadow = client.get("/stats").json()["tasks"][0]

            client.post(f"/admin/tasks/{task}/mode", json={"mode": "serve"})
            served = correct = 0
            student_ms = []
            for ex in test:
                t0 = time.perf_counter()
                r = send(ex.input)
                if r.headers["x-distiller-route"] == "student":
                    student_ms.append(1000 * (time.perf_counter() - t0))
                    served += 1
                    correct += r.json()["choices"][0]["message"]["content"] == gold[ex.input]
            elapsed = time.time() - start

    n = len(test)
    in_tok = sum((len(system) + len(ex.input)) // 4 for ex in test) / n
    out_tok = sum(max(1, len(gold[ex.input]) // 4) for ex in test) / n
    cost_per_call = (in_tok * args.price_in + out_tok * args.price_out) / 1e6
    coverage = served / n
    monthly = cost_per_call * args.calls_per_month
    report = {
        "task": task,
        "test_requests": n,
        "shadow_agreement_all_traffic": round(shadow["shadow_agreement"], 4),
        "serve_share_answered_by_student": round(coverage, 4),
        "serve_agreement_on_student_answers": round(correct / served, 4) if served else None,
        "student_latency_ms_p50": sorted(student_ms)[len(student_ms) // 2] if student_ms else None,
        "est_teacher_tokens_per_call": {"input": round(in_tok), "output": round(out_tok)},
        "est_teacher_cost_per_1k_calls_usd": round(1000 * cost_per_call, 4),
        f"est_monthly_teacher_cost_usd_at_{args.calls_per_month}_calls": round(monthly, 2),
        "est_monthly_saving_usd": round(monthly * coverage, 2),
        "prices_usd_per_1m_tokens": {"input": args.price_in, "output": args.price_out},
        "wall_seconds": round(elapsed, 1),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
