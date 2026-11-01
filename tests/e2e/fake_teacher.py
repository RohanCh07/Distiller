"""A local OpenAI-compatible server standing in for the teacher in end-to-end tests.

It answers two kinds of prompts the way a real teacher would:
- "write N new inputs" prompts (teacher augment) -> a JSON array of new inputs;
- labelling prompts -> a label or a JSON extraction, from simple rules.
"""

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MERCHANTS = {
    "swiggy": ("Swiggy", "food"),
    "zomato": ("Zomato", "food"),
    "amazon": ("Amazon", "shopping"),
    "uber": ("Uber", "travel"),
    "jio": ("Jio", "recharge"),
    "bigbasket": ("BigBasket", "groceries"),
}


def extract_transaction(text: str) -> dict:
    low = text.lower()
    amount = re.search(r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)", low)
    merchant, category = None, "other"
    for key, (name, cat) in MERCHANTS.items():
        if key in low:
            merchant, category = name, cat
    if "salary" in low:
        category = "salary"
    if "upi" in low or "vpa" in low:
        method = "upi"
    elif "card" in low:
        method = "card"
    elif "atm" in low:
        method, category = "atm", "cash"
    elif "neft" in low or "imps" in low:
        method = "neft_imps"
    else:
        method = "other"
    return {
        "direction": "credit" if ("credit" in low or "received" in low) else "debit",
        "amount": float(amount.group(1).replace(",", "")) if amount else None,
        "merchant": merchant,
        "category": category,
        "method": method,
    }


def classify_intent(text: str) -> str:
    low = text.lower()
    if "card" in low:
        return "card_arrival"
    if "rate" in low or "currency" in low or "euro" in low:
        return "exchange_rate"
    return "top_up"


def write_inputs(prompt: str, counter: list[int]) -> list[str]:
    n = int(re.search(r"Write (\d+) new", prompt).group(1))
    out = []
    for _ in range(n):
        i = counter[0]
        counter[0] += 1
        key = list(MERCHANTS)[i % len(MERCHANTS)]
        how = ["via UPI", "on card XX1234", "by NEFT"][i % 3]
        verb = "credited to" if i % 5 == 0 else "debited from"
        out.append(f"Rs {100 + 7 * i}.00 {verb} a/c XX{1000 + i} {how} at {key.upper()} ref {i}")
    return out


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep test output quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(body)
        msgs = body["messages"]
        if len(msgs) == 1 and "Write " in msgs[0]["content"]:
            content = json.dumps(write_inputs(msgs[0]["content"], self.server.counter))
        else:
            system, user = msgs[0]["content"], msgs[-1]["content"]
            if "JSON" in system:
                content = "Here it is:\n" + json.dumps(extract_transaction(user))
            else:
                content = classify_intent(user)
        data = json.dumps(
            {
                "id": "fake",
                "object": "chat.completion",
                "model": "fake-teacher",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class FakeTeacherServer:
    def __enter__(self) -> "FakeTeacherServer":
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.requests = []
        self.httpd.counter = [0]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    @property
    def requests(self) -> list[dict]:
        return self.httpd.requests

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
