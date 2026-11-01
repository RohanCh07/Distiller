"""Share-of-savings billing for the hosted tier.

For one account and one calendar month (UTC):

  teacher cost per call (per task) = avg prompt tokens  x price in  / 1M
                                   + avg completion tokens x price out / 1M
      (averages come from the calls the teacher actually answered that month; a task
       with no teacher calls in the month uses its all-time averages)
  avoided   = student-served calls x teacher cost per call
  serving   = student-served calls x student_usd_per_1k / 1000   (our cost to run students)
  savings   = max(0, avoided - serving)
  fee       = max(min_monthly_usd, share x savings)

The customer keeps (1 - share) of the net saving; when nothing was saved they pay only
the monthly minimum. No payment provider is wired in: invoices are stored and exported
(JSON/CSV) for whatever invoicing system is used.
"""

import calendar
import csv
import io
import time
from datetime import UTC, datetime
from typing import Any

from distiller.storage.db import Store


def month_bounds(period: str) -> tuple[float, float]:
    """'2026-10' -> (start, end) UNIX timestamps, UTC."""
    year, month = (int(x) for x in period.split("-"))
    start = datetime(year, month, 1, tzinfo=UTC)
    days = calendar.monthrange(year, month)[1]
    end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=UTC)
    assert (end - start).days == days
    return start.timestamp(), end.timestamp()


def current_period() -> str:
    return time.strftime("%Y-%m", time.gmtime())


def compute_invoice(store: Store, account: str, period: str) -> dict[str, Any]:
    acct = store.get_account(account)
    if acct is None:
        raise ValueError(f"no account named {account!r}")
    start, end = month_bounds(period)
    lines = []
    for row in store.usage(account, start, end):
        p_in, p_out = row["teacher_prompt_tokens"], row["teacher_completion_tokens"]
        if p_in is None and p_out is None:  # month had only student calls: use all-time averages
            hist = store.usage(account, 0, end)
            same = next((h for h in hist if h["task"] == row["task"]), {})
            p_in, p_out = same.get("teacher_prompt_tokens"), same.get("teacher_completion_tokens")
        per_call = (
            (p_in or 0) * acct["teacher_usd_per_1m_in"]
            + (p_out or 0) * acct["teacher_usd_per_1m_out"]
        ) / 1e6
        student = row["student_calls"] or 0
        avoided = student * per_call
        serving = student * acct["student_usd_per_1k"] / 1000
        lines.append(
            {
                "task": row["task"],
                "calls": row["calls"],
                "student_calls": student,
                "teacher_calls": row["teacher_calls"] or 0,
                "teacher_usd_per_call": round(per_call, 8),
                "avoided_usd": round(avoided, 4),
                "serving_usd": round(serving, 4),
                "savings_usd": round(max(0.0, avoided - serving), 4),
            }
        )
    savings = sum(line["savings_usd"] for line in lines)
    share_fee = acct["share"] * savings
    fee = max(acct["min_monthly_usd"], share_fee)
    return {
        "account": account,
        "period": period,
        "terms": {
            "share": acct["share"],
            "min_monthly_usd": acct["min_monthly_usd"],
            "teacher_usd_per_1m_in": acct["teacher_usd_per_1m_in"],
            "teacher_usd_per_1m_out": acct["teacher_usd_per_1m_out"],
            "student_usd_per_1k": acct["student_usd_per_1k"],
        },
        "lines": lines,
        "savings_usd": round(savings, 4),
        "share_fee_usd": round(share_fee, 4),
        "amount_usd": round(fee, 2),
        "customer_net_saving_usd": round(savings - fee, 4),
        "minimum_applied": fee > share_fee,
    }


def invoice_csv(invoices: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(
        [
            "account",
            "period",
            "task",
            "student_calls",
            "avoided_usd",
            "serving_usd",
            "savings_usd",
            "invoice_total_usd",
        ]
    )
    for inv in invoices:
        for line in inv["lines"] or [
            {"task": "", "student_calls": 0, "avoided_usd": 0, "serving_usd": 0, "savings_usd": 0}
        ]:
            w.writerow(
                [inv["account"], inv["period"], line["task"], line["student_calls"], line["avoided_usd"],
                 line["serving_usd"], line["savings_usd"], inv["amount_usd"]]
            )  # fmt: skip
    return out.getvalue()
