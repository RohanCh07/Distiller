"""SQLite store for every gateway call, current task modes, and events (e.g. auto-demotion).

SQLite keeps self-hosting to a single file with no extra service. One connection guarded
by a lock is enough for a single gateway process; Postgres can replace this later.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    task TEXT NOT NULL,
    mode TEXT NOT NULL,
    route TEXT NOT NULL,             -- teacher | student | fallback | passthrough
    audited INTEGER NOT NULL DEFAULT 0,
    input TEXT,
    request TEXT,                    -- original messages (JSON)
    teacher_output TEXT,
    student_output TEXT,
    student_confidence REAL,
    agree INTEGER,                   -- 1/0 when both answered, else NULL
    teacher_ms REAL,
    student_ms REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    model TEXT
);
CREATE INDEX IF NOT EXISTS calls_task_ts ON calls(task, ts);
CREATE TABLE IF NOT EXISTS task_modes (task TEXT PRIMARY KEY, mode TEXT NOT NULL, ts REAL);
CREATE TABLE IF NOT EXISTS events (ts REAL, task TEXT, kind TEXT, detail TEXT);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)

    def log_call(self, **fields: Any) -> int:
        fields.setdefault("ts", time.time())
        if isinstance(fields.get("request"), (list, dict)):
            fields["request"] = json.dumps(fields["request"], ensure_ascii=False)
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self._lock:
            cur = self._db.execute(
                f"INSERT INTO calls ({cols}) VALUES ({marks})", list(fields.values())
            )
            self._db.commit()
            return cur.lastrowid

    def update_call(self, call_id: int, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE calls SET {sets} WHERE id = ?", [*fields.values(), call_id])
            self._db.commit()

    # ---------- task modes ----------

    def get_mode(self, task: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT mode FROM task_modes WHERE task = ?", (task,)).fetchone()
        return row["mode"] if row else None

    def set_mode(self, task: str, mode: str, reason: str = "manual") -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO task_modes (task, mode, ts) VALUES (?, ?, ?) "
                "ON CONFLICT(task) DO UPDATE SET mode = excluded.mode, ts = excluded.ts",
                (task, mode, time.time()),
            )
            self._db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?)",
                (time.time(), task, "mode", f"{mode} ({reason})"),
            )
            self._db.commit()

    # ---------- monitoring ----------

    def recent_agreement(self, task: str, window: int) -> tuple[int, float]:
        """(samples, agreement) over the last `window` audited requests since the last mode change."""
        with self._lock:
            changed = self._db.execute(
                "SELECT ts FROM task_modes WHERE task = ?", (task,)
            ).fetchone()
            rows = self._db.execute(
                "SELECT agree FROM calls WHERE task = ? AND audited = 1 AND agree IS NOT NULL "
                "AND ts > ? ORDER BY id DESC LIMIT ?",
                (task, changed["ts"] if changed else 0, window),
            ).fetchall()
        if not rows:
            return 0, 1.0
        return len(rows), sum(r["agree"] for r in rows) / len(rows)

    def stats(self) -> list[dict[str, Any]]:
        q = """
        SELECT task,
               COUNT(*) AS calls,
               SUM(route = 'student') AS served_by_student,
               SUM(route = 'fallback') AS fallbacks,
               SUM(route = 'teacher') AS teacher_only,
               AVG(CASE WHEN mode = 'shadow' THEN agree END) AS shadow_agreement,
               SUM(CASE WHEN mode = 'shadow' AND agree IS NOT NULL THEN 1 ELSE 0 END) AS shadow_samples,
               AVG(CASE WHEN audited = 1 THEN agree END) AS audit_agreement,
               SUM(CASE WHEN audited = 1 AND agree IS NOT NULL THEN 1 ELSE 0 END) AS audit_samples,
               AVG(teacher_ms) AS teacher_ms,
               AVG(CASE WHEN route = 'student' THEN student_ms END) AS student_ms,
               AVG(prompt_tokens + completion_tokens) AS teacher_tokens_per_call
        FROM calls GROUP BY task ORDER BY calls DESC
        """
        with self._lock:
            return [dict(r) for r in self._db.execute(q).fetchall()]

    def events(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- export ----------

    def export_training_rows(self, task: str):
        """Logged teacher answers in chat format, ready for `distiller train`."""
        with self._lock:
            rows = self._db.execute(
                "SELECT request, teacher_output FROM calls "
                "WHERE task = ? AND teacher_output IS NOT NULL ORDER BY id",
                (task,),
            ).fetchall()
        for r in rows:
            messages = json.loads(r["request"]) if r["request"] else []
            messages = [m for m in messages if m.get("role") != "assistant"]
            yield {"messages": messages + [{"role": "assistant", "content": r["teacher_output"]}]}
