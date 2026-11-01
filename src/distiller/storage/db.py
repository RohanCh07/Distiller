"""Store for every gateway call, task modes, events and API keys.

Backends:
- SQLite (default): `database: distiller.db`. One file, no extra service. WAL mode and a
  busy timeout let several gateway processes on one machine share it.
- Postgres: `database: postgresql://user:pass@host/db` (needs `distiller[postgres]`).
  Use it when gateways run on more than one machine.

Customer content (input, request, outputs) passes through one write path, where it is
PII-scrubbed and encrypted if configured, and one read path, where it is decrypted.
"""

import json
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from distiller.security.crypto import FieldCipher
from distiller.security.pii import Scrubber

CONTENT_FIELDS = ("input", "request", "teacher_output", "student_output")

# {ID}, {FLOAT} differ between SQLite and Postgres. Everything else is portable SQL.
SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id {ID},
    ts {FLOAT} NOT NULL,
    task TEXT NOT NULL,
    mode TEXT NOT NULL,
    route TEXT NOT NULL,
    audited INTEGER NOT NULL DEFAULT 0,
    input TEXT,
    request TEXT,
    teacher_output TEXT,
    student_output TEXT,
    student_confidence {FLOAT},
    agree INTEGER,
    teacher_ms {FLOAT},
    student_ms {FLOAT},
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    model TEXT
);
CREATE INDEX IF NOT EXISTS calls_task_ts ON calls(task, ts);
CREATE TABLE IF NOT EXISTS task_modes (task TEXT PRIMARY KEY, mode TEXT NOT NULL, ts {FLOAT});
CREATE TABLE IF NOT EXISTS events (ts {FLOAT}, task TEXT, kind TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS api_keys (
    id {ID},
    name TEXT NOT NULL UNIQUE,
    key_hash TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL,
    created_ts {FLOAT} NOT NULL,
    revoked_ts {FLOAT},
    last_used_ts {FLOAT}
);
"""

# Columns added after the first release: added to existing databases on start-up.
CALL_COLUMNS_ADDED = {"api_key": "TEXT", "streamed": "INTEGER"}


def is_postgres(url: str | Path) -> bool:
    return str(url).startswith(("postgres://", "postgresql://"))


class Store:
    def __init__(
        self,
        url: str | Path,
        cipher: FieldCipher | None = None,
        scrubber: Scrubber | None = None,
    ):
        self.url = str(url)
        self.postgres = is_postgres(url)
        self.cipher = cipher
        self.scrubber = scrubber
        self._local = threading.local()
        self._sqlite_lock = threading.Lock()  # one writer at a time inside this process
        if not self.postgres:
            Path(self.url).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @property
    def path(self) -> Path:
        return Path(self.url)

    # ---------- connections ----------

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if self.postgres:
                try:
                    import psycopg
                    from psycopg.rows import dict_row
                except ImportError as e:  # pragma: no cover
                    raise RuntimeError("Postgres needs: pip install 'distiller[postgres]'") from e
                conn = psycopg.connect(self.url, autocommit=True, row_factory=dict_row)
            else:
                import sqlite3

                conn = sqlite3.connect(self.url, timeout=30, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    def _sql(self, sql: str) -> str:
        return sql.replace("?", "%s") if self.postgres else sql

    def _run(self, sql: str, params=(), fetch: str | None = None):
        """Execute one statement. fetch: None | 'one' | 'all' | 'rowcount'."""
        conn = self._conn()
        if self.postgres:
            with conn.cursor() as cur:
                cur.execute(self._sql(sql), params)
                return self._fetch(cur, fetch)
        with self._sqlite_lock:
            cur = conn.execute(sql, params)
            out = self._fetch(cur, fetch)
            conn.commit()
            return out

    @staticmethod
    def _fetch(cur, fetch):
        if fetch == "one":
            row = cur.fetchone()
            return _plain(row) if row is not None else None
        if fetch == "all":
            return [_plain(r) for r in cur.fetchall()]
        if fetch == "rowcount":
            return cur.rowcount
        return None

    def query(self, sql: str, params=()) -> list[dict[str, Any]]:
        """Read-only SQL with ? placeholders (admin tools and tests)."""
        return self._run(sql, params, fetch="all")

    def _init_schema(self) -> None:
        ddl = SCHEMA.format(
            ID="BIGSERIAL PRIMARY KEY" if self.postgres else "INTEGER PRIMARY KEY AUTOINCREMENT",
            FLOAT="DOUBLE PRECISION" if self.postgres else "REAL",
        )
        for stmt in filter(str.strip, ddl.split(";")):
            self._run(stmt)
        have = set(self._columns("calls"))
        for col, typ in CALL_COLUMNS_ADDED.items():
            if col not in have:
                self._run(f"ALTER TABLE calls ADD COLUMN {col} {typ}")

    def _columns(self, table: str) -> list[str]:
        if self.postgres:
            rows = self._run(
                "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                (table,),
                fetch="all",
            )
            return [r["column_name"] for r in rows]
        return [r["name"] for r in self._run(f"PRAGMA table_info({table})", fetch="all")]

    # ---------- content protection ----------

    def _protect(self, fields: dict[str, Any]) -> dict[str, Any]:
        out = dict(fields)
        if isinstance(out.get("request"), (list, dict)):
            req = out["request"]
            if self.scrubber and isinstance(req, list):
                req = self.scrubber.scrub_messages(req)
            out["request"] = json.dumps(req, ensure_ascii=False)
        elif self.scrubber and isinstance(out.get("request"), str):
            try:
                out["request"] = json.dumps(
                    self.scrubber.scrub_messages(json.loads(out["request"])), ensure_ascii=False
                )
            except (ValueError, TypeError):
                out["request"] = self.scrubber.scrub(out["request"])
        for f in CONTENT_FIELDS:
            if f in out and isinstance(out[f], str):
                if self.scrubber and f != "request":
                    out[f] = self.scrubber.scrub(out[f])
                if self.cipher:
                    out[f] = self.cipher.encrypt(out[f])
        return out

    def _reveal(self, row: dict[str, Any]) -> dict[str, Any]:
        if self.cipher:
            for f in CONTENT_FIELDS:
                if f in row:
                    row[f] = self.cipher.decrypt(row[f])
        return row

    # ---------- calls ----------

    def log_call(self, **fields: Any) -> int:
        fields.setdefault("ts", time.time())
        fields = self._protect(fields)
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        row = self._run(
            f"INSERT INTO calls ({cols}) VALUES ({marks}) RETURNING id",
            list(fields.values()),
            fetch="one",
        )
        return row["id"]

    def update_call(self, call_id: int, **fields: Any) -> None:
        if not fields:
            return
        fields = self._protect(fields)
        sets = ", ".join(f"{k} = ?" for k in fields)
        self._run(f"UPDATE calls SET {sets} WHERE id = ?", [*fields.values(), call_id])

    def get_call(self, call_id: int) -> dict[str, Any] | None:
        row = self._run("SELECT * FROM calls WHERE id = ?", (call_id,), fetch="one")
        return self._reveal(row) if row else None

    def iter_calls(
        self, task: str | None = None, task_prefix: str | None = None, since_ts: float = 0
    ):
        """Decrypted call rows, oldest first."""
        sql, params = "SELECT * FROM calls WHERE ts >= ?", [since_ts]
        if task is not None:
            sql += " AND task = ?"
            params.append(task)
        if task_prefix is not None:
            sql += " AND task LIKE ?"
            params.append(task_prefix + "%")
        for row in self._run(sql + " ORDER BY id", params, fetch="all"):
            yield self._reveal(row)

    def rename_task(self, old: str, new: str) -> int:
        """Move logged calls to another task name (used by task discovery)."""
        return self._run("UPDATE calls SET task = ? WHERE task = ?", (new, old), fetch="rowcount")

    def rotate_encryption(self) -> int:
        """Re-encrypt every stored content value with the current first key."""
        if not self.cipher:
            raise ValueError("no encryption key configured")
        cols = ", ".join(CONTENT_FIELDS)
        n = 0
        for row in self._run(f"SELECT id, {cols} FROM calls", fetch="all"):
            new = {f: self.cipher.rotate(row[f]) for f in CONTENT_FIELDS if row[f] is not None}
            if new:
                sets = ", ".join(f"{k} = ?" for k in new)
                self._run(f"UPDATE calls SET {sets} WHERE id = ?", [*new.values(), row["id"]])
                n += 1
        return n

    # ---------- task modes ----------

    def get_mode(self, task: str) -> str | None:
        row = self._run("SELECT mode FROM task_modes WHERE task = ?", (task,), fetch="one")
        return row["mode"] if row else None

    def set_mode(
        self, task: str, mode: str, reason: str = "manual", expect: str | None = None
    ) -> bool:
        """Set a task's mode. With expect, only if the stored mode is expect (or unset).

        The conditional form makes concurrent gateways safe: when two instances detect
        the same drift, only one demotion (and one event) happens. Returns True if changed.
        """
        now = time.time()
        sql = (
            "INSERT INTO task_modes (task, mode, ts) VALUES (?, ?, ?) "
            "ON CONFLICT(task) DO UPDATE SET mode = excluded.mode, ts = excluded.ts"
        )
        params: list[Any] = [task, mode, now]
        if expect is not None:
            sql += " WHERE task_modes.mode = ?"
            params.append(expect)
        changed = self._run(sql, params, fetch="rowcount")
        if changed:
            self.add_event(task, "mode", f"{mode} ({reason})", ts=now)
        return bool(changed)

    def mode_changed_at(self, task: str) -> float:
        row = self._run("SELECT ts FROM task_modes WHERE task = ?", (task,), fetch="one")
        return row["ts"] if row else 0.0

    # ---------- events ----------

    def add_event(self, task: str, kind: str, detail: str, ts: float | None = None) -> None:
        self._run(
            "INSERT INTO events (ts, task, kind, detail) VALUES (?, ?, ?, ?)",
            (ts or time.time(), task, kind, detail),
        )

    def events(self, limit: int = 20, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            return self._run(
                "SELECT * FROM events WHERE kind = ? ORDER BY ts DESC LIMIT ?",
                (kind, limit),
                fetch="all",
            )
        return self._run("SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,), fetch="all")

    # ---------- monitoring ----------

    def recent_agreement(
        self, task: str, window: int, audited_only: bool = True
    ) -> tuple[int, float]:
        """(samples, agreement) over the last `window` compared requests since the last mode change.

        audited_only=True: serve-mode audits (drift). False: shadow comparisons (promotion).
        """
        cond = "audited = 1" if audited_only else "mode = 'shadow'"
        rows = self._run(
            f"SELECT agree FROM calls WHERE task = ? AND {cond} AND agree IS NOT NULL "
            "AND ts > ? ORDER BY id DESC LIMIT ?",
            (task, self.mode_changed_at(task), window),
            fetch="all",
        )
        if not rows:
            return 0, 1.0
        return len(rows), sum(r["agree"] for r in rows) / len(rows)

    def stats(self) -> list[dict[str, Any]]:
        q = """
        SELECT task,
               COUNT(*) AS calls,
               SUM(CASE WHEN route = 'student' THEN 1 ELSE 0 END) AS served_by_student,
               SUM(CASE WHEN route = 'fallback' THEN 1 ELSE 0 END) AS fallbacks,
               SUM(CASE WHEN route = 'teacher' THEN 1 ELSE 0 END) AS teacher_only,
               AVG(CASE WHEN mode = 'shadow' THEN agree END) AS shadow_agreement,
               SUM(CASE WHEN mode = 'shadow' AND agree IS NOT NULL THEN 1 ELSE 0 END)
                   AS shadow_samples,
               AVG(CASE WHEN audited = 1 THEN agree END) AS audit_agreement,
               SUM(CASE WHEN audited = 1 AND agree IS NOT NULL THEN 1 ELSE 0 END)
                   AS audit_samples,
               AVG(teacher_ms) AS teacher_ms,
               AVG(CASE WHEN route = 'student' THEN student_ms END) AS student_ms,
               AVG(prompt_tokens + completion_tokens) AS teacher_tokens_per_call,
               AVG(prompt_tokens) AS teacher_prompt_tokens,
               AVG(completion_tokens) AS teacher_completion_tokens
        FROM calls GROUP BY task ORDER BY calls DESC
        """
        return self._run(q, fetch="all")

    def timeseries(self, bucket_seconds: int = 3600, since_ts: float = 0) -> list[dict[str, Any]]:
        """Calls per task, time bucket and route (for dashboard charts)."""
        q = """
        SELECT task, CAST(ts / ? AS INTEGER) * ? AS bucket, route, COUNT(*) AS calls,
               AVG(agree) AS agreement
        FROM calls WHERE ts >= ? GROUP BY task, bucket, route ORDER BY bucket
        """
        if self.postgres:
            q = q.replace("CAST(ts / ? AS INTEGER) * ?", "FLOOR(ts / ?) * ?")
        return self._run(q, (bucket_seconds, bucket_seconds, since_ts), fetch="all")

    # ---------- export ----------

    def export_training_rows(self, task: str):
        """Logged teacher answers in chat format, ready for `distiller train`."""
        rows = self._run(
            "SELECT request, teacher_output FROM calls "
            "WHERE task = ? AND teacher_output IS NOT NULL ORDER BY id",
            (task,),
            fetch="all",
        )
        for r in rows:
            r = self._reveal(r)
            messages = json.loads(r["request"]) if r["request"] else []
            messages = [m for m in messages if m.get("role") != "assistant"]
            yield {"messages": messages + [{"role": "assistant", "content": r["teacher_output"]}]}

    # ---------- API keys ----------

    def add_api_key(self, name: str, key_hash: str, role: str) -> None:
        self._run(
            "INSERT INTO api_keys (name, key_hash, role, created_ts) VALUES (?, ?, ?, ?)",
            (name, key_hash, role, time.time()),
        )

    def find_api_key(self, key_hash: str) -> dict[str, Any] | None:
        return self._run(
            "SELECT * FROM api_keys WHERE key_hash = ? AND revoked_ts IS NULL",
            (key_hash,),
            fetch="one",
        )

    def touch_api_key(self, name: str) -> None:
        self._run("UPDATE api_keys SET last_used_ts = ? WHERE name = ?", (time.time(), name))

    def revoke_api_key(self, name: str) -> bool:
        return bool(
            self._run(
                "UPDATE api_keys SET revoked_ts = ? WHERE name = ? AND revoked_ts IS NULL",
                (time.time(), name),
                fetch="rowcount",
            )
        )

    def list_api_keys(self) -> list[dict[str, Any]]:
        return self._run(
            "SELECT name, role, created_ts, revoked_ts, last_used_ts FROM api_keys ORDER BY id",
            fetch="all",
        )

    def has_admin_key(self) -> bool:
        return bool(
            self._run(
                "SELECT 1 AS x FROM api_keys WHERE role = 'admin' AND revoked_ts IS NULL LIMIT 1",
                fetch="one",
            )
        )


def _plain(row) -> dict[str, Any]:
    """Row -> dict with plain Python numbers (Postgres AVG returns Decimal)."""
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, Decimal):
            d[k] = float(v)
    return d
