"""SQLite state for the ambient agent (spec-a7): queue, seen events, annotations, approvals, alerts, outbox.
One file, so a crash leaves a consistent state and a restart resumes from it."""
import json
import os
import sqlite3
from pathlib import Path

from ticketpilot.config import ROOT

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE, payload TEXT, priority INTEGER, status TEXT,
  duplicate_of TEXT, claimed_at REAL, attempts INTEGER DEFAULT 0, error TEXT, ts REAL, customer_id TEXT, norm TEXT);
CREATE INDEX IF NOT EXISTS ev_status ON events(status, priority, seq);
CREATE INDEX IF NOT EXISTS ev_cust ON events(customer_id, ts);
CREATE TABLE IF NOT EXISTS annotations(event_id TEXT PRIMARY KEY, data TEXT);
CREATE TABLE IF NOT EXISTS approvals(
  id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT, kind TEXT, status TEXT, recipient TEXT, level INTEGER DEFAULT 0,
  urgent INTEGER DEFAULT 0, payload TEXT, created_ts REAL, decided_by TEXT, decided_ts REAL, reason TEXT,
  UNIQUE(event_id, kind));
CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, bucket_ts REAL, data TEXT, UNIQUE(bucket_ts));
CREATE TABLE IF NOT EXISTS outbox(approval_id INTEGER PRIMARY KEY, event_id TEXT, text TEXT, sent_ts REAL);
"""


def default_path() -> Path:
    return Path(os.environ.get("TICKETPILOT_AMBIENT_DIR") or ROOT / "runs" / "ambient")


class Store:
    def __init__(self, path: Path | str | None = None):
        path = Path(path) if path else default_path() / "ambient.db"
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)  # autocommit: each statement is durable on its own
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # --- events / queue -----------------------------------------------------------------------------------------------
    def event_row(self, event_id: str):
        return self.db.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()

    def insert_event(self, event, priority: int, status: str, norm: str, duplicate_of: str | None = None) -> None:
        self.db.execute("INSERT INTO events(event_id,payload,priority,status,duplicate_of,ts,customer_id,norm) VALUES(?,?,?,?,?,?,?,?)",
                        (event.event_id, event.model_dump_json(), priority, status, duplicate_of, event.epoch, event.customer_id, norm))

    def recent_for_customer(self, customer_id: str, since_ts: float, statuses: tuple[str, ...]):
        q = ",".join("?" * len(statuses))
        return self.db.execute(f"SELECT event_id, ts, norm FROM events WHERE customer_id=? AND ts>=? AND status IN ({q})",
                               (customer_id, since_ts, *statuses)).fetchall()

    def depth(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM events WHERE status IN ('pending','processing')").fetchone()[0]

    def count(self, status: str) -> int:
        return self.db.execute("SELECT COUNT(*) FROM events WHERE status=?", (status,)).fetchone()[0]

    def claim(self, now: float, lease_s: float, max_attempts: int = 3):
        """Next pending event (priority, then arrival), or one whose worker died (lease expired). Marks it processing."""
        row = self.db.execute("SELECT * FROM events WHERE status='pending' ORDER BY priority, seq LIMIT 1").fetchone() or \
            self.db.execute("SELECT * FROM events WHERE status='processing' AND claimed_at<? ORDER BY seq LIMIT 1",
                            (now - lease_s,)).fetchone()
        if row is None:
            return None
        if row["attempts"] >= max_attempts:
            self.db.execute("UPDATE events SET status='dead', error='too many attempts' WHERE seq=?", (row["seq"],))
            return self.claim(now, lease_s, max_attempts)
        self.db.execute("UPDATE events SET status='processing', claimed_at=?, attempts=attempts+1 WHERE seq=?", (now, row["seq"]))
        return self.event_row(row["event_id"])

    def finish(self, event_id: str, status: str = "done", error: str | None = None) -> None:
        self.db.execute("UPDATE events SET status=?, error=? WHERE event_id=?", (status, error, event_id))

    def retry(self, event_id: str, error: str) -> None:
        self.db.execute("UPDATE events SET status='pending', error=? WHERE event_id=?", (error, event_id))

    # --- results ------------------------------------------------------------------------------------------------------
    def annotate(self, event_id: str, data: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO annotations(event_id,data) VALUES(?,?)", (event_id, json.dumps(data, default=str)))

    def annotation(self, event_id: str) -> dict | None:
        r = self.db.execute("SELECT data FROM annotations WHERE event_id=?", (event_id,)).fetchone()
        return json.loads(r[0]) if r else None

    def annotations(self) -> dict[str, dict]:
        return {r[0]: json.loads(r[1]) for r in self.db.execute("SELECT event_id,data FROM annotations")}

    def add_approval(self, event_id: str, kind: str, payload: dict, recipient: str, created_ts: float, urgent: bool = False):
        cur = self.db.execute("INSERT OR IGNORE INTO approvals(event_id,kind,status,recipient,urgent,payload,created_ts) "
                              "VALUES(?,?,'pending',?,?,?,?)", (event_id, kind, recipient, int(urgent), json.dumps(payload, default=str), created_ts))
        return cur.lastrowid if cur.rowcount else None

    def approvals(self, status: str | None = None) -> list[dict]:
        q = "SELECT * FROM approvals" + (" WHERE status=?" if status else "") + " ORDER BY id"
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in self.db.execute(q, (status,) if status else ())]

    def approval(self, approval_id: int) -> dict | None:
        r = self.db.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
        return (dict(r) | {"payload": json.loads(r["payload"])}) if r else None

    def update_approval(self, approval_id: int, **fields) -> None:
        if "payload" in fields:
            fields["payload"] = json.dumps(fields["payload"], default=str)
        sets = ",".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE approvals SET {sets} WHERE id=?", (*fields.values(), approval_id))

    def add_alert(self, bucket_ts: float, data: dict) -> bool:
        return bool(self.db.execute("INSERT OR IGNORE INTO alerts(bucket_ts,data) VALUES(?,?)",
                                    (bucket_ts, json.dumps(data, default=str))).rowcount)

    def alerts(self) -> list[dict]:
        return [json.loads(r[0]) for r in self.db.execute("SELECT data FROM alerts ORDER BY bucket_ts")]

    def send(self, approval_id: int, event_id: str, text: str, ts: float) -> None:
        self.db.execute("INSERT OR IGNORE INTO outbox(approval_id,event_id,text,sent_ts) VALUES(?,?,?,?)", (approval_id, event_id, text, ts))

    def outbox(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM outbox ORDER BY approval_id")]

    def admitted_tickets(self) -> list[tuple[float, str, str]]:
        """(event time, event id, category) of every deduplicated, admitted ticket: the series the anomaly detector counts."""
        rows = self.db.execute("SELECT e.ts, e.event_id, a.data FROM events e LEFT JOIN annotations a ON a.event_id=e.event_id "
                               "WHERE e.status IN ('pending','processing','done') ORDER BY e.ts").fetchall()
        return [(r[0], r[1], (json.loads(r[2]).get("category") if r[2] else None) or "unclassified") for r in rows]
