"""Append-only audit log with a hash chain (spec-a7 'Audit entry'): editing or removing an entry breaks verification."""
import hashlib
import json
import time
from pathlib import Path

GENESIS = "0" * 64


def _digest(prev: str, entry: dict) -> str:
    return hashlib.sha256((prev + json.dumps(entry, sort_keys=True, default=str)).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: Path, clock=time.time):
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.read()
        self.seq = rows[-1]["seq"] + 1 if rows else 0
        self.last = rows[-1]["hash"] if rows else GENESIS

    def append(self, event_id: str | None, step: str, decision: str, *, tier: str = "", actor: str = "system",
               reason: str = "", **detail) -> dict:
        entry = {"seq": self.seq, "ts": round(self.clock(), 3), "prev_hash": self.last, "event_id": event_id, "step": step,
                 "decision": decision, "tier": tier, "actor": actor, "reason": reason, "detail": detail}
        entry["hash"] = _digest(self.last, {k: v for k, v in entry.items() if k != "hash"})
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        self.seq, self.last = self.seq + 1, entry["hash"]
        return entry

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def verify(self) -> tuple[bool, str]:
        prev = GENESIS
        for i, e in enumerate(self.read()):
            body = {k: v for k, v in e.items() if k != "hash"}
            if e.get("seq") != i or e.get("prev_hash") != prev or e.get("hash") != _digest(prev, body):
                return False, f"entry {i} does not match the chain"
            prev = e["hash"]
        return True, "ok"
