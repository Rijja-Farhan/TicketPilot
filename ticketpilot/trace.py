"""Per-run trace: one JSONL line per phase and tool call (spec-a5 'Observability'). Never breaks a run."""
import json
import os
import time
from pathlib import Path

from ticketpilot.config import RUN_TRACE_DIR


def trace_dir() -> Path:
    return Path(os.environ.get("TICKETPILOT_RUN_TRACE_DIR") or RUN_TRACE_DIR)


class RunTrace:
    def __init__(self, run_id: str, directory: Path | None = None):
        self.run_id = run_id
        self.path = Path(directory or trace_dir()) / f"{run_id}.jsonl"

    def event(self, phase: str, **fields) -> None:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "phase": phase, **fields}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
