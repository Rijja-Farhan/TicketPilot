"""Durable run state (spec-a5 'Durable state'): the whole run is one JSON file, rewritten atomically after every
tool call and phase, so a killed run resumes from its last complete step with budgets carried over."""
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ticketpilot.config import RUNS_DIR

PHASES = ("plan", "act", "reflect", "report", "done")


class RunStateError(Exception):
    """A saved run cannot be loaded (missing, corrupt or incomplete)."""


def runs_dir() -> Path:
    return Path(os.environ.get("TICKETPILOT_RUNS_DIR") or RUNS_DIR)


@dataclass
class RunState:
    run_id: str
    task: str
    ticket_id: str
    created_at: str
    phase: str = "plan"
    plan: dict | None = None
    pending: list[dict] = field(default_factory=list)   # plan steps not yet acted on
    ledger: list[dict] = field(default_factory=list)    # evidence: id, tool, arguments, result, timestamp, ok
    budget: dict = field(default_factory=dict)
    no_progress: int = 0
    ledger_at_reflect: int = 0
    revised: bool = False
    tool_failures: dict = field(default_factory=dict)   # tool -> calls that ended in a retryable error; 2 = unavailable
    stop_reason: str = ""
    stop_detail: str = ""
    report: dict | None = None


def _path(run_id: str, directory: Path | None) -> Path:
    return Path(directory or runs_dir()) / f"{run_id}.json"


def save(state: RunState, directory: Path | None = None) -> None:
    path = _path(state.run_id, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), indent=1))
    tmp.replace(path)  # atomic: a crash leaves the previous complete checkpoint, never half a file


def load(run_id: str, directory: Path | None = None) -> RunState:
    path = _path(run_id, directory)
    if not path.exists():
        raise RunStateError(f"No saved run {run_id!r} (looked for {path}).")
    try:
        data = json.loads(path.read_text())
        state = RunState(**data)
    except (ValueError, TypeError) as e:
        raise RunStateError(f"Saved run {run_id!r} is corrupt and cannot be resumed ({type(e).__name__}).") from e
    if state.phase not in PHASES:
        raise RunStateError(f"Saved run {run_id!r} has an unknown phase {state.phase!r}.")
    return state


def list_runs(directory: Path | None = None, limit: int = 10) -> list[tuple[str, str, str, str]]:
    """(run_id, ticket, phase, stop_reason) of the most recent runs."""
    base = Path(directory or runs_dir())
    if not base.exists():
        return []
    out = []
    for p in sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            s = load(p.stem, base)
            out.append((s.run_id, s.ticket_id, s.phase, s.stop_reason))
        except RunStateError:
            out.append((p.stem, "?", "corrupt", ""))
    return out
