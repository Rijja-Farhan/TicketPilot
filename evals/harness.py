"""Run one eval case and return an Outcome (spec-a6 'Runner'). Uses the real code paths: router, cli.dispatch,
agent.run (with a recording toolbox), coworker.investigate, Memory and Classifier. Isolates memory and run files per case so
the suite never touches data/memory.jsonl."""
import os
import re
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from evals.core import Case, Outcome, report_text
from ticketpilot import agent, cli, coworker, llm, runstate
from ticketpilot.budget import Budget
from ticketpilot.mcp_client import open_toolbox
from ticketpilot.memory import Memory
from ticketpilot.router import route
from ticketpilot.trace import RunTrace

TICKET = re.compile(r"\bT-\d+\b", re.I)


class GuardedToolBox:
    """Wraps a ToolBox: records every call that reaches it, and can force failures (`fail`: tool name or '*' to a code)."""

    def __init__(self, inner, fail: dict | None = None):
        self.inner, self.fail, self.calls = inner, fail or {}, []

    async def openai_tools(self):
        return await self.inner.openai_tools()

    async def call(self, name: str, arguments: dict) -> dict:
        code = self.fail.get(name) or self.fail.get("*")
        if code:
            result = {"error": {"code": code, "message": f"{name} is unavailable (forced by the eval case)", "retryable": True}}
        else:
            result = await self.inner.call(name, arguments)
        self.calls.append({"tool": name, "args": arguments, "ok": "error" not in result,
                           "error": result.get("error", {}).get("code"), "result": result})
        return result


@contextmanager
def counting():
    """Count tokens and model calls for everything that goes through llm.complete."""
    totals = {"prompt": 0, "completion": 0, "calls": 0}
    real = llm.complete

    def wrapped(*a, **kw):
        res = real(*a, **kw)
        totals["prompt"] += res.prompt_tokens
        totals["completion"] += res.completion_tokens
        totals["calls"] += 1
        return res

    llm.complete = wrapped
    try:
        yield totals
    finally:
        llm.complete = real


@contextmanager
def isolated_env(trace_dir: Path | None = None):
    """Per-case memory file and run-state dir; coworker traces go to `trace_dir` when given."""
    tmp = tempfile.TemporaryDirectory()
    keys = {"TICKETPILOT_MEMORY_PATH": str(Path(tmp.name) / "memory.jsonl"), "TICKETPILOT_RUNS_DIR": str(Path(tmp.name) / "runs")}
    if trace_dir:
        keys["TICKETPILOT_RUN_TRACE_DIR"] = str(trace_dir)
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(keys)
    try:
        yield Path(tmp.name)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        tmp.cleanup()


# ---- memory scenarios (no model) -----------------------------------------------------------------------------------

def _clock(start: datetime):
    box = {"now": start}
    return (lambda: box["now"]), box


def memory_scenario(name: str, path: Path) -> Outcome:
    now, box = _clock(datetime(2026, 11, 1, tzinfo=timezone.utc))
    mem = Memory(path, now=now)
    trace = [("get_ticket", {"ticket_id": "T-1002"}, {"ticket_id": "T-1002", "customer_id": "C-17", "status": "open",
                                                      "category": "fees_charges", "priority": "p3"})]
    if name == "reuse":
        mem.record_run(trace, "answered")
    elif name == "expired":
        mem.record_run(trace, "answered")
        box["now"] += timedelta(days=100)
    elif name == "forget":
        mem.record_run(trace, "answered")
        mem.remember_fact("C-17", "contact preference", "phone")
        mem.forget("C-17")
    elif name == "newer_fact":
        mem.remember_fact("C-17", "contact preference", "email")
        box["now"] += timedelta(days=1)
        mem.remember_fact("C-17", "contact preference", "phone")
    else:
        return Outcome(error=f"unknown memory scenario {name}")
    file = path.read_text() if path.exists() else ""
    return Outcome(text=mem.block_for(["C-17"]) + ("\n--file--\n" + file if name == "forget" else ""), extra={"memory_file": file})


# ---- classifier ------------------------------------------------------------------------------------------------------

def classify(text: str) -> Outcome:
    from ticketpilot.classifier import MODEL_PATH, Classifier
    if not MODEL_PATH.exists():
        return Outcome(error="classifier model missing: run ml/train.py")
    p = Classifier.load().predict(text)
    return Outcome(text=p.label, label=p.label, probability=round(p.probability, 3), deferred=p.deferred)


# ---- main entry -------------------------------------------------------------------------------------------------------

def _budget(setup: dict) -> Budget | None:
    return Budget(**setup["budget"]) if setup.get("budget") else None


async def _coworker(case: Case, factory, trace_dir: Path | None) -> Outcome:
    out = Outcome()
    ticket = TICKET.search(case.input).group().upper()
    budget = _budget(case.setup)
    async with factory() as tb:
        guarded = GuardedToolBox(tb, case.setup.get("tool_failure"))
        report = await coworker.investigate(case.input, ticket, guarded, budget)
    out.run_id, out.report = report.run_id, report.model_dump()
    out.text = report_text(out.report)
    out.events = RunTrace(report.run_id).read()
    out.ledger = runstate.load(report.run_id).ledger
    out.tool_calls = [{"tool": e["tool"], "args": e["args"], "ok": e["ok"], "error": e.get("error"), "result": {}}
                      for e in out.events if e["phase"] == "act" and e.get("kind") == "tool"]
    out.budget = {k: getattr(budget or Budget(), k) for k in ("max_tool_calls", "max_reflect_rounds", "max_tokens", "max_seconds")}
    return out


async def _assistant(case: Case, factory) -> Outcome:
    out = Outcome()
    r = route(case.input)
    out.route = r.kind
    if r.kind in ("ticket_question", "customer_question") and case.component != "router":
        async with factory() as tb:
            guarded = GuardedToolBox(tb, case.setup.get("tool_failure"))
            out.text = await agent.run(case.input, guarded)
        out.tool_calls = guarded.calls
    elif r.kind == "investigate" or (case.component == "router" and r.kind in ("ticket_question", "customer_question")):
        out.text = ""   # a router case checks the route only; it never starts the agent or the coworker
    else:
        out.text = cli.dispatch(case.input, r)
    return out


async def run_case(case: Case, factory=open_toolbox, trace_dir: Path | None = None) -> Outcome:
    """Never raises: a crash becomes Outcome.error, which fails every grader (a crash is a failure, not a skip)."""
    start = time.perf_counter()
    with isolated_env(trace_dir) as tmp, counting() as tot:
        try:
            if case.component == "memory":
                out = memory_scenario(case.setup["scenario"], tmp / "memory.jsonl")
            elif case.component == "classifier":
                out = classify(case.input)
            elif case.component == "coworker":
                out = await _coworker(case, factory, trace_dir)
            elif case.setup.get("no_tools"):
                out = Outcome(route=route(case.input).kind, text=await agent.run(case.input, None))
                path = tmp / "memory.jsonl"
                out.extra["memory_file"] = path.read_text() if path.exists() else ""
            else:
                out = await _assistant(case, factory)
        except Exception as e:  # noqa: BLE001
            out = Outcome(error=f"{type(e).__name__}: {str(e)[:200]}")
    out.prompt_tokens, out.completion_tokens, out.model_calls = tot["prompt"], tot["completion"], tot["calls"]
    out.latency_s = round(time.perf_counter() - start, 2)
    return out
