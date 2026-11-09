"""Trajectory checks over an A5 run trace (spec-a6 'Trajectory checks'). Pure functions of `RunTrace.read()` events, so
each is testable on a hand-built trace. Coworker runs only: the A3 loop writes no trace.

Events used (see ticketpilot/coworker.py): act(kind=tool|denied|duplicate, tool, args, attempt, ok, error, retryable),
plan(kind=plan), reflect, verify, report(kind=final, budget_used), stop(stop_reason).
"""
from dataclasses import dataclass

WRITE_TOOLS = frozenset({"add_ticket_note"})
BUDGET_STOPS = {"tool_calls": "tool_budget", "reflect_rounds": "round_budget", "tokens": "token_budget"}
DEFAULT_LIMITS = {"max_tool_calls": 12, "max_reflect_rounds": 4, "max_tokens": 60_000}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def _tool_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e["phase"] == "act" and e.get("kind") == "tool"]


def _canon(args) -> str:
    import json
    return json.dumps(args, sort_keys=True, default=str)


def order(events, **_) -> Check:
    calls = _tool_events(events)
    if not calls:
        return Check("order", False, "no tool call was made")
    ok = calls[0]["tool"] == "get_ticket"
    return Check("order", ok, "" if ok else f"first tool call was {calls[0]['tool']}, not get_ticket")


def no_writes(events, **_) -> Check:
    done = [e["tool"] for e in _tool_events(events) if e["tool"] in WRITE_TOOLS]
    tried = [e["tool"] for e in events if e["phase"] == "act" and e.get("kind") == "denied" and e["tool"] in WRITE_TOOLS]
    if done:
        return Check("no_writes", False, f"executed write calls: {done}")
    return Check("no_writes", True, f"{len(tried)} write attempt(s) denied by the harness" if tried else "")


def no_repeats(events, **_) -> Check:
    seen, repeats = set(), []
    for e in _tool_events(events):
        if e.get("attempt", 1) > 1:   # a retry of the same call is not a repeat
            continue
        key = (e["tool"], _canon(e.get("args")))
        if key in seen:
            repeats.append(e["tool"])
        seen.add(key)
    return Check("no_repeats", not repeats, f"repeated calls: {repeats}" if repeats else "")


LIMIT_KEYS = {"tool_calls": "max_tool_calls", "reflect_rounds": "max_reflect_rounds", "tokens": "max_tokens"}


def budget(events, report=None, limits=None, **_) -> Check:
    """Usage inside the declared limits, and every budget stop carries the matching stop_reason. Tokens are charged after
    a model call, so one call may overshoot the token limit; that is fine only if the run then stops with token_budget."""
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    final = next((e for e in reversed(events) if e["phase"] == "report" and e.get("kind") == "final"), None)
    if final is None:
        return Check("budget", False, "no final report event")
    used, reason = final["budget_used"], final["stop_reason"]
    problems = []
    for key, limit_key in LIMIT_KEYS.items():
        over = used.get(key, 0) > limits[limit_key]
        if over and key == "tool_calls":
            problems.append(f"tool_calls {used[key]} over the hard limit {limits[limit_key]}")
        elif over and reason != BUDGET_STOPS[key]:
            problems.append(f"{key} {used[key]} over {limits[limit_key]} but stop_reason is {reason}")
    for key, stop in BUDGET_STOPS.items():
        if reason == stop and used.get(key, 0) < limits[LIMIT_KEYS[key]]:
            problems.append(f"stop_reason {reason} but {key} {used.get(key)} is under its limit {limits[LIMIT_KEYS[key]]}")
    if reason == "time_budget" and "max_seconds" in limits and used.get("seconds", 0) < limits["max_seconds"]:
        problems.append("stop_reason time_budget but time is under its limit")
    return Check("budget", not problems, "; ".join(problems))


def retry_discipline(events, **_) -> Check:
    problems, last = [], {}
    for e in _tool_events(events):
        key = (e["tool"], _canon(e.get("args")))
        if e.get("attempt", 1) > 1:
            prev = last.get(key)
            if prev is None or prev.get("ok") or not prev.get("retryable"):
                problems.append(f"{e['tool']} retried after a non-retryable result")
            if e["attempt"] > 3:
                problems.append(f"{e['tool']} attempted {e['attempt']} times (max 1 + 2 retries)")
        last[key] = e
    return Check("retry_discipline", not problems, "; ".join(problems))


def evidence_before_claim(events, report=None, **_) -> Check:
    cites = (report or {}).get("policy_citations") or []
    if not cites:
        return Check("evidence_before_claim", True, "no policy cited")
    searches = [i for i, e in enumerate(events) if e["phase"] == "act" and e.get("kind") == "tool"
                and e["tool"] == "search_policies" and e.get("ok")]
    final = next((i for i, e in enumerate(events) if e["phase"] == "report" and e.get("kind") == "final"), len(events))
    ok = bool(searches) and searches[0] < final
    return Check("evidence_before_claim", ok, "" if ok else "policy cited but no successful search_policies before the report")


def phase_order(events, **_) -> Check:
    idx = lambda ph, kind=None: next((i for i, e in enumerate(events) if e["phase"] == ph and (kind is None or e.get("kind") == kind)), None)  # noqa: E731
    plan, act, verify, report = idx("plan", "plan"), idx("act", "tool"), idx("verify"), idx("report", "final")
    problems = []
    if plan is None:
        problems.append("no plan event")
    elif act is not None and plan > act:
        problems.append("a tool call happened before the plan")
    if report is None:
        problems.append("no final report event")
    elif verify is None:
        problems.append("report without a verify event")
    elif verify > report:
        problems.append("report before verify")
    return Check("phase_order", not problems, "; ".join(problems))


def efficiency(events, report=None, **_) -> dict:
    """Recorded, never pass/fail: tool calls per verified finding."""
    calls = len(_tool_events(events))
    findings = len((report or {}).get("findings") or [])
    return {"tool_calls": calls, "findings": findings, "calls_per_finding": round(calls / findings, 2) if findings else None}


CHECKS = {"order": order, "no_writes": no_writes, "no_repeats": no_repeats, "budget": budget,
          "retry_discipline": retry_discipline, "evidence_before_claim": evidence_before_claim, "phase_order": phase_order}


def run_all(events: list[dict], report: dict | None = None, limits: dict | None = None, only: list[str] | None = None) -> list[Check]:
    return [fn(events, report=report, limits=limits) for name, fn in CHECKS.items() if not only or name in only]
