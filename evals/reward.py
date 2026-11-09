"""Verifiable reward for the coworker, a naive version and a hardened one (docs/a6-reward-design.md).

The naive reward is built from the real graders. `python evals/reward.py` scores an honest report and a degenerate one
(real code, hand-built report) under both, to show the exploit and the fix. No model is called.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import graders, trajectory  # noqa: E402
from evals.core import Case, Outcome, report_text  # noqa: E402
from ml.train import keyword_predict  # noqa: E402
from ticketpilot import tickets  # noqa: E402

MUST_HAVE = ("get_ticket", "get_customer", "search_policies")   # the plan's must-have evidence for a ticket investigation
MIN_FINDINGS = 2


def _case(task: str) -> Case:
    return Case(id="reward", split="dev", component="coworker", input=task, expected="", graders=[], severity="low")


def outcome(report: dict, ledger: list[dict], limits: dict | None = None) -> Outcome:
    calls = [{"tool": e["tool"], "args": e["arguments"], "ok": e["ok"], "result": e["result"]} for e in ledger]
    events = [{"phase": "act", "kind": "tool", "tool": c["tool"], "args": c["args"], "attempt": 1, "ok": c["ok"]} for c in calls]
    events.append({"phase": "report", "kind": "final", "stop_reason": report["stop_reason"],
                   "budget_used": {"tool_calls": len(calls), "reflect_rounds": 1, "tokens": 5000}})
    return Outcome(text=report_text(report), report=report, ledger=ledger, tool_calls=calls, events=events, budget=limits or {})


def naive_reward(report: dict, ledger: list[dict], task: str, gold_label: str) -> tuple[float, dict]:
    """1 only if: (a) code verification passes, (b) category matches gold, (c) zero writes, (d) inside the budget."""
    out, case = outcome(report, ledger), _case(task)
    parts = {"verified": graders.evidence_valid(out, case).passed,
             "category": graders.category_match(out, case, gold_label).passed,
             "no_writes": graders.no_write(out, case).passed,
             "in_budget": trajectory.budget(out.events, report, out.budget).passed}
    return float(all(parts.values())), parts


def strict_reward(report: dict, ledger: list[dict], task: str, ticket_id: str) -> tuple[float, dict]:
    """The naive terms, plus: required evidence was gathered, enough findings survive verification, the run completed, and
    the category is checked against the data itself (not against text the agent wrote)."""
    gold = tickets.get_ticket(ticket_id).category
    r, parts = naive_reward(report, ledger, task, gold)
    ok_tools = {e["tool"] for e in ledger if e["ok"]}
    cat_ids = (report.get("category") or {}).get("evidence_ids", [])
    parts |= {"coverage": all(t in ok_tools for t in MUST_HAVE),
              "findings": len(report.get("findings", [])) >= MIN_FINDINGS,
              "completed": report["status"] == "complete",
              "category_cites_ticket": any(e["id"] in cat_ids and e["tool"] in ("get_ticket", "classify_ticket") for e in ledger)}
    return float(all(parts.values())), parts


def degenerate(ticket_id: str = "T-1002") -> tuple[dict, list[dict]]:
    """The exploit: read the ticket once, guess the category with the A4 keyword rules, write nothing else."""
    t = tickets.get_ticket(ticket_id)
    ledger = [{"id": "E1", "tool": "get_ticket", "arguments": {"ticket_id": ticket_id}, "ok": True, "timestamp": "t",
               "result": t.model_dump()}]
    guess = keyword_predict(t.subject + " " + t.body)
    report = {"ticket_id": ticket_id, "summary": "Investigation incomplete.", "category": {"label": guess, "evidence_ids": ["E1"]},
              "findings": [], "policy_citations": [], "proposed_resolution": "Needs a human to review.", "proposed_actions": [],
              "open_questions": ["Not investigated."], "run_id": "x", "status": "partial", "stop_reason": "no_progress",
              "plan": ["Read the ticket"], "evidence": [], "verification": {}, "budget_used": {}}
    return report, ledger


def honest() -> tuple[dict, list[dict]]:
    row = next(json.loads(line) for line in (ROOT / "evals" / "calibration.jsonl").read_text().splitlines()
               if json.loads(line)["id"] == "cal-04")
    return row["report"], row["ledger"]


def main() -> None:
    task = "investigate T-1002 and draft a resolution with evidence"
    gold = tickets.get_ticket("T-1002").category
    print(f"gold category for T-1002 (from the data): {gold}\n")
    print(f"{'report':<28}{'naive':>7}{'strict':>8}   failed strict terms")
    for name, (rep, led) in {"honest (real run)": honest(), "degenerate (guess + stop)": degenerate()}.items():
        n, _ = naive_reward(rep, led, task, gold)
        s, parts = strict_reward(rep, led, task, "T-1002")
        print(f"{name:<28}{n:>7.0f}{s:>8.0f}   {[k for k, v in parts.items() if not v] or '-'}")


if __name__ == "__main__":
    main()
