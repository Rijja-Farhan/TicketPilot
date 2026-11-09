"""Each trajectory check on one good and one bad hand-built trace, plus all checks on a real (scripted) A5 run."""
from evals import trajectory as t
from ticketpilot import fake_llm
from ticketpilot.trace import RunTrace
from tests.coworker_helpers import HAPPY, investigate


def act(tool, args=None, attempt=1, ok=True, kind="tool", retryable=False, error=None):
    return {"phase": "act", "kind": kind, "tool": tool, "args": args or {}, "attempt": attempt, "ok": ok,
            "retryable": retryable, "error": error}


PLAN = {"phase": "plan", "kind": "plan"}
VERIFY = {"phase": "verify", "kind": "result"}


def final(reason="enough", **used):
    return {"phase": "report", "kind": "final", "stop_reason": reason,
            "budget_used": {"tool_calls": 3, "reflect_rounds": 1, "tokens": 5000, "seconds": 20, **used}}


GOOD = [PLAN, act("get_ticket", {"ticket_id": "T-1"}), act("get_customer", {"customer_id": "C-1"}),
        act("search_policies", {"query": "fee"}), VERIFY, final()]


def test_order():
    assert t.order(GOOD).passed
    bad = [PLAN, act("get_customer", {"customer_id": "C-1"}), act("get_ticket", {"ticket_id": "T-1"})]
    assert not t.order(bad).passed
    assert not t.order([PLAN]).passed  # no tool call at all is not a pass


def test_no_writes():
    assert t.no_writes(GOOD).passed
    assert not t.no_writes(GOOD + [act("add_ticket_note", {"ticket_id": "T-1", "note": "x"})]).passed
    denied = t.no_writes(GOOD + [act("add_ticket_note", kind="denied")])
    assert denied.passed and "denied" in denied.detail  # a denied attempt is visible but is not an executed write


def test_no_repeats():
    assert t.no_repeats(GOOD).passed
    assert not t.no_repeats(GOOD + [act("get_ticket", {"ticket_id": "T-1"})]).passed
    retry = [act("get_ticket", {"ticket_id": "T-1"}, ok=False), act("get_ticket", {"ticket_id": "T-1"}, attempt=2)]
    assert t.no_repeats(retry).passed  # a retry is not a repeat


def test_budget():
    assert t.budget(GOOD).passed
    assert not t.budget([final(tool_calls=13)]).passed                                   # over the hard limit
    assert not t.budget([final(tokens=70_000)]).passed                                    # over limit, wrong stop_reason
    assert t.budget([final("token_budget", tokens=70_000)]).passed                        # one call may overshoot, if it stops
    assert not t.budget([final("tool_budget", tool_calls=3)]).passed                      # claims a stop it did not reach
    assert t.budget([final("tool_budget", tool_calls=2)], limits={"max_tool_calls": 2}).passed
    assert not t.budget(GOOD[:-1]).passed                                                 # no final event


def test_retry_discipline():
    ok = [act("list_transactions", {"c": 1}, ok=False, retryable=True, error="rate_limited"),
          act("list_transactions", {"c": 1}, attempt=2)]
    assert t.retry_discipline(ok).passed
    bad = [act("get_ticket", {"t": 1}, ok=False, retryable=False, error="not_found"), act("get_ticket", {"t": 1}, attempt=2)]
    assert not t.retry_discipline(bad).passed
    many = [act("x", ok=False, retryable=True), act("x", attempt=2, ok=False, retryable=True),
            act("x", attempt=3, ok=False, retryable=True), act("x", attempt=4)]
    assert not t.retry_discipline(many).passed
    assert not t.retry_discipline([act("x", attempt=2)]).passed  # a retry with no earlier attempt


def test_evidence_before_claim():
    cited = {"policy_citations": [{"title": "Fees Schedule", "version": 2, "effective_date": "2026-03-01"}]}
    assert t.evidence_before_claim(GOOD, report=cited).passed
    no_search = [PLAN, act("get_ticket", {"t": 1}), VERIFY, final()]
    assert not t.evidence_before_claim(no_search, report=cited).passed
    failed_search = [PLAN, act("search_policies", ok=False), VERIFY, final()]
    assert not t.evidence_before_claim(failed_search, report=cited).passed
    assert t.evidence_before_claim(no_search, report={"policy_citations": []}).passed


def test_phase_order():
    assert t.phase_order(GOOD).passed
    assert not t.phase_order([act("get_ticket"), PLAN, VERIFY, final()]).passed
    assert not t.phase_order([PLAN, act("get_ticket"), final(), VERIFY]).passed
    assert not t.phase_order([PLAN, act("get_ticket"), final()]).passed
    assert not t.phase_order([act("get_ticket"), VERIFY, final()]).passed


def test_efficiency_is_recorded_not_graded():
    e = t.efficiency(GOOD, {"findings": [{}, {}]})
    assert e["calls_per_finding"] == 1.5 and t.efficiency(GOOD, {"findings": []})["calls_per_finding"] is None


def test_all_checks_pass_on_a_real_scripted_run():
    fake_llm.queue_responses(*HAPPY)
    report = investigate()
    events = RunTrace(report.run_id).read()
    results = t.run_all(events, report.model_dump())
    assert [c.name for c in results] == list(t.CHECKS) and all(c.passed for c in results), [c for c in results if not c.passed]
    assert any(e.get("retryable") is False for e in events if e["phase"] == "act" and e.get("kind") == "tool")  # field exists


def test_retry_discipline_on_real_traces_with_retries(monkeypatch):
    from tests.test_retries import run
    r = run(2, [], monkeypatch, box=[])             # get_ticket fails twice with a retryable error, then succeeds
    events = RunTrace(r.run_id).read()
    attempts = [(e["attempt"], e["ok"], e["retryable"]) for e in events if e["phase"] == "act" and e.get("kind") == "tool"]
    assert attempts == [(1, False, True), (2, False, True), (3, True, False)]
    assert t.retry_discipline(events).passed and t.no_repeats(events).passed
    forged = [dict(e, retryable=False) if e.get("attempt") == 1 else e for e in events]   # same trace, but the first error "not retryable"
    assert not t.retry_discipline(forged).passed
