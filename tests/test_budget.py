import pytest

from ticketpilot import fake_llm
from ticketpilot.budget import Budget, StopRun
from tests.coworker_helpers import HAPPY, checks, investigate, plan, reflect, report, tool


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_tool_call_budget_stops_the_run():
    fake_llm.queue_responses(*HAPPY[:4], report(findings=[]), checks())
    r = investigate(budget=Budget(max_tool_calls=2))
    assert r.stop_reason == "tool_budget" and r.status == "partial" and r.budget_used["tool_calls"] == 2


def test_round_budget_stops_the_run():
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-1002"), "DONE", "DONE",
                             reflect("need_more", missing="a"), tool("get_customer", customer_id="C-17"),
                             reflect("need_more", missing="b"), report(findings=[]), checks())
    r = investigate(budget=Budget(max_reflect_rounds=2))
    assert r.stop_reason == "round_budget" and r.status == "partial"


def test_token_budget_stops_the_run_and_still_returns_a_report():
    fake_llm.queue_responses(*HAPPY)
    r = investigate(budget=Budget(max_tokens=1))
    assert r.stop_reason == "token_budget" and r.status == "failed" and r.findings == []
    assert r.open_questions


def test_time_budget_stops_the_run(monkeypatch):
    clock = Clock()

    orig = fake_llm.respond

    def slow(*a, **k):  # every model call takes 500 s on the fake clock
        clock.t += 500
        return orig(*a, **k)

    monkeypatch.setattr(fake_llm, "respond", slow)
    fake_llm.queue_responses(*HAPPY)
    r = investigate(budget=Budget(max_seconds=100, clock=clock))
    assert r.stop_reason == "time_budget"


def test_per_tool_cap_denies_only_that_tool():
    b = Budget(max_calls_per_tool=1)
    fake_llm.queue_responses(plan(), {"tool_calls": [{"name": "get_customer", "arguments": {"customer_id": "C-1"}},
                                                     {"name": "get_customer", "arguments": {"customer_id": "C-2"}}]},
                             "DONE", "DONE", reflect(), report(findings=[], summary="Two customers were looked up."),
                             checks())
    r = investigate(budget=b)
    codes = [e.ok for e in r.evidence]
    assert codes == [True, False] and r.status == "complete"


def test_no_progress_after_two_empty_reflect_rounds():
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-1002"), "DONE", "DONE",
                             reflect("need_more", missing="a"), "DONE",
                             reflect("need_more", missing="b"), "DONE",
                             reflect("need_more", missing="c"), report(findings=[]), checks())
    r = investigate()
    assert r.stop_reason == "no_progress" and r.status == "partial"


def test_budget_roundtrip_and_cost():
    b = Budget()
    b.charge_tokens(1000, 500)
    b.count_tool("get_ticket")
    b2 = Budget.from_dict(b.to_dict())
    assert b2.tokens == 1500 and b2.tool_calls == 1 and b2.calls_by_tool == {"get_ticket": 1}
    assert b2.cost_usd > 0
    with pytest.raises(StopRun):
        Budget(max_tool_calls=0).check_tool("get_ticket")
