from ticketpilot import coworker, fake_llm
from tests.coworker_helpers import checks, investigate, plan, reflect, report, tool

RETRYABLE = {"error": {"code": "upstream_unavailable", "message": "down", "retryable": True}}


class Flaky:
    """Wraps a ToolBox: the named tool returns `result` for the first `fails` calls."""

    def __init__(self, tb, name, result, fails):
        self.tb, self.name, self.result, self.fails, self.calls = tb, name, result, fails, 0

    async def openai_tools(self):
        return await self.tb.openai_tools()

    async def call(self, name, args):
        if name == self.name:
            self.calls += 1
            if self.calls <= self.fails:
                return self.result
        return await self.tb.call(name, args)


def run(fails, delays, monkeypatch, steps=1, box=None):
    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(coworker, "_sleep", fake_sleep)
    calls = [tool("get_ticket", ticket_id="T-1002")] + ["DONE", "DONE"]
    fake_llm.queue_responses(plan(), *calls, reflect(), report(findings=[], summary="Ticket looked up."), checks())

    def wrap(tb):
        box.append(Flaky(tb, "get_ticket", RETRYABLE, fails))
        return box[0]
    return investigate(toolbox_wrap=wrap)


def test_retryable_error_is_retried_with_backoff_then_succeeds(monkeypatch):
    delays, box = [], []
    r = run(2, delays, monkeypatch, box=box)
    assert box[0].calls == 3 and len(delays) == 2 and delays[1] > delays[0] * 1.5 - 0.5
    assert r.evidence[0].ok and r.budget_used["tool_calls"] == 3


def test_tool_that_keeps_failing_is_recorded_and_bounded(monkeypatch):
    delays, box = [], []
    r = run(99, delays, monkeypatch, box=box)
    assert box[0].calls == 3 and not r.evidence[0].ok  # 1 call + 2 retries, then give up


def test_second_failed_call_marks_the_tool_unavailable(monkeypatch):
    async def fake_sleep(d):
        pass

    monkeypatch.setattr(coworker, "_sleep", fake_sleep)
    box = []
    fake_llm.queue_responses(plan(), {"tool_calls": [{"name": "get_ticket", "arguments": {"ticket_id": "T-1002"}},
                                                     {"name": "get_ticket", "arguments": {"ticket_id": "T-1003"}},
                                                     {"name": "get_ticket", "arguments": {"ticket_id": "T-1004"}}]},
                             "DONE", "DONE", reflect(), report(findings=[], summary="Tool failed."), checks())
    r = investigate(toolbox_wrap=lambda tb: box.append(Flaky(tb, "get_ticket", RETRYABLE, 99)) or box[0])
    assert box[0].calls == 6  # third call never reaches the tool
    assert [e.ok for e in r.evidence] == [False, False, False]


def test_not_found_is_never_retried(monkeypatch):
    delays = []

    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(coworker, "_sleep", fake_sleep)
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-424242"), "DONE", "DONE", reflect(),
                             report(findings=[], summary="Not found."), checks())
    r = investigate()
    assert delays == [] and r.budget_used["tool_calls"] == 1 and not r.evidence[0].ok


def test_identical_call_is_not_repeated():
    fake_llm.queue_responses(plan(), {"tool_calls": [{"name": "get_ticket", "arguments": {"ticket_id": "T-1002"}},
                                                     {"name": "get_ticket", "arguments": {"ticket_id": "T-1002"}}]},
                             "DONE", "DONE", reflect(), report(findings=[], summary="Looked up."), checks())
    r = investigate()
    assert [e.tool for e in r.evidence] == ["get_ticket"] and r.budget_used["tool_calls"] == 1
