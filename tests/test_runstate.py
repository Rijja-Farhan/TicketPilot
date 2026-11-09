import asyncio

import httpx
import pytest
from mcp import Client

from mcp_server.server import build_server
from mockapi.main import app
from ticketpilot import coworker, fake_llm, runstate
from ticketpilot.budget import Budget
from ticketpilot.mcp_client import open_toolbox
from tests.coworker_helpers import checks, investigate, plan, reflect, report, tool


def test_checkpoint_after_every_tool_call_and_phase():
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-1002"), "DONE", "DONE", reflect(), report(), checks("supported"))
    r = investigate()
    s = runstate.load(r.run_id)
    assert s.phase == "done" and s.report["run_id"] == r.run_id and s.ledger[0]["id"] == "E1" and s.budget["tool_calls"] == 1


def test_resume_skips_completed_calls_and_carries_budgets(monkeypatch):
    # Run is killed after the first act step: the model is asked for step 2 and the process "dies".
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-1002"))
    orig = coworker.Coworker.act_phase
    calls = {"n": 0}

    async def die_after_first_step(self):
        calls["n"] += 1
        if calls["n"] == 1:
            step = self.state.pending[0]
            self.state.pending.pop(0)
            res = self._ask("act", [{"role": "user", "content": "x"}], tools=self.tools)
            for c in res.tool_calls:
                await self.execute(c.name, c.arguments)
            self._save()
            raise KeyboardInterrupt
        return await orig(self)

    monkeypatch.setattr(coworker.Coworker, "act_phase", die_after_first_step)
    with pytest.raises(BaseException):
        investigate()
    run_id = runstate.list_runs()[0][0]
    saved = runstate.load(run_id)
    assert saved.phase == "act" and len(saved.ledger) == 1 and len(saved.pending) == 2
    used = saved.budget["tool_calls"]

    monkeypatch.setattr(coworker.Coworker, "act_phase", orig)
    fake_llm.queue_responses(tool("get_customer", customer_id="C-17"), tool("search_policies", query="fx fee"),
                             reflect(), report(), checks("supported"))

    async def go():
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
        async with open_toolbox(Client(build_server(http))) as tb:
            return await coworker.resume(run_id, tb)

    r = asyncio.run(go())
    assert r.status == "complete" and [e.tool for e in r.evidence] == ["get_ticket", "get_customer", "search_policies"]
    assert r.budget_used["tool_calls"] == used + 2  # carried over, plus only the new calls


def test_resume_of_a_finished_run_returns_the_saved_report():
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-1002"), "DONE", "DONE", reflect(), report(), checks("supported"))
    first = investigate()

    async def go():
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
        async with open_toolbox(Client(build_server(http))) as tb:
            return await coworker.resume(first.run_id, tb)

    assert asyncio.run(go()) == first and not fake_llm._queue


def test_corrupt_and_missing_state_give_clear_errors(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "bad.json").write_text("{not json")
    with pytest.raises(runstate.RunStateError, match="corrupt"):
        runstate.load("bad")
    (tmp_path / "runs" / "odd.json").write_text('{"run_id": "odd"}')
    with pytest.raises(runstate.RunStateError, match="corrupt"):
        runstate.load("odd")
    with pytest.raises(runstate.RunStateError, match="No saved run"):
        runstate.load("nope")
    assert ("bad", "?", "corrupt", "") in runstate.list_runs()


def test_budget_elapsed_time_carries_over():
    b = Budget(clock=lambda: 10.0)
    b.elapsed_before = 30
    assert Budget.from_dict(b.to_dict(), clock=lambda: 10.0).elapsed >= 30
