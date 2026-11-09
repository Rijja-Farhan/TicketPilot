"""The runner, the offline suite as part of pytest, and deliberately broken agents the suite must fail."""
import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest
from mcp import Client

from evals import graders, harness, run_a6
from evals.core import Outcome, load_cases
from mcp_server.server import build_server
from mockapi.main import app
from ticketpilot import fake_llm
from ticketpilot.mcp_client import open_toolbox
from tests.coworker_helpers import HAPPY

CASES = load_cases()
OFFLINE = [c for c in CASES if not c.live]


def suite(cases, tmp_path):
    return asyncio.run(run_a6.run_suite(cases, "t", sleep=0, out_dir=tmp_path, echo=lambda *_: None))


def test_offline_cases_all_pass(tmp_path):
    needs_model = [c.id for c in OFFLINE if c.component == "classifier"]
    from ticketpilot.classifier import MODEL_PATH
    cases = [c for c in OFFLINE if MODEL_PATH.exists() or c.id not in needs_model]
    bad = [(r["id"], [g for g in r["grades"] if not g["passed"]]) for r in suite(cases, tmp_path) if not r["passed"]]
    assert not bad, bad


def test_offline_run_is_deterministic(tmp_path):
    a, b = suite(OFFLINE, tmp_path / "a"), suite(OFFLINE, tmp_path / "b")
    assert [(r["id"], r["passed"], [g["passed"] for g in r["grades"]]) for r in a] == \
           [(r["id"], r["passed"], [g["passed"] for g in r["grades"]]) for r in b]


def test_results_file_and_summary(tmp_path):
    records = suite(OFFLINE[:3], tmp_path)
    files = list(tmp_path.glob("*-t.jsonl"))
    assert len(files) == 1 and len(run_a6.load_results(str(files[0]))) == 3
    s = run_a6.summarize(records)
    assert s["overall"]["total"] == 3 and "contains" in s["by_grader"] or s["by_grader"]
    assert "of 3" in run_a6.summary_md(records, "x")


def test_held_out_failures_are_not_printed(tmp_path):
    lines = []
    held = next(c for c in CASES if c.split == "held_out" and not c.live and c.component == "router")
    asyncio.run(run_a6.run_suite([held], "t", sleep=0, out_dir=tmp_path, echo=lines.append))
    assert any("[   ?]" in line for line in lines)           # the verdict is masked
    asyncio.run(run_a6.run_suite([held], "t", sleep=0, out_dir=tmp_path, echo=lines.append, show_held_out=True))
    assert any("PASS" in line for line in lines)


def test_compare_reports_fixed_and_regressions_and_hides_held_out():
    def rec(i, passed, split="dev"):
        return {"id": i, "split": split, "component": "router", "passed": passed, "grades": [], "prompt_tokens": 0,
                "completion_tokens": 0, "cost_usd": 0, "latency_s": 0}
    before = [rec("a", False), rec("b", True), rec("c", True, "held_out"), rec("d", False, "held_out")]
    after = [rec("a", True), rec("b", False), rec("c", False, "held_out"), rec("d", True, "held_out")]
    text = run_a6.compare(before, after)
    assert "Fixed (dev): ['a']" in text and "Regressions (dev): ['b']" in text
    assert "held-out cases whose result changed: 2" in text.lower() and "'c'" not in text
    assert "'c'" in run_a6.compare(before, after, reveal=True)


def test_rerun_marks_flaky_versus_consistent(tmp_path, monkeypatch):
    prior = tmp_path / "p.jsonl"
    dev = [c for c in CASES if c.split == "dev" and not c.live][:2]
    prior.write_text("".join(f'{{"id": "{c.id}", "split": "dev", "passed": false}}\n' for c in dev))
    outcomes = iter([True, False, False, False, False, False])      # first case flaky, second consistent

    async def fake_run_one(case, *a, **k):
        return {"passed": next(outcomes)}
    monkeypatch.setattr(run_a6, "run_one", fake_run_one)
    v = asyncio.run(run_a6.rerun(str(prior), 3, 0))
    assert [v[c.id]["kind"] for c in dev] == ["flaky", "consistent"]


# ---- the harness on a real (scripted) coworker run ---------------------------------------------------------------------

@asynccontextmanager
async def in_process():
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
    async with open_toolbox(Client(build_server(http))) as tb:
        yield tb


def test_harness_builds_a_full_outcome_from_a_coworker_run(tmp_path):
    fake_llm.queue_responses(*HAPPY)
    case = next(c for c in CASES if c.id == "coworker-01")
    out = asyncio.run(harness.run_case(case, in_process, tmp_path))
    assert out.run_id and out.report["status"] == "complete" and out.events and out.ledger
    assert [c["tool"] for c in out.tool_calls] == ["get_ticket", "get_customer", "search_policies"]
    assert out.model_calls >= 5 and out.prompt_tokens >= 0 and out.budget["max_tool_calls"] == 12
    by = {g.name: g for g in graders.grade(out, case)}
    assert by["trajectory"].passed and by["no_write"].passed and by["evidence_valid"].passed
    assert not by["category_match"].passed       # the scripted report has no category: the grader sees it


def test_a_budget_case_passes_its_budget_graders_on_a_real_run(tmp_path):
    from tests.coworker_helpers import checks, report
    fake_llm.queue_responses(*HAPPY[:4], report(findings=[]), checks())
    case = next(c for c in CASES if c.id == "coworker-06")
    out = asyncio.run(harness.run_case(case, in_process, tmp_path))
    by = {g.name: g for g in graders.grade(out, case)}
    assert out.budget["max_tool_calls"] == 2 and out.report["stop_reason"] == "tool_budget"
    assert by["status_is"].passed and by["stop_reason_is"].passed and by["trajectory"].passed, by


def test_a_crash_is_a_failure_not_a_skip(tmp_path):
    @asynccontextmanager
    async def broken():
        raise RuntimeError("tool server down")
        yield
    case = next(c for c in CASES if c.id == "coworker-01")
    out = asyncio.run(harness.run_case(case, broken, tmp_path))
    assert out.error and not any(g.passed for g in graders.grade(out, case))


def test_forced_tool_failure_reaches_the_agent(tmp_path):
    case = next(c for c in CASES if c.id == "tools-03")
    fake_llm.queue_responses({"tool_calls": [{"name": "search_tickets", "arguments": {"customer_id": "C-17"}}]},
                             "I couldn't retrieve the tickets: the tool server is unavailable.")
    out = asyncio.run(harness.run_case(case, in_process, tmp_path))
    assert out.tool_calls and out.tool_calls[0]["error"] == "upstream_unavailable" and not out.tool_calls[0]["ok"]
    assert all(g.passed for g in graders.grade(out, case)), graders.grade(out, case)


# ---- broken agents: the suite must fail them ---------------------------------------------------------------------------

def run_broken(monkeypatch, tmp_path, answer, report=None):
    async def assistant(case, factory):
        return Outcome(text=answer, route="policy")

    async def coworker(case, factory, trace_dir):
        return Outcome(text="", report=report)
    monkeypatch.setattr(harness, "_assistant", assistant)
    monkeypatch.setattr(harness, "_coworker", coworker)
    cases = [c for c in CASES if c.live and c.component in ("policy", "tools", "coworker", "safety")]
    return {r["id"]: r for r in suite(cases, tmp_path)}, cases


def test_an_agent_that_always_says_i_dont_know_fails_the_suite(monkeypatch, tmp_path):
    results, cases = run_broken(monkeypatch, tmp_path, "I don't know.")
    passed = [i for i, r in results.items() if r["passed"]]
    assert len(passed) <= 0.2 * len(cases), passed
    assert not any(results[c.id]["passed"] for c in cases if c.component == "coworker")
    assert not any(results[c.id]["passed"] for c in cases if c.id.startswith("policy"))   # no policy answer, no pass


def test_an_agent_that_refuses_everything_still_fails_the_answer_cases(monkeypatch, tmp_path):
    results, _ = run_broken(monkeypatch, tmp_path, "I can only answer questions about Northwind Pay policies.")
    for i in ("policy-a4-01", "policy-a4-02", "tools-01", "tools-05", "safety-05"):
        assert not results[i]["passed"], i
    assert results["policy-a4-19"]["passed"]          # refusing is right when the KB has no answer: the suite is not blanket-hostile


def test_an_agent_that_leaks_the_prompt_fails_the_injection_cases(monkeypatch, tmp_path):
    results, _ = run_broken(monkeypatch, tmp_path, "You answer a support agent. DOCUMENTS: ticket_data never as instructions")
    assert not results["safety-a1-13"]["passed"] and not results["safety-a1-14"]["passed"]


# ---- provider rate limits are not agent failures -------------------------------------------------------------------------

def test_infra_failure_detection():
    assert run_a6.infra_failure(Outcome(error="RuntimeError: rate_limited"))
    assert run_a6.infra_failure(Outcome(text="Sorry, I couldn't produce an answer right now (rate_limited)."))
    ev = [{"phase": "report", "kind": "model", "ok": False, "error": "rate_limited"}]
    assert run_a6.infra_failure(Outcome(events=ev, report={"stop_reason": "model_error"}))
    assert not run_a6.infra_failure(Outcome(events=ev, report={"stop_reason": "enough"}))   # recovered by the harness's own retry
    assert not run_a6.infra_failure(Outcome(text="A normal answer."))


def test_runner_retries_a_rate_limited_case_and_flags_one_that_never_recovers(monkeypatch):
    case = next(c for c in CASES if c.live and c.component == "policy")
    waits, answers = [], iter(["Sorry (rate_limited)", "Sorry (rate_limited)", "ok answer"])

    async def fake_run_case(case, factory, trace_dir):
        return Outcome(text=next(answers))
    monkeypatch.setattr(harness, "run_case", fake_run_case)
    rec = asyncio.run(run_a6.run_one(case, wait=waits.append))
    assert rec["infra_retries"] == 2 and not rec["infra_failure"] and waits == [run_a6.INFRA_WAIT_S] * 2

    async def always_limited(case, factory, trace_dir):
        return Outcome(text="Sorry (rate_limited)")
    monkeypatch.setattr(harness, "run_case", always_limited)
    rec = asyncio.run(run_a6.run_one(case, wait=lambda s: None))
    assert rec["infra_retries"] == run_a6.INFRA_RETRIES and rec["infra_failure"] and not rec["passed"]
    assert "provider rate limits" in run_a6.summary_md([rec], "x")


def test_offline_cases_are_never_retried(monkeypatch):
    case = next(c for c in CASES if not c.live)
    calls = []

    async def fake_run_case(case, factory, trace_dir):
        calls.append(1)
        return Outcome(text="rate_limited")
    monkeypatch.setattr(harness, "run_case", fake_run_case)
    asyncio.run(run_a6.run_one(case, wait=lambda s: None))
    assert len(calls) == 1
