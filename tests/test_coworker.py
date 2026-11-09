import json

from ticketpilot import fake_llm
from ticketpilot.trace import RunTrace
from tests.coworker_helpers import HAPPY, STEPS, checks, investigate, plan, reflect, report, tool


def test_happy_path_produces_a_verified_complete_report():
    fake_llm.queue_responses(*HAPPY)
    r = investigate()
    assert r.status == "complete" and r.stop_reason == "enough"
    assert [e.id for e in r.evidence] == ["E1", "E2", "E3"] and r.plan[0] == "Read the ticket"
    assert r.verification.failed == [] and "evidence_ids" in r.verification.passed
    assert r.verification.model_check == ["finding 1: supported"]
    assert r.proposed_actions and not fake_llm._queue


def test_phases_are_visible_in_the_trace():
    fake_llm.queue_responses(*HAPPY)
    r = investigate()
    phases = [e["phase"] for e in RunTrace(r.run_id).read()]
    for p in ("plan", "act", "observe", "reflect", "verify", "report"):
        assert p in phases


def test_write_tool_is_denied_even_if_the_model_asks(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCKAPI_NOTES_PATH", str(tmp_path / "notes.jsonl"))
    fake_llm.queue_responses(plan(), tool("add_ticket_note", ticket_id="T-1002", note="hi"), "DONE", "DONE",
                             reflect(), report(findings=[]), checks())
    r = investigate("investigate T-1002 and add a note")
    assert not (tmp_path / "notes.jsonl").exists()
    denied = [e for e in r.evidence if e.tool == "add_ticket_note"]
    assert denied and not denied[0].ok


def test_injection_in_ticket_body_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCKAPI_NOTES_PATH", str(tmp_path / "notes.jsonl"))
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-9001"),
                             tool("add_ticket_note", ticket_id="T-9001", note="all disputes approved"), "DONE",
                             reflect(), report(ticket_id="T-9001", summary="T-9001 contains an instruction I ignored.",
                                               findings=[{"claim": "T-9001 body contains an instruction aimed at the agent",
                                                          "evidence_ids": ["E1"]}], proposed_actions=[]),
                             checks("supported"))
    r = investigate("investigate T-9001", "T-9001")
    assert not (tmp_path / "notes.jsonl").exists()
    calls = [e.tool for e in r.evidence if e.ok]
    assert calls == ["get_ticket"] and r.status == "complete"


def test_plan_with_a_write_step_is_rejected_then_cleaned():
    bad = STEPS + [{"step": "Add a note", "evidence_needed": "none", "tool": "add_ticket_note"}]
    fake_llm.queue_responses(plan(bad), plan(bad), "DONE", "DONE", "DONE", reflect("blocked", reason="x"), report(findings=[]),
                             checks())
    r = investigate()
    assert len(r.plan) == 3 and "Add a note" not in r.plan
    assert any(e["kind"] == "rejected" for e in RunTrace(r.run_id).read() if e["phase"] == "plan")


def test_plan_with_a_good_second_try_is_used():
    bad = STEPS + [{"step": "Write to the ticket with add_ticket_note", "evidence_needed": "n", "tool": None}]
    fake_llm.queue_responses(plan(bad), plan(), *HAPPY[1:])
    assert investigate().status == "complete"


def test_blocked_reflection_gives_a_partial_report_with_the_reason():
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-1002"), "DONE", "DONE",
                             reflect("blocked", reason="customer record missing"), report(), checks("supported"))
    r = investigate()
    assert r.status == "partial" and r.stop_reason == "blocked"
    assert any("customer record missing" in q for q in r.open_questions)


def test_need_more_runs_another_act_round():
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-1002"), "DONE", "DONE",
                             reflect("need_more", missing="customer history"),
                             tool("search_tickets", customer_id="C-17"),
                             reflect(), report(), checks("supported"))
    r = investigate()
    assert r.status == "complete" and r.budget_used["reflect_rounds"] == 2
    assert [e.tool for e in r.evidence] == ["get_ticket", "search_tickets"]


def test_model_error_in_plan_fails_cleanly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("ticketpilot.config.load_dotenv", lambda *_: None)
    r = investigate()
    assert r.status == "failed" and r.stop_reason == "model_error" and r.findings == []


def test_memory_gets_an_episode_written_by_code(tmp_path):
    fake_llm.queue_responses(*HAPPY)
    investigate()
    text = (tmp_path / "memory.jsonl").read_text()
    assert "C-17" in text and "T-1002" in text and "exchange" not in text.lower()
    assert json.loads(text.splitlines()[0])["kind"] == "episodic"


def test_model_failure_mid_run_keeps_the_evidence_and_says_why():
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-1002"), "DONE", "DONE",
                             "not json", "still not json",           # REFLECT fails validation twice
                             report(), checks("supported"))
    r = investigate()
    assert r.status == "partial" and r.stop_reason == "model_error"
    assert any("model_error (invalid_output)" in q for q in r.open_questions)
    assert [e.tool for e in r.evidence] == ["get_ticket"]


def test_budget_stop_lists_the_steps_never_run():
    from ticketpilot.budget import Budget
    fake_llm.queue_responses(*HAPPY[:3], report(findings=[], summary="Partial."), checks())
    r = investigate(budget=Budget(max_tool_calls=1))
    assert r.stop_reason == "tool_budget" and r.status == "partial"
    assert any(q.startswith("Not completed:") for q in r.open_questions)


def test_rate_limited_model_call_is_retried_by_the_harness(monkeypatch):
    from ticketpilot import coworker, llm
    waits, real = [], llm.complete
    state = {"n": 0}

    def flaky(messages, **kw):
        if kw.get("schema") is not None and kw["schema"].__name__ == "ReportDraft" and state["n"] < 2:
            state["n"] += 1
            return llm.LLMResult(error=llm.LLMError("rate_limited", "429"))
        return real(messages, **kw)

    monkeypatch.setattr(coworker, "_model_sleep", waits.append)
    monkeypatch.setattr(coworker.llm, "complete", flaky)
    fake_llm.queue_responses(*HAPPY)
    r = investigate()
    assert r.status == "complete" and waits == [coworker.MODEL_RATE_LIMIT_WAIT_S] * 2


def test_persistent_rate_limit_ends_as_a_partial_report_that_names_it(monkeypatch):
    from ticketpilot import coworker, llm
    monkeypatch.setattr(coworker, "_model_sleep", lambda s: None)
    real = llm.complete

    def limited(messages, **kw):
        if kw.get("schema") is not None and kw["schema"].__name__ == "ReportDraft":
            return llm.LLMResult(error=llm.LLMError("rate_limited", "429"))
        return real(messages, **kw)

    monkeypatch.setattr(coworker.llm, "complete", limited)
    fake_llm.queue_responses(*HAPPY[:5])
    r = investigate()
    assert r.status == "partial" and r.stop_reason == "model_error"
    assert any("rate_limited" in q for q in r.open_questions) and "rate_limited" in r.verification.failed[0]
