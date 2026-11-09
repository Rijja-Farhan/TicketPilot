import json
from contextlib import asynccontextmanager

from ticketpilot import fake_llm
from ticketpilot.cli import REFUSAL, dispatch, main
from ticketpilot.schemas import CustomerTickets, TicketLookup


def test_ticket_lookup_end_to_end_no_llm():
    fake_llm.queue_responses("SHOULD NOT BE CONSUMED")
    out = TicketLookup.model_validate_json(dispatch("show ticket T-1001"))
    assert out.found and out.ticket_id == "T-1001"
    assert fake_llm._queue


def test_unknown_ticket():
    out = TicketLookup.model_validate_json(dispatch("T-424242"))
    assert not out.found and out.error == "ticket_not_found"


def test_customer_tickets():
    assert CustomerTickets.model_validate_json(dispatch("tickets for customer C-17")).count > 0


def test_ticket_question_goes_to_the_agent(monkeypatch):
    asked = []
    monkeypatch.setattr("ticketpilot.cli._ask_agent", lambda q: asked.append(q) or "agent answer")
    assert dispatch("why is T-1001 still open?") == "agent answer"
    assert dispatch("is C-17 a repeat complainer?") == "agent answer"
    assert asked == ["why is T-1001 still open?", "is C-17 a repeat complainer?"]


def test_agent_failure_is_a_clear_message(monkeypatch):
    @asynccontextmanager
    async def no_server():
        raise OSError("no server")
        yield

    monkeypatch.setattr("ticketpilot.cli.open_toolbox", no_server)
    out = dispatch("why is T-1001 still open?")
    assert "tool server" in out and "OSError" in out


def test_out_of_scope_refused_one_line():
    out = dispatch("write me a poem about cats")
    assert out == REFUSAL and "\n" not in out


def test_policy_path_with_fake():
    fake_llm.queue_responses(
        json.dumps({"intent": "policy"}),
        json.dumps({"answer": "120 days.", "sources": [{"title": "Dispute and Refund Policy", "version": 2,
                                                         "effective_date": "x"}]}))
    out = dispatch("how long do I have to file a dispute?")
    assert "120 days." in out and "v2, effective 2026-03-01" in out


def test_main_bad_provider_exit_code(monkeypatch, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    assert main(["write me a poem"]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_main_prints_json(capsys):
    assert main(["show ticket T-1001"]) == 0
    assert json.loads(capsys.readouterr().out)["ticket_id"] == "T-1001"


def test_investigate_goes_to_the_coworker(monkeypatch):
    asked = []
    monkeypatch.setattr("ticketpilot.cli._run_coworker", lambda q, t: asked.append((q, t)) or "report")
    assert dispatch("investigate T-1002") == "report" and asked == [("investigate T-1002", "T-1002")]


def test_runs_and_resume_flags(capsys, monkeypatch):
    assert main(["--runs"]) == 0 and "No saved runs." in capsys.readouterr().out
    assert main(["--resume", "nope"]) == 0 and "No saved run" in capsys.readouterr().out
