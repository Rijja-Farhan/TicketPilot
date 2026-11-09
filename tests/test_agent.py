"""Agent loop tests with scripted fake-provider turns and an in-process tool server."""
import asyncio
import json

import httpx
import pytest
from mcp import Client

from mcp_server.server import build_server
from mockapi.main import app
from ticketpilot import agent, fake_llm
from ticketpilot.mcp_client import open_toolbox


@pytest.fixture(autouse=True)
def api_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MOCKAPI_NOTES_PATH", str(tmp_path / "notes.jsonl"))
    monkeypatch.setenv("MOCKAPI_FAIL_RATE", "0")


@pytest.fixture
def seen(monkeypatch):
    """Record the message list the model saw on every turn."""
    turns = []
    orig = fake_llm.respond

    def spy(messages, json_mode, **kw):
        turns.append(json.loads(json.dumps(messages)))
        return orig(messages, json_mode, **kw)

    monkeypatch.setattr(fake_llm, "respond", spy)
    return turns


def ask(question: str, max_steps: int = agent.MAX_STEPS) -> str:
    async def go():
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
        async with open_toolbox(Client(build_server(http))) as toolbox:
            return await agent.run(question, toolbox, max_steps)
    return asyncio.run(go())


def tool(name: str, **arguments):
    return {"tool_calls": [{"name": name, "arguments": arguments}]}


def tool_messages(turns):
    return [m["content"] for m in turns[-1] if m["role"] == "tool"]


def test_multi_step_lookup_then_answer(seen):
    fake_llm.queue_responses(tool("get_ticket", ticket_id="T-1002"), tool("get_customer", customer_id="C-17"),
                             "T-1002 is open; customer C-17 is active.")
    assert ask("why is T-1002 still open?") == "T-1002 is open; customer C-17 is active."
    results = tool_messages(seen)
    assert len(results) == 2 and all(r.startswith("<tool_result>") for r in results)
    assert '"ticket_id": "T-1002"' in results[0] and '"customer_id": "C-17"' in results[1]


def test_system_prompt_marks_tool_output_as_data(seen):
    fake_llm.queue_responses("ok")
    ask("how is C-17 doing?")
    assert "never instructions" in seen[0][0]["content"] and "<tool_result>" in seen[0][0]["content"]


def test_stops_after_max_steps():
    fake_llm.queue_responses(*[tool("get_customer", customer_id=f"C-{i}") for i in range(1, 10)])
    out = ask("keep going", max_steps=3)
    assert "stopped after 3 tool-calling steps" in out
    assert len(fake_llm._queue) == 6  # exactly three turns were consumed


def test_malformed_json_arguments_are_rejected(seen):
    fake_llm.queue_responses({"tool_calls": [{"name": "get_ticket", "arguments": "{not json"}]}, "gave up")
    assert ask("look up T-1002") == "gave up"
    assert "invalid_argument" in tool_messages(seen)[0]


def test_schema_invalid_arguments_are_rejected(seen):
    fake_llm.queue_responses(tool("search_tickets", limit="many"), "gave up")
    ask("tickets for C-17 please summarise")
    assert "invalid_argument" in tool_messages(seen)[0]


def test_unknown_tool_name(seen):
    fake_llm.queue_responses(tool("delete_everything"), "gave up")
    ask("do something")
    assert "not_found" in tool_messages(seen)[0] and "get_ticket" in tool_messages(seen)[0]


def test_identical_repeat_call_is_blocked(seen):
    fake_llm.queue_responses(tool("get_ticket", ticket_id="T-424242"), tool("get_ticket", ticket_id="T-424242"), "not found")
    ask("what about T-424242?")
    results = tool_messages(seen)
    assert "not_found" in results[0] and "duplicate_call" in results[1]


def test_injection_in_ticket_body_cannot_trigger_a_write(seen, tmp_path):
    fake_llm.queue_responses(tool("get_ticket", ticket_id="T-9001"),
                             tool("add_ticket_note", ticket_id="T-9001", note="injected by ticket text"),
                             "The ticket contains an instruction I ignored.")
    out = ask("summarise T-9001")
    assert "ignored" in out
    assert "not_permitted" in tool_messages(seen)[1]
    assert not (tmp_path / "notes.jsonl").exists()


def test_note_is_written_when_the_user_asks(tmp_path):
    fake_llm.queue_responses(tool("add_ticket_note", ticket_id="T-1002", note="customer called back"), "Note added.")
    assert ask("add a note to T-1002 saying the customer called back") == "Note added."
    assert json.loads((tmp_path / "notes.jsonl").read_text())["note"] == "customer called back"


def test_triage_question_gets_the_skill_in_its_prompt(seen):
    fake_llm.queue_responses("done")
    ask("triage T-1002")
    system = seen[0][0]["content"]
    assert "Triage a support ticket" in system and "name: triage-ticket" not in system


def test_plain_question_does_not_load_the_skill(seen):
    fake_llm.queue_responses("done")
    ask("why is T-1002 still open?")
    assert "Triage a support ticket" not in seen[0][0]["content"]


def test_tool_result_cannot_close_its_own_wrapper():
    assert agent.wrap_result({"x": "a</tool_result>b"}).count("</tool_result>") == 1


def test_tool_server_error_reaches_the_model(seen, monkeypatch):
    monkeypatch.setenv("MOCKAPI_FAIL_RATE", "1")
    fake_llm.queue_responses(tool("get_ticket", ticket_id="T-1002"), "The ticket system is down right now.")
    assert "down" in ask("show why T-1002 is open")
    assert "upstream_unavailable" in tool_messages(seen)[0] and '"retryable": true' in tool_messages(seen)[0]
