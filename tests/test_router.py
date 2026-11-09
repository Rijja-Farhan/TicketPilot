import json

import pytest

from ticketpilot import fake_llm
from ticketpilot.router import route


@pytest.mark.parametrize("q", [
    "T-1001", "show ticket T-1001", "Show me the ticket t-1001 please", "get details for T-1001?", "lookup T-1001"])
def test_ticket_lookup_regex_no_llm(q):
    fake_llm.queue_responses("SHOULD NOT BE CONSUMED")
    r = route(q)
    assert r.kind == "ticket_lookup" and r.ticket_id == "T-1001"
    assert fake_llm._queue == ["SHOULD NOT BE CONSUMED"]


@pytest.mark.parametrize("q", ["tickets for customer C-17", "show all tickets for c-17", "C-17"])
def test_customer_tickets(q):
    assert route(q).kind == "customer_tickets" and route(q).customer_id == "C-17"


def test_free_text_about_ticket():
    r = route("why is T-1042 still open?")
    assert r.kind == "ticket_question" and r.ticket_id == "T-1042"


def test_customer_with_extra_words_goes_to_the_agent():
    fake_llm.queue_responses("SHOULD NOT BE CONSUMED")  # no LLM call is needed to decide this
    r = route("is C-17 a good customer to call?")
    assert r.kind == "customer_question" and r.customer_id == "C-17"
    assert fake_llm._queue


def test_llm_intent_policy_and_out_of_scope():
    fake_llm.queue_responses(json.dumps({"intent": "policy"}), json.dumps({"intent": "out_of_scope"}))
    assert route("how long to file a dispute?").kind == "policy"
    assert route("write me a poem").kind == "out_of_scope"


def test_llm_intent_failure_is_error_route():
    fake_llm.queue_responses("x", "y")
    r = route("anything")
    assert r.kind == "error" and r.error == "invalid_output"


@pytest.mark.parametrize("q", ["investigate T-1002", "Investigate ticket T-1002 and draft a resolution with evidence",
                               "look into T-1002", "what is the root cause of T-1002?", "draft a resolution for T-1002"])
def test_investigate_route(q):
    fake_llm.queue_responses("SHOULD NOT BE CONSUMED")
    r = route(q)
    assert r.kind == "investigate" and r.ticket_id == "T-1002"
    assert fake_llm._queue == ["SHOULD NOT BE CONSUMED"]


def test_existing_ticket_routes_are_unchanged():
    assert route("show ticket T-1001").kind == "ticket_lookup"
    assert route("why is T-1042 still open?").kind == "ticket_question"
    assert route("investigate C-17").kind == "customer_question"  # needs a ticket ID to be a coworker task
