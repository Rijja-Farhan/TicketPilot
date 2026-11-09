import json
import time

from ticketpilot import tickets
from ticketpilot.schemas import TicketLookup


def test_known_ticket_is_schema_valid():
    r = tickets.lookup("T-1001")
    assert r.found and r.error is None
    TicketLookup.model_validate(r.model_dump())
    assert set(r.model_dump()) == {
        "ticket_id", "found", "customer_id", "category", "status", "priority",
        "subject", "created_at", "summary", "error"}


def test_lookup_is_case_insensitive_and_fast():
    start = time.perf_counter()
    r = tickets.lookup("t-1001")
    assert (time.perf_counter() - start) < 0.05
    assert r.ticket_id == "T-1001"


def test_unknown_ticket_has_no_invented_fields():
    r = tickets.lookup("T-424242")
    assert r.found is False and r.error == "ticket_not_found"
    data = r.model_dump(exclude={"ticket_id", "found", "error"})
    assert all(v is None for v in data.values())


def test_body_not_in_lookup_json():
    assert "body" not in tickets.lookup("T-1001").model_dump()


def test_customer_filter():
    result = tickets.tickets_for_customer("c-17")
    assert result.count == len(result.tickets) > 0
    assert all(t.customer_id == "C-17" for t in result.tickets)


def test_unknown_customer_is_empty_not_error():
    assert tickets.tickets_for_customer("C-0").count == 0


def test_custom_path(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"ticket_id": "T-1", "customer_id": "C-1", "category": "other", "status": "open",
                             "priority": "p3", "subject": "s", "body": "b", "created_at": "2026-01-01T00:00:00Z"}) + "\n")
    assert tickets.lookup("T-1", p).found
    assert not tickets.lookup("T-1001", p).found
