"""Ticket store: plain-code lookups by ID or customer. No LLM involved."""
import json
from functools import lru_cache
from pathlib import Path

from ticketpilot.config import TICKETS_PATH
from ticketpilot.schemas import CustomerTickets, Ticket, TicketLookup


@lru_cache(maxsize=4)
def _load(path: Path) -> dict[str, Ticket]:
    with path.open() as f:
        return {t.ticket_id: t for t in (Ticket.model_validate_json(line) for line in f if line.strip())}


def _store(path: Path | None) -> dict[str, Ticket]:
    return _load(path or TICKETS_PATH)


def all_tickets(path: Path | None = None) -> list[Ticket]:
    return list(_store(path).values())


def _lookup(t: Ticket) -> TicketLookup:
    return TicketLookup(found=True, **t.model_dump(exclude={"body", "assignee"}))


def get_ticket(ticket_id: str, path: Path | None = None) -> Ticket | None:
    return _store(path).get(ticket_id.upper())


def lookup(ticket_id: str, path: Path | None = None) -> TicketLookup:
    t = get_ticket(ticket_id, path)
    if t is None:
        return TicketLookup(ticket_id=ticket_id.upper(), found=False, error="ticket_not_found")
    return _lookup(t)


def tickets_for_customer(customer_id: str, path: Path | None = None) -> CustomerTickets:
    cid = customer_id.upper()
    found = [_lookup(t) for t in _store(path).values() if t.customer_id == cid]
    return CustomerTickets(customer_id=cid, count=len(found), tickets=found)
