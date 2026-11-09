"""Mock operations API over synthetic data (tickets, customers, transactions, policies, notes).

Stands in for the company's internal systems so the MCP server wraps a real HTTP service.
Run: python -m uvicorn mockapi.main:app --port 8000
Env: MOCKAPI_FAIL_RATE (0..1, returns 503), MOCKAPI_LATENCY_MS, MOCKAPI_NOTES_PATH.
"""
import asyncio
import json
import os
import random
import time
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ticketpilot import policies, tickets
from ticketpilot.config import CUSTOMERS_PATH, NOTES_PATH, TRANSACTIONS_PATH
from ticketpilot.schemas import Category, Status

app = FastAPI(title="Northwind Pay mock API")
MAX_LIMIT = 50


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    first = exc.errors()[0]
    where = ".".join(str(p) for p in first["loc"])
    return JSONResponse(status_code=422, content={"detail": {"code": "invalid_argument", "message": f"{where}: {first['msg']}"}})


@app.middleware("http")
async def flaky_upstream(request: Request, call_next):
    latency = float(os.environ.get("MOCKAPI_LATENCY_MS", "0") or 0)
    if latency:
        await asyncio.sleep(latency / 1000)
    if random.random() < float(os.environ.get("MOCKAPI_FAIL_RATE", "0") or 0):
        return JSONResponse(status_code=503, content={"detail": {"code": "upstream_unavailable", "message": "simulated outage"}})
    return await call_next(request)


@lru_cache(maxsize=4)
def _rows(path: Path) -> tuple[dict, ...]:
    with path.open() as f:
        return tuple(json.loads(line) for line in f if line.strip())


def _customer(customer_id: str) -> dict:
    for c in _rows(CUSTOMERS_PATH):
        if c["customer_id"] == customer_id.upper():
            return c
    raise _error(404, "not_found", f"customer {customer_id} not found")


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    t = tickets.get_ticket(ticket_id)
    if t is None:
        raise _error(404, "not_found", f"ticket {ticket_id} not found")
    return t.model_dump()


@app.get("/tickets")
def search_tickets(customer_id: str | None = None, status: Status | None = None, category: Category | None = None,
                   limit: int = Query(10, ge=1, le=MAX_LIMIT)):
    found = [t for t in tickets.all_tickets()
             if (not customer_id or t.customer_id == customer_id.upper())
             and (not status or t.status == status) and (not category or t.category == category)]
    found.sort(key=lambda t: t.created_at, reverse=True)
    return {"total": len(found), "tickets": [t.model_dump() for t in found[:limit]]}


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str):
    return _customer(customer_id)


@app.get("/customers/{customer_id}/transactions")
def list_transactions(customer_id: str, since: date | None = None, limit: int = Query(10, ge=1, le=MAX_LIMIT)):
    _customer(customer_id)
    rows = [t for t in _rows(TRANSACTIONS_PATH)
            if t["customer_id"] == customer_id.upper() and (not since or t["date"] >= since.isoformat())]
    rows.sort(key=lambda t: t["date"], reverse=True)
    return {"customer_id": customer_id.upper(), "total": len(rows), "transactions": rows[:limit]}


class NoteIn(BaseModel):
    note: str = Field(min_length=1, max_length=500)


@app.post("/tickets/{ticket_id}/notes", status_code=201)
def add_note(ticket_id: str, body: NoteIn):
    t = tickets.get_ticket(ticket_id)
    if t is None:
        raise _error(404, "not_found", f"ticket {ticket_id} not found")
    record = {"note_id": f"N-{time.time_ns()}", "ticket_id": t.ticket_id, "note": body.note,
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = Path(os.environ.get("MOCKAPI_NOTES_PATH") or NOTES_PATH)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _slug(doc: policies.Doc) -> str:
    return doc.path.removesuffix(".md")


def _policy_json(doc: policies.Doc, with_body: bool) -> dict:
    out = {"slug": _slug(doc), "title": doc.title, "version": doc.version,
           "effective_date": doc.effective_date, "status": doc.status}
    if with_body:
        out["body"] = doc.body
    return out


@app.get("/policies")
def search_policies(query: str = Query(min_length=1, max_length=300), include_superseded: bool = False,
                    limit: int = Query(3, ge=1, le=10)):
    docs = policies.search(query, k=limit, include_superseded=include_superseded)
    return {"policies": [_policy_json(d, with_body=True) for d in docs]}


@app.get("/policies/{slug}")
def get_policy(slug: str):
    doc = policies.get_doc(slug)
    if doc is not None:
        return _policy_json(doc, with_body=True)
    raise _error(404, "not_found", f"policy {slug} not found")
