"""Webhook trigger (spec-a7 'Triggers'): POST /events validates the event and admits it to the durable queue. It returns at once;
the worker (`python -m ticketpilot.ambient serve`) does the work.

Also serves the demo UI (GET /ui) and the small read/decide API it uses: GET /events/{id} (one ticket's progress, built from the
store and the audit log) and POST /approvals/{id}/approve|reject (the same Approvals path as the CLI, so the policy is re-checked).
There is no auth: run it on localhost only (uvicorn binds 127.0.0.1 by default)."""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ValidationError

from ticketpilot.ambient.approvals import ApprovalError, Approvals
from ticketpilot.ambient.events import Event
from ticketpilot.ambient.queue import Intake
from ticketpilot.config import ROOT

UI_FILE = ROOT / "ui" / "index.html"
UI_SHELL = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>{}</body></html>')


class Decision(BaseModel):
    by: str = "ui-reviewer"
    reason: str = ""


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def event_status(store, audit, event_id: str) -> dict | None:
    """Everything the UI needs about one event: queue status, its audit steps, the annotation and its approval items."""
    row = store.event_row(event_id)
    if row is None:
        return None
    steps = [{k: e.get(k) for k in ("seq", "ts", "step", "decision", "reason", "detail")} for e in audit.read() if e.get("event_id") == event_id]
    return {"event_id": event_id, "status": row["status"], "duplicate_of": row["duplicate_of"], "attempts": row["attempts"],
            "error": row["error"], "customer_id": row["customer_id"], "annotation": store.annotation(event_id),
            "approvals": [a for a in store.approvals() if a["event_id"] == event_id], "steps": steps}


def create_app(store, audit, intake: Intake | None = None) -> FastAPI:
    app = FastAPI(title="TicketPilot ambient intake")
    intake = intake or Intake(store, audit)
    approvals = Approvals(store, audit)

    @app.post("/events", status_code=202)
    async def post_event(request: Request):
        try:
            raw = await request.json()
            event = Event.model_validate(raw)
        except (ValidationError, ValueError) as e:
            audit.append(None, "validate", "rejected", tier="auto", actor="webhook", reason="invalid_event", detail_text=str(e)[:200])
            return _error(422, "invalid_event", "event failed validation")
        event = event.model_copy(update={"source": "webhook"})
        return intake.admit(event)

    @app.get("/events/{event_id}")
    async def get_event(event_id: str):
        status = event_status(store, audit, event_id)
        return status if status is not None else _error(404, "not_found", f"no event {event_id}")

    @app.post("/approvals/{approval_id}/{action}")
    async def decide(approval_id: int, action: str, decision: Decision):
        try:
            if action == "approve":
                return approvals.approve(approval_id, decision.by)
            if action == "reject":
                return approvals.reject(approval_id, decision.by, decision.reason)
        except ApprovalError as e:
            return _error(409, "refused", str(e))
        return _error(404, "not_found", f"unknown action {action}")

    @app.get("/health")
    async def health():
        return {"queue_depth": store.depth(), "pending_approvals": len(store.approvals("pending")), "alerts": len(store.alerts())}

    @app.get("/ui", response_class=HTMLResponse)
    async def ui():
        if not UI_FILE.exists():
            return HTMLResponse("ui/index.html not found", status_code=404)
        head, tail = UI_SHELL.split("{}")          # not str.format: the page is full of braces
        return head + UI_FILE.read_text() + tail

    return app
