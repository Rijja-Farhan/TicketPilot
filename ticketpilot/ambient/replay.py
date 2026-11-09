"""Replay an event stream through the ambient pipeline on event time (spec-a7 'Triggers'): admit, work the queue, and run the
scheduled detector tick whenever a 5 minute bucket closes. Deterministic: no sleeping, the stream's own timestamps are the clock."""
from contextlib import asynccontextmanager

import httpx
from mcp import Client

from ticketpilot.ambient import anomaly
from ticketpilot.ambient.audit import AuditLog
from ticketpilot.ambient.events import Event
from ticketpilot.ambient.queue import Intake, IntakeConfig
from ticketpilot.ambient.store import Store
from ticketpilot.ambient.triage import Triage, drain


@asynccontextmanager
async def inprocess_toolbox():
    """The real MCP server over the real mock API, in this process (no ports, no subprocess)."""
    from mcp_server.server import build_server
    from mockapi.main import app
    from ticketpilot.mcp_client import open_toolbox
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
    async with open_toolbox(Client(build_server(http))) as tb:
        yield tb


async def replay(events: list[Event], store: Store, audit: AuditLog, toolbox, *, params: anomaly.Params = anomaly.Params(),
                 start: float | None = None, intake_config: IntakeConfig | None = None, drain_every: int = 1) -> dict:
    intake, triage = Intake(store, audit, intake_config), Triage(store, audit, toolbox)
    start = start if start is not None else (min(e.epoch for e in events) // anomaly.BUCKET_S) * anomaly.BUCKET_S
    results, latest, next_tick, pending = [], start, start + anomaly.BUCKET_S, 0
    for ev in events:
        results.append(intake.admit(ev))
        pending += 1
        latest = max(latest, ev.epoch)
        if pending >= drain_every:
            await drain(store, triage)
            pending = 0
        while latest >= next_tick:
            await drain(store, triage)
            anomaly.tick(store, audit, start, next_tick, params)
            next_tick += anomaly.BUCKET_S
    await drain(store, triage)
    anomaly.tick(store, audit, start, ((latest - start) // anomaly.BUCKET_S + 1) * anomaly.BUCKET_S + start, params)
    return {"admit": results, "usage": triage.usage, "tool_calls": triage.tools.calls, "start": start}
