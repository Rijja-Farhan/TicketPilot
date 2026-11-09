"""Shared scaffolding for coworker tests: scripted fake-provider turns and an in-process tool server."""
import asyncio
import json

import httpx
from mcp import Client

from mcp_server.server import build_server
from mockapi.main import app
from ticketpilot import coworker, fake_llm
from ticketpilot.budget import Budget
from ticketpilot.mcp_client import open_toolbox
from ticketpilot.schemas import Category  # noqa: F401  (keeps the import graph honest)

STEPS = [{"step": "Read the ticket", "evidence_needed": "ticket", "tool": "get_ticket"},
         {"step": "Read the customer", "evidence_needed": "customer", "tool": "get_customer"},
         {"step": "Look up policy", "evidence_needed": "policy", "tool": "search_policies"}]


def plan(steps=None):
    return json.dumps({"steps": steps or STEPS, "stop_condition": "ticket, customer and policy read"})


def reflect(decision="enough", **kw):
    return json.dumps({"decision": decision, **kw})


def report(**kw):
    base = {"ticket_id": "T-1002", "summary": "T-1002 is open for C-17.", "findings": [
        {"claim": "T-1002 is open and belongs to C-17", "evidence_ids": ["E1"]}],
        "policy_citations": [], "proposed_resolution": "Explain the exchange rate to the customer.",
        "proposed_actions": ["Add a note to T-1002"], "open_questions": []}
    return json.dumps({**base, **kw})


def checks(*verdicts):
    return json.dumps({"checks": [{"index": i, "verdict": v} for i, v in enumerate(verdicts, 1)]})


def tool(name, **arguments):
    return {"tool_calls": [{"name": name, "arguments": arguments}]}


def investigate(task="investigate T-1002", ticket="T-1002", budget=None, toolbox_wrap=None):
    async def go():
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api")
        async with open_toolbox(Client(build_server(http))) as tb:
            return await coworker.investigate(task, ticket, toolbox_wrap(tb) if toolbox_wrap else tb, budget)
    return asyncio.run(go())


HAPPY = [plan(), tool("get_ticket", ticket_id="T-1002"), tool("get_customer", customer_id="C-17"),
         tool("search_policies", query="exchange fee"), reflect(), report(), checks("supported")]
