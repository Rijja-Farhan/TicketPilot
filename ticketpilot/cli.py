"""CLI: python -m ticketpilot "question"."""
import argparse
import asyncio
import sys

from mcp import MCPError

from ticketpilot import agent, coworker, policies, runstate, tickets
from ticketpilot.memory import Memory
from ticketpilot.mcp_client import open_toolbox
from ticketpilot.router import Route, route

REFUSAL = ("I can only answer questions about Northwind Pay policies (disputes, refunds, fees, fraud) "
           "and look up tickets by ID (e.g. T-1001) or customer (e.g. C-17).")

AGENT_FAILED = "Sorry, I couldn't reach the TicketPilot tool server ({reason}). Is the mock API running?"


def _ask_agent(question: str) -> str:
    async def go() -> str:
        async with open_toolbox() as toolbox:
            return await agent.run(question, toolbox)

    try:
        return asyncio.run(go())
    except (OSError, MCPError, ExceptionGroup) as e:
        return AGENT_FAILED.format(reason=type(e).__name__)


def _run_coworker(question: str | None = None, ticket_id: str | None = None, resume_id: str | None = None) -> str:
    async def go() -> str:
        if resume_id:
            runstate.load(resume_id)  # fail fast on a missing or corrupt run, before starting the tool server
        async with open_toolbox() as toolbox:
            if resume_id:
                return coworker.render(await coworker.resume(resume_id, toolbox))
            return coworker.render(await coworker.investigate(question, ticket_id, toolbox))

    try:
        return asyncio.run(go())
    except KeyboardInterrupt:  # the checkpoint is already on disk
        return "Interrupted. The run was checkpointed: list runs with --runs, continue with --resume RUN_ID."
    except runstate.RunStateError as e:
        return str(e)
    except (OSError, MCPError, ExceptionGroup) as e:
        return AGENT_FAILED.format(reason=type(e).__name__)


def dispatch(question: str, r: Route | None = None) -> str:
    r = r or route(question)
    if r.kind == "ticket_lookup":
        return tickets.lookup(r.ticket_id).model_dump_json(indent=2)
    if r.kind == "customer_tickets":
        return tickets.tickets_for_customer(r.customer_id).model_dump_json(indent=2)
    if r.kind in ("ticket_question", "customer_question"):
        return _ask_agent(question)
    if r.kind == "investigate":
        return _run_coworker(question, r.ticket_id)
    if r.kind == "policy":
        return policies.answer(question)
    if r.kind == "error":
        return f"Sorry, I couldn't work out what you asked right now ({r.error})."
    return REFUSAL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ticketpilot", description="TicketPilot assistant")
    parser.add_argument("question", nargs="?", help="e.g. 'show ticket T-1001' or 'how long to file a dispute?'")
    parser.add_argument("--forget", metavar="CUSTOMER_ID", help="delete everything remembered about a customer, then exit")
    parser.add_argument("--show-memory", metavar="CUSTOMER_ID", help="print what is remembered about a customer, then exit")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue a saved coworker run from its checkpoint, then exit")
    parser.add_argument("--runs", action="store_true", help="list recent coworker runs, then exit")
    args = parser.parse_args(argv)
    if args.runs:
        rows = runstate.list_runs()
        print("\n".join(f"{rid}  {tid}  phase={phase}  {stop}".rstrip() for rid, tid, phase, stop in rows) or "No saved runs.")
        return 0
    if args.resume:
        print(_run_coworker(resume_id=args.resume))
        return 0
    if args.forget:
        print(f"Forgot {Memory().forget(args.forget)} entries for {args.forget.upper()}.")
        return 0
    if args.show_memory:
        print(Memory().block_for([args.show_memory.upper()]) or f"Nothing remembered about {args.show_memory.upper()}.")
        return 0
    if not args.question:
        parser.error("a question is required (or use --forget / --show-memory / --resume / --runs)")
    try:
        print(dispatch(args.question))
    except ValueError as e:  # bad configuration, e.g. unknown LLM_PROVIDER
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2
    return 0
