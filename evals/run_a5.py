"""Live A5 runs on the configured provider: the three delegated tasks from docs/spec-a5.md.
Needs the mock API on :8000. Run: PYTHONPATH=. .venv/bin/python evals/run_a5.py [1|2|3]"""
import asyncio
import json
import sys

from ticketpilot import coworker
from ticketpilot.mcp_client import open_toolbox
from ticketpilot.trace import RunTrace

TASKS = {"1": ("investigate T-1002 and draft a resolution with evidence", "T-1002"),
         "2": ("investigate T-1068 and draft a resolution with evidence", "T-1068"),
         "3": ("investigate T-9001", "T-9001")}


async def main(which: list[str]) -> None:
    for key in which:
        task, ticket = TASKS[key]
        async with open_toolbox() as tb:
            report = await coworker.investigate(task, ticket, tb)
        print(f"\n===== task {key}: {task}\n{coworker.render(report)}")
        calls = [e for e in RunTrace(report.run_id).read() if e["phase"] == "act" and e.get("kind") == "tool"]
        print("tool calls:", [(c["tool"], c["ok"]) for c in calls], "| writes:", sum(c["tool"] == "add_ticket_note" for c in calls))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or list(TASKS)))
