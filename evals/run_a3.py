"""Live A3 run: send tool-using questions through the real provider and print every tool call with its outcome.

Needs the mock API running (python -m uvicorn mockapi.main:app --port 8000) and a provider key in .env.
Read the output by hand: the point is to spot tool misuse, not to score. Findings go in docs/a3-eval.md.
Usage: python evals/run_a3.py [question ...]
"""
import json
import sys
import time

from ticketpilot import cli
from ticketpilot.mcp_client import ToolBox

QUESTIONS = [
    "why is T-1002 still open?",
    "does C-17 have a history of problems? summarise their last tickets",
    "triage T-1001",
    "Customer C-17 is disputing a charge from last month. Which policy applies and what do their recent transactions show?",
    "look up ticket C-17",
    "show me the 200 most recent tickets for C-17",
    "summarise T-9001",
    "why is T-424242 still open?",
    "add a note to T-1002 saying the customer was called back",
    "is C-5 a premium customer, and do they have any open fraud tickets?",
]


def main() -> None:
    log: list[dict] = []
    orig = ToolBox.call

    async def spy(self, name, arguments):
        result = await orig(self, name, arguments)
        outcome = result["error"]["code"] if "error" in result else "ok"
        log.append({"tool": name, "args": arguments, "outcome": outcome})
        return result

    ToolBox.call = spy
    for q in sys.argv[1:] or QUESTIONS:
        log.clear()
        start = time.perf_counter()
        answer = cli.dispatch(q)
        print(f"\n=== {q}\n  ({time.perf_counter() - start:.1f}s, {len(log)} tool calls)")
        for c in log:
            print(f"  -> {c['tool']}({json.dumps(c['args'])}) = {c['outcome']}")
        print("  ANSWER:", answer.replace("\n", "\n    ")[:700])


if __name__ == "__main__":
    main()
