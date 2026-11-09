import asyncio
import sys

from ticketpilot.mcp_client import open_toolbox


def test_stdio_roundtrip_with_unreachable_api(monkeypatch):
    """Launches the real server as a subprocess; the API port is closed, so tools must fail cleanly."""
    monkeypatch.setenv("TICKETPILOT_API_URL", "http://127.0.0.1:9")

    async def go():
        async with open_toolbox() as toolbox:
            tools = await toolbox.openai_tools()
            return tools, await toolbox.call("get_ticket", {"ticket_id": "T-1002"})

    tools, result = asyncio.run(go())
    assert {t["function"]["name"] for t in tools} >= {"get_ticket", "search_policies"}
    assert all(t["type"] == "function" and t["function"]["parameters"]["type"] == "object" for t in tools)
    assert result["error"]["code"] == "upstream_unavailable" and result["error"]["retryable"] is True
