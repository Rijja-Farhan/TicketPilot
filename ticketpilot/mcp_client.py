"""MCP client side: connect to the tool server, expose its tools in OpenAI format, execute calls.

Every call returns a dict: the tool's data, or {"error": {"code", "message", "retryable"}} (the same contract
the server uses), so the agent loop never has to catch exceptions from tools.
"""
import json
import os
import sys
from contextlib import asynccontextmanager

import anyio
import jsonschema
from mcp import Client, MCPError, StdioServerParameters

from ticketpilot.config import ROOT

TRANSPORT_ERRORS = (MCPError, OSError, anyio.ClosedResourceError, anyio.BrokenResourceError)


def _error(code: str, message: str, retryable: bool = False) -> dict:
    return {"error": {"code": code, "message": message, "retryable": retryable}}


class ToolBox:
    def __init__(self, client: Client):
        self._client = client
        self._schemas: dict[str, dict] = {}

    async def openai_tools(self) -> list[dict]:
        tools = (await self._client.list_tools()).tools
        self._schemas = {t.name: t.input_schema for t in tools}
        return [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                  "parameters": t.input_schema}} for t in tools]

    async def call(self, name: str, arguments: dict) -> dict:
        if not self._schemas:
            await self.openai_tools()
        schema = self._schemas.get(name)
        if schema is None:
            return _error("not_found", f"unknown tool {name!r}; available: {sorted(self._schemas)}")
        try:
            jsonschema.validate(arguments, schema)
        except jsonschema.ValidationError as e:
            return _error("invalid_argument", f"bad arguments for {name}: {e.message}"[:300])
        try:
            res = await self._client.call_tool(name, arguments)
        except TRANSPORT_ERRORS as e:
            return _error("upstream_unavailable", f"tool server failed: {type(e).__name__}", True)
        text = "".join(getattr(c, "text", "") for c in res.content)
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict):
            return data
        if res.is_error:  # raised inside the SDK (e.g. schema validation), not by our own error contract
            return _error("invalid_argument", text.splitlines()[0][:200] if text else "tool failed")
        return {"result": text}


@asynccontextmanager
async def open_toolbox(client: Client | None = None):
    """Yield a ToolBox. Default: launch `python -m mcp_server` over stdio. Tests pass an in-process Client."""
    if client is not None:
        async with client:
            yield ToolBox(client)
        return
    params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server"], cwd=str(ROOT),
                                   env={k: v for k, v in os.environ.items() if k in ("PATH", "TICKETPILOT_API_URL", "HOME")})
    async with Client(params) as stdio_client:
        yield ToolBox(stdio_client)
