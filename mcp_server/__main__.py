"""Run the MCP server over stdio: python -m mcp_server (set TICKETPILOT_API_URL if the API is not on :8000)."""
import logging

from mcp_server.server import build_server

if __name__ == "__main__":
    logging.getLogger("httpx").setLevel(logging.WARNING)  # per-request INFO lines would flood the CLI
    build_server().run()
