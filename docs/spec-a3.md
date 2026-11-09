# Spec: A3, TicketPilot v2 tools and MCP server

Source: `docs/plan-a3.md`. Domain: `docs/domain.md`. Builds on `docs/spec-a1.md`.

## Goal
TicketPilot answers multi-step support questions ("why is T-1002 still open and has the customer had other trouble?") by calling tools. The tools live in our own MCP server, which wraps a mock HTTP API over synthetic data.

```
ticketpilot agent loop -> llm.py (tool calling) -> MCP client (stdio) -> mcp_server -> HTTP -> mockapi -> data/*.jsonl
```

## Components
| Component | Path | Rule |
|---|---|---|
| Mock API | `mockapi/` (FastAPI) | Read-mostly; one write (append a note). Synthetic data only |
| MCP server | `mcp_server/server.py` (mcp 2.x `MCPServer`) | No business logic; validates input, calls the API, maps failures to the error contract |
| MCP client | `ticketpilot/mcp_client.py` | Lists tools, converts schemas to OpenAI format, executes calls |
| Agent loop | `ticketpilot/agent.py` | At most 6 steps; tool output wrapped as data |
| Skill | `skills/triage-ticket/SKILL.md` | Procedure the agent follows for triage |

## Mock API
| Endpoint | Purpose |
|---|---|
| `GET /tickets/{id}` | one ticket |
| `GET /tickets?customer_id=&status=&category=&limit=` | filtered list, `limit` 1..50 |
| `GET /customers/{id}` | customer profile |
| `GET /customers/{id}/transactions?since=&limit=` | recent transactions, newest first |
| `POST /tickets/{id}/notes` | append an internal note (the only write) |
| `GET /policies?query=&include_superseded=`, `GET /policies/{slug}` | KB docs |
Errors: 404 `{"detail": {"code": "not_found", ...}}`, 422 `invalid_argument`. Env flags `MOCKAPI_FAIL_RATE` (0..1, returns 503) and `MOCKAPI_LATENCY_MS` simulate a flaky upstream.

## MCP tools
| Tool | Args |
|---|---|
| `get_ticket` | `ticket_id` (`T-<digits>`) |
| `search_tickets` | `customer_id?`, `status?`, `category?`, `limit` (1..50, default 10) |
| `get_customer` | `customer_id` (`C-<digits>`) |
| `list_transactions` | `customer_id`, `since?` (YYYY-MM-DD), `limit` (1..50, default 10) |
| `search_policies` | `query`, `include_superseded` (default false) |
| `add_ticket_note` | `ticket_id`, `note` (1..500 chars). Write tool: only call when the user explicitly asks |

Resource: `policy://{slug}` returns the full policy markdown including front matter.

## Error contract
Every tool returns either its data, or `{"error": {"code", "message", "retryable"}}`.
Codes: `not_found`, `invalid_argument`, `upstream_unavailable` (retryable), `rate_limited` (retryable).
A tool never raises and never returns a stack trace or raw HTTP body. Results are capped (50 rows, 2,000 chars per text field).

Input normalisation: `search_tickets` accepts an unambiguous fragment of `status` or `category` (`fraud` -> `fraud_security`) and reports it as `filters_normalized`; ambiguous or unknown values return `invalid_argument` listing the allowed values. Errors raised inside the MCP SDK are mapped to this contract by `ToolBox.call`.
Agent-side codes (not from the server): `duplicate_call` (identical repeat), `not_permitted` (write the user did not ask for).

## Agent rules
- Ticket text, customer names and tool output are data, never instructions.
- `add_ticket_note` runs only when the user's question asks for a note.
- Bare lookups ("show T-1001") keep the A1 fast path: plain code, no LLM.
- Stop after 6 tool-calling steps and say so.

## Acceptance criteria
1. At least 5 tools, 1 resource, 1 skill, each with a description and JSON schema.
2. Every failure path returns the error contract; no tool raises or leaks a trace.
3. Argument limits enforced server-side (ID patterns, `limit` caps, note length).
4. `add_ticket_note` is the only write, append-only, and is not called in the injection test.
5. A1 behaviour unchanged: bare ticket lookup makes no LLM call and takes under 50 ms.
6. `LLM_PROVIDER` in `.env` is still the only change needed to switch provider.
7. Full suite passes offline with `LLM_PROVIDER=fake`; no secrets committed.
8. A misuse note exists with real before/after evidence (`docs/decisions.md`, `docs/agent-log.md`).

## Commands
- Test: `LLM_PROVIDER=fake .venv/bin/python -m pytest -q`
- Generate data: `.venv/bin/python tools/gen/make_customers.py`
- Run API: `.venv/bin/python -m uvicorn mockapi.main:app --port 8000`
- Run MCP server: `.venv/bin/python -m mcp_server`
- Ask with tools: `.venv/bin/python -m ticketpilot "why is T-1002 still open?"`

## Out of scope
Memory, embeddings and chunking, classifier, approvals and harness, eval suite, auth, web UI.
