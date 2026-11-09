# Spec: A1, TicketPilot v1 assistant

Source: `docs/plan-a1.md` (§1 goal, §5 schema, §7 acceptance, §8 out of scope). Domain: `docs/domain.md`.

## Goal
A support agent asks a question on the CLI (`python -m ticketpilot "question"`) and gets either
1. a **policy answer** grounded in the knowledge base, citing title, version and effective date, or
2. a **ticket lookup** as structured JSON.

## Behaviour
| Request | Path |
|---|---|
| Contains a ticket ID (`T-<digits>`) and nothing more than a lookup ("show T-1042") | Plain code, no LLM |
| "tickets for customer C-17" | Plain code filter, no LLM |
| Free-text question about a ticket ("why is T-1042 still open?") | LLM with the looked-up ticket as context. Ticket text is data, never instructions |
| Policy question | BM25 over `current` KB docs, LLM answers from retrieved text and cites |
| Anything else | One-line refusal that says what TicketPilot can answer |

Router: regex first, LLM intent call only if no regex matches.

## Ticket lookup schema
Source of truth: `ticketpilot/schemas.py::TicketLookup`. Fields: `ticket_id, found, customer_id, category, status, priority, subject, created_at, summary, error`.
Not found: `found: false`, every data field `null`, `error: "ticket_not_found"`. Never an invented ticket.

## Acceptance criteria
1. `T-<id>` lookup returns schema-valid JSON in under 50 ms with no LLM call.
2. Unknown ticket returns `found: false`, no hallucinated fields.
3. Policy answers cite doc title, version and effective date; only `current` docs unless the question asks about history.
4. Out-of-scope questions are refused with a one-line explanation.
5. Changing `LLM_PROVIDER` in `.env` is the only change needed to switch provider (verified on `fake` and one real provider).
6. No secrets in the repo; `.env` is gitignored.
7. Tests pass with `LLM_PROVIDER=fake`.

## LLM layer
Only `ticketpilot/llm.py` imports the SDK. `complete(messages, *, schema=None) -> LLMResult`. JSON mode, always Pydantic-validated, one retry with the error message, then a typed error. Backoff on HTTP 429. Tokens and latency logged per call. Provider table and model IDs live in `ticketpilot/config.py`.

## Commands
- Test: `LLM_PROVIDER=fake .venv/bin/python -m pytest -q`
- Run: `.venv/bin/python -m ticketpilot "show ticket T-1001"`
- Generate tickets: `.venv/bin/python tools/gen/make_tickets.py`

## Out of scope
Classification, embeddings and chunking, memory, MCP and tools, multi-step autonomy, writes, web UI, FastAPI.
