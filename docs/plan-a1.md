# Plan: A1, Spec-driven assistant build (TicketPilot v1)

Rubric focus: D2 spec and context, D3 architecture, D4 verification, D7 communication. Due end of Week 1 with A2.

## 1. Goal
TicketPilot v1 is an **assistant**: a support agent asks a question and gets an answer. It must
1. answer questions about **policies** (cited from the knowledge base), and
2. answer **ticket lookups** as **structured JSON**.

## 2. Key design decision: LLM only where language is needed
| Request | Handled by | Why |
|---|---|---|
| "Show ticket T-1042" / "tickets for customer C-17" | **Plain code** (lookup by ID/filter) | Exact, deterministic, free. An LLM here only adds error. |
| Free-text question about a ticket ("why is T-1042 still open?") | LLM given the looked-up ticket as context | Needs language |
| Policy question ("how long do I have to file a dispute?") | LLM + retrieved policy text, must cite the source doc and version | Needs language and grounding |
| Anything else | Refuse politely and say what it can answer | Out of scope |

A small router decides the path: regex for ticket IDs first, an LLM intent call only if that does not match. Record this reasoning in `docs/decisions.md`, since it shows D1/D3.

## 3. Provider abstraction (Groq now, one-line switch later)
Groq, OpenAI, Together, OpenRouter, Ollama, Gemini and Anthropic all expose OpenAI-compatible chat endpoints. So use the single `openai` Python SDK and put everything provider-specific in one table. No LiteLLM-style framework is needed.

```
ticketpilot/llm.py      # the only module that imports the SDK
ticketpilot/config.py   # reads .env
.env                 # LLM_PROVIDER=groq   <- the one line you change
.env.example         # committed; no secrets
```

`config.py` holds a provider table:
```python
PROVIDERS = {
    "groq":   {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",   "model": "<groq model id>"},
    "openai": {"base_url": "https://api.openai.com/v1",      "key_env": "OPENAI_API_KEY", "model": "<model id>"},
    "ollama": {"base_url": "http://localhost:11434/v1",      "key_env": None,             "model": "<local model>"},
}
```
`llm.py` exposes one function the rest of the code uses:
```python
complete(messages, *, schema=None) -> LLMResult   # text or validated Pydantic object, plus tokens, latency
```
Rules:
- Nothing outside `llm.py` mentions Groq or any provider.
- Pick the Groq model ID from Groq's current model list when you set it up (verify it supports JSON mode); keep it in config, never in code.
- Structured output: request JSON mode, then **always validate with Pydantic**. On failure, retry once with the error message, then return a typed error. Never trust provider-side schema enforcement alone, because support differs by provider.
- Log tokens and latency per call into a trace. A5 and A6 reuse this.
- A fake provider (`LLM_PROVIDER=fake`) returns canned responses, so tests run offline and free.
- Groq free tier is rate limited: add backoff on HTTP 429.

## 4. Data slice A1 needs (small)
- `data/tickets.jsonl`: about 300 tickets from Banking77, seeded generator in `tools/gen/make_tickets.py`. Fields: `ticket_id, customer_id, category (our 8 labels), subject, body, status, priority, created_at, assignee`. IDs and statuses are synthetic. Fixed seed.
- `data/kb/*.md`: 5 short policy docs written by hand, each with front matter (`title, version, effective_date, status: current|superseded`). Include **dispute policy v1 (superseded) and v2 (current)** and a fees schedule v1/v2. A1 retrieves only `current` docs by default and cites version and date. The stale-data failure comes in A4.
- Retrieval in A1 is deliberately simple: keyword/BM25 over a handful of docs, or load them whole. Embeddings and chunking are A4.

## 5. Ticket lookup JSON schema (spec this precisely)
```json
{
  "ticket_id": "T-1042",
  "found": true,
  "customer_id": "C-17",
  "category": "card_issues",
  "status": "open",
  "priority": "p2",
  "subject": "...",
  "created_at": "2026-10-03T09:12:00Z",
  "summary": "one sentence, LLM-written, optional",
  "error": null
}
```
Not found returns `found: false` plus an error code, never an invented ticket. Pydantic model is the source of truth; the spec links to it.

## 6. Architecture
```
CLI (python -m ticketpilot "question")
  -> router (regex, then LLM intent)
       -> ticket_store (plain code) -> JSON
       -> policy_search (BM25) + llm.complete -> answer + citations
  -> llm.py -> provider (from .env)
```
Modules: `router.py`, `tickets.py`, `policies.py`, `schemas.py`, `llm.py`, `config.py`, `cli.py`. A FastAPI wrapper is out of scope for A1; a CLI is enough.

## 7. Acceptance criteria (copy into the spec)
1. `T-<id>` lookup returns schema-valid JSON in under 50 ms with no LLM call.
2. Unknown ticket returns `found: false`, no hallucinated fields.
3. Policy answers cite the doc title, version and effective date, and use only `current` docs unless asked about history.
4. A question outside scope is refused with a one-line explanation.
5. Changing `LLM_PROVIDER` in `.env` is the only change needed to switch provider. Verified by running the suite against `fake` and one real provider.
6. No secrets in the repo; `.env` is gitignored.
7. Tests pass with `LLM_PROVIDER=fake`.

## 8. Out of scope for A1
Classification, embeddings and chunking, memory, MCP and tools, autonomous multi-step work, writes to any system, a web UI.

## 9. Verification (D4)
- Unit tests: store lookups, schema validation, router decisions, config switching, retry on bad JSON.
- A small live eval of about 15 questions (policy facts, ticket lookups, out-of-scope, an injection attempt) with expected answers, run manually against Groq and recorded in `docs/a1-eval.md`.
- Read every diff the agent produces, and log each mistake in `docs/agent-log.md` with how you caught it. Expect things like invented fields, a hardcoded model name, duplicated helpers, and a swallowed exception.

## 10. Process (D2)
1. Write `docs/spec-a1.md` from sections 1, 5, 7, 8 above and `AGENTS.md` commands. Spec first, no code.
2. Open a **fresh coding-agent session in plan mode**, give it the spec, and review the plan before approving.
3. One module per session or branch, small diffs, tests with each. Restart with a short summary if the session drifts.
4. Commit messages cite the spec section.
5. Final pass: README with run instructions, agent log, and a short "what I would change" note, which is the D7 5/5 behaviour.

## 11. Order of work
| Step | Output |
|---|---|
| 1 | `.env.example`, `config.py`, `llm.py` + fake provider + tests |
| 2 | `make_tickets.py`, `tickets.py`, schema, lookup tests |
| 3 | KB docs, `policies.py`, BM25 search, citation format |
| 4 | `router.py`, `cli.py` |
| 5 | Live eval on Groq, then a provider-switch check |
| 6 | README, agent log, decisions log |

## 12. Risks
- Groq rate limits: backoff, and use the fake provider for tests.
- Model returns invalid JSON: validate and retry once.
- Agent reaches for a framework (LangChain etc.): `AGENTS.md` says plain SDK only.
- Prompt injection inside ticket text: treat ticket body as data, never as instructions. Add one eval case.
- Secrets: key only in `.env`; never paste it into a prompt.
