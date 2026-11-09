# TicketPilot

An operations agent for Northwind Pay, a fictional digital bank (see `docs/domain.md`): assistant (v1) -> tools/MCP/retrieval/memory + classifier (v2) -> coworker (v3) -> ambient (v4).

## Layout

| Path | Purpose |
|---|---|
| `docs/` | Domain bible, plans and specs (`spec-a1.md`, `spec-a3.md`), decisions log, agent log, eval notes |
| `data/` | `tickets.jsonl`, `customers.jsonl`, `transactions.jsonl` (generated), `kb/` policy docs; raw data is gitignored |
| `tools/gen/` | Seeded scripts that build or ingest the data |
| `mockapi/` | Mock REST API (tickets, customers, transactions, policies, notes) |
| `mcp_server/` | MCP server wrapping the mock API: 7 tools (incl. `classify_ticket`), 1 resource |
| `skills/` | Agent skills (`triage-ticket/SKILL.md`) |
| `ticketpilot/` | The agent (see below) |
| `ml/` | Classifier training, metrics and LLM comparison (anomaly detector comes later) |
| `evals/` | Eval datasets and graders |
| `tests/` | Unit and integration tests |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set LLM_PROVIDER and the matching key
```

## Run (A1)

```bash
python -m ticketpilot "show ticket T-1001"              # JSON, no LLM
python -m ticketpilot "tickets for customer C-17"       # JSON, no LLM
python -m ticketpilot "how long do I have to file a dispute?"   # LLM + cited policy
```

Switch provider by changing `LLM_PROVIDER` in `.env` (`groq`, `openai`, `gemini`, `ollama`, `fake`). Only `ticketpilot/llm.py` and `ticketpilot/config.py` know about providers.

## Run (A3: tools over MCP)

```bash
python -m uvicorn mockapi.main:app --port 8000        # terminal 1: the mock API
python -m ticketpilot "why is T-1002 still open?"        # terminal 2: agent starts the MCP server itself (stdio)
python -m ticketpilot "does C-17 have a history of problems?"
python -m ticketpilot "triage T-1001"                    # loads skills/triage-ticket
python evals/run_a3.py                                # live run; prints every tool call
```

Free-text questions about a ticket or customer go through the tool-calling agent; bare lookups and policy questions keep the A1 paths. Set `TICKETPILOT_API_URL` if the API is not on `127.0.0.1:8000`. `MOCKAPI_FAIL_RATE` and `MOCKAPI_LATENCY_MS` simulate a flaky upstream. Inspect the server with `npx @modelcontextprotocol/inspector .venv/bin/python -m mcp_server`.

## Run (A5: coworker)

```bash
python -m uvicorn mockapi.main:app --port 8000                    # terminal 1
python -m ticketpilot "investigate T-1002 and draft a resolution"    # plan, act, observe, reflect, verified report
python -m ticketpilot --runs                                         # recent runs (checkpoints in runs/)
python -m ticketpilot --resume RUN_ID                                # continue an interrupted run
PYTHONPATH=. python evals/run_a5.py                               # the three live tasks
```

The coworker is read-only: writes are denied in code and listed as proposed actions. Traces: `traces/runs/<run_id>.jsonl`. Design: `docs/harness-a5.md`; real runs: `docs/a5-traces.md`; single vs multi-agent: `docs/a5-multi-agent.md`.

## Evals (A6)
```bash
LLM_PROVIDER=fake python -m pytest -q                                   # includes the offline eval cases, grader and trajectory tests
python evals/run_a6.py --offline --label local                          # the cases that need no model
PYTHONPATH=. python evals/run_a6.py --label baseline                    # all 45 cases, live (mock API on :8000)
python evals/run_a6.py --rerun evals/results/FILE.jsonl --times 3       # flaky vs consistent failures
python evals/run_a6.py --compare BEFORE.jsonl AFTER.jsonl               # before/after, dev vs held-out
python ml/a6_classifier.py                                              # held-out report, cost table, threshold (live LLM, cached)
python evals/reward.py                                                  # reward-hacking demo (no model)
```
45 cases in `evals/cases/` (spec: `docs/spec-a6.md`), 14 held out. Results go to `evals/results/` (gitignored). Write-up: `docs/a6-eval.md`, `docs/a6-classifier.md`, `docs/a6-judge-calibration.md`, `docs/a6-reward-design.md`.


## Ambient agent (A7)
```bash
python -m ticketpilot.ambient replay data/stream/eval.jsonl --offline      # whole pipeline, scripted model, no network
python -m ticketpilot.ambient replay data/stream/eval.jsonl --offline --obedient   # the scripted model obeys every injection
python -m ticketpilot.ambient approvals list|approve ID|edit ID --text "..."|reject ID --reason "..."
python -m ticketpilot.ambient verify-audit
PYTHONPATH=. python evals/run_a7.py --offline        # graded against data/stream/eval_gold.json; add --live --inprocess --limit 45 for a real model
PYTHONPATH=. python evals/redteam_a7.py [--live]     # red team
python -m ticketpilot.ambient serve                     # webhook on :8100 plus a worker (needs the mock API on :8000 and an LLM key)
```
Events enter by webhook, replay or schedule, go through a SQLite queue (dedupe, rate cap, back-pressure), a guard, the classifier tool (the LLM only for unsure tickets), a policy layer and a human approval queue. Nothing customer-facing is sent without approval; every step is in a hash-chained audit log (`runs/ambient/`). Design: `docs/spec-a7.md`, `docs/a7-policy.md`; results: `docs/a7-eval.md`, `docs/a7-cost.md`, `docs/a7-redteam.md`.

## Test and eval

```bash
LLM_PROVIDER=fake python -m pytest -q       # offline, free
LLM_PROVIDER=groq python evals/run_a1.py    # live eval, record in docs/a1-eval.md
python tools/gen/make_tickets.py            # regenerate data/tickets.jsonl (needs data/raw/banking77)
python tools/gen/make_customers.py          # regenerate customers and transactions
```

## A1 modules
`router.py` (regex, then LLM intent), `tickets.py` (plain-code lookups), `policies.py` (BM25 + cited answers), `schemas.py` (Pydantic source of truth), `llm.py` (`complete()`, retry, backoff, trace), `fake_llm.py` (offline provider), `config.py`, `cli.py`.

## A3 modules
`mockapi/main.py` (FastAPI), `mcp_server/server.py` (tools, resource, error contract), `ticketpilot/mcp_client.py` (`ToolBox`: OpenAI-format tools, schema check, error mapping), `ticketpilot/agent.py` (bounded loop, write gate, repeat guard).

## What I would change
- Verify model IDs and run the live eval before trusting any answer quality claim.
- Policy retrieval loads whole docs with keyword search; A4 should chunk and embed, and handle staleness properly.
- The history-question check is a keyword list; an LLM or better parse would be sturdier.
- The ticket-vs-question split relies on a lookup-word list and will misroute unusual phrasings to the LLM path.
- Traces go to `traces/llm_calls.jsonl` as plain JSONL; A5/A6 will want structure and IDs.
- A3: the agent answers more than the tool results show (docs/a3-eval.md #1); a "state only what the tools returned" check belongs in the A6 evals.
- A3: the mock API must be started by hand; a single dev command would help.

## A4: data plumbing, memory, classifier
```bash
python tools/gen/make_messy_kb.py && python -m ticketpilot.ingest   # build data/kb_raw, then the indexes and docs/a4-ingest-report.md
python evals/run_a4_retrieval.py                                   # chunking x retriever grid (docs/a4-retrieval-grid.md)
python ml/train.py                                                 # classifier + baselines (docs/a4-classifier-metrics.md)
python -m ticketpilot --show-memory C-17                              # what is remembered; --forget C-17 deletes it
```
Policy answers use the built index when `data/index/` exists, else the A1 BM25 over `data/kb/`. Results, deviations and the recommendation are in `docs/a4-eval.md`. Embeddings need a one-time model download (`EMBEDDINGS_PROVIDER=hash` avoids it for tests).
