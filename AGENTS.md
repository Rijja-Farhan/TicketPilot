# AGENTS.md

Project conventions for coding agents. Keep this file current; conventions live here, not in chat history.

## Project
TicketPilot: an operations agent. See `docs/domain.md` for the business context and `README.md` for layout.

## Rules
- Search the repo before creating a helper; do not duplicate existing utilities.
- One feature per branch; keep PRs under ~400 changed lines.
- Cite the relevant spec in `docs/` in commit messages.
- Never paste credentials or real customer data. Use only the provided/generated data.
- Fix train/test splits once, before inspecting data; never train on the test set.
- Record design decisions in `docs/decisions.md`; record agent mistakes in `docs/agent-log.md`.

## Commands
- Test: `LLM_PROVIDER=fake .venv/bin/python -m pytest -q`
- Run: `.venv/bin/python -m ticketpilot "question"`
- Generate tickets: `.venv/bin/python tools/gen/make_tickets.py`
- Live eval: `.venv/bin/python evals/run_a1.py`
- Mock API: `.venv/bin/python -m uvicorn mockapi.main:app --port 8000`
- MCP server (stdio, normally launched by the agent): `.venv/bin/python -m mcp_server`
- Generate customers and transactions: `.venv/bin/python tools/gen/make_customers.py`
- Live tool run: `PYTHONPATH=. .venv/bin/python evals/run_a3.py`
- Build the messy KB, then ingest and index it: `.venv/bin/python tools/gen/make_messy_kb.py && .venv/bin/python -m ticketpilot.ingest`
- Retrieval eval, stale example (live): `.venv/bin/python evals/run_a4_retrieval.py`, `.venv/bin/python evals/run_a4_stale.py`
- Classifier: `.venv/bin/python ml/train.py`, then `.venv/bin/python ml/compare_llm.py` (live)
- Memory: `.venv/bin/python -m ticketpilot --show-memory C-17`, `--forget C-17`
- Coworker: `.venv/bin/python -m ticketpilot "investigate T-1002"`, `--runs`, `--resume RUN_ID`; live tasks: `PYTHONPATH=. .venv/bin/python evals/run_a5.py`
- Evals: `.venv/bin/python evals/run_a6.py --offline --label local`; live `PYTHONPATH=. .venv/bin/python evals/run_a6.py --label NAME`; `--rerun`, `--compare` (see README)
- Classifier report: `.venv/bin/python ml/a6_classifier.py`; judge calibration: `.venv/bin/python evals/calibrate_judge.py run`; reward demo: `.venv/bin/python evals/reward.py`
- Ambient (A7): `.venv/bin/python -m ticketpilot.ambient replay data/stream/eval.jsonl --offline`; `approvals list`; `verify-audit`; eval `PYTHONPATH=. .venv/bin/python evals/run_a7.py --offline`; red team `PYTHONPATH=. .venv/bin/python evals/redteam_a7.py`; cost `python -m ticketpilot.ambient.cost A7_OFFLINE.json A7_LIVE.json`.
- The A7 streams in `data/stream/` are fixed (checksums); never regenerate them or tune the detector's k on `eval.jsonl`. The ambient agent never sends or executes: replies wait for `approvals`, and unknown actions are denied in `ticketpilot/ambient/policy.py`.
- Eval gold is written before a run; held-out case outputs are not read until the final report; the dev/held-out split (`evals/cases/SPLIT.sha256`) is fixed. Never tune on the classifier test set.
- Plain `openai` SDK only; no LangChain-style frameworks.
- The train/val/test split in `data/splits/` is fixed; never regenerate it, never tune on the test set.
- Tools return data or the error contract `{error: {code, message, retryable}}`; never raise from a tool. Writes are gated in `ticketpilot/agent.py` (assistant) and denied in `ticketpilot/coworker.py` (coworker), not only in prompts.

## Do not
- Commit `.venv/`, `.env`, or generated raw data.
- Commit code you cannot explain.
