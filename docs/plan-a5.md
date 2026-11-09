# Plan: A5, Coworker agent with harness (TicketPilot v3)

Source: syllabus A5 (Week 3, 9% of the evaluation phase). Rubric focus: D3 architecture, D5 safety, D2 spec, D4 verification.

## Context
Syllabus task: turn TicketPilot into a coworker that takes a delegated task ("investigate ticket T-1002 and draft a resolution with evidence"), runs an explicit **plan, act, observe, reflect** loop, and returns a report for human review.

Deliverables (each maps to a section below):
| Deliverable | Section |
|---|---|
| The agent | 1, 2 |
| Harness design: permissions, budgets, retries, verification step, stopping rule, durable state | 3 |
| Three traces of real runs, one of them a failure | 5 |
| One-page justification of multi-agent vs single agent | 6 |

What we build on:
- `ticketpilot/agent.py` (A3) is a bounded Q&A loop with a repeat guard and a write gate. It stays as the assistant path. The coworker is a new module so A1, A3 and A4 behaviour and tests do not change.
- Tools (6 MCP tools), the triage skill, memory (A4), hybrid retrieval and the classifier (A4) already exist and are reused, not rebuilt.
- A3's lesson holds: rules that matter are enforced in code, not in the prompt.

Gaps today:
- No explicit plan, no evidence record, no self-check, no budgets beyond `MAX_STEPS`, no saved run state, and no structured report.
- `llm.complete` takes `tools=` and `schema=`; check in step 0 whether both can be used in one call on Groq (the earlier enum 400 shows schemas are fragile). If not, plan and report use `schema=` calls and the act phase uses `tools=` calls.

## 0. Spec first
- `docs/spec-a5.md` with the task contract, report schema, harness table and acceptance criteria (section 9), written before code.
- Write 3 delegated tasks and their expected evidence **before** running anything: one clean case, one that needs customer history and a policy lookup, one hostile case (ticket body contains an instruction aimed at the agent). The hostile one is a test, not a staged failure for the trace deliverable.

## 1. The loop (`ticketpilot/coworker.py`)
One run is a small state machine. Each phase is a function, so each is testable on its own.

| Phase | Who decides | What happens |
|---|---|---|
| PLAN | model, `schema=Plan` | Lists 3 to 6 steps, the evidence each needs, and a stop condition. Code rejects a plan that proposes a write tool. |
| ACT | model proposes, code runs | Tool-calling steps run through the harness (section 3). One plan step at a time. |
| OBSERVE | code | Every tool result gets an ID (`E1`, `E2`, ...) in an **evidence ledger**: tool, arguments, result, timestamp. Errors are recorded too. |
| REFLECT | model, `schema=Reflection` | Given the plan and the ledger: `enough`, `need_more` (with the missing item), or `blocked` (with the reason). Can revise the plan once. |
| VERIFY | code first, then model | See section 4. Gate before the report leaves the harness. |
| REPORT | model, `schema=Report` | Draft built only from the ledger, then verified. |

Loop: PLAN once, then ACT, OBSERVE, REFLECT until REFLECT says `enough` or a stop rule fires, then VERIFY, then REPORT.

**Report schema (`schemas.py`):** `ticket_id`, `status` (`complete | partial | failed`), `summary`, `category` (with classifier label and probability when used), `findings[]` (each `claim` plus `evidence_ids`), `policy_citations[]` (title, version, effective_date), `proposed_resolution` (draft text for the human), `proposed_actions[]` (what a human could approve, never executed), `open_questions[]`, `verification` (passed checks, failed checks), `budget_used`.

The report is a draft. It never says an action was taken.

## 2. Wiring
- `ticketpilot/router.py`: new route `investigate` for "investigate", "look into", "draft a resolution", "root cause" plus a ticket ID. Regex first, as for the other routes. Bare ticket lookups and A3 questions are unchanged.
- `ticketpilot/cli.py`: `python -m ticketpilot "investigate T-1002"` runs the coworker and prints the report. `--resume RUN_ID` continues a saved run. `--runs` lists recent runs.
- Add one tool through the MCP server if it earns its place: `classify_ticket(text)` wrapping `ticketpilot/classifier.py`, returning label, probability and whether it is below the deferral threshold. Read-only. If it is not added, the coworker calls the classifier in code and records it in the ledger like any tool.
- Reuse `Memory` as in A3: load a `<memory>` block for the customer at start, `record_run` at the end, written by code.
- Fake provider: scripted PLAN, tool calls, REFLECT and REPORT sequences so every test is offline.

## 3. Harness design (the graded design document, `docs/harness-a5.md`)
| Concern | Design | Enforced where |
|---|---|---|
| **Permissions** | Read tools allowed. `add_ticket_note` and any future write are **denied inside the coworker**, always. The report lists them under `proposed_actions`; a human runs them separately. Tool allowlist per run, from the task type. | `coworker.py`, not the prompt |
| **Budgets** | Per run: max 12 tool calls, max 4 reflect rounds, a token ceiling summed from `LLMResult` usage, a wall-clock limit (default 120 s), a cost estimate for the report. Per tool: max 3 calls. Numbers are defaults in `config.py`, changeable per run. | `Budget` class checked before every model and tool call |
| **Retries** | Only when the tool error has `retryable: true`. Max 2 retries, exponential backoff with jitter, counted against the budget. `not_found` and `invalid_argument` are never retried; a repeated identical call returns `duplicate_call` as in A3. A tool that fails twice is marked unavailable for the run. | `coworker.py` |
| **Verification** | Section 4. | `verify.py` |
| **Stopping rule** | Stop on the first of: REFLECT says `enough` and VERIFY passes; REFLECT says `blocked`; any budget exhausted; two REFLECT rounds that add no new evidence (no-progress); user abort. Every stop sets an explicit `stop_reason` in the report. Stopping short gives `status: partial` with what is missing, never a confident guess. | `coworker.py` |
| **Durable state** | After each step the run is checkpointed to `runs/<run_id>.json` (task, plan, ledger, phase, budget used, stop_reason). `--resume` reloads it and continues from the last complete step; budgets carry over. Completed tool calls are not repeated. Gitignored. | `runstate.py` |
| **Untrusted input** | Ticket bodies, customer names and memory stay `<tool_result>` or `<memory>` data. An injected instruction cannot add a tool, change the plan's allowlist, or write: those live in code. | `coworker.py` |
| **Observability** | One JSONL line per phase and tool call in `traces/runs/<run_id>.jsonl`: phase, tool, args, result summary, tokens, latency, budget left. Feeds the A6 trajectory checks. | `trace.py` |

Record each design choice and its alternative in `docs/decisions.md`: for example, deny writes outright versus an approval prompt (approval flows are Week 4), and checkpoint per step versus per phase.

## 4. Verification step (`ticketpilot/verify.py`)
Two layers, cheapest first.

**Code checks (deterministic, no model):**
1. Every `evidence_id` in the report exists in the ledger.
2. Every ticket, customer and transaction ID, amount and date quoted in a finding appears in the cited evidence. A claim with a number or ID the ledger does not contain fails.
3. Every policy citation matches a policy the ledger actually retrieved, with its version and date, and is `current` unless the task asked for history.
4. No claim of an action taken ("I refunded", "I added a note"); `proposed_actions` only.
5. Report validates against the schema; `status` matches the stop reason.

**Model check (a fresh call with only the ledger and the report, no plan or chat history):** for each finding, is it supported by the cited evidence? Returns `supported | unsupported | partly` per finding. Treated as advisory evidence, never as the only gate: code checks decide pass or fail.

On failure: one repair round (feed back the specific failed checks, regenerate the report from the same ledger). If it fails again, return `status: partial` with the unsupported findings removed and listed under `open_questions`. Do not loop.

## 5. The three traces (graded deliverable)
Real runs on Groq, saved from `traces/runs/` into `docs/a5-traces.md` with a short reading of each. Do not stage a failure.
1. **Clean:** a ticket that needs ticket, customer, history and a current policy. Show plan, evidence, verify pass, report.
2. **Multi-hop with memory or a stale-policy trap:** use a customer with an earlier run, or a question where a superseded policy exists, and show the harness choosing the current one.
3. **A real failure.** Candidates, in order of preference: whatever actually goes wrong in runs 1 and 2 or in the injection test; the mock API killed mid-run (`upstream_unavailable`, retries, then `partial`); a budget stop. Report what the trace shows, what the harness did, and what we changed afterward. If everything passes, kill the API mid-run and say that is how the failure was produced.

Each trace includes the stop reason, tokens, cost estimate, latency and the verification result. Log the fix in `docs/agent-log.md`.

## 6. Multi-agent justification (one page, `docs/a5-multi-agent.md`)
Default position, to be tested against the traces rather than assumed: **single agent with phases**, plus a fresh-context verifier call (a second model call, not a second agent with its own tools).
- For: one investigation is a sequential dependency chain (ticket, then customer, then policy), so parallel workers add coordination cost for little gain; shared state (the ledger) is simple in one process; cost and latency scale with each extra agent's context.
- Where multi-agent would help: independent evidence gathering at much larger scale, different permission scopes per agent, or a reviewer that must not see the planner's reasoning. The fresh-context verifier gets most of the third benefit.
- Use measured numbers from the traces: tokens, latency and cost of the verifier call, and an estimate of what a planner plus researcher plus reviewer split would add. State what would change the answer.
- Reference the Week 3 workflow ideas (chaining, routing, parallelism) and say which one this design uses.

## 7. Tests (offline, `LLM_PROVIDER=fake`)
- `tests/test_coworker.py`: full happy path with scripted model; write tool denied even if the model requests it; injection in a ticket body does not change the plan or call a write; plan with a write step is rejected.
- `tests/test_budget.py`: each budget (tool calls, tokens, time, rounds) stops the run with the right `stop_reason`; no-progress rule fires after two empty reflect rounds.
- `tests/test_retries.py`: retryable error retried with backoff (clock patched), then tool marked unavailable; `not_found` not retried.
- `tests/test_verify.py`: unknown evidence ID fails, invented amount fails, superseded policy citation fails, "I refunded" fails, repair round then `partial`.
- `tests/test_runstate.py`: checkpoint after each step, resume skips completed calls and carries budgets, corrupt state file gives a clear error.
- `tests/test_router.py`: `investigate` route added, existing routes unchanged.
- The existing suite must still pass.

## 8. Order of work (one branch each, under about 400 lines each)
| Step | Output |
|---|---|
| 0 | `docs/spec-a5.md`, 3 delegated tasks with expected evidence, check `tools=` plus `schema=` support |
| 1 | Schemas (`Plan`, `Reflection`, `Report`), `Budget`, trace writer, tests |
| 2 | `runstate.py` checkpoint and resume, tests |
| 3 | `coworker.py` loop with fake-provider tests (permissions, retries, stopping rule) |
| 4 | `verify.py` code checks, model check, repair round, tests |
| 5 | Router and CLI wiring, optional `classify_ticket` tool |
| 6 | Live runs on Groq, three traces, one failure, fix and re-run |
| 7 | `docs/harness-a5.md`, `docs/a5-multi-agent.md`, README, AGENTS.md commands, decisions and agent-log |

## 9. Acceptance criteria (copy into the spec)
1. A delegated task produces a structured report with a plan, evidence IDs, citations, a drafted resolution and open questions.
2. The loop has explicit PLAN, ACT, OBSERVE, REFLECT phases, visible in the trace.
3. The coworker never executes a write; proposed actions are listed for a human.
4. Budgets (tool calls, rounds, tokens, time) are enforced in code; each stop has a named `stop_reason`.
5. Retries follow the error contract's `retryable` flag, are bounded and back off.
6. Verification runs before the report is returned; unsupported claims never appear as findings.
7. A killed run resumes from its checkpoint without repeating completed tool calls.
8. An injected instruction in a ticket does not change tools, plan or writes.
9. Three real traces exist, one a failure, each with tokens, cost, latency and stop reason.
10. The multi-agent page uses measured numbers and states what would change the decision.
11. A1, A3 and A4 behaviour unchanged; `LLM_PROVIDER` still the only switch; offline suite passes; no secrets or run files committed.

## 10. Verification
- `LLM_PROVIDER=fake .venv/bin/python -m pytest -q`.
- Live: start the mock API, run `python -m ticketpilot "investigate T-1002"` and read the trace line by line.
- Kill the API mid-run and confirm `upstream_unavailable`, bounded retries and a `partial` report, no invented data.
- Interrupt a run, then `--resume`, and confirm no duplicate tool calls in the trace.
- Run the injection ticket and confirm zero write calls in the trace.
- Run A1, A3 and A4 live scripts to confirm no regression.
- Grep tracked files for key-shaped strings before any commit.

## 11. Risks
- **Over-engineering:** this is a concept-level harness. Keep each piece small and testable; do not build a framework.
- **Verifier theatre:** a model checking a model can pass bad claims. Code checks are the gate; the model check is advisory and measured in A6.
- **Cost and rate limits:** each run is several model calls plus a verifier call. Budgets are the control; cache nothing silently, report the cost.
- **Provider quirks:** Groq's tool-calling and schema limits. Keep the fake provider as the oracle and test live early (step 0).
- **Checkpoint drift:** resume after a policy or data change could mix old and new evidence. Store timestamps in the ledger and show evidence age in the report.
- **Scope creep:** approvals, ambient triggers and webhooks are Week 4 (A7); the eval suite and the reward note are A6. Plan the trace format so A6 can read it.

## Open questions (defaults chosen, change if you disagree)
1. **Writes:** coworker denies all writes and proposes them (default), versus a CLI approval prompt for notes. Default is safer and matches "report for human review".
2. **Classifier as an MCP tool or called in code:** default is a read-only MCP tool, so the trace shows it as a normal step.
3. **Model check in VERIFY:** default on, advisory only. Turn off if it doubles cost without catching anything in the traces.
4. **Run storage:** `runs/` and `traces/runs/` gitignored; the three graded traces are copied into `docs/a5-traces.md`.
