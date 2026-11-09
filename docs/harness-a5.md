# Harness design (A5)

Source of truth: `docs/spec-a5.md`. Code: `ticketpilot/coworker.py`, `budget.py`, `runstate.py`, `trace.py`, `verify.py`.

The model proposes; the code decides. Every row below is enforced in code, none of it in the prompt alone.

| Concern | Design | Where | Test |
|---|---|---|---|
| Permissions | Per-task allowlist (`TASK_TOOLS`). The model is shown only allowed tools; a call outside the list (writes, unknown names) is recorded as `not_permitted` and never reaches the tool server. Plans that name a write tool are rejected once, then the offending steps are dropped. Writes are never executed: they appear as `proposed_actions`. | `coworker.execute`, `plan_problems` | `test_coworker.py` |
| Budgets | 12 tool calls, 4 reflect rounds, 60k tokens, 240 s, 3 calls per tool. Checked before every model and tool call. A run-wide limit raises `StopRun(<reason>)`; the per-tool cap only denies that tool. Cost estimate and usage are in every report. | `Budget` | `test_budget.py` |
| Retries | Tools: only `retryable: true`, 2 retries, exponential backoff with jitter, every attempt counts against the run budget; `not_found` and `invalid_argument` never retried; an identical call is skipped; a tool that ends in a retryable error twice is `tool_unavailable` for the run. Model calls: `rate_limited` retried twice at 20 s. | `execute`, `_ask` | `test_retries.py`, `test_coworker.py` |
| Verification | Code checks decide (evidence IDs exist; IDs, dates and amounts quoted are in the cited evidence; policy citations were retrieved and are current unless history was asked; no first-person claim of an action; classifier label matches; status consistent with stop reason). Model check is advisory. One repair round from the same ledger; a second failure strips the failing items into `open_questions` and gives `partial` / `verification_failed`. | `verify.py`, `report_phase` | `test_verify.py` |
| Stopping | First of: `enough`, `blocked`, `tool_budget`, `round_budget`, `token_budget`, `time_budget`, `no_progress` (two reflect rounds, no new evidence), `model_error`, `verification_failed`. `complete` requires `enough` and no failures. Anything else is `partial` (or `failed` with no usable evidence). | `reflect_phase`, `run` | `test_budget.py` |
| Durable state | One JSON file per run, written atomically after every tool call and phase. Resume reloads it; budgets and elapsed time carry over; a step whose tool calls finished before the crash is not repeated (identical calls are skipped). Corrupt files give a clear error before any server starts. | `runstate.py` | `test_runstate.py` |
| Untrusted input | Evidence is shown inside `<tool_result>`, memory inside `<memory>`. The tool list, allowlist, plan checks and report fields (`status`, `verification`, `budget_used`) are written by code, so injected text cannot change them. | `coworker.py` | injection tests |
| Observability | `traces/runs/<run_id>.jsonl`: phase, kind, tool, args, error, tokens, latency, budget left, one line per event. | `trace.py` | `test_phases_are_visible_in_the_trace` |

## Known limits
- Code checks cover IDs, dates, amounts and percentages; they do not check that a prose claim is true, only that its numbers appear in the cited evidence. The model check is another model and is not independent proof.
- The ACT phase does one model turn per plan step and does not feed a step's results back within the same step; the next step or REFLECT sees them. This is why a run takes 5 to 6 tool-calling turns.
- Resume after the data or policies change would mix old and new evidence; the ledger timestamps are kept and shown in the report's `evidence` list, but nothing refreshes stale evidence.
- Only Groq was exercised live. Latency and rate limits are provider-dependent.
