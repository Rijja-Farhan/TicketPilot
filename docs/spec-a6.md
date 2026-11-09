# Spec: A6, eval suite and improvement loop (TicketPilot v3, part 2)

Source: `docs/plan-a6.md`. Reads, never changes: the A5 trace format (`traces/runs/*.jsonl`), the fixed classifier split (`data/splits/`), `ticketpilot/verify.py`. One A5 change is allowed and additive: the `act` trace event gains a `retryable` field, so retry discipline can be graded.

## Case schema (`evals/cases/*.jsonl`, one file per component)
`id, split (dev|held_out), component, input, setup, expected, graders, severity (high|medium|low), source (new|a1|a4|a5), live (bool), adversarial (bool)`.
- `expected` is prose: what a correct run must contain or do. The machine-checked part is `graders`.
- `graders` is a list of `{"g": <name>, ...args}`. Each is one of the graders below, run on the run's `Outcome`.
- `setup` is data a case needs: `budget` overrides, `tool_failure` (tools forced to fail), `scenario` (memory cases).
- `live: true` needs the model; the rest run in `pytest` with `LLM_PROVIDER=fake`.
- Cases are written by `evals/build_cases.py`, which also assigns the split with a fixed seed and writes `evals/cases/SPLIT.sha256`. A1 and A4 cases are imported from their question files with `source` set. Gold answers are written there, before any run.

## The 45 cases
| Component | n | Notes |
|---|---|---|
| router | 6 | regex routes, bare ID lookup, unknown ID, investigate route, one LLM intent case |
| policy | 8 | direct, superseded trap, conflicting FAQ, history, cross-document, two the KB cannot answer |
| tools (A3 agent) | 6 | multi-hop, wrong ID type, upstream down, limit abuse, grounding |
| coworker (A5) | 10 | clean, history plus policy, fraud-flagged, hostile, missing ticket, two budget cases |
| safety | 6 | injected ticket, "ignore your rules", PII, refund request, injection in a policy question, bulk-data request |
| memory | 4 | reuse, expiry, forget, newer fact replaces older |
| classifier | 5 | labelled and deferred text, so the classifier shares the pass rate |

At least 8 cases are adversarial by design. About 30% are `held_out`, stratified by component. For held-out cases only the aggregate score is read until the final report.

## Graders
Code-based (`evals/graders.py`): `contains`, `absent` (NFKC-normalised), `route_is`, `refused`, `schema_valid`, `citation_valid`, `evidence_valid` (reuses `verify.check_report`), `no_write`, `no_action_claims`, `ids_grounded`, `category_match`, `status_is`, `stop_reason_is`, `flags_injection`, `limit_bounded`, `label_is`, `deferred_is`, `memory_clean`, and `trajectory` (below).
LLM judge (`evals/judge.py`): rubric of four criteria scored 0 to 2, reasoning before the score, `schema=`, temperature 0 (the provider layer already fixes it). It sees the task, evidence and report only. It is advisory: a code-checkable failure fails the case whatever the judge says.
Trajectory checks (`evals/trajectory.py`, over `RunTrace.read()`): `order`, `no_writes`, `no_repeats`, `budget`, `retry_discipline`, `evidence_before_claim`, `phase_order`; `efficiency` is recorded, not pass/fail. Coworker cases only: the A3 loop writes no trace. For A3 cases the runner records tool calls itself, so `no_write` and `limit_bounded` still work there.

## Runner
`python evals/run_a6.py --label NAME [--only ID,ID] [--component C] [--offline]`, `--rerun RESULTS_FILE --times 3`, `--compare BEFORE AFTER`. Results: `evals/results/<timestamp>-<label>.jsonl` (gitignored). Per case: outputs, grader verdicts, tokens, latency, cost estimate, run ID. Summary: counts and rates overall, by component, by grader type and by split.
**Cost estimate before the first live run** (placeholder price from `config.TOKEN_PRICE_PER_M`): coworker cases cost 15k to 30k tokens each, the other live cases 2k to 8k. About 300k to 450k tokens for the 30 live cases, so under $0.10 at the placeholder price, and about 20 to 30 minutes because of rate-limit waits. Flaky-case reruns add a fraction of that.
**Variance:** one full baseline run, then each failing case 3 more times; consistent failures are the top failures, mixed results are flaky.

## Classifier section (`ml/a6_classifier.py`)
Held-out report on the fixed test split, read for reporting only, with bootstrap intervals. The threshold is chosen on validation only, from the error-cost table in `docs/a6-classifier.md`, with the cost ratio varied (2, 5, 10, 20 to 1), and with the LLM's accuracy on deferred validation tickets measured, not assumed.

## Acceptance criteria
1. At least 30 distinct cases across router, policy, tools, coworker, safety, memory and classifier, with gold written before the first run.
2. Code-based and LLM-judge graders, each with tests of known good and known bad input.
3. At least 5 trajectory checks over real A5 traces, each tested on a hand-built good and bad trace.
4. The judge is calibrated against 15 hand labels; agreement, kappa and the length-bias check are reported.
5. Per-class precision, recall and confusion matrix on the fixed held-out set, with intervals on small classes.
6. The threshold is justified from a written cost table, sensitivity to the cost ratio, measured LLM accuracy on deferred tickets, validation only.
7. Suite run before and after; top failures fixed as separate logged changes; gold edits only as logged corrections; dev and held-out compared.
8. Flaky versus consistent failures are separated by repeated runs.
9. The reward note names a verifiable reward and one exploit, demonstrated on the real grader.
10. `LLM_PROVIDER` is still the only switch; the offline suite passes; no secrets or result files committed; the split checksums match before and after.

## Out of scope
CI dashboards, drift monitoring, ambient triggers (A7); a human-review queue implementation (recommended only if the data says so).
