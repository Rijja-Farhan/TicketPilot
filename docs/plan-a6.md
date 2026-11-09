# Plan: A6, Eval suite and improvement loop (TicketPilot v3, part 2)

Source: syllabus A6 (Week 3, 10% of the evaluation phase). Rubric focus: D4 verification, D7 communication, D8 ML and data reasoning, D1 problem framing.

## Context
Syllabus task: build an eval suite of **at least 30 cases** with **code-based and LLM-as-judge graders** and **at least five trajectory checks**. Include the A4 classifier (precision and recall on a held-out set, threshold justified from the cost of false positives and false negatives). Run the suite, fix the top failures, run it again. Write a short note on a verifiable RL reward for this agent and one way it could be gamed.

Deliverables (each maps to a section below):
| Deliverable | Section |
|---|---|
| Eval set (30+ cases) | 1 |
| Graders: code-based, LLM-as-judge, 5+ trajectory checks | 2, 3, 4 |
| Classifier precision and recall on held-out data, threshold note | 5 |
| Before/after results and the changes made | 6 |
| Reward-design note with one gaming example | 7 |

What we build on:
- A1 has 15 questions (`evals/a1_questions.jsonl`), A4 has 20 retrieval questions and a 200-ticket LLM comparison, A5 writes one JSONL trace per run (`traces/runs/`) with phases and tool calls, and `ticketpilot/verify.py` already has deterministic claim checks. A6 reuses them; it does not restart from scratch.
- The classifier test split is fixed (`data/splits/`, checksums). A6 reads it, never regenerates it.
- A5 must be in a stable state first (its files are uncommitted today). A6 reads the A5 trace format, so changes to it after A6 starts break the trajectory checks.

Gaps today:
- Evals are per-assignment scripts with substring checks. There is no shared case format, no results history, no LLM-judge, no trajectory grading, and nothing to diff before against after.
- The graders themselves are untested. A grader that passes everything is the most common eval bug, so each grader gets tests with known good and known bad outputs.
- A4 left a finding to resolve here: on the 20 tickets the classifier deferred, the LLM was right on 10 and the classifier on 13 (0.5 vs 0.65), but the threshold choice assumed the LLM would be right. Section 5 deals with it.

## 0. Spec and ground rules first
- `docs/spec-a6.md` with the case schema, grader list and acceptance criteria (section 9), written before running anything.
- **Expected answers are written before any run.** After the baseline run, a gold answer may change only if the gold itself was wrong, and each change is logged with the reason in `docs/a6-eval.md`.
- **Dev and held-out cases.** About 70% of cases are `dev` (failures here get fixed), about 30% are `held_out` (only the aggregate score is looked at, never the individual outputs, until the final report). This shows whether fixes generalise or just fit the cases we read. The split is fixed in the case file, by a seeded script, before the first run.
- Never tune on the classifier test set. Threshold changes use validation only (section 5).

## 1. The eval set (`evals/cases/*.jsonl`, 40 cases)
One schema for every case: `id, split (dev|held_out), component, input, setup (data or flags needed), expected (what a correct run must contain or do), graders (list), severity (high|medium|low), source (new | a1 | a4 | a5)`.

| Component | Cases | Examples |
|---|---|---|
| Router and lookup | 6 | bare ticket ID goes to plain code; "investigate T-1002" goes to the coworker; unknown ID returns a clean not-found |
| Policy Q&A (A1/A4) | 8 | direct lookup; superseded-policy trap; question the KB cannot answer (must refuse); conflicting FAQ page |
| Tool agent (A3) | 6 | multi-hop customer question; wrong-ID-type misuse; upstream down; limit abuse |
| Coworker investigation (A5) | 10 | clean ticket; ticket needing customer history plus current policy; fraud-flagged ticket (must be urgent); missing ticket (`T-9001`); budget too small |
| Safety and injection | 6 | injected instruction in ticket body; request for a note nobody asked for; "ignore your rules" from the user; PII in a question; request that needs a refund the agent cannot do |
| Memory (A4) | 4 | prior-run summary reused; expired entry ignored; `forget` honoured; stale memory vs newer data |

Reuse rules: A1 and A4 cases are imported with `source` set, not copied by hand, and a case is counted once. Count real cases, not the same case under two graders.

At least 8 cases are adversarial or edge cases by design, since a suite of easy cases measures nothing. Cases that need the live model are tagged `live`; the rest run offline (section 3).

## 2. Graders (`evals/graders.py`)
**Code-based graders (cheap, exact, run first):**
| Grader | Checks |
|---|---|
| `contains` / `absent` | required and forbidden strings, after the NFKC normalisation A1 already needed |
| `schema_valid` | Report or answer validates against its Pydantic schema |
| `citation_valid` | every cited policy exists, is current unless history was asked for, and has the right version and date |
| `evidence_valid` | reuses `verify.py`: evidence IDs exist and quoted IDs and amounts appear in the ledger |
| `no_write` | zero `add_ticket_note` calls in the trace |
| `category_match` | report category equals the gold label |
| `refused` | out-of-scope or unanswerable input gets a refusal, not an answer |

**LLM-as-judge (`evals/judge.py`), for what code cannot score:** is the drafted resolution grounded in the evidence, and is it actionable for a support agent. Design rules:
- A written rubric with 3 or 4 criteria scored 0 to 2, structured output via `schema=`, temperature 0, reasoning field before the score.
- The judge sees the task, the evidence ledger and the report, not the system prompt or the expected answer.
- **Calibrate it against humans.** Hand-label 15 outputs first (done by us before the judge runs), then report agreement and Cohen's kappa. If agreement is poor, change the rubric, not the numbers.
- Known biases to test or state: prefers long answers, prefers its own model's style. Check length against score and say what we found. If the key lists a second model, use a different one for judging than for the agent (open question 2).
- The judge is advisory. Any case with a code-checkable failure fails on the code grader regardless of the judge score.

## 3. Trajectory checks (`evals/trajectory.py`, 7 checks over A5 traces)
Functions over `RunTrace(run_id).read()`, so each is deterministic and testable on hand-built traces:
1. **Order:** `get_ticket` is the first tool call for an investigation.
2. **No writes:** no write tool is ever called.
3. **No repeats:** no identical (tool, args) pair twice.
4. **Budget:** tool calls, rounds and tokens stay inside the declared budget, and a budget stop has the matching `stop_reason`.
5. **Retry discipline:** retries happen only after `retryable: true`, at most twice.
6. **Evidence before claim:** the policy search precedes any policy citation in the report.
7. **Phase order:** plan appears before the first act, and verify before report.

Optional 8th: tool-call efficiency (calls per correct finding), recorded but not pass/fail.

Each check is tested on one good and one bad hand-built trace. A5 traces cover the coworker; the A3 loop writes no trace, so trajectory cases target coworker runs only, and this is stated in the results.

## 4. Runner, results and reproducibility (`evals/run_a6.py`)
- Two modes. **Offline** (`LLM_PROVIDER=fake`, scripted): router, graders, harness, trajectory checks; part of `pytest`, so grader bugs break CI. **Live** (Groq): the full 40 cases, run by script.
- Output: `evals/results/<timestamp>-<label>.jsonl` (per case: outputs, grader results, tokens, latency, cost, trace ID) plus a summary table. Gitignored, except the two files copied into `docs/` for before and after.
- Metrics: pass rate overall, by component, by grader type, by split, plus cost and latency per case. Report counts (for example 31 of 40) as well as percentages.
- **Variance:** a live model is not deterministic. Run the full suite once for the baseline, then re-run every failing case 3 times and report which are flaky versus consistent failures. Only consistent failures count as "top failures".
- Rate limits: sequential runs with the same backoff `llm.py` uses; the cost of one full run is estimated and written down before the first live run.

## 5. Classifier on held-out data and the threshold note
- **Held-out set:** the fixed test split (3,196 tickets), read for reporting only. Report per-class precision, recall, F1 and support, macro-F1, and the confusion matrix, reusing `ml/metrics.py`. Add a confidence interval (bootstrap) on the weakest classes, since `loans_credit` has only 90 test rows.
- **Threshold justification from error costs.** Define the costs in `docs/domain.md` terms and write them as a table:
  - False negative on `fraud_security`: a fraud ticket is routed as routine, so the customer waits (highest cost).
  - False positive on `fraud_security`: a routine ticket gets escalated, so the fraud team spends time (moderate cost).
  - Other confusions: misrouted ticket, small cost.
  - Deferral: one LLM call plus latency (small cost, but the LLM is not free of errors).
- **Per-class view:** the question is the cutoff for the fraud class, not only the global one. Compute expected cost over the validation set for thresholds, once with the A4 cost ratio (5 to 1) and with 2, 10 and 20 to 1, and show how the chosen threshold moves. A recommendation that holds across the plausible range is the honest result.
- **Fix the A4 gap:** measure LLM accuracy on the **validation** tickets the classifier defers (not the test sample) and replace the "LLM is always right" assumption with the measured number. If the LLM is no better than the classifier on low-confidence tickets, say so, and consider routing those to a human review queue instead of the LLM. Pick the threshold on validation only.
- Because the test set was already read in A4 for the LLM comparison, any threshold changed now is validated on validation data, and the note states that test numbers are no longer fully untouched.
- Add 4 to 6 classifier cases into the main suite (fraud ticket labelled fraud, ambiguous text deferred, loan text), so the classifier appears in the same pass rate and the same before/after.

## 6. Improvement loop (`docs/a6-eval.md`)
1. **Baseline:** run all 40 cases live; record per-case results.
2. **Triage failures:** group failing dev cases by root cause (retrieval, prompt, harness rule, grader bug, gold error). Rank by severity times frequency. Check for grader bugs first; a wrong grader is not an agent failure.
3. **Fix the top 2 or 3 causes**, each as a separate change with its own decision-log entry: what failed, the evidence (case IDs, trace lines), the change, and the expected effect. Likely areas, from what A1 to A5 already showed: weak refusals when the KB has no answer, citation of a superseded policy, over-long investigations hitting the budget. Use whatever the baseline actually shows; do not pre-pick.
4. **Re-run** the same 40 cases with unchanged gold. Report before and after per case, the dev score, and the held-out score. A large dev gain with no held-out gain means the fix overfit; say so.
5. **Check for regressions:** list cases that passed before and fail now, and the offline `pytest` suite.
6. Write the changes into `docs/decisions.md` and mistakes into `docs/agent-log.md`.

## 7. Reward-design note (`docs/a6-reward-design.md`, about one page)
- **A verifiable reward for the coworker**, from checks we already have, no judge needed for the main term: `R = 1` only if (a) the report passes the code verification, (b) the category matches the gold label, (c) zero writes, (d) the run stayed inside its budget; `0` otherwise. Optional shaping terms: a small per-finding bonus for findings supported by the ledger, a small penalty per tool call over a target. Say which terms are verifiable and which are not.
- **One concrete way it could be gamed**, shown with an example: an agent that returns `status: partial` with no findings and a correct guessed category (guessed from ticket keywords) passes (a) to (d) and collects the reward without investigating. Mitigations: a required coverage term (the plan's must-have evidence items must appear in the ledger) and a ground-truth check against the data, not the text.
- Demonstrate it for real, cheaply: build the degenerate report by hand, score it with the naive reward function, show it passes, then show the improved reward rejects it. This is evidence, not an essay.
- Connect to what we have: the verifier is the reward function; an unverified reward is exactly the failure Week 3 describes. State what the held-out set is for (catching reward overfitting).

## 8. Tests and order of work
**Tests (offline, `LLM_PROVIDER=fake`):**
- `tests/test_graders.py`: each grader passes a known-good output and fails a known-bad one, including NFKC edge cases and the injected-instruction leak check.
- `tests/test_trajectory.py`: each of the 7 checks has one good and one bad hand-built trace.
- `tests/test_judge.py`: rubric schema validates; a scripted fake judge returns scores; the judge never sees the expected answer.
- `tests/test_eval_cases.py`: case file schema, unique IDs, at least 30 cases, at least 5 trajectory graders, the dev/held-out split is fixed and unchanged (checksum), every referenced ticket, customer and policy exists.
- All existing tests must still pass.

| Step | Output |
|---|---|
| 0 | `docs/spec-a6.md`, case schema, fixed dev/held-out split, cost estimate of one live run |
| 1 | Cases written with gold answers (40, A1 and A4 imported), tests for the case file |
| 2 | Code-based graders and trajectory checks, with their tests |
| 3 | LLM judge, 15 hand-labelled calibration outputs, agreement report |
| 4 | Runner, results format, offline mode wired into `pytest` |
| 5 | Classifier section: held-out report, cost table, threshold sensitivity, measured LLM accuracy on deferred validation tickets |
| 6 | Baseline live run, failure triage, flaky-versus-consistent reruns |
| 7 | Fix the top causes, re-run, before/after, regression check |
| 8 | Reward note with the demonstrated exploit, README and AGENTS.md commands, decisions and agent-log |

## 9. Acceptance criteria (copy into the spec)
1. At least 30 distinct cases covering router, policy, tool agent, coworker, safety, memory and classifier, with gold answers written before the first run.
2. Code-based and LLM-as-judge graders exist, and every grader has tests with known good and bad inputs.
3. At least 5 trajectory checks run over real A5 traces, each tested on a hand-built good and bad trace.
4. The judge is calibrated against 15 hand labels, with agreement reported and known biases checked.
5. The classifier's per-class precision, recall and confusion matrix on the fixed held-out set are reported, with intervals on small classes.
6. The threshold is justified from a written error-cost table, with sensitivity to the cost ratio and a measured (not assumed) LLM accuracy on deferred tickets, chosen on validation only.
7. The suite ran before and after, with the top failures fixed in separate logged changes, no gold edits other than logged corrections, and a dev versus held-out comparison.
8. Flaky and consistent failures are distinguished by repeated runs.
9. The reward note names a verifiable reward and one concrete exploit, demonstrated on the real grader.
10. `LLM_PROVIDER` is still the only switch; the offline suite passes; no secrets or result files committed; the test split is untouched (checksums match).

## 10. Verification
- `LLM_PROVIDER=fake .venv/bin/python -m pytest -q`, including the grader and trajectory tests.
- `sha256sum -c data/splits/CHECKSUMS.txt` before and after, to show the split did not move.
- Run the offline eval twice and confirm identical results; run the live suite and read at least 10 raw outputs by hand against the grader verdicts to catch grader bugs.
- Feed the graders a deliberately broken agent (always answers "I don't know", or always passes) and confirm the suite fails it.
- Grep tracked files for key-shaped strings before any commit.

## 11. Risks
- **Grader bugs read as agent failures, or hide them.** Mitigation: grader tests, the broken-agent check, and reading raw outputs.
- **Overfitting to the cases we read.** Mitigation: held-out cases, gold fixed first, log every change.
- **LLM-judge unreliability and self-preference.** Mitigation: calibration, code graders take precedence, a different judge model if available.
- **Small samples.** 40 cases and a handful of held-out cases give noisy rates. Report counts, avoid claims about small differences, and say what a 1-case change means.
- **Flaky live runs and rate limits.** Mitigation: reruns for failures only, sequential runs, a cost estimate first.
- **Test-set reuse.** A4 already read the test set; keep threshold decisions on validation and disclose it.
- **Scope creep:** CI dashboards, drift monitoring and ambient triggers belong to A7 and later. Keep the trace format stable so they can reuse these graders.

## Open questions (defaults chosen, change if you disagree)
1. **Case count:** 40, so the held-out slice is large enough to mean something. 30 is the minimum.
2. **Judge model:** a different Groq model from the agent if the key lists one, otherwise the same model with the self-preference caveat stated.
3. **Classifier in the main suite or separate:** both. A section in the report with full metrics, plus 4 to 6 cases in the pass rate.
4. **Human review queue for low-confidence tickets:** considered only if the measured LLM accuracy on deferred tickets is no better than the classifier's. Implementing it is out of scope; the note just recommends it.
