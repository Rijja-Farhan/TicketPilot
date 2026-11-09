# Plan: A4, Data plumbing, memory and a classifier baseline (TicketPilot v2, part 2)

Source: syllabus A4 (Week 2, 9% of the evaluation phase). Rubric focus: D3 architecture, D4 verification, D8 ML and data reasoning, D1 problem framing.

## Context
Syllabus task: ingest a messy knowledge base, build retrieval and compare two chunking strategies, add long-term memory of customer history, then train a ticket-category classifier and compare it with LLM classification on held-out tickets.

Deliverables (each maps to a section below):
| Deliverable | Section |
|---|---|
| Ingestion pipeline | 1 |
| Retrieval quality comparison on 20 test questions | 2 |
| Memory design: what is stored, when, how it expires | 3 |
| One example where stale data caused a wrong answer, and the fix | 4 |
| Classifier vs LLM table: per-class precision and recall, cost per 1,000 tickets, latency, plus a recommendation | 5, 6 |

Gaps in the repo today:
- `data/kb/` has 5 clean hand-written docs. The syllabus KB is messy (duplicates, outdated policies, scanned PDFs), and we have no starter kit, so we build the messy corpus ourselves.
- A1 retrieval is BM25 over whole docs (`ticketpilot/policies.py`). A4 replaces it, but A1 behaviour and tests must keep passing.
- No `scikit-learn`, no embedding model and no OCR installed. Installs are a decision point (see Open questions).
- `tickets.jsonl` (300 rows) was sampled from Banking77 `train.csv` and has no `loans_credit` rows and only 8 `disputes_refunds` rows. Too small and skewed to train or test on. A4 builds a bigger labelled set from `data/raw/`.
- No memory exists. The agent forgets everything between questions.

## 0. Spec and splits first (before touching data)
- `docs/spec-a4.md` with acceptance criteria (section 9), written before code.
- **Fix the train/test split once, before inspecting data** (AGENTS.md rule). A script writes `data/splits/{train,val,test}_ids.json` plus a checksum file; nothing may change them afterwards. Details in section 5.
- Write the 20 retrieval test questions and their gold answers **before** looking at what any retriever returns.

## 1. Ingestion pipeline (`ticketpilot/ingest/`)
Messy corpus: extend `data/kb/` to about 20 files via a seeded generator `tools/gen/make_messy_kb.py`, all synthetic:
- exact duplicates and near-duplicates (reformatted copy, one changed number),
- three versions of one policy (v1 and v2 superseded, v3 current) plus a draft that was never effective,
- a conflicting FAQ page that quotes an old window,
- 2 text PDFs, and 1 image-only PDF ("scanned").

Pipeline stages, each a small pure function so each is testable:
1. **Load**: `.md` and `.txt` directly, PDFs through `pdftotext` (already installed). A PDF with no extractable text is **quarantined** with reason `needs_ocr` and listed in the report, not silently dropped. OCR itself is optional (Open questions).
2. **Clean**: normalise whitespace and unicode, strip headers, footers and page numbers.
3. **Dedupe**: exact by content hash, near-duplicate by shingle Jaccard (threshold chosen on a few labelled pairs, not by feel). Keep the newest or most authoritative copy and log what was dropped and why.
4. **Metadata**: `title, version, effective_date, status (current | superseded | draft), supersedes`. Missing metadata is inferred from file name or text and flagged `inferred`.
5. **Chunk**: strategy A or B (section 2).
6. **Index**: write `data/index/<strategy>/` (chunks, vectors, BM25 corpus). Gitignored, rebuilt by one command.

Output: `docs/a4-ingest-report.md` with counts: files in, duplicates removed, quarantined, chunks per strategy.

## 2. Retrieval and the chunking comparison
- **Chunking A, fixed-size:** about 500 characters, 100 overlap, no awareness of structure.
- **Chunking B, structure-aware:** split on headings and sections, and prepend the doc header `[title | version | effective_date | status]` to every chunk so version context survives the split.
- **Retrievers:** BM25 (reuse `rank-bm25`), dense embeddings, and hybrid (reciprocal rank fusion of both). Embeddings sit behind one `embeddings.py` interface: a real local model for live runs, a hashing-based fake for offline tests, matching how `llm.py` and the fake provider work.
- **Eval:** 20 questions in `evals/a4_retrieval.jsonl`, each with gold doc and section. Mix: 10 direct lookups, 5 stale traps (an old version answers differently), 3 questions answered only across two chunks, 2 with no answer in the KB.
- **Metrics:** recall@3, MRR, and **stale-hit rate** (top result is superseded while a current version exists). Report a 2 x 3 grid (chunking x retriever) in `docs/a4-eval.md`. Keep the comparison honest: 20 questions is small, so report counts, not just percentages, and say which differences are within noise.
- Wire the winner into `policies.py` behind the existing `search` and `answer` interface so A1 and A3 callers do not change. Update the mock API `/policies` endpoint to use it.

## 3. Long-term memory of customer history (`ticketpilot/memory.py`)
Design (also written up in `docs/decisions.md`):
| Kind | Stored | Written when | Expires |
|---|---|---|---|
| Episodic | one short summary per session: customer ID, ticket IDs touched, what was found, outcome | at the end of an agent run that made at least one tool call, written **by our code, not by the model** | 90 days, or sooner on `forget` |
| Semantic | stable facts the agent learned (for example "prefers email", verified by a human note) with `valid_from` | only from explicit user statements, never from ticket text | replaced when a newer fact conflicts; old one kept as superseded |
| Procedural | out of scope in A4 (skills already cover it) | n/a | n/a |

Rules:
- Store IDs and short summaries only. No raw ticket bodies, no names, no card or account numbers.
- Keyed by `customer_id`. Retrieval returns at most 3 most recent unexpired entries, each shown with its age.
- Memory enters the prompt as a `<memory>` data block with dates and "as of" wording, like tool results: data, never instructions. Memory text from past sessions is untrusted, since it may contain echoed ticket content.
- Storage is an append-only JSONL file (`data/memory.jsonl`, gitignored) with a `forget(customer_id)` function and a CLI flag. No vector store needed at this scale.
- Memory is a source of stale facts, so the memory tests in section 7 check expiry.

## 4. The stale-data example (graded deliverable)
Do not invent one. Procedure:
1. Run the 5 stale-trap questions through naive fixed-size chunks with embeddings and **no** metadata filter.
2. Record any case where the model answers with a superseded filing window or fee, with the exact retrieved chunk.
3. Fix with the index-time status filter, version header in every chunk, and the `supersedes` link, then re-run and log before and after in `docs/agent-log.md` and `docs/decisions.md`.
4. If naive retrieval does not fail, say so and use a stale-memory case (an episodic entry from before a policy change) instead.

## 5. Classifier (`ml/` and `ticketpilot/classifier.py`)
**Data.** `make_tickets.py` already maps Banking77 intents to 7 of the 8 categories (decision recorded in `docs/decisions.md`). Banking77 has no loan or credit intents, so `loans_credit` comes from CFPB complaints via a product-to-category mapping in `tools/gen/make_labelled.py`.
- CFPB narratives are long and Banking77 queries are short. A model can learn the text style instead of the topic. Mitigation: clip CFPB narratives to the first one or two sentences, cap the class size, and report the confound in `docs/a4-eval.md`.

**Split (fixed once, in section 0).** Test set = Banking77 `test.csv` (mapped) plus a held-out CFPB slice. `tickets.jsonl` and the A1 and A3 demo tickets are drawn from `train.csv`, so they are training-side only and never appear in test. Validation is carved from train with a fixed seed and used for tuning and the threshold. A test asserting zero ID and text overlap between train and test guards against leakage.

**Model.** TF-IDF (word and character n-grams) plus logistic regression with `class_weight="balanced"`. Baselines to beat: majority class and a keyword rule set. Add `scikit-learn` to `requirements.txt`; the trained model is saved with `joblib` to `ml/model.joblib` (gitignored) with the training script and metrics committed.

**Metrics.** Per-class precision, recall and F1, macro-F1, confusion matrix, and the class counts. No accuracy-only reporting (rubric D8).

**Threshold.** The probability cutoff below which the classifier defers to the LLM is chosen on validation from the cost of errors (a missed `fraud_security` is worse than a misrouted `other`), not at 0.5 by default. Report the threshold, the share of tickets deferred, and the resulting accuracy.

## 6. Classifier vs LLM comparison
- **Same held-out tickets** for both. LLM calls cost money and hit rate limits, so use a stratified sample of 200 from test (fixed seed, IDs saved), and report the classifier on both the full test set and the same 200 so the numbers are comparable.
- LLM classification uses `llm.complete(..., schema=...)` with the 8 labels, temperature 0, through the existing provider abstraction (Groq in `.env`).
- **Table in `docs/a4-eval.md`:** per-class precision and recall for each method, macro-F1, **cost per 1,000 tickets** (LLM from measured token counts times the provider's current price, which must be looked up and cited, not recalled; classifier from compute time), **latency** (median and p95 per ticket), and **consistency** (LLM run twice on 50 tickets, share of identical labels).
- **Recommendation** written from the numbers: likely classifier-first with LLM fallback below the threshold, but the data decides. State when each is preferable (volume, latency budget, new categories with no labelled data).

## 7. Tests (offline, `LLM_PROVIDER=fake`, fake embeddings)
- `tests/test_ingest.py`: exact and near-duplicate removal, image-only PDF is quarantined with a reason, metadata inference is flagged.
- `tests/test_chunking.py`: no chunk exceeds the size cap, structure-aware chunks carry the doc header, no text is lost.
- `tests/test_retrieval.py`: superseded chunks are excluded by default and included for history questions; hybrid ranking is deterministic.
- `tests/test_memory.py`: expiry at 90 days, `forget`, conflicting semantic fact supersedes the old one, no ticket body or name is stored, memory text is wrapped as data.
- `tests/test_classifier.py`: train and test IDs and texts do not overlap (leak guard), training is reproducible with the fixed seed, prediction returns label plus probability, below-threshold deferral works.
- All existing 113 tests must still pass.

## 8. Order of work (one branch each, under about 400 lines each)
| Step | Output |
|---|---|
| 0 | `docs/spec-a4.md`, split script and saved split files, 20 retrieval questions with gold answers |
| 1 | Messy KB generator, ingestion pipeline and report, tests |
| 2 | Chunking A and B, embeddings interface, retrievers, 20-question eval, comparison grid |
| 3 | Wire the winner into `policies.py` and the mock API; stale-data example, fix and before/after |
| 4 | Memory module, agent integration, CLI forget flag, tests |
| 5 | Labelled set builder, train and evaluate the classifier, threshold on validation |
| 6 | LLM comparison on the 200 sample, comparison table and recommendation |
| 7 | README, AGENTS.md commands, `docs/decisions.md`, `docs/agent-log.md` |

## 9. Acceptance criteria (copy into the spec)
1. The ingestion pipeline handles duplicates, outdated versions and an image-only PDF, and reports what it dropped and why.
2. Two chunking strategies are compared on 20 questions with recall@3, MRR and stale-hit rate.
3. Superseded content is never returned by default; history questions can still reach it.
4. Memory stores IDs and short summaries only, expires entries, supports `forget`, and enters the prompt as data.
5. One real stale-data failure is documented with before and after evidence (or an honest statement plus a stale-memory case).
6. The train and test sets are fixed once, with a passing leak test, and the test set is never used for tuning.
7. The classifier beats both baselines, with per-class precision and recall, a confusion matrix and a justified threshold.
8. The comparison table covers cost per 1,000 tickets and latency from measured numbers, with a recommendation.
9. `LLM_PROVIDER` in `.env` is still the only change needed to switch provider; no secrets or raw data committed; the offline suite passes.

## 10. Verification
- `LLM_PROVIDER=fake .venv/bin/python -m pytest -q` for old and new tests.
- Rebuild the index from scratch with one command and confirm the report counts match the generator.
- Run the retrieval eval and the classifier eval twice and confirm identical numbers (fixed seeds).
- Run the A1 and A3 live scripts again to confirm no regression from the new retriever.
- Grep tracked files for key-shaped strings before any commit.

## 11. Risks
- **Test leakage:** tickets already come from Banking77 `train.csv`. Mitigation: split fixed first, test is `test.csv`, leak test in CI.
- **Class imbalance and the CFPB style gap** make `loans_credit` numbers unreliable. Mitigation: clip text, report per-class support, and say when a class has too few test rows to trust.
- **Small eval sets** (20 questions, 200 tickets) give noisy comparisons. Report counts and avoid claiming small differences.
- **Embeddings dependency weight.** A local model is a large download and slows tests. Mitigation: hide it behind an interface, use the fake in tests, and keep the dependency optional for the offline suite.
- **LLM cost and rate limits** on the 200-ticket run. Mitigation: sample size fixed up front, cache responses to disk so reruns are free.
- **Scope creep:** serving the classifier as an agent tool, approvals and drift detection belong to A5 to A7.

## Open questions (defaults chosen, change if you disagree)
1. **Embeddings:** default is a small local model (for example via `fastembed`) because Groq has no embeddings endpoint. Alternative: TF-IDF vectors only, which needs no download but is weaker for the "dense" comparison.
2. **OCR for the scanned PDF:** default is quarantine with a report and no OCR install. Alternative: install `tesseract` and OCR it, which makes the pipeline more realistic.
3. **Sample size for the LLM comparison:** default 200 stratified tickets to limit cost.

## Out of scope for A4
Classifier as an MCP tool and approvals (A5), the full eval suite and trajectory checks (A6), anomaly detection and drift (A7), procedural memory, any web UI, and auth.
