# Spec: A4, data plumbing, memory and a classifier baseline

Source: `docs/plan-a4.md`. Builds on `docs/spec-a1.md` (policy answers) and `docs/spec-a3.md` (agent, mock API).

## Order of work and rules fixed before code
1. This spec, then the train/val/test split (`data/splits/`), then the 20 retrieval questions with gold answers. All three are written before any retriever or classifier output is looked at.
2. The split is written once by `tools/gen/make_splits.py` with a checksum file. Nothing may change it afterwards; tuning uses validation only.
3. Retrieval questions are written from the known content of the generated KB (`tools/gen/make_messy_kb.py`), never from what a retriever returns.

## Components
| Component | Path | Notes |
|---|---|---|
| Messy KB generator | `tools/gen/make_messy_kb.py` | Seeded, synthetic. Writes `data/kb_raw/` (committed; ~19 files incl. 3 PDFs) |
| Ingestion | `ticketpilot/ingest/` | load, clean, dedupe, metadata, chunk, index. One pure function per stage |
| Embeddings | `ticketpilot/embeddings.py` | One interface; `fastembed` (live), `hash` (offline tests) |
| Retrieval | `ticketpilot/retrieval.py` | BM25, dense, hybrid (reciprocal rank fusion) over chunk indexes |
| Policy wiring | `ticketpilot/policies.py` | Same `search`/`answer` interface; uses the index when one is built |
| Memory | `ticketpilot/memory.py` | Episodic and semantic, JSONL, expiry, `forget` |
| Labelled set and splits | `tools/gen/make_labelled.py`, `tools/gen/make_splits.py` | Banking77 + CFPB narratives |
| Classifier | `ticketpilot/classifier.py`, `ml/train.py` | TF-IDF + logistic regression |
| LLM comparison | `ml/compare_llm.py` | Same held-out sample, measured tokens and latency |

## Ingestion rules
- Load `.md`/`.txt` directly and PDFs via `pdftotext`. A PDF with no extractable text is **quarantined** with reason `needs_ocr`; unsupported types and empty files are quarantined too. Nothing is dropped silently: every drop is in `docs/a4-ingest-report.md` with a reason.
- Clean: NFKC, whitespace, page numbers, headers/footers repeated on every PDF page.
- Dedupe: exact by content hash; near-duplicate by 3-word shingle Jaccard at a threshold chosen on labelled pairs (`evals/a4_dedupe_pairs.jsonl`). Two documents with different **explicit** versions are never duplicates (versions of one policy are naturally similar). The keeper is the copy with explicit metadata, then the newest, then the shortest file name.
- Metadata: `title, version, effective_date, status, supersedes, applies_to`. Missing values are inferred from file name or text and flagged.
- **Status is derived, not trusted.** Within a policy (`applies_to` or title), the latest effective non-draft document is `current`; earlier ones are `superseded` and get a `supersedes` link; drafts stay `draft`. A stale front-matter claim of `current` is overridden.
- Chunking A: fixed 500 characters, 100 overlap, no structure. Chunking B: split on headings, with `[title | version N | effective DATE | status]` prepended to every chunk.

## Retrieval rules
- Default search returns `current` chunks only. History questions (A1 `wants_history`) add `superseded`. `draft` is never returned.
- Metrics on 20 questions (`evals/a4_retrieval.jsonl`): recall@3, MRR, stale-hit rate. Reported as counts as well as percentages.
- `policies.search`/`answer` keep their A1 signatures. Without a built index they fall back to A1 BM25 over `data/kb/`, so A1 behaviour and tests are unchanged.

## Memory
| Kind | Stored | Written | Expires |
|---|---|---|---|
| Episodic | customer ID, ticket IDs, short outcome | by code at the end of an agent run that made a tool call | 90 days or `forget` |
| Semantic | explicit user-stated fact with `valid_from` | only from explicit user statements, never from ticket text | replaced by a newer conflicting fact; old kept as superseded |
Store IDs and short summaries only; no ticket bodies, names, or card/account numbers. At most 3 most recent unexpired entries per customer, shown with age, inside a `<memory>` data block.

## Classifier
- Labels: the 8 categories in `docs/domain.md`. `loans_credit` comes from CFPB narratives (clipped to the first two sentences) because Banking77 has none. The text-style confound is reported.
- Test set = Banking77 `test.csv` (mapped) plus a held-out CFPB slice. `tickets.jsonl` and demo tickets come from `train.csv` and are training-side only. Validation is carved from train with a fixed seed and is used for tuning and the deferral threshold.
- Model: TF-IDF (word + char n-grams) + logistic regression, `class_weight="balanced"`. Baselines: majority class and a keyword rule set.
- Metrics: per-class precision/recall/F1, macro-F1, confusion matrix, class counts. Threshold chosen on validation from error cost, not 0.5.
- LLM comparison: stratified 200 test tickets (IDs saved), measured tokens and latency, cost from a cited price, consistency over two runs on 50.

## Acceptance criteria
1. The ingestion pipeline handles duplicates, outdated versions and an image-only PDF, and reports what it dropped and why.
2. Two chunking strategies are compared on 20 questions with recall@3, MRR and stale-hit rate.
3. Superseded content is never returned by default; history questions can still reach it.
4. Memory stores IDs and short summaries only, expires entries, supports `forget`, and enters the prompt as data.
5. One real stale-data failure is documented with before and after evidence (or an honest statement plus a stale-memory case).
6. The train and test sets are fixed once, with a passing leak test, and the test set is never used for tuning.
7. The classifier beats both baselines, with per-class precision and recall, a confusion matrix and a justified threshold.
8. The comparison table covers cost per 1,000 tickets and latency from measured numbers, with a recommendation.
9. `LLM_PROVIDER` in `.env` is still the only change needed to switch provider; no secrets or raw data committed; the offline suite passes (113 existing tests plus new ones).

## Commands
- Build the messy KB: `python tools/gen/make_messy_kb.py`
- Ingest and index: `python -m ticketpilot.ingest` (writes `data/index/<strategy>/` and `docs/a4-ingest-report.md`)
- Retrieval eval: `python evals/run_a4_retrieval.py`
- Labelled set, splits, train, compare: `tools/gen/make_labelled.py`, `tools/gen/make_splits.py`, `ml/train.py`, `ml/compare_llm.py`
- Tests: `LLM_PROVIDER=fake .venv/bin/python -m pytest -q`

## Out of scope
Classifier as an MCP tool and approvals (A5), full eval suite (A6), anomaly detection and drift (A7), procedural memory, web UI, auth.
