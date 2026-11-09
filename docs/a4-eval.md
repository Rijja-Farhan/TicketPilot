# A4 evaluation: data plumbing, memory, classifier

Spec: `docs/spec-a4.md`. Full generated tables: [ingest report](a4-ingest-report.md), [retrieval grid](a4-retrieval-grid.md), [stale example](a4-stale-example.md), [classifier metrics](a4-classifier-metrics.md), [LLM comparison](a4-llm-comparison.md). Everything below is reproducible from the commands in the spec; seeded runs were repeated and gave identical numbers (retrieval grid, classifier metrics; the LLM comparison is cached).

## Deviations from the plan (read first)
1. **No CFPB narratives.** The plan builds `loans_credit` from CFPB complaint text. The CFPB search API and the CSV export no longer carry the narrative (0 of 9,680 fetched records have text). Substitute: the LLM wrote 444 short customer-style `loans_credit` messages from 30 real CFPB topics (product, issue, sub-issue), plus a 120-message `fees_charges` control set (`tools/gen/make_synth.py`, output committed because LLM output is not seed-reproducible). Consequence: **`loans_credit` results measure recognition of LLM-written messages, not real customer language.**
2. **Embeddings:** `fastembed` with `BAAI/bge-small-en-v1.5` (open question 1 default). **OCR:** none; the image-only PDF is quarantined (open question 2 default). **LLM sample:** 200 (default).
3. The retrieval-strategy selection rule (highest recall@3 in the default condition, ties broken by unfiltered recall, then MRR) was written after seeing a first grid. That first grid was invalid (chunker bug, see agent log), so the rule was applied to the corrected grid; the rule itself was not changed after that.

## 1. Ingestion (criterion 1)
19 files in, 13 documents kept, 5 duplicates dropped with reasons (2 exact, 3 near), 1 file quarantined (`scanned-chargeback-memo.pdf`, `needs_ocr`; its content is not searchable until OCR is added). Two stale "current" claims were overridden by the version chain (dispute v2, disputes FAQ). A near-duplicate with one changed number ("100 days") was dropped in favour of the copy with declared metadata. Near-duplicate threshold 0.68, chosen from 14 labelled pairs (lowest true duplicate 0.813, highest non-duplicate 0.537; pairs of different declared versions never count as duplicates).

## 2. Retrieval and chunking (criterion 2)
20 questions (18 answerable; 10 direct, 5 stale traps, 3 cross-chunk, 2 with no answer), written before any retriever ran. Counts out of 18; one question is 5.6 points, so differences of one or two are within noise.

Default condition (current chunks only; what ships):

| chunking | retriever | recall@3 | MRR |
|---|---|---|---|
| fixed 500/100 | BM25 | 16/18 | 0.91 |
| fixed | dense | 17/18 | 0.92 |
| fixed | hybrid | 16/18 | 0.92 |
| structured + header | BM25 | 16/18 | 0.96 |
| structured + header | dense | 17/18 | 0.88 |
| **structured + header** | **hybrid** | **18/18** | 0.91 |

Unfiltered condition (all statuses searched, to see stale hits): recall@3 14 to 16 of 18, and the top-1 chunk was a superseded or draft chunk in **5 to 10 of 18** questions depending on the retriever; with the status filter it was 0 of 18 for every combination. Structure-aware chunking with the header helped most without the filter (structured dense: 5/18 stale top-1 vs 8/18 for fixed dense). Shipped: structured + hybrid (`policies.RETRIEVAL_STRATEGY/MODE`), with the status filter. The 18/18 vs 17/18 difference is a single question and should not be over-read; the filter, not the retriever, is what removes stale hits.

Not solved: question 4 ("Who is paged first for a P1 incident?") is missed by all three fixed-chunk retrievers in the default condition, and the two no-answer questions always return a top-1 chunk (retrieval has no abstention score). Abstention comes from the answer prompt plus the citation check, and from a lexical gate for unrelated questions: "savings account maintenance fee" abstains, "capital of France" never reaches the LLM.

## 3. Memory design (criterion 4)
Implemented as specified (`ticketpilot/memory.py`; decisions in `docs/decisions.md`): episodic summaries written by code from enumerated fields only (`status`, `category`, `priority`, tool names, IDs), 90-day expiry, newest three per customer shown with age inside a `<memory>` data block, `forget` purges the file, semantic facts only from an explicit "remember that C-17 ... is ..." statement and a newer fact with the same key replaces the old one. Tests cover expiry at day 90, forget, conflict, no ticket body or name in the file (including the `T-9001` injection ticket), redaction of long digit runs, and the data wrapper. Limits: names cannot be detected in free text, so safety rests on never copying free text in; CLI `--forget` and `--show-memory` exist.

## 4. The stale-data failure (criterion 5)
Real, not invented. Naive setup (fixed chunks, dense, no status filter, no version header) vs the shipped path, on the 5 stale-trap questions with the live model (`openai/gpt-oss-120b`):

| | wrong (only superseded value) | mixed (hedges across versions) | ok |
|---|---|---|---|
| before | 1 (#15: "60 days", from the stale FAQ) | 2 (#11, #13: lists 60, 90 and 120 days) | 2 |
| after | 0 | 0 | 5 |

The fix bundled five changes (derived status across versions, status filter, version header in every chunk, dedupe, structure-aware chunks). I did not ablate them, so I cannot say which one mattered most; the unfiltered grid suggests the filter does most of the work.

## 5. Classifier (criteria 6 and 7)
Split fixed once before any training (`data/splits/`, checksummed; the split script refuses to overwrite). Test = Banking77 `test.csv` plus held-out synthetic topics (whole topics, so paraphrases never cross the line): train 8,822, val 1,569, test 3,196. Leak tests check ID and text disjointness on the real split.

| class | test support | classifier P | classifier R | keyword rules P | keyword rules R |
|---|---|---|---|---|---|
| card_issues | 999 | 0.97 | 0.98 | 0.82 | 0.81 |
| transfers_payments | 878 | 0.97 | 0.95 | 0.87 | 0.69 |
| account_access | 280 | 0.98 | 1.00 | 0.68 | 0.66 |
| fees_charges | 429 | 0.97 | 0.95 | 0.87 | 0.85 |
| fraud_security | 200 | 0.94 | 0.95 | 0.71 | 0.27 |
| loans_credit | 90 | 0.90 | 0.90 | 0.30 | 0.68 |
| disputes_refunds | 160 | 0.94 | 0.96 | 0.82 | 0.51 |
| other | 160 | 0.94 | 0.98 | 0.24 | 0.68 |
| **macro-F1** | | **0.954** | | 0.613 | |

Majority-class baseline: macro-F1 0.054. The classifier beats both on every class's precision. Confusion matrix and the threshold curve are in the [metrics file](a4-classifier-metrics.md). Deferral threshold **0.7**, chosen on validation (wrong = 1, missed fraud = 5, deferral = 0.3): 7.4% of test tickets deferred, accuracy on the rest 0.986 (all: 0.962).

**How far to trust 0.954.** Banking77 test messages come from the same templates as training, so this is an in-distribution score. The `loans_credit` row rests on 90 rows from 6 topics written by an LLM. The style confound is measurable: 4 of the 30 LLM-written `fees_charges` control messages (13%) were predicted `loans_credit`, against 5 of 3,076 Banking77 rows (0.16%), so `loans_credit` precision on real tickets is probably lower than 0.90. Serving: median 1.8 ms, 1.85 CPU-seconds per 1,000 tickets.

## 6. Classifier vs LLM (criterion 8)
200 stratified test tickets, same for every method; `openai/gpt-oss-120b`, temperature 0, 0 failed calls.

| | classifier | LLM | classifier + LLM fallback |
|---|---|---|---|
| macro-F1 | 0.939 | 0.779 | 0.923 |
| accuracy | 0.935 | 0.805 | 0.920 |
| cost per 1,000 tickets | 1.85 CPU-seconds | $0.117 | $0.012 |
| latency median / p95 | 1.8 / 2.5 ms | 1,755 / 4,003 ms | LLM only on deferred tickets |
| consistency (same label on rerun) | deterministic | 49 of 50 | |

Price: $0.15 per 1M input and $0.60 per 1M output tokens from [Groq's model page](https://console.groq.com/docs/model/openai/gpt-oss-120b), read 2026-11-01 (410 input and 93 output tokens per ticket measured). Per-class precision and recall for all three are in the [comparison file](a4-llm-comparison.md); LLM recall was weakest on `loans_credit` (0.46, 13 rows) and `other` (0.44, 16 rows).

**The assumption behind the threshold failed.** The threshold was chosen assuming the LLM resolves deferred tickets correctly. On the 20 deferred tickets the classifier was right 65% of the time and the LLM 50%, and the hybrid ended up slightly worse than the classifier alone (0.920 vs 0.935). With n = 20 this is noisy, but it gives no support for LLM fallback.

Caveats on fairness to the LLM: the labels follow our mapping of Banking77's 77 intents (for example "payment not recognised" is `fraud_security`), which a zero-shot prompt cannot know and the classifier learned. A few-shot prompt carrying those conventions would likely narrow the gap; I did not test it.

### Recommendation
- **Classify with the classifier.** It is more accurate on this data, about 1,000 times faster, deterministic and has no per-call fee. At this model's price the LLM is cheap in absolute terms ($117 per million tickets), so cost is not the deciding factor; accuracy and latency are.
- **Do not add the LLM fallback as built.** Send tickets below the 0.7 threshold (about 7 to 10%) to human review, which also matches `docs/domain.md` ("anything the agent is not confident about"). Reconsider LLM fallback only after a few-shot prompt beats the classifier on the deferred subset in a larger sample.
- **Prefer the LLM** when there is no labelled data for a category (a new category, a new product), at low volume where training is not worth it, or when a written explanation is needed. Retrain the classifier on real tickets before trusting `loans_credit`.

## Acceptance criteria
| # | Criterion | Status |
|---|---|---|
| 1 | Ingestion handles duplicates, outdated versions, image-only PDF, reports drops | Met (report and tests) |
| 2 | Two chunkings compared on 20 questions, recall@3, MRR, stale-hit | Met (6-cell grid, two conditions) |
| 3 | Superseded never returned by default; history reachable | Met (tests; live check of a history question) |
| 4 | Memory: IDs and summaries only, expiry, forget, data block | Met (tests); free-text names cannot be detected, only avoided |
| 5 | One real stale-data failure with before and after | Met (1 wrong, 2 mixed before; 5 of 5 correct after; fix not ablated) |
| 6 | Split fixed once, leak test, test never used for tuning | Met; 2 demo tickets share text with a test row (pinned in a test, not used for training) |
| 7 | Classifier beats both baselines, per-class P/R, confusion matrix, justified threshold | Met, with the `loans_credit` and in-distribution caveats above |
| 8 | Cost and latency from measured numbers, recommendation | Met; price from a cited page |
| 9 | Provider switch is one `.env` line; no secrets; offline suite passes | 199 tests pass offline; only Groq was run live in A4 |

## What I would change
- Replace the LLM-written `loans_credit` set with real narratives once a source exists, and evaluate on real tickets rather than Banking77 templates.
- Test a few-shot LLM prompt and a larger sample before ruling the LLM out as a fallback.
- Ablate the five stale-data fixes, and add a score floor so retrieval can abstain on its own.
- Add OCR so the quarantined chargeback memo becomes searchable.
