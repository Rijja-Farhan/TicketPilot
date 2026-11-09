# A7: cost per 1,000 tickets, with and without the classifier

Code: `ticketpilot/ambient/cost.py`. Run: `PYTHONPATH=. python -m ticketpilot.ambient.cost evals/results/a7-offline.json evals/results/a7-live-sample.json`.

## What is measured and what is extrapolated
| Quantity | Source | Status |
|---|---|---|
| Tokens per draft call: 413 prompt, 374 completion | 36 live calls, `openai/gpt-oss-120b` | measured (completion includes the model's reasoning tokens) |
| Tokens per LLM classification call: 413 prompt, 87 completion | 3 live calls | measured, **n = 3, treat as rough** |
| Share of tickets that get a draft: 95% (the rest are quarantined, unknown customers or shed) | full eval stream, real classifier | measured |
| Share the classifier is unsure about: 13.8% of processed tickets (10% of the normal tickets) | full eval stream | measured |
| Price per million tokens: 0.15 input, 0.60 output (USD) | `config.TOKEN_PRICE_PER_M` | **placeholder, check against the provider's price page** |
| Scaling to 1,000 tickets, and the "without classifier" scenario | arithmetic on the above | extrapolated |

## Result (placeholder prices)
| Scenario | Model calls per ticket | USD per 1,000 tickets |
|---|---|---|
| **With the classifier** (confident predictions free, unsure ones to the LLM) | 0.95 draft + 0.14 classify | **0.29** |
| **Without it** (every drafted ticket is also classified by the LLM) | 0.95 draft + 0.95 classify | **0.38** |
| Saving | | 0.09 (24%) |

Sensitivity to the unsure share: 5% gives 0.28, 10% 0.28, 20% 0.29, 30% 0.31 (with the classifier). Without it the figure does not depend on that share.

## What the numbers say
- The classifier saves a quarter of the cost, not most of it, because the **draft call dominates**: a draft is about 790 tokens against about 500 for an LLM classification, and every drafted ticket needs one while only 14% need a classification. A cheaper way to cut cost is to draft only for categories that need a reply, or to use a smaller model for drafts.
- The classifier's other advantages are not in this table: it answers in milliseconds, makes no call that can be rate-limited (the live runs hit that), and gives the same answer for the same text.
- **Do not read this as "the LLM is better on unsure tickets".** In A6 the LLM was right on 64.4% of the 118 low-confidence validation tickets and the classifier on 62.7%, a difference of 1.7 points (interval -10 to +13). Sending unsure tickets to the LLM here follows the A7 brief and is a cost and consistency choice. Sending them to a person would cost more per ticket and is the alternative `docs/a6-classifier.md` describes.
- The quarantine step is a saving too: flagged tickets make no LLM call at all.
