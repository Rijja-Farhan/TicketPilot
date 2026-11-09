# A7: guardrail and permission policy

Code: `ticketpilot/ambient/policy.py` (tiers, output checks), `guard.py` (input scan), `triage.py` (`GuardedTools`), `approvals.py`. Spec: `docs/spec-a7.md`.

The rule behind every choice: **the model may propose, only code decides.** A fooled model must not be able to do anything the policy would not allow a calm one to do.

## Layers, in the order a ticket meets them
| # | Layer | What it does | Where enforced |
|---|---|---|---|
| 1 | Schema validation | `event_id`, `customer_id`, timestamp, field lengths; bad events get 422 and an audit entry | `events.py`, `webhook.py` |
| 2 | Admission | exact and near duplicates linked, per-customer rate cap, back-pressure | `queue.py` |
| 3 | Input guard | normalise (NFKC, strip hidden characters), scan for injection signals | `guard.py` |
| 4 | Quarantine | a flagged ticket is never shown to the model: classifier label only, no LLM label, no draft, read-only enrichment, security on-call notified | `triage.py` |
| 5 | Tool allowlist | the ambient worker may call `get_customer`, `classify_ticket`, `search_policies` only; every call audited | `GuardedTools` |
| 6 | No tools during drafting | the draft call has no tools at all, so a fooled model cannot fetch other customers' data | `triage.draft` |
| 7 | Action tiers | every action, by name, maps to auto, approval or denied; unknown names are denied | `policy.decide` |
| 8 | Output checks | a drafted reply is rejected if it promises a refund, claims an action was taken, contains a link or email, names another customer's ID, or quotes an amount that is not in the ticket | `policy.check_draft` |
| 9 | Human approval | every customer-facing reply waits for a person; edits are re-checked; approving re-runs the policy | `approvals.py` |
| 10 | Audit | every step above is written to a hash-chained log | `audit.py` |

## Action tiers
| Tier | Actions | Who decides |
|---|---|---|
| auto | `set_category`, `set_priority`, `link_duplicate`, `quarantine`, `raise_alert`, `notify` | code, no human (reversible, internal) |
| approval | `send_reply`, `change_status`, `escalate_to_human` | a named person |
| denied | `refund`, `freeze_account`, `close_account`, `read_other_customer`, `add_ticket_note`, anything not listed | nobody: refused in code and audited |

In quarantine only `set_priority`, `link_duplicate`, `quarantine`, `raise_alert`, `notify` and `escalate_to_human` are allowed.

Model-proposed action names are normalised first (`Issue-Refund`, `refund` with a hidden character, `refund_customer` all become `refund`), then looked up. A name that is not in the table is denied, which is also why a real model that suggests a harmless-sounding name such as `review_dispute_details` is recorded as refused. The draft prompt lists the two names it may use, so the refusals stay rare.

## Who gets notified
| Situation | Recipient |
|---|---|
| reply for a normal ticket | the team that owns the category (cards, payments, disputes, billing, access, lending) |
| fraud wording, p1, or a restricted customer | the fraud team, flagged urgent |
| quarantined ticket | security on-call, urgent |
| unknown customer, or no valid draft | support lead |
| nobody answers for 4 hours | the next person up (support lead, then head of support) |

## What is deliberately not protected
- The guard is a pattern list. It does not catch paraphrases, other languages, ROT13 or spaced-out letters (`docs/a7-redteam.md` has the measured numbers). It exists to keep obvious attacks away from the model, not to be the defence.
- A harmless-looking draft still reaches a person. The approver is the last line, so the approval screen shows the ticket text beside the draft.
- Dedupe hides appended text on a near duplicate: nobody reviews the extra words.
- The per-customer cap does not stop a flood spread over many customers; back-pressure keeps the queue bounded and still admits fraud wording, but it sheds ordinary tickets while it does.

## What an approver can and cannot do
Approve, edit (re-validated, then approved), or reject with a reason. An approver cannot approve a denied kind: `approve()` re-checks the tier, so a forged `refund` row in the queue is refused.
