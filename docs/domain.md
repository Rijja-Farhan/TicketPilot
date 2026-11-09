# Domain: Northwind Pay (fictional)

Northwind Pay is a fictional digital bank / payments company for consumers and small businesses. It replaces the original SaaS framing so the project can use public banking-support datasets (Banking77, Bitext, CFPB complaints, NAB). All customers, accounts and transactions are synthetic.

## Products
- **Current account** with a debit card (physical and virtual)
- **Transfers and payments:** domestic, international, standing orders, direct debits
- **Savings account**
- **Personal loans and a credit card**

## Ticket categories (target label set, 8 classes)
| Label | Covers | Main public sources |
|---|---|---|
| `card_issues` | lost/stolen, declined, activation, PIN, delivery | Banking77, Bitext, CFPB (credit card) |
| `transfers_payments` | pending, failed, wrong amount, limits, international | Banking77, Bitext |
| `account_access` | login, password, lockout, verification, closing | Banking77, Bitext, CFPB |
| `fees_charges` | unexpected fees, exchange rates, interest | Banking77, CFPB |
| `fraud_security` | unauthorised transactions, phishing, account takeover | Banking77, CFPB |
| `loans_credit` | applications, repayments, credit reporting | CFPB |
| `disputes_refunds` | chargebacks, merchant refunds, complaints | Bitext, CFPB |
| `other` | everything else, including out-of-scope | any |

Label mapping from each source lives in `tools/gen/` and must be written down in `docs/decisions.md`. Classes are expected to be imbalanced.

## Policies (the knowledge base is built from these)
Each policy has versions with an effective date; older versions stay in the KB as deliberate stale data.
- **Dispute and refund policy:** v1 (superseded) and v2 (current), with different filing windows.
- **Fraud policy:** liability rules, freeze-then-investigate.
- **Fees schedule:** v1 superseded by v2.
- **Support SLA:** response targets by priority (P1 outage, P2 degraded, P3 normal).
- **Escalation policy:** who gets paged and when.
- **Data and privacy policy:** what may be shown to an agent or put in a draft.

## Incidents
Platform incidents (card processing down, transfers delayed, login outage, app crash) with severity, affected service, timeline, root cause and resolution. Incident clusters cause ticket bursts.

## Users of TicketPilot
Internal support agents and on-call operations staff, not end customers.

## What a human must approve (initial view)
- Any customer-facing reply that mentions money, fees, refunds or fraud decisions
- Account freezes, refunds, limit changes or any write action
- Anything the agent is not confident about
The agent may alone: categorise, enrich with context, draft replies (not send), and raise internal volume alerts.

## Cost of a wrong answer
- Wrong policy cited to a customer: complaint, possible regulatory exposure
- Missed fraud: direct financial loss. For `fraud_security`, a false negative is worse than a false positive, which informs the classifier threshold.
- Unneeded escalation: wasted staff time

## Synthetic data to build
Customers, accounts, transactions, orders/payments, tickets (linked to customers), incidents (linked to ticket clusters), a replayable ticket stream (duplicates, one burst, prompt-injection tickets), and customer history for memory.
