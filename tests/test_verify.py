import json

from ticketpilot import fake_llm
from ticketpilot.schemas import Citation, Finding, ReportDraft
from ticketpilot.verify import REMOVED, check_report, strip_failures, summarize
from tests.coworker_helpers import checks, investigate, plan, reflect, report, tool

POLICY = {"policies": [{"slug": "fees-schedule-v2", "title": "Fees Schedule", "version": 2,
                        "effective_date": "2026-03-01", "status": "current", "body": "FX fee 1.5%"},
                       {"slug": "fees-schedule-v1", "title": "Fees Schedule", "version": 1,
                        "effective_date": "2025-01-01", "status": "superseded", "body": "FX fee 3%"}]}
LEDGER = [
    {"id": "E1", "tool": "get_ticket", "ok": True, "arguments": {}, "timestamp": "",
     "result": {"ticket_id": "T-1002", "customer_id": "C-17", "status": "open", "created_at": "2026-09-22T17:25:00Z"}},
    {"id": "E2", "tool": "list_transactions", "ok": True, "arguments": {}, "timestamp": "",
     "result": {"transactions": [{"txn_id": "X-00006", "amount": 265.99, "date": "2026-07-12"}]}},
    {"id": "E3", "tool": "search_policies", "ok": True, "arguments": {}, "timestamp": "", "result": POLICY},
]


def draft(**kw):
    base = dict(ticket_id="T-1002", summary="T-1002 is open.", findings=[Finding(claim="T-1002 is open", evidence_ids=["E1"])],
                policy_citations=[], proposed_resolution="Explain the fee.")
    return ReportDraft(**{**base, **kw})


def failed(d, task="investigate T-1002"):
    return {f.check for f in check_report(d, LEDGER, task, "partial", "enough")}


def test_clean_report_passes_everything():
    d = draft(findings=[Finding(claim="C-17 owns T-1002, opened 2026-09-22", evidence_ids=["E1"]),
                        Finding(claim="Transaction X-00006 was $265.99 on 2026-07-12", evidence_ids=["E2"])],
              policy_citations=[Citation(title="Fees Schedule", version=2, effective_date="2026-03-01")])
    assert check_report(d, LEDGER, "investigate T-1002", "complete", "enough") == []


def test_unknown_evidence_id_fails():
    assert "evidence_ids" in failed(draft(findings=[Finding(claim="T-1002 is open", evidence_ids=["E9"])]))
    assert "evidence_ids" in failed(draft(findings=[Finding(claim="T-1002 is open", evidence_ids=[])]))


def test_invented_amount_id_and_date_fail():
    assert "facts_in_evidence" in failed(draft(findings=[Finding(claim="Charged $300.00", evidence_ids=["E2"])]))
    assert "facts_in_evidence" in failed(draft(findings=[Finding(claim="Ticket T-7777 is related", evidence_ids=["E1"])]))
    assert "facts_in_evidence" in failed(draft(findings=[Finding(claim="Opened 2026-01-01", evidence_ids=["E1"])]))
    assert "facts_in_evidence" in failed(draft(summary="Refund of 12.50 is due"))


def test_fact_must_be_in_the_cited_evidence_not_just_anywhere():
    assert "facts_in_evidence" in failed(draft(findings=[Finding(claim="Amount 265.99", evidence_ids=["E1"])]))


def test_superseded_policy_citation_fails_unless_history_was_asked():
    old = [Citation(title="Fees Schedule", version=1, effective_date="2025-01-01")]
    assert failed(draft(policy_citations=old)) == {"policy_citations"}
    assert failed(draft(policy_citations=old), task="investigate T-1002 and show the fee history, previous versions") == set()
    never = [Citation(title="Fees Schedule", version=3, effective_date="2027-01-01")]
    assert failed(draft(policy_citations=never), task="what did the old policy say, history") == {"policy_citations"}


def test_claims_of_actions_taken_fail():
    for text in ("I refunded the customer", "I have added a note to the ticket", "We've escalated this", "A note was added"):
        assert "no_action_claims" in failed(draft(proposed_resolution=text)), text
    assert failed(draft(proposed_resolution="A human could refund the customer; I will not act.")) == set()


def test_classifier_label_must_match_evidence():
    from ticketpilot.schemas import CategoryResult
    led = LEDGER + [{"id": "E4", "tool": "classify_ticket", "ok": True, "arguments": {}, "timestamp": "",
                     "result": {"label": "fees_charges", "probability": 0.91}}]
    good = draft(category=CategoryResult(label="fees_charges", classifier_label="fees_charges", probability=0.91, evidence_ids=["E4"]))
    bad = draft(category=CategoryResult(label="fees_charges", classifier_label="fees_charges", probability=0.5, evidence_ids=["E4"]))
    assert check_report(good, led, "t", "partial", "enough") == []
    assert {f.check for f in check_report(bad, led, "t", "partial", "enough")} == {"classifier_matches"}


def test_complete_requires_the_enough_stop_reason():
    assert {f.check for f in check_report(draft(), LEDGER, "t", "complete", "blocked")} == {"status_consistent"}


def test_strip_removes_only_what_failed_and_keeps_a_record():
    d = draft(findings=[Finding(claim="T-1002 is open", evidence_ids=["E1"]), Finding(claim="Charged $300.00", evidence_ids=["E2"])],
              proposed_resolution="I refunded it")
    fs = check_report(d, LEDGER, "t", "partial", "enough")
    s = strip_failures(d, fs)
    assert [f.claim for f in s.findings] == ["T-1002 is open"] and s.proposed_resolution == REMOVED
    assert any("Charged $300.00" in q for q in s.open_questions)
    passed, lines = summarize(fs)
    assert "facts_in_evidence" not in passed and len(lines) == len(fs)


def test_repair_round_fixes_a_bad_draft():
    bad = report(findings=[{"claim": "Charged $999.00", "evidence_ids": ["E1"]}])
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-1002"), "DONE", "DONE", reflect(), bad, report(), checks("supported"))
    r = investigate()
    assert r.status == "complete" and r.findings[0].claim.startswith("T-1002") and not fake_llm._queue


def test_second_failure_gives_partial_with_findings_removed():
    bad = report(findings=[{"claim": "Charged $999.00", "evidence_ids": ["E1"]},
                           {"claim": "T-1002 is open", "evidence_ids": ["E1"]}])
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-1002"), "DONE", "DONE", reflect(), bad, bad, checks("supported"))
    r = investigate()
    assert r.status == "partial" and r.stop_reason == "verification_failed"
    assert [f.claim for f in r.findings] == ["T-1002 is open"]
    assert any("$999.00" in q for q in r.open_questions) and r.verification.failed


def test_model_check_is_advisory_only():
    fake_llm.queue_responses(plan(), tool("get_ticket", ticket_id="T-1002"), "DONE", "DONE", reflect(), report(), checks("unsupported"))
    r = investigate()
    assert r.status == "complete" and r.verification.model_check == ["finding 1: unsupported"]


# ---- A6 baseline findings -------------------------------------------------------------------------------------------

def _draft(**kw):
    from ticketpilot.schemas import ReportDraft
    base = dict(ticket_id="T-1", summary="s", findings=[], policy_citations=[], proposed_resolution="r")
    return ReportDraft(**{**base, **kw})


def test_action_claims_now_cover_freezes_and_blocks():
    from ticketpilot.verify import find_action_claim
    for bad in ("We've placed a freeze on your card as you requested.", "We froze the card.", "I have flagged the account.",
                "A freeze has been placed on the card.", "We've contacted the merchant."):
        assert find_action_claim(bad), bad
    for ok in ("We will place a freeze once a human approves.", "Verify whether a refund has been issued for the transaction.",
               "If the merchant has issued the refund, close the ticket.", "A human could place a freeze.",
               "Check if a freeze has been placed.", "The customer asked us to put a freeze on the card."):
        assert find_action_claim(ok) is None, ok


def test_category_from_a_deferred_classifier_result_is_rejected():
    from ticketpilot.schemas import CategoryResult
    from ticketpilot.verify import check_report
    ledger = [{"id": "E1", "tool": "get_ticket", "ok": True, "arguments": {}, "result": {"ticket_id": "T-1", "category": "other"}},
              {"id": "E2", "tool": "classify_ticket", "ok": True, "arguments": {},
               "result": {"label": "fraud_security", "probability": 0.39, "deferred": True}}]
    bad = _draft(category=CategoryResult(label="fraud_security", classifier_label="fraud_security", probability=0.39, evidence_ids=["E2"]))
    failures = check_report(bad, ledger, "investigate T-1", "complete", "enough")
    assert [f.check for f in failures] == ["category_source"] and failures[0].target == "category"
    good = _draft(category=CategoryResult(label="other", evidence_ids=["E1"]))
    assert check_report(good, ledger, "investigate T-1", "complete", "enough") == []
    confident = [dict(ledger[0]), {**ledger[1], "result": {"label": "fraud_security", "probability": 0.9, "deferred": False}}]
    ok = _draft(category=CategoryResult(label="fraud_security", classifier_label="fraud_security", probability=0.9, evidence_ids=["E2"]))
    assert check_report(ok, confident, "investigate T-1", "complete", "enough") == []   # a confident classifier result is allowed


def test_policy_citation_for_a_missing_ticket_is_rejected():
    from ticketpilot.schemas import Citation
    from ticketpilot.verify import check_report
    ledger = [{"id": "E1", "tool": "get_ticket", "ok": False, "arguments": {}, "result": {"error": {"code": "not_found", "message": "x", "retryable": False}}},
              {"id": "E2", "tool": "search_policies", "ok": True, "arguments": {},
               "result": {"policies": [{"title": "Fees Schedule", "version": 2, "effective_date": "2026-03-01", "status": "current"}]}}]
    cite = Citation(title="Fees Schedule", version=2, effective_date="2026-03-01")
    failures = check_report(_draft(policy_citations=[cite]), ledger, "investigate T-9", "partial", "no_progress")
    assert [f.check for f in failures] == ["missing_ticket"]
    assert check_report(_draft(), ledger, "investigate T-9", "partial", "no_progress") == []
