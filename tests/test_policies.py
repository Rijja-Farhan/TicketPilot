import json

from ticketpilot import fake_llm, policies


def _reply(*sources, answer="ok"):
    return json.dumps({"answer": answer, "sources": [
        {"title": t, "version": v, "effective_date": "WRONG"} for t, v in sources]})


def test_all_docs_have_front_matter():
    docs = policies.load_docs()
    assert len(docs) == 5
    assert {d.status for d in docs} == {"current", "superseded"}


def test_search_defaults_to_current_only():
    hits = policies.search("how long do I have to file a dispute?")
    assert hits and all(d.status == "current" for d in hits)
    assert hits[0].title == "Dispute and Refund Policy" and hits[0].version == 2


def test_search_history_includes_superseded():
    hits = policies.search("dispute filing window", include_superseded=True)
    assert {(d.title, d.version) for d in hits} >= {("Dispute and Refund Policy", 1), ("Dispute and Refund Policy", 2)}


def test_wants_history_true_cases():
    for q in ["what was the old dispute filing window before it changed?",
              "what is the dispute policy history?",
              "what did the previous fees schedule say?",
              "show me the superseded version of the fees schedule",
              "how did the transfer fee used to work?",
              "what is the v1 dispute window?",
              "has the international fee changed since 2025?"]:
        assert policies.wants_history(q), q


def test_wants_history_false_positives_regression():
    for q in ["how many days before I must file a dispute?",
              "what is the exchange rate fee?",
              "my card is on hold what do I do?",
              "what is the past due fee?",
              "can I change my PIN?",
              "my old card expired, how do I get a new one?",
              "how long do I have to file a dispute?"]:
        assert not policies.wants_history(q), q


def test_before_question_cites_only_current_doc():
    fake_llm.queue_responses(_reply(("Dispute and Refund Policy", 2), answer="120 days."))
    out = policies.answer("how many days before I must file a dispute?")
    assert "v2" in out and "v1" not in out


def test_irrelevant_query_finds_nothing_and_skips_llm():
    assert policies.search("what is the capital of France") == []
    assert policies.answer("what is the capital of France") == policies.NOT_FOUND


def test_answer_cites_title_version_and_date_from_doc_not_model():
    fake_llm.queue_responses(_reply(("Dispute and Refund Policy", 2), answer="120 days."))
    out = policies.answer("how long do I have to file a dispute?")
    assert "120 days." in out
    assert "Dispute and Refund Policy, v2, effective 2026-03-01" in out
    assert "WRONG" not in out


def test_fabricated_source_is_dropped():
    fake_llm.queue_responses(_reply(("Made Up Policy", 9), ("Fees Schedule", 2)))
    out = policies.answer("what is the international transfer fee?")
    assert "Made Up Policy" not in out and "Fees Schedule, v2" in out


def test_no_valid_source_means_no_answer():
    fake_llm.queue_responses(_reply(("Made Up Policy", 9)))
    assert "won't give one" in policies.answer("what is the international transfer fee?")


def test_superseded_source_not_citable_in_default_mode():
    fake_llm.queue_responses(_reply(("Fees Schedule", 1)))
    assert "won't give one" in policies.answer("what is the international transfer fee?")


def test_llm_failure_is_reported_not_swallowed():
    fake_llm.queue_responses("bad", "still bad")
    assert "invalid_output" in policies.answer("what is the international transfer fee?")
