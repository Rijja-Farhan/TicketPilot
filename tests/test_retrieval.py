from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ticketpilot import fake_llm, policies
from ticketpilot.embeddings import HashEmbedder, get_embedder
from ticketpilot.ingest.pipeline import build_indexes, ingest
from ticketpilot.retrieval import ALL_STATUSES, CURRENT, WITH_HISTORY, Index
from mockapi.main import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("index")
    result = ingest(ROOT / "data" / "kb_raw")
    build_indexes(result.docs, out, HashEmbedder())
    return out


@pytest.fixture
def index(built):
    return Index.load(built / "structured")


def statuses(hits):
    return {h.chunk.status for h in hits}


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_default_search_excludes_superseded_and_draft(index, mode):
    hits = index.search("what is the dispute filing window", k=8, mode=mode, statuses=CURRENT)
    assert hits and statuses(hits) == {"current"}


def test_history_includes_superseded_but_never_draft(index):
    hits = index.search("dispute filing window", k=20, mode="hybrid", statuses=WITH_HISTORY)
    assert "superseded" in statuses(hits) and "draft" not in statuses(hits)


def test_unfiltered_search_can_return_drafts(index):
    hits = index.search("dispute filing window 45 days proposal", k=20, mode="bm25", statuses=ALL_STATUSES)
    assert "draft" in statuses(hits)


def test_hybrid_ranking_is_deterministic(index):
    a = [(h.chunk.chunk_id, h.score) for h in index.search("international transfer fee", k=5, mode="hybrid")]
    b = [(h.chunk.chunk_id, h.score) for h in index.search("international transfer fee", k=5, mode="hybrid")]
    assert a == b and len(a) == 5


def test_save_and_load_round_trip(built, index):
    again = Index.load(built / "structured")
    q = "who is paged first for a P1 incident"
    assert [h.chunk.chunk_id for h in index.search(q, mode="dense")] == [h.chunk.chunk_id for h in again.search(q, mode="dense")]


def test_require_lexical_returns_nothing_for_unrelated_question(index):
    assert index.search("what is the capital of France", mode="dense", require_lexical=True) == []
    assert index.search("what is the capital of France", mode="dense")  # dense alone always ranks something


def test_unknown_mode_rejected(index):
    with pytest.raises(ValueError):
        index.search("x", mode="nope")


def test_hash_embedder_is_deterministic_and_normalised():
    e = get_embedder("hash")
    a, b = e.embed_documents(["dispute window 90 days"]), e.embed_documents(["dispute window 90 days"])
    assert (a == b).all() and abs(float((a * a).sum()) - 1.0) < 1e-5


def test_unknown_embedder_rejected():
    with pytest.raises(ValueError):
        get_embedder("nope")


# ---- policies.py wiring ----

@pytest.fixture
def use_index(built, monkeypatch):
    monkeypatch.setenv("TICKETPILOT_INDEX_DIR", str(built))
    policies._load_index.cache_clear()
    yield
    policies._load_index.cache_clear()


def test_policies_search_uses_index_and_returns_current_version(use_index):
    docs = policies.search("what is the dispute filing window")
    assert docs and {d.status for d in docs} == {"current"}
    assert docs[0].title == "Dispute and Refund Policy" and docs[0].version == 3


def test_policies_history_reaches_superseded_versions(use_index):
    docs = policies.search("what was the old dispute filing window before it changed?", k=6,
                           include_superseded=policies.wants_history("what was the old dispute filing window before it changed?"))
    assert {d.version for d in docs if d.title == "Dispute and Refund Policy"} & {1, 2}
    assert all(d.status != "draft" for d in docs)


def test_history_question_retrieves_the_old_value_not_just_a_title(use_index):
    q = "what was the old dispute filing window before it changed?"
    docs = policies.search(q, k=3, include_superseded=policies.wants_history(q))
    old = {d.version: d.body for d in docs if d.title == "Dispute and Refund Policy" and d.status == "superseded"}
    assert "60 days" in old.get(1, "") or "120 days" in old.get(2, "")


def test_policies_unrelated_question_skips_the_llm(use_index):
    assert policies.search("what is the capital of France") == []
    assert policies.answer("what is the capital of France") == policies.NOT_FOUND


def test_policies_answer_cites_the_current_version_from_the_index(use_index):
    fake_llm.queue_responses('{"answer": "90 days.", "sources": [{"title": "Dispute and Refund Policy", "version": 3, "effective_date": "x"}]}')
    out = policies.answer("what is the dispute filing window")
    assert "Dispute and Refund Policy, v3, effective 2026-09-01" in out


def test_policies_answer_cannot_cite_a_superseded_version_by_default(use_index):
    fake_llm.queue_responses('{"answer": "60 days.", "sources": [{"title": "Dispute and Refund Policy", "version": 1, "effective_date": "x"}]}')
    assert "won't give one" in policies.answer("what is the dispute filing window")


def test_get_doc_and_mock_api_serve_index_documents(use_index):
    assert policies.get_doc("dispute-refund-policy-v3").version == 3
    assert policies.get_doc("nope") is None
    client = TestClient(app)
    assert client.get("/policies/dispute-refund-policy-v3").json()["status"] == "current"
    assert client.get("/policies/nope").status_code == 404
    found = client.get("/policies", params={"query": "dispute filing window"}).json()["policies"]
    assert found and all(p["status"] == "current" for p in found)
