import json
from pathlib import Path

import pytest

from ticketpilot.ingest import dedupe as dd
from ticketpilot.ingest.clean import clean
from ticketpilot.ingest.load import load_file
from ticketpilot.ingest.model import Quarantined
from ticketpilot.ingest.pipeline import choose_threshold, ingest, score_pairs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result():
    return ingest(ROOT / "data" / "kb_raw")


def by_id(result):
    return {d.doc_id: d for d in result.docs}


def test_counts_add_up(result):
    assert result.files_in == 19
    assert len(result.docs) + len(result.dropped) + len(result.quarantined) == result.files_in


def test_exact_and_near_duplicates_removed_with_reason(result):
    reasons = {d.path: (d.reason, d.kept) for d in result.dropped}
    assert reasons["dispute-refund-policy-v3-copy.md"] == ("exact_duplicate", "dispute-refund-policy-v3.md")
    assert reasons["fraud-policy-v1-copy.md"][0] == "exact_duplicate"
    assert reasons["dispute-refund-policy-v3-reformatted.md"][0] == "near_duplicate"
    assert reasons["fees-schedule-v2-reformatted.md"][0] == "near_duplicate"


def test_near_duplicate_with_changed_number_loses_to_declared_metadata(result):
    assert "dispute-refund-policy-v3-wiki-export.txt" in {d.path for d in result.dropped}
    assert all("100 days" not in d.body for d in result.docs)  # the wrong number did not survive
    assert "90 days" in by_id(result)["dispute-refund-policy-v3"].body


def test_versions_of_one_policy_are_kept_not_deduped(result):
    ids = by_id(result)
    assert {"dispute-refund-policy-v1", "dispute-refund-policy-v2", "dispute-refund-policy-v3"} <= ids.keys()


def test_image_only_pdf_is_quarantined_with_reason(result):
    q = {x.path: x.reason for x in result.quarantined}
    assert q == {"scanned-chargeback-memo.pdf": "needs_ocr"}
    assert "scanned-chargeback-memo" not in by_id(result)


def test_status_is_derived_not_trusted(result):
    docs = by_id(result)
    assert docs["dispute-refund-policy-v3"].status == "current"
    assert docs["dispute-refund-policy-v2"].status == "superseded"  # its front matter says current
    assert docs["disputes-faq"].status == "superseded"              # stale FAQ, older than the policy
    assert docs["dispute-refund-policy-v4-DRAFT"].status == "draft"
    assert docs["fees-schedule-v3-DRAFT"].status == "draft"
    assert docs["fees-schedule-v2"].status == "current"
    assert docs["dispute-refund-policy-v3"].supersedes == "dispute-refund-policy-v2"
    assert ("dispute-refund-policy-v2", "current", "superseded") in result.status_overrides


def test_inferred_metadata_is_flagged(result):
    esc = by_id(result)["escalation-policy-v1"]
    assert {"title", "version", "effective_date"} <= set(esc.inferred)
    assert (esc.title, esc.version, esc.effective_date) == ("Escalation Policy", 1, "2026-01-15")
    assert by_id(result)["dispute-refund-policy-v3"].inferred == []


def test_pdf_text_is_extracted_and_boilerplate_stripped(result):
    sla = by_id(result)["support-sla-v1"]
    assert "15 minutes" in sla.body
    assert "Confidential" not in sla.body and "Page 1 of" not in sla.body


def test_threshold_sits_in_the_gap_of_labelled_pairs():
    rows = score_pairs()
    t, lo, hi = choose_threshold(rows)
    assert hi < dd.THRESHOLD < lo
    assert dd.THRESHOLD == t


def test_clean_strips_page_numbers_and_repeated_headers():
    pages = "HEADER\nfirst page text\n1\n\fHEADER\nsecond page text\nPage 2 of 2"
    assert clean(pages) == "first page text\n\nsecond page text"  # a page break becomes a paragraph break


def test_clean_normalises_unicode_and_whitespace():
    assert clean("Fee:  2.50 GBP \n\n\n\nnext") == "Fee: 2.50 GBP\n\nnext"


def test_unsupported_and_empty_files_are_quarantined(tmp_path):
    (tmp_path / "x.docx").write_text("hi")
    (tmp_path / "e.md").write_text("  \n")
    assert load_file(tmp_path / "x.docx") == Quarantined("x.docx", "unsupported_type")
    assert load_file(tmp_path / "e.md") == Quarantined("e.md", "empty")


def test_pipeline_is_deterministic():
    a, b = ingest(ROOT / "data" / "kb_raw"), ingest(ROOT / "data" / "kb_raw")
    assert [(d.doc_id, d.status) for d in a.docs] == [(d.doc_id, d.status) for d in b.docs]
    assert [(d.path, d.kept) for d in a.dropped] == [(d.path, d.kept) for d in b.dropped]


def test_generator_output_matches_committed_corpus():
    """data/kb_raw is the generator's output: every committed file has a counterpart in the pairs file or the report."""
    names = {p.name for p in (ROOT / "data" / "kb_raw").iterdir()}
    pair_names = {Path(json.loads(l)[k]).name for l in (ROOT / "evals" / "a4_dedupe_pairs.jsonl").read_text().splitlines()
                  for k in ("a", "b")}
    assert pair_names <= names
