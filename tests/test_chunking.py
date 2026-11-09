import re

from ticketpilot.ingest.chunk import FIXED_OVERLAP, FIXED_SIZE, chunk_fixed, chunk_structured
from ticketpilot.ingest.metadata import make_document


def doc(body: str, **meta):
    meta = {"title": "Test Policy", "version": "2", "effective_date": "2026-03-01", "status": "current", **meta}
    return make_document("test-policy-v2.md", meta, body)


LONG = " ".join(f"word{i}" for i in range(400))
SECTIONED = "# Test Policy (v2)\n\n## Filing window\nThirty days to file.\n\n## Refunds\nFive days for refunds.\n"


def test_fixed_chunks_respect_size_cap():
    chunks = chunk_fixed(doc(LONG))
    assert len(chunks) > 1 and all(len(c.text) <= FIXED_SIZE for c in chunks)


def test_fixed_chunks_overlap_and_lose_nothing():
    body = doc(LONG).body
    chunks = chunk_fixed(doc(LONG))
    rebuilt = chunks[0].text + "".join(c.text[FIXED_OVERLAP:] for c in chunks[1:])
    assert rebuilt == body
    assert chunks[0].text[-FIXED_OVERLAP:] == chunks[1].text[:FIXED_OVERLAP]


def test_fixed_chunks_have_no_header_and_short_doc_is_one_chunk():
    chunks = chunk_fixed(doc("Short."))
    assert [c.text for c in chunks] == ["Short."]


def test_structured_chunks_carry_the_doc_header():
    chunks = chunk_structured(doc(SECTIONED))
    assert len(chunks) >= 2
    assert all(c.text.startswith("[Test Policy | version 2 | effective 2026-03-01 | current]") for c in chunks)


def test_structured_chunks_split_on_headings_and_name_sections():
    chunks = chunk_structured(doc(SECTIONED))
    by_section = {c.section: c.text for c in chunks}
    assert "Thirty days" in by_section["Filing window"] and "Five days" in by_section["Refunds"]
    assert "Five days" not in by_section["Filing window"]


def test_structured_chunks_lose_no_text():
    d = doc(SECTIONED)
    body_without_title = re.sub(r"^# .*$", "", d.body, flags=re.M)  # the document title is carried by every chunk header
    words = set(re.findall(r"[a-z0-9]+", body_without_title.lower()))
    covered = set(re.findall(r"[a-z0-9]+", " ".join(c.text for c in chunk_structured(d)).lower()))
    assert words <= covered


def test_structured_splits_oversized_section_and_handles_headingless_text():
    paras = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(8))
    chunks = chunk_structured(doc(paras), max_chars=700)
    assert len(chunks) >= 3 and all(len(c.text) < 700 + 120 for c in chunks)


def test_chunk_ids_are_unique_and_inherit_status():
    d = doc(SECTIONED, status="superseded")
    ids = [c.chunk_id for c in chunk_structured(d)]
    assert len(ids) == len(set(ids)) and all(c.status == "superseded" for c in chunk_structured(d))


def test_no_chunk_is_only_a_heading_or_header():
    """Regression: a document-title heading with no text under it used to become its own fact-free chunk, which then
    ranked highly for on-topic queries and crowded out the chunks holding the answer."""
    for text in (SECTIONED, "# Title only\n\n## One\nBody one.\n\n## Two\nBody two.\n"):
        for c in chunk_structured(doc(text)):
            content = c.text.split("\n", 1)[1]
            assert re.sub(r"[#\s]", "", content) and not re.fullmatch(r"(#+ .*\n?)+", content.strip() + "\n")


def test_every_structured_chunk_of_the_real_corpus_has_body_text():
    from pathlib import Path
    from ticketpilot.ingest.pipeline import ingest
    for d in ingest(Path(__file__).resolve().parents[1] / "data" / "kb_raw").docs:
        for c in chunk_structured(d):
            lines = [ln for ln in c.text.split("\n")[1:] if ln.strip() and not ln.startswith("#")]
            assert lines, f"{c.chunk_id} has no body text"
