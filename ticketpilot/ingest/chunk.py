"""Stage 5, chunk. A: fixed-size windows, no structure. B: heading-aware, with the doc header on every chunk."""
import re
from dataclasses import dataclass

from ticketpilot.ingest.model import Document

FIXED_SIZE = 500
FIXED_OVERLAP = 100
MAX_SECTION = 1200
HEADING = re.compile(r"^#{1,4}\s+(.*)$", re.M)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    version: int
    effective_date: str
    status: str
    section: str
    text: str


def _chunk(doc: Document, n: int, section: str, text: str) -> Chunk:
    return Chunk(f"{doc.doc_id}#{n}", doc.doc_id, doc.title, doc.version, doc.effective_date, doc.status, section, text)


def chunk_fixed(doc: Document, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP) -> list[Chunk]:
    text, step = doc.body, size - overlap
    return [_chunk(doc, i, "", text[s:s + size]) for i, s in enumerate(range(0, max(len(text) - overlap, 1), step))]


def _sections(body: str) -> list[tuple[str, str]]:
    """Split on markdown headings; text without headings splits on blank lines instead."""
    marks = list(HEADING.finditer(body))
    if marks:
        out = [("", body[:marks[0].start()].strip())] if body[:marks[0].start()].strip() else []
        for m, nxt in zip(marks, marks[1:] + [None]):
            out.append((m.group(1).strip(), body[m.end():nxt.start() if nxt else len(body)].strip()))
        return [(h, t) for h, t in out if t or h]
    return [("", p.strip()) for p in re.split(r"\n\s*\n", body) if p.strip()]


def chunk_structured(doc: Document, max_chars: int = MAX_SECTION) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending: list[str] = []
    pending_heading = ""

    def flush() -> None:
        if pending:
            text = f"{doc.header}\n" + (f"## {pending_heading}\n" if pending_heading else "") + "\n\n".join(pending)
            chunks.append(_chunk(doc, len(chunks), pending_heading, text))
            pending.clear()

    for heading, text in _sections(doc.body):
        if not text:  # a heading with nothing under it (a document title) is already in the chunk header
            continue
        if heading:  # a heading always starts a new chunk
            flush()
            pending_heading = heading
        if pending and sum(map(len, pending)) + len(text) > max_chars:
            flush()
        pending.append(text)
    flush()
    return chunks


STRATEGIES = {"fixed": chunk_fixed, "structured": chunk_structured}
