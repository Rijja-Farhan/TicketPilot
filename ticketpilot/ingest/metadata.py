"""Stage 4, metadata: read front matter, infer what is missing (and flag it), derive status across versions."""
import re
from pathlib import Path

from ticketpilot.ingest.model import Document

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n(.*)", re.S)
EFFECTIVE = re.compile(r"^\s*effective(?: date)?\s*:?\s*(\d{4}-\d{2}-\d{2})", re.I | re.M)
VERSION_LINE = re.compile(r"^\s*version\s*:?\s*(\d+)\s*$", re.I | re.M)
VERSION_NAME = re.compile(r"-v(\d+)\b", re.I)
VERSION_HEAD = re.compile(r"\(v(\d+)\b", re.I)


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    m = FRONT_MATTER.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def norm_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower().replace("&", " and ")).strip()


def _infer_title(body: str, stem: str) -> str:
    m = re.search(r"^\s*title\s*:\s*(.+)$", body, re.I | re.M)
    if m:
        return m.group(1).strip()
    first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    first = re.sub(r"^#+\s*", "", first)
    first = re.sub(r"\s*\(v\d+.*?\)\s*$", "", first)   # "Fees Schedule (v2)" -> "Fees Schedule"
    first = first.split(" - ")[0].replace("&", "and").strip()
    return first or stem.replace("-", " ").title()


def make_document(path: str, meta: dict[str, str], body: str) -> Document:
    """Declared metadata where present; otherwise inferred from text or file name and flagged."""
    stem = Path(path).stem
    inferred: list[str] = []
    title = meta.get("title")
    if not title:
        title = _infer_title(body, stem)
        inferred.append("title")
    version = meta.get("version")
    if version is None:
        m = VERSION_LINE.search(body) or VERSION_NAME.search(stem) or VERSION_HEAD.search(body[:200])
        version = m.group(1) if m else "1"
        inferred.append("version")
    date = meta.get("effective_date")
    if date is None:
        m = EFFECTIVE.search(body)
        date = m.group(1) if m else ""
        inferred.append("effective_date")
    status = meta.get("status")
    if status is None:
        status = "draft" if re.search(r"\bdraft\b", stem + " " + body[:200], re.I) else "current"
        inferred.append("status")
    return Document(doc_id=stem, path=path, title=title, version=int(version), effective_date=date,
                    status=status, body=body.strip(), applies_to=meta.get("applies_to", ""), inferred=inferred)


def resolve_status(docs: list[Document]) -> None:
    """Status is derived, not trusted: within one policy the latest effective non-draft document is current,
    earlier ones are superseded (a stale front-matter 'current' is overridden). Drafts stay drafts."""
    groups: dict[str, list[Document]] = {}
    for d in docs:
        if d.status != "draft":
            groups.setdefault(norm_key(d.applies_to or d.title), []).append(d)
    for group in groups.values():
        group.sort(key=lambda d: (d.effective_date, d.version))
        for d in group[:-1]:
            d.status = "superseded"
        group[-1].status = "current"
    by_title: dict[str, list[Document]] = {}
    for d in docs:
        if d.status != "draft":
            by_title.setdefault(norm_key(d.title), []).append(d)
    for chain in by_title.values():
        chain.sort(key=lambda d: (d.effective_date, d.version))
        for prev, d in zip(chain, chain[1:]):
            d.supersedes = prev.doc_id
