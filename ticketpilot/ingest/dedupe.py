"""Stage 3, dedupe: exact and near duplicates by 3-word shingle Jaccard. Every drop is reported with its reason."""
import re

from ticketpilot.ingest.model import Document, Dropped

# Chosen on labelled pairs (evals/a4_dedupe_pairs.jsonl), see docs/a4-ingest-report.md: it sits in the gap between
# the lowest true-duplicate score and the highest score of any labelled non-duplicate pair.
THRESHOLD = 0.68


def shingles(text: str, n: int = 3) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(len(words) - n + 1, 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def _explicit_versions_differ(a: Document, b: Document) -> bool:
    """Versions of one policy are naturally near-identical text; two declared, different versions are history,
    not duplicates."""
    return "version" not in a.inferred and "version" not in b.inferred and a.version != b.version


def authority(d: Document) -> tuple:
    """Higher sorts first: declared metadata, then newest, then shortest file name, then name for determinism."""
    return ("version" not in d.inferred and "effective_date" not in d.inferred, d.effective_date, -len(d.path), d.path)


def dedupe(docs: list[Document], threshold: float = THRESHOLD) -> tuple[list[Document], list[Dropped]]:
    ordered = sorted(docs, key=authority, reverse=True)
    kept: list[Document] = []
    cache: dict[str, set[str]] = {}
    drops: list[Dropped] = []
    for d in ordered:
        cache[d.doc_id] = shingles(d.body)
        best, best_j = None, 0.0
        for k in kept:
            if _explicit_versions_differ(d, k):
                continue
            j = jaccard(cache[d.doc_id], cache[k.doc_id])
            if j > best_j:
                best, best_j = k, j
        if best is not None and best_j >= threshold:
            exact = re.sub(r"\s+", " ", d.body.lower()) == re.sub(r"\s+", " ", best.body.lower())
            drops.append(Dropped(d.path, "exact_duplicate" if exact else "near_duplicate", best.path, round(best_j, 3)))
        else:
            kept.append(d)
    return sorted(kept, key=lambda d: d.path), sorted(drops, key=lambda x: x.path)
