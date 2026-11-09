"""Exact and near-duplicate detection (spec-a7 'Pipeline' step 1). A duplicate is linked to the original, never deleted.

Near duplicate = same customer, word-set Jaccard >= THRESHOLD, inside WINDOW_S of event time. A customer reporting the same
thing hours later is a new ticket (the gold stream has such a case), so the window matters as much as the threshold."""
import re

THRESHOLD = 0.7
WINDOW_S = 30 * 60


def norm(text: str) -> str:
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", text.lower()))))


def jaccard(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / len(x | y) if x | y else 0.0


def find_near(store, customer_id: str, ts: float, normalised: str, threshold: float = THRESHOLD, window_s: float = WINDOW_S):
    """Event ID of the earliest admitted event this one duplicates, or None."""
    rows = store.recent_for_customer(customer_id, ts - window_s, ("pending", "processing", "done"))
    best = [(r["ts"], r["event_id"]) for r in rows if abs(ts - r["ts"]) <= window_s and jaccard(normalised, r["norm"]) >= threshold]
    return min(best)[1] if best else None
