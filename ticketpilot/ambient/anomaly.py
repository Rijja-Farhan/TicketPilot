"""Volume anomaly detector (spec-a7 'Detector'): rolling mean and z-score over 5 minute buckets of deduplicated tickets.

The baseline for a bucket uses only earlier buckets, so a burst cannot inflate its own baseline. The alert text is built from the
numbers by code; no model writes them."""
import math
import statistics
from dataclasses import dataclass, field

BUCKET_S = 300


@dataclass(frozen=True)
class Params:
    window: int = 12       # buckets of history for the baseline
    k: float = 3.0         # z-score threshold
    min_count: int = 8     # ignore small absolute counts
    cooldown: int = 3      # buckets after an alert during which no new alert fires


@dataclass
class Alert:
    bucket_ts: float
    count: int
    expected: float
    z: float
    category: str | None
    category_count: int
    sample_events: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        from datetime import datetime, timezone
        t = datetime.fromtimestamp(self.bucket_ts, timezone.utc).strftime("%H:%M")
        top = f" Most of it is {self.category} ({self.category_count} tickets)." if self.category else ""
        return (f"Ticket volume alert: {self.count} new tickets in the 5 minutes from {t} UTC, against about {self.expected:.1f} "
                f"expected (z = {self.z:.1f}).{top} Examples: {', '.join(self.sample_events[:3])}.")


def series(points: list[tuple[float, str, str]], start: float, end: float) -> tuple[list[float], list[list[tuple[str, str]]]]:
    """Bucket start times and, per bucket, the (event_id, category) of the tickets in it. `points` = (ts, event_id, category)."""
    n = max(int(math.ceil((end - start) / BUCKET_S)), 1)
    buckets: list[list[tuple[str, str]]] = [[] for _ in range(n)]
    for ts, eid, cat in points:
        if start <= ts < end:
            buckets[int((ts - start) // BUCKET_S)].append((eid, cat))
    return [start + i * BUCKET_S for i in range(n)], buckets


def detect(points: list[tuple[float, str, str]], start: float, end: float, params: Params = Params()) -> list[Alert]:
    starts, buckets = series(points, start, end)
    counts = [len(b) for b in buckets]
    alerts, quiet_until = [], -1
    for i, c in enumerate(counts):
        hist = counts[max(0, i - params.window):i]
        if len(hist) < min(params.window, 6) or i <= quiet_until:
            continue
        mean = statistics.fmean(hist)
        std = max(statistics.pstdev(hist), math.sqrt(mean), 1.0)
        z = (c - mean) / std
        if c >= params.min_count and z >= params.k:
            cats: dict[str, int] = {}
            for _, cat in buckets[i]:
                cats[cat] = cats.get(cat, 0) + 1
            top = max(cats.items(), key=lambda kv: kv[1]) if cats else (None, 0)
            alerts.append(Alert(starts[i], c, mean, z, top[0], top[1], [eid for eid, cat in buckets[i] if cat == top[0]][:3]))
            quiet_until = i + params.cooldown
    return alerts


GRID_K = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]


def calibrate(points, start, end, burst: tuple[float, float], base: Params = Params()) -> tuple[Params, list[dict]]:
    """Choose k on the calibration stream only. Every k in the grid that catches the labelled burst with no alert outside it is
    acceptable; take the middle of that plateau (robust to noise either way) rather than its edge. Returns (params, table)."""
    table, ok = [], []
    for k in GRID_K:
        alerts = detect(points, start, end, Params(base.window, k, base.min_count, base.cooldown))
        inside = [a for a in alerts if burst[0] - BUCKET_S < a.bucket_ts <= burst[1]]
        false = [a for a in alerts if a not in inside]
        table.append({"k": k, "alerts": len(alerts), "caught_burst": bool(inside), "false_alerts": len(false)})
        if inside and not false:
            ok.append(k)
    k = ok[len(ok) // 2] if ok else GRID_K[-1]
    return Params(base.window, k, base.min_count, base.cooldown), table


def tick(store, audit, start: float, now: float, params: Params = Params()) -> list[Alert]:
    """The scheduled trigger: look at the buckets that have closed by `now`, raise an alert once per bucket, audit it."""
    end = start + ((now - start) // BUCKET_S) * BUCKET_S
    new = []
    for a in detect(store.admitted_tickets(), start, end, params):
        if store.add_alert(a.bucket_ts, {"bucket_ts": a.bucket_ts, "count": a.count, "expected": round(a.expected, 2), "z": round(a.z, 2),
                                          "category": a.category, "category_count": a.category_count, "samples": a.sample_events,
                                          "text": a.text}):
            audit.append(None, "raise_alert", "raised", tier="auto", actor="detector", reason=f"z={a.z:.1f}", count=a.count,
                         expected=round(a.expected, 2), category=a.category, text=a.text)
            new.append(a)
    return new
