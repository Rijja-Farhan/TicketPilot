"""Cost per 1,000 tickets, with and without the classifier (spec-a7 'Cost').

Per-call token averages come from a live sample (measured). The path mix (how many tickets are confident, deferred or quarantined)
comes from the full eval stream replayed with the real classifier (measured). The scaling to 1,000 tickets and the
'without classifier' scenario are arithmetic on those two (extrapolated). Prices are the placeholders in config.TOKEN_PRICE_PER_M."""
import json
import sys
from pathlib import Path

from ticketpilot import config


def per_call(report: dict) -> dict[str, tuple[float, float]]:
    """Average (prompt, completion) tokens per model call, by path, from a result file of a live run."""
    u = report["usage"]
    return {p: (u["prompt_tokens"][p] / n, u["completion_tokens"].get(p, 0) / n) for p, n in u["calls"].items() if n}


def mix(report: dict) -> dict[str, float]:
    u, done = report["usage"]["calls"], report["queue"]["done"]
    return {"processed": done, "draft_rate": u.get("draft", 0) / done, "deferred_rate": u.get("classify_llm", 0) / done}


def usd(tokens: tuple[float, float]) -> float:
    return (tokens[0] * config.TOKEN_PRICE_PER_M[0] + tokens[1] * config.TOKEN_PRICE_PER_M[1]) / 1e6


def estimate(calls: dict[str, tuple[float, float]], m: dict[str, float], deferred_rate: float | None = None) -> dict:
    """USD per 1,000 tickets. `deferred_rate` overrides the measured share of tickets the classifier is unsure about."""
    d = m["deferred_rate"] if deferred_rate is None else deferred_rate
    draft, classify = usd(calls["draft"]), usd(calls["classify_llm"])
    with_clf = 1000 * (m["draft_rate"] * draft + d * classify)
    without = 1000 * (m["draft_rate"] * draft + m["draft_rate"] * classify)     # every ticket that gets a draft is also LLM-classified
    return {"with_classifier": round(with_clf, 4), "without_classifier": round(without, 4), "saving": round(without - with_clf, 4),
            "saving_share": round((without - with_clf) / without, 3) if without else 0.0, "deferred_rate": round(d, 3)}


def main(argv=None) -> int:
    offline_path, live_path = (Path(a) for a in (argv or sys.argv[1:3]))
    off, live = json.loads(offline_path.read_text()), json.loads(live_path.read_text())
    calls, m = per_call(live), mix(off)
    out = {"per_call_tokens_measured": {k: [round(a), round(b)] for k, (a, b) in calls.items()}, "mix_measured": {k: round(v, 3) for k, v in m.items()},
           "price_per_m_tokens_placeholder": config.TOKEN_PRICE_PER_M, "estimate": estimate(calls, m),
           "sensitivity": {f"deferred={r:.2f}": estimate(calls, m, r) for r in (0.05, 0.10, 0.20, 0.30)}}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
