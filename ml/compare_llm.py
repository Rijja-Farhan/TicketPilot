"""Classifier vs LLM on the same held-out tickets (spec-a4 'Classifier vs LLM comparison').

Sample: 200 stratified test tickets, fixed seed, IDs saved in ml/compare_sample_ids.json. LLM responses are cached on disk
(ml/artifacts/llm_cache.jsonl) so reruns are free. Consistency: the first 50 tickets are classified a second time.
Usage: python ml/compare_llm.py [--sleep SECONDS]      (uses LLM_PROVIDER from .env)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data import CATEGORIES, load_split  # noqa: E402
from ml.metrics import per_class, table  # noqa: E402
from ticketpilot.classifier import Classifier, llm_classify  # noqa: E402

SAMPLE_IDS = ROOT / "ml" / "compare_sample_ids.json"
CACHE = ROOT / "ml" / "artifacts" / "llm_cache.jsonl"
SEED, N_SAMPLE, MIN_PER_CLASS, N_CONSISTENCY = 42, 200, 10, 50
# Groq model page https://console.groq.com/docs/model/openai/gpt-oss-120b, read 2026-11-01 (USD per 1M tokens).
PRICE_IN, PRICE_OUT = 0.15, 0.60
PRICE_SOURCE = "https://console.groq.com/docs/model/openai/gpt-oss-120b (read 2026-11-01)"


def stratified_ids(test: list[dict]) -> list[str]:
    rng = random.Random(SEED)
    by_cat = {c: sorted(r["id"] for r in test if r["category"] == c) for c in CATEGORIES}
    n = len(test)
    take = {c: min(len(ids), MIN_PER_CLASS) for c, ids in by_cat.items()}
    spare = N_SAMPLE - sum(take.values())
    for c, ids in by_cat.items():
        take[c] = min(len(ids), take[c] + round(spare * len(ids) / n))
    while sum(take.values()) > N_SAMPLE:
        take[max(take, key=take.get)] -= 1
    while sum(take.values()) < N_SAMPLE:
        take[max(by_cat, key=lambda c: len(by_cat[c]) - take[c])] += 1
    return sorted(i for c, ids in by_cat.items() for i in rng.sample(ids, take[c]))


def load_cache() -> dict:
    if not CACHE.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, CACHE.read_text().splitlines())}


def call(row: dict, run: int, cache: dict, sleep: float) -> dict:
    key = f"{row['id']}|run{run}"
    if key in cache:
        return cache[key]
    label, res = llm_classify(row["text"])
    rec = {"key": key, "id": row["id"], "run": run, "label": label, "model": res.model,
           "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
           "latency_ms": res.latency_ms, "error": res.error.code if res.error else None}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    cache[key] = rec
    time.sleep(sleep)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()
    test = load_split("test")
    if not SAMPLE_IDS.exists():
        SAMPLE_IDS.write_text(json.dumps(stratified_ids(test)) + "\n")
    ids = json.loads(SAMPLE_IDS.read_text())
    by_id = {r["id"]: r for r in test}
    sample = [by_id[i] for i in ids]
    clf = Classifier.load()
    cache = load_cache()

    run1 = [call(r, 1, cache, args.sleep) for r in sample]
    run2 = [call(r, 2, cache, args.sleep) for r in sample[:N_CONSISTENCY]]
    ok = [(r, c) for r, c in zip(sample, run1) if c["label"]]
    failures = len(sample) - len(ok)

    y = [r["category"] for r in sample]
    preds = clf.predict_many([r["text"] for r in sample])
    clf_pred = [p.label for p in preds]
    llm_pred = [c["label"] for r, c in ok]
    results = {"classifier (200)": per_class(y, clf_pred),
               "LLM (200)": per_class([r["category"] for r, _ in ok], llm_pred)}
    full = json.loads((ROOT / "ml" / "metrics.json").read_text())
    results["classifier (full test)"] = full["test"]["classifier"]

    # hybrid: classifier first, LLM below the threshold (falls back to the classifier if the LLM call failed)
    hybrid = [c["label"] if (p.deferred and c["label"]) else p.label for p, c in zip(preds, run1)]
    deferred = [i for i, p in enumerate(preds) if p.deferred]
    results["classifier + LLM fallback (200)"] = per_class(y, hybrid)
    on_def = lambda labels: float(np.mean([labels[i] == y[i] for i in deferred])) if deferred else None  # noqa: E731
    llm_by_i = [c["label"] for c in run1]

    toks_in = float(np.mean([c["prompt_tokens"] for _, c in ok]))
    toks_out = float(np.mean([c["completion_tokens"] for _, c in ok]))
    llm_cost = (toks_in * PRICE_IN + toks_out * PRICE_OUT) / 1e6 * 1000
    lat = [c["latency_ms"] for _, c in ok]
    same = [a["label"] == b["label"] for a, b in zip(run1, run2) if a["label"] and b["label"]]
    cost = {"llm_cost_per_1000_usd": llm_cost, "hybrid_cost_per_1000_usd": llm_cost * len(deferred) / len(sample),
            "llm_mean_prompt_tokens": toks_in, "llm_mean_completion_tokens": toks_out,
            "llm_latency_median_ms": float(np.median(lat)), "llm_latency_p95_ms": float(np.percentile(lat, 95)),
            "clf_latency_median_ms": full["serving"]["median_ms"], "clf_latency_p95_ms": full["serving"]["p95_ms"],
            "clf_cpu_seconds_per_1000": full["serving"]["cpu_seconds_per_1000"],
            "consistency_identical": sum(same), "consistency_n": len(same), "llm_failures": failures,
            "deferred_n": len(deferred), "deferred_share": len(deferred) / len(sample),
            "accuracy_on_deferred_classifier": on_def(clf_pred), "accuracy_on_deferred_llm": on_def(llm_by_i),
            "model": run1[0]["model"], "price_source": PRICE_SOURCE}
    out = {"cost_latency": cost, "results": {k: {kk: vv for kk, vv in v.items() if kk != "confusion"} for k, v in results.items()}}
    (ROOT / "ml" / "llm_comparison.json").write_text(json.dumps(out, indent=2) + "\n")

    L = ["# A4 classifier vs LLM (generated by `ml/compare_llm.py`)", "",
         f"Sample: {len(sample)} stratified test tickets (IDs in `ml/compare_sample_ids.json`), LLM `{cost['model']}` at "
         f"temperature 0, {failures} failed calls. Classes with fewer than {MIN_PER_CLASS} test rows are taken whole; "
         "small supports make per-class numbers noisy.", "",
         table({k: results[k] for k in ("classifier (200)", "LLM (200)", "classifier + LLM fallback (200)")}), "",
         "## Cost, latency, consistency", "",
         "| | classifier | LLM | classifier + LLM fallback |", "|---|---|---|---|",
         f"| cost per 1,000 tickets | {cost['clf_cpu_seconds_per_1000']:.2f} CPU-seconds (no per-call fee) | "
         f"${cost['llm_cost_per_1000_usd']:.3f} | ${cost['hybrid_cost_per_1000_usd']:.3f} |",
         f"| latency median / p95 per ticket | {cost['clf_latency_median_ms']:.1f} / {cost['clf_latency_p95_ms']:.1f} ms | "
         f"{cost['llm_latency_median_ms']:.0f} / {cost['llm_latency_p95_ms']:.0f} ms | "
         "LLM latency on deferred tickets only |",
         f"| macro-F1 (200) | {results['classifier (200)']['macro_f1']:.3f} | {results['LLM (200)']['macro_f1']:.3f} | "
         f"{results['classifier + LLM fallback (200)']['macro_f1']:.3f} |",
         f"| accuracy (200) | {results['classifier (200)']['accuracy']:.3f} | {results['LLM (200)']['accuracy']:.3f} | "
         f"{results['classifier + LLM fallback (200)']['accuracy']:.3f} |", "",
         f"- LLM tokens per ticket: {toks_in:.0f} in, {toks_out:.0f} out. Price: ${PRICE_IN}/1M input, ${PRICE_OUT}/1M output "
         f"({PRICE_SOURCE}). Completion tokens include any reasoning tokens the model reports.",
         f"- LLM latency includes any 429 backoff waits, so the p95 is an upper bound for a quiet account.",
         f"- Consistency: the LLM gave the identical label on {sum(same)} of {len(same)} tickets classified twice.",
         f"- Deferred by the classifier at threshold {full['threshold']['threshold']}: {len(deferred)} of {len(sample)} "
         f"({len(deferred) / len(sample):.1%}). On those tickets: classifier accuracy "
         f"{cost['accuracy_on_deferred_classifier']}, LLM accuracy {cost['accuracy_on_deferred_llm']} "
         "(the threshold choice assumed the LLM would be right on them).",
         f"- Full test set classifier (n={full['split_sizes']['test']}): macro-F1 {full['test']['classifier']['macro_f1']:.3f}."]
    (ROOT / "docs" / "a4-llm-comparison.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
