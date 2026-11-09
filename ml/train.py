"""Train and evaluate the ticket classifier (spec-a4 'Classifier').

Tuning (C) and the deferral threshold use train and validation only. The test set is read once, at the end.
Writes ml/artifacts/model.joblib (gitignored), ml/metrics.json and docs/a4-classifier-metrics.md.
Usage: python ml/train.py
"""
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data import CATEGORIES, load_split  # noqa: E402
from ml.metrics import confusion_md, per_class, table  # noqa: E402
from ticketpilot.classifier import Classifier, build_pipeline  # noqa: E402

C_GRID = [1.0, 3.0, 10.0, 30.0]
THRESHOLD_GRID = [round(x * 0.05, 2) for x in range(0, 20)]
COST_WRONG = 1.0
COST_MISSED_FRAUD = 5.0   # a missed fraud_security ticket is worse than any other mistake (docs/domain.md)
COST_DEFER = 0.3          # one LLM call plus latency; assumes the LLM resolves a deferred ticket correctly (checked in compare_llm.py)

# Keyword baseline: first match wins, in this order. Written before any test result was seen.
RULES = [
    ("fraud_security", r"unauthori[sz]ed|fraud|scam|hack|stolen|compromis|not recogni[sz]e|didn'?t make|phishing|someone (?:else )?(?:used|took)"),
    ("loans_credit", r"\bloans?\b|mortgage|credit (?:score|report|card)|lender|repay|borrow|lease|repossess|student"),
    ("disputes_refunds", r"refund|dispute|chargeback|charged twice|double|wrong amount|money back|reverse"),
    ("fees_charges", r"\bfees?\b|charge|exchange rate|commission|cost"),
    ("card_issues", r"\bcard|\bpin\b|\batm\b|contactless|activate|apple pay|google pay"),
    ("transfers_payments", r"transfer|top.?up|payment|pending|deposit|cheque|beneficiary|direct debit|sepa|transaction"),
    ("account_access", r"account|password|passcode|log ?in|verif|identity|personal details|close"),
]


def keyword_predict(text: str) -> str:
    low = text.lower()
    return next((cat for cat, pat in RULES if re.search(pat, low)), "other")


def cost_at(threshold: float, proba: np.ndarray, classes: list[str], y_true: list[str]) -> dict:
    idx = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    cost, deferred, correct_kept, kept = 0.0, 0, 0, 0
    for i, t in enumerate(y_true):
        if conf[i] < threshold:
            cost += COST_DEFER
            deferred += 1
            continue
        kept += 1
        if classes[idx[i]] == t:
            correct_kept += 1
        else:
            cost += COST_MISSED_FRAUD if t == "fraud_security" else COST_WRONG
    n = len(y_true)
    return {"threshold": threshold, "expected_cost": cost / n, "deferred_share": deferred / n,
            "accuracy_on_kept": correct_kept / kept if kept else None}


def choose_threshold(proba: np.ndarray, classes: list[str], y_true: list[str]) -> tuple[float, list[dict]]:
    curve = [cost_at(t, proba, classes, y_true) for t in THRESHOLD_GRID]
    return min(curve, key=lambda r: (r["expected_cost"], r["threshold"]))["threshold"], curve


def tune_C(train: list[dict], val: list[dict]) -> tuple[float, dict]:
    scores = {}
    for C in C_GRID:
        pipe = build_pipeline(C).fit([r["text"] for r in train], [r["category"] for r in train])
        pred = pipe.predict([r["text"] for r in val])
        scores[C] = per_class([r["category"] for r in val], list(pred))["macro_f1"]
    return max(scores, key=lambda c: (scores[c], -c)), scores


def serving_timing(clf: Classifier, texts: list[str]) -> dict:
    lat = []
    cpu0 = time.process_time()
    for t in texts:
        s = time.perf_counter()
        clf.predict(t)
        lat.append((time.perf_counter() - s) * 1000)
    cpu = time.process_time() - cpu0
    return {"n": len(texts), "median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
            "cpu_seconds_per_1000": cpu / len(texts) * 1000}


def main() -> None:
    train, val = load_split("train"), load_split("val")
    best_C, c_scores = tune_C(train, val)
    pipe = build_pipeline(best_C).fit([r["text"] for r in train], [r["category"] for r in train])
    clf = Classifier(pipe)
    proba, classes = clf.predict_proba([r["text"] for r in val])
    threshold, curve = choose_threshold(proba, classes, [r["category"] for r in val])
    clf.threshold = threshold
    clf.save()

    # ---- the test set is read here, once ----
    test = load_split("test")
    y = [r["category"] for r in test]
    texts = [r["text"] for r in test]
    t_proba, _ = clf.predict_proba(texts)
    model_pred = [classes[i] for i in t_proba.argmax(axis=1)]
    majority = Counter(r["category"] for r in train).most_common(1)[0][0]
    results = {
        "classifier": per_class(y, model_pred),
        "keyword_rules": per_class(y, [keyword_predict(t) for t in texts]),
        "majority_class": per_class(y, [majority] * len(y)),
    }
    kept = t_proba.max(axis=1) >= threshold
    deferral = {"threshold": threshold, "deferred_share": float(1 - kept.mean()),
                "accuracy_on_kept": float(np.mean([p == t for p, t, k in zip(model_pred, y, kept) if k])),
                "accuracy_all": results["classifier"]["accuracy"]}

    by_source = {}
    for src in sorted({r["source"] for r in test}):
        rows = [(p, r["category"]) for p, r in zip(model_pred, test) if r["source"] == src]
        by_source[src] = {"n": len(rows), "accuracy": float(np.mean([p == t for p, t in rows])),
                          "predicted": dict(Counter(p for p, _ in rows))}
    sample_texts = [texts[i] for i in range(0, len(texts), max(len(texts) // 300, 1))][:300]
    out = {"best_C": best_C, "val_macro_f1_by_C": c_scores, "majority_class": majority, "threshold": deferral,
           "threshold_curve_val": curve, "test": results, "accuracy_by_source": by_source,
           "serving": serving_timing(clf, sample_texts),
           "split_sizes": {"train": len(train), "val": len(val), "test": len(test)}}
    (ROOT / "ml" / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")

    L = ["# A4 classifier metrics (generated by `ml/train.py`)", "",
         f"Split sizes: {out['split_sizes']}. Best C on validation macro-F1: {best_C} (val macro-F1 by C: "
         + ", ".join(f"{c}: {v:.3f}" for c, v in c_scores.items()) + ").", "",
         "## Test set: per-class precision (P) and recall (R)", "", table(results), "",
         "## Classifier confusion matrix (test; rows are the true class)", "", confusion_md(results["classifier"]["confusion"]), "",
         "## Deferral threshold (chosen on validation)", "",
         f"Costs: wrong = {COST_WRONG}, missed fraud_security = {COST_MISSED_FRAUD}, deferral = {COST_DEFER} (assumes the LLM "
         "resolves a deferred ticket correctly; verified in the LLM comparison).", "",
         "| threshold | expected cost | deferred share | accuracy on kept |", "|---|---|---|---|"]
    L += [f"| {r['threshold']} | {r['expected_cost']:.3f} | {r['deferred_share']:.3f} | "
          f"{'-' if r['accuracy_on_kept'] is None else format(r['accuracy_on_kept'], '.3f')} |" for r in curve]
    L += ["", f"Chosen threshold **{threshold}**. On test: {deferral['deferred_share']:.1%} deferred, accuracy on the "
          f"kept tickets {deferral['accuracy_on_kept']:.3f} (all tickets: {deferral['accuracy_all']:.3f}).", "",
          "## Accuracy by source (test)", "", "| source | n | accuracy | predicted labels |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['accuracy']:.3f} | {', '.join(f'{a}={b}' for a, b in sorted(v['predicted'].items(), key=lambda x: -x[1]))} |"
          for k, v in by_source.items()]
    s = out["serving"]
    L += ["", f"Serving: median {s['median_ms']:.1f} ms, p95 {s['p95_ms']:.1f} ms per ticket, "
          f"{s['cpu_seconds_per_1000']:.2f} CPU-seconds per 1,000 tickets (one ticket per call, {s['n']} tickets)."]
    (ROOT / "docs" / "a4-classifier-metrics.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
