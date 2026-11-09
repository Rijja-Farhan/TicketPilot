"""A6 classifier section (spec-a6 'Classifier section'): held-out report with intervals, the error-cost table, threshold
sensitivity to the fraud-miss cost ratio, and the LLM's measured accuracy on tickets the classifier defers.

Threshold and rules are chosen on VALIDATION only. The test split is read once, for reporting. A4 already read it for the
LLM comparison, so test numbers are not fully untouched (stated in docs/a6-classifier.md).
LLM labels for validation tickets are cached in ml/artifacts/a6_llm_val_cache.jsonl, so reruns are free.
Usage: python ml/a6_classifier.py [--sleep SECONDS]      (LLM_PROVIDER from .env; writes ml/a6_metrics.json, docs/a6-classifier.md)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data import CATEGORIES, load_split  # noqa: E402
from ml.metrics import confusion_md, per_class  # noqa: E402
from ticketpilot.classifier import Classifier, llm_classify  # noqa: E402

CACHE = ROOT / "ml" / "artifacts" / "a6_llm_val_cache.jsonl"
FRAUD = "fraud_security"
# Error costs (docs/domain.md 'Cost of a wrong answer'), in units of "one misrouted ticket".
COST_MISROUTE = 1.0      # any confusion that does not involve fraud_security: a ticket in the wrong queue
COST_FRAUD_FP = 2.0      # routine ticket escalated as fraud: the fraud team spends time
COST_DEFER = 0.3         # one LLM call plus latency
RATIOS = [2.0, 5.0, 10.0, 20.0]   # cost of a missed fraud ticket (false negative); A4 used 5
T_GRID = [round(x * 0.05, 2) for x in range(0, 19)]   # 0.0 to 0.9: the LLM is called only below 0.9
ALERT_GRID = [None, 0.30, 0.20, 0.15, 0.10]   # defer when P(fraud) >= alert although another class won
LLM_BELOW, ALERT_MIN = 0.90, 0.10
N_BOOT, SEED = 1000, 42


def err_cost(pred: str, true: str, ratio: float) -> float:
    if pred == true:
        return 0.0
    if true == FRAUD:
        return ratio
    return COST_FRAUD_FP if pred == FRAUD else COST_MISROUTE


def llm_labels(rows: list[dict], sleep: float) -> dict[str, str | None]:
    cache = {r["id"]: r for r in map(json.loads, CACHE.read_text().splitlines())} if CACHE.exists() else {}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        if row["id"] in cache:
            continue
        label, res = llm_classify(row["text"])
        rec = {"id": row["id"], "label": label, "error": res.error.code if res.error else None,
               "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens}
        with CACHE.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        cache[row["id"]] = rec
        if (i + 1) % 25 == 0:
            print(f"  LLM labelled {i + 1}/{len(rows)}", flush=True)
        time.sleep(sleep)
    return {k: v["label"] for k, v in cache.items()}


def decisions(proba: np.ndarray, classes: list[str], t: float, alert: float | None) -> tuple[np.ndarray, np.ndarray]:
    """(predicted index, deferred mask) for threshold t and the optional fraud alert level."""
    idx, conf = proba.argmax(axis=1), proba.max(axis=1)
    deferred = conf < t
    if alert is not None:
        f = classes.index(FRAUD)
        deferred |= (proba[:, f] >= alert) & (idx != f)
    return idx, deferred


def expected_cost(proba, classes, y, llm, t, alert, ratio, assume_perfect_llm=False, human_cost=None) -> dict:
    """Expected cost per ticket. Deferred tickets go to the LLM (its measured labels) unless `assume_perfect_llm` (A4's
    assumption) or `human_cost` (a review queue: that cost per ticket, assumed always right) is given."""
    idx, deferred = decisions(proba, classes, t, alert)
    total, n_def, wrong_after = 0.0, int(deferred.sum()), 0
    for i, true in enumerate(y):
        if deferred[i] and human_cost is not None:
            pred = true
            total += human_cost
        elif deferred[i]:
            pred = true if assume_perfect_llm else (llm.get(i) or classes[idx[i]])
            total += COST_DEFER + err_cost(pred, true, ratio)
        else:
            pred = classes[idx[i]]
            total += err_cost(pred, true, ratio)
        wrong_after += pred != true
    n = len(y)
    return {"t": t, "alert": alert, "ratio": ratio, "cost": total / n, "deferred_share": n_def / n, "accuracy": 1 - wrong_after / n}


def paired_diff_ci(correct_a: list[bool], correct_b: list[bool]) -> tuple[float, float, float]:
    """Accuracy of b minus a on the same tickets, with a 95% bootstrap interval."""
    rng = np.random.default_rng(SEED)
    a, b = np.array(correct_a, float), np.array(correct_b, float)
    diffs = [(b[s] - a[s]).mean() for s in (rng.integers(0, len(a), len(a)) for _ in range(N_BOOT))]
    return float((b - a).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def bootstrap_ci(y_true: list[str], y_pred: list[str]) -> dict:
    rng = np.random.default_rng(SEED)
    ti = np.array([CATEGORIES.index(c) for c in y_true])
    pi = np.array([CATEGORIES.index(c) for c in y_pred])
    k, n = len(CATEGORIES), len(ti)
    prec, rec = [], []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        cm = np.bincount(ti[s] * k + pi[s], minlength=k * k).reshape(k, k)
        tp = np.diag(cm).astype(float)
        prec.append(tp / np.maximum(cm.sum(axis=0), 1))
        rec.append(tp / np.maximum(cm.sum(axis=1), 1))
    lo = lambda a: np.percentile(a, 2.5, axis=0)  # noqa: E731
    hi = lambda a: np.percentile(a, 97.5, axis=0)  # noqa: E731
    return {c: {"precision_ci": [float(lo(prec)[j]), float(hi(prec)[j])], "recall_ci": [float(lo(rec)[j]), float(hi(rec)[j])]}
            for j, c in enumerate(CATEGORIES)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()
    clf = Classifier.load()
    val = load_split("val")
    y_val = [r["category"] for r in val]
    proba, classes = clf.predict_proba([r["text"] for r in val])
    idx, conf = proba.argmax(axis=1), proba.max(axis=1)
    f = classes.index(FRAUD)
    need = [i for i in range(len(val)) if conf[i] < LLM_BELOW or (proba[i, f] >= ALERT_MIN and idx[i] != f)]
    print(f"{len(val)} validation tickets; {len(need)} need an LLM label (below {LLM_BELOW} or P(fraud) >= {ALERT_MIN})", flush=True)
    labels = llm_labels([val[i] for i in need], args.sleep)
    llm = {i: labels.get(val[i]["id"]) for i in need}
    failed = sum(1 for v in llm.values() if v is None)

    # ---- measured LLM accuracy on the tickets A4's threshold defers ------------------------------------------------
    a4_t = clf.threshold
    deferred_a4 = [i for i in range(len(val)) if conf[i] < a4_t]
    scored = [i for i in deferred_a4 if llm[i] is not None]
    clf_acc = float(np.mean([classes[idx[i]] == y_val[i] for i in deferred_a4]))
    llm_acc = float(np.mean([llm[i] == y_val[i] for i in scored]))
    diff, d_lo, d_hi = paired_diff_ci([classes[idx[i]] == y_val[i] for i in scored], [llm[i] == y_val[i] for i in scored])
    fraud_def = [i for i in deferred_a4 if y_val[i] == FRAUD]
    measured = {"a4_threshold": a4_t, "n_deferred": len(deferred_a4), "n_llm_failed": len(deferred_a4) - len(scored),
                "classifier_accuracy_on_deferred": clf_acc, "llm_accuracy_on_deferred": llm_acc,
                "fraud_among_deferred": len(fraud_def),
                "diff": diff, "diff_ci": [d_lo, d_hi], "llm_better": d_lo > 0}

    # ---- cost curves and sensitivity --------------------------------------------------------------------------------
    grid, best = [], {}
    for ratio in RATIOS:
        rows = [expected_cost(proba, classes, y_val, llm, t, a, ratio) for t in T_GRID for a in ALERT_GRID]
        grid += rows
        no_alert = [r for r in rows if r["alert"] is None]
        best[ratio] = {"threshold_only": min(no_alert, key=lambda r: (r["cost"], r["t"])),
                       "with_fraud_alert": min(rows, key=lambda r: (r["cost"], r["t"], -(r["alert"] or 0))),
                       "never_defer": next(r for r in no_alert if r["t"] == 0.0),
                       "a4_threshold_measured_llm": expected_cost(proba, classes, y_val, llm, a4_t, None, ratio),
                       "perfect_llm": min((expected_cost(proba, classes, y_val, llm, t, None, ratio, True) for t in T_GRID),
                                          key=lambda r: (r["cost"], r["t"])),
                       "human_0.5": min((expected_cost(proba, classes, y_val, llm, t, None, ratio, human_cost=0.5) for t in T_GRID),
                                        key=lambda r: (r["cost"], r["t"])),
                       "human_1.0": min((expected_cost(proba, classes, y_val, llm, t, None, ratio, human_cost=1.0) for t in T_GRID),
                                        key=lambda r: (r["cost"], r["t"]))}
    curve5 = [r for r in grid if r["ratio"] == 5.0 and r["alert"] is None]

    # ---- held-out test report (read for reporting only) --------------------------------------------------------------
    test = load_split("test")
    y_test = [r["category"] for r in test]
    t_proba, _ = clf.predict_proba([r["text"] for r in test])
    t_pred = [classes[i] for i in t_proba.argmax(axis=1)]
    rep = per_class(y_test, t_pred)
    ci = bootstrap_ci(y_test, t_pred)
    out = {"measured_deferred_accuracy": measured, "best_by_ratio": {str(k): v for k, v in best.items()}, "llm_failed_calls": failed,
           "test": {"per_class": rep["classes"], "macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"], "n": rep["n"],
                    "confusion": rep["confusion"], "ci": ci},
           "costs": {"misroute": COST_MISROUTE, "fraud_fp": COST_FRAUD_FP, "defer": COST_DEFER, "ratios": RATIOS}}
    (ROOT / "ml" / "a6_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / "docs" / "a6-classifier.md").write_text(render(out, curve5, len(val), len(need)))
    print((ROOT / "docs" / "a6-classifier.md").read_text())


def render(out: dict, curve5: list[dict], n_val: int, n_llm: int) -> str:
    m, best, test = out["measured_deferred_accuracy"], out["best_by_ratio"], out["test"]
    L = ["# A6 classifier: held-out report, error costs, threshold (generated by `ml/a6_classifier.py`)", "",
         "## 1. Held-out performance (fixed test split, read for reporting only)", "",
         f"n = {test['n']}, macro-F1 {test['macro_f1']:.3f}, accuracy {test['accuracy']:.3f}. Intervals are 95% bootstrap "
         f"percentile intervals ({N_BOOT} resamples of the test tickets, seed {SEED}). Small classes have wide intervals.", "",
         "| class | support | precision | 95% CI | recall | 95% CI | F1 |", "|---|---|---|---|---|---|---|"]
    for c in CATEGORIES:
        r, i = test["per_class"][c], test["ci"][c]
        L.append(f"| {c} | {r['support']} | {r['precision']:.3f} | {i['precision_ci'][0]:.2f} to {i['precision_ci'][1]:.2f} | "
                 f"{r['recall']:.3f} | {i['recall_ci'][0]:.2f} to {i['recall_ci'][1]:.2f} | {r['f1']:.3f} |")
    L += ["", "Confusion matrix (rows are the true class):", "", confusion_md(test["confusion"]), "",
          "**Disclosure:** A4 already read this test set to compare the classifier with the LLM, so these numbers are not fully "
          "untouched. Nothing below was tuned on it: the threshold and the fraud rule are chosen on validation only.", "",
          "## 2. Error costs (units: one misrouted ticket)", "", "| error | who pays | cost |", "|---|---|---|",
          f"| false negative on `fraud_security` (fraud routed as routine) | the customer waits while money may be leaving | R (2, 5, 10 or 20; A4 used 5) |",
          f"| false positive on `fraud_security` (routine escalated as fraud) | the fraud team spends time | {COST_FRAUD_FP} |",
          f"| any other confusion | ticket in the wrong queue, re-triaged | {COST_MISROUTE} |",
          f"| deferral to the LLM | one call plus latency, and the LLM can be wrong too | {COST_DEFER} + the LLM's own error cost |", "",
          "## 3. The A4 gap: how good is the LLM on the tickets the classifier defers?", "",
          f"Measured on the **validation** tickets below A4's threshold ({m['a4_threshold']}): {m['n_deferred']} tickets "
          f"({m['n_llm_failed']} LLM calls failed and are excluded from the LLM figure).", "",
          "| | accuracy on the deferred tickets |", "|---|---|",
          f"| classifier's own answer | {m['classifier_accuracy_on_deferred']:.3f} |",
          f"| LLM (`LLM_PROMPT`, temperature 0) | {m['llm_accuracy_on_deferred']:.3f} |", "",
          f"Difference (LLM minus classifier): {m['diff']:+.3f}, 95% bootstrap interval {m['diff_ci'][0]:+.3f} to {m['diff_ci'][1]:+.3f} "
          f"on {m['n_deferred'] - m['n_llm_failed']} paired tickets. {m['fraud_among_deferred']} of them are true `fraud_security`.", "",
          ("The LLM is better than the classifier on the tickets the classifier is unsure about (the interval excludes zero)."
           if m["llm_better"] else
           "**The LLM is not shown to be better than the classifier on these tickets** (the interval includes zero), and it is far from "
           "the 'always right' assumption A4 used. Sending low-confidence tickets to the LLM buys no measurable accuracy; a "
           "human-review queue is the better destination (recommended, not built: out of scope)."), "",
          "## 4. Threshold choice on validation, and how it moves with the cost ratio", "",
          f"Expected cost per ticket over the {n_val} validation tickets, using the measured LLM labels for the {n_llm} tickets that "
          "can be deferred (below 0.90, or with P(fraud) of at least 0.10).", "",
          "| missed-fraud cost R | never defer | best threshold | cost at best | deferred share | A4 threshold, measured LLM | "
          "if the LLM were perfect (A4's assumption): best t, cost |", "|---|---|---|---|---|---|---|"]
    for r, b in best.items():
        L.append(f"| {r} | {b['never_defer']['cost']:.3f} | {b['threshold_only']['t']} | {b['threshold_only']['cost']:.3f} | "
                 f"{b['threshold_only']['deferred_share']:.1%} | {b['a4_threshold_measured_llm']['cost']:.3f} | {b['perfect_llm']['t']}, {b['perfect_llm']['cost']:.3f} |")
    L += ["", "If deferred tickets went to a human review queue instead (a human assumed always right; cost per ticket is an "
          "assumption, not a measurement):", "", "| R | review cost 0.5: best t, cost, deferred share | review cost 1.0: best t, cost, deferred share |",
          "|---|---|---|"]
    for r, b in best.items():
        h5, h1 = b["human_0.5"], b["human_1.0"]
        L.append(f"| {r} | {h5['t']}, {h5['cost']:.3f}, {h5['deferred_share']:.1%} | {h1['t']}, {h1['cost']:.3f}, {h1['deferred_share']:.1%} |")
    L += ["", "Cost curve at R = 5 (threshold only):", "", "| threshold | expected cost | deferred share | accuracy after deferral |", "|---|---|---|---|"]
    L += [f"| {r['t']} | {r['cost']:.3f} | {r['deferred_share']:.1%} | {r['accuracy']:.3f} |" for r in curve5]
    L += ["", "### Per-class view: a lower alert level for fraud", "",
          "Rule: also defer when P(`fraud_security`) is at least the alert level although another class won, so a likely fraud ticket "
          "is never silently routed as routine.", "", "| R | best threshold, no alert | cost | best with alert | alert | threshold | cost |",
          "|---|---|---|---|---|---|---|"]
    for r, b in best.items():
        a, t = b["with_fraud_alert"], b["threshold_only"]
        L.append(f"| {r} | {t['t']} | {t['cost']:.3f} | | {a['alert']} | {a['t']} | {a['cost']:.3f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
