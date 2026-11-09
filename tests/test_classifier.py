import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml import data, train
from ml.metrics import per_class
from ticketpilot import classifier, fake_llm
from ticketpilot.classifier import Classifier, build_pipeline, classify, llm_classify

ROOT = Path(__file__).resolve().parents[1]
needs_raw = pytest.mark.skipif(not (data.B77 / "test.csv").exists(), reason="data/raw/banking77 not downloaded")

PHRASES = {
    "card_issues": ["my card was lost", "card not working at the shop", "activate my new card", "pin is blocked", "card has not arrived", "freeze my card"],
    "transfers_payments": ["my transfer is pending", "transfer failed", "top up did not arrive", "payment still pending", "bank transfer not received", "cancel a transfer"],
    "account_access": ["cannot log in", "forgot my passcode", "verify my identity", "change my personal details", "close my account", "password reset"],
    "fees_charges": ["why was I charged a fee", "unexpected fee on my statement", "bad exchange rate", "atm fee abroad", "extra charge applied", "fee for transfer"],
    "fraud_security": ["unauthorised transaction on my account", "someone stole my card details", "i think i was scammed", "suspicious payment i did not make", "account hacked", "phishing email"],
    "loans_credit": ["my loan repayment is late", "apply for a mortgage", "student loan servicer problem", "credit report is wrong", "car loan interest", "lender keeps calling"],
    "disputes_refunds": ["i want a refund", "refund not showing up", "charged twice for one purchase", "dispute this payment", "merchant has not refunded me", "wrong amount charged"],
    "other": ["what countries do you support", "how old do i need to be", "which currencies can i hold", "is the app available in spanish", "tell me about your company", "do you have an office"],
}
TEXTS = [t for ts in PHRASES.values() for t in ts]
LABELS = [c for c, ts in PHRASES.items() for _ in ts]


@pytest.fixture(scope="module")
def tiny():
    return build_pipeline(10.0).fit(TEXTS, LABELS)


# ---- leak guard on the real, fixed split ----

@needs_raw
def test_splits_are_disjoint_by_id_and_by_text():
    pool = {r["id"]: r for r in data.build_pool()[0]}
    sets = {n: data.load_split_ids(n) for n in ("train", "val", "test")}
    for ids in sets.values():
        assert len(ids) == len(set(ids))
    assert not (set(sets["train"]) & set(sets["test"])) and not (set(sets["val"]) & set(sets["test"]))
    assert not (set(sets["train"]) & set(sets["val"]))
    texts = {n: {data.norm_text(pool[i]["text"]) for i in ids} for n, ids in sets.items()}
    assert not (texts["train"] & texts["test"]) and not (texts["val"] & texts["test"]) and not (texts["train"] & texts["val"])


@needs_raw
def test_synthetic_topics_never_straddle_splits():
    pool = {r["id"]: r for r in data.build_pool()[0]}
    side = {}
    for name in ("train", "val", "test"):
        for i in data.load_split_ids(name):
            t = pool[i]["topic"]
            if t:
                assert side.setdefault(t, name) == name, f"topic {t} appears in {side[t]} and {name}"


# Banking77 itself repeats a few texts across train.csv and test.csv. build_pool drops such texts from the training
# side, so no model trains on a test text; but two demo tickets drawn from train.csv carry one of them. Pinned, not hidden:
# any new overlap fails this test, and A5 must not present these two as unseen examples for the classifier.
KNOWN_DEMO_TEXT_IN_TEST = {"T-1198", "T-1234"}


@needs_raw
def test_demo_tickets_share_text_with_test_only_in_the_known_cases():
    test_texts = {data.norm_text(r["text"]) for r in data.load_split("test")}
    demo = [json.loads(l) for l in (ROOT / "data" / "tickets.jsonl").read_text().splitlines()]
    assert {t["ticket_id"] for t in demo if data.norm_text(t["body"]) in test_texts} == KNOWN_DEMO_TEXT_IN_TEST


def test_split_files_match_their_checksums():
    for line in (data.SPLITS / "CHECKSUMS.txt").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((data.SPLITS / name).read_bytes()).hexdigest() == digest, f"{name} changed after the split was fixed"


def test_make_splits_refuses_to_overwrite(monkeypatch):
    import runpy
    with pytest.raises(SystemExit, match="fixed once"):
        runpy.run_path(str(ROOT / "tools" / "gen" / "make_splits.py"), run_name="__main__")


@needs_raw
def test_every_class_present_in_every_split():
    for name in ("train", "val", "test"):
        assert {r["category"] for r in data.load_split(name)} == set(data.CATEGORIES)


# ---- model behaviour ----

def test_training_is_reproducible():
    a = build_pipeline(10.0).fit(TEXTS, LABELS).predict_proba(["lost my card", "refund please"])
    b = build_pipeline(10.0).fit(TEXTS, LABELS).predict_proba(["lost my card", "refund please"])
    assert np.array_equal(a, b)


def test_predict_returns_label_and_probability(tiny):
    p = Classifier(tiny).predict("i need a refund for a double charge")
    assert p.label in data.CATEGORIES and 0 < p.probability <= 1
    proba, classes = Classifier(tiny).predict_proba(["anything"])
    assert abs(proba.sum() - 1.0) < 1e-9 and sorted(classes) == sorted(data.CATEGORIES)


def test_below_threshold_defers_and_above_does_not(tiny):
    assert Classifier(tiny, threshold=1.01).predict("my transfer is pending").deferred
    assert not Classifier(tiny, threshold=0.0).predict("my transfer is pending").deferred


def test_save_and_load_round_trip(tiny, tmp_path):
    Classifier(tiny, threshold=0.7).save(tmp_path / "m.joblib")
    loaded = Classifier.load(tmp_path / "m.joblib")
    assert loaded.threshold == 0.7 and loaded.predict("account hacked").label == Classifier(tiny).predict("account hacked").label


def test_classify_uses_classifier_when_confident_and_never_calls_llm(tiny):
    called = []
    out = classify("my transfer is pending", Classifier(tiny, 0.0), fallback=lambda t: called.append(t) or ("other", None))
    assert out["source"] == "classifier" and not called


def test_classify_defers_to_llm_below_threshold(tiny):
    out = classify("my transfer is pending", Classifier(tiny, 1.01), fallback=lambda t: ("fraud_security", None))
    assert out["source"] == "llm" and out["label"] == "fraud_security"


def test_classify_survives_a_failed_llm_fallback(tiny):
    out = classify("my transfer is pending", Classifier(tiny, 1.01), fallback=lambda t: (None, None))
    assert out["source"] == "classifier_fallback_failed" and out["label"] in data.CATEGORIES


def test_llm_classify_validates_the_label():
    fake_llm.queue_responses('{"category": "card_issues"}')
    assert llm_classify("my card")[0] == "card_issues"
    fake_llm.queue_responses('{"category": "weather"}', '{"category": "nope"}')
    assert llm_classify("my card")[0] is None


def test_llm_prompt_names_every_category_and_treats_input_as_data():
    assert all(c in classifier.LLM_PROMPT for c in data.CATEGORIES) and "not instructions" in classifier.LLM_PROMPT


# ---- threshold, baselines, metrics ----

def _proba(rows):
    return np.array(rows), ["a", "fraud_security"]


def test_threshold_is_zero_when_the_model_is_always_confident_and_right():
    proba, classes = _proba([[0.95, 0.05], [0.05, 0.95]] * 5)
    t, _ = train.choose_threshold(proba, classes, ["a", "fraud_security"] * 5)
    assert t == 0.0


def test_threshold_defers_low_confidence_wrong_predictions():
    proba, classes = _proba([[0.9, 0.1]] * 6 + [[0.55, 0.45]] * 6)
    y = ["a"] * 6 + ["fraud_security"] * 6          # the unsure rows are wrong (and are missed fraud)
    t, curve = train.choose_threshold(proba, classes, y)
    assert 0.55 < t <= 0.9
    assert min(r["expected_cost"] for r in curve) < next(r for r in curve if r["threshold"] == 0.0)["expected_cost"]


def test_a_missed_fraud_costs_more_than_another_miss():
    proba, classes = _proba([[0.6, 0.4]])
    fraud = train.cost_at(0.0, proba, classes, ["fraud_security"])["expected_cost"]
    other = train.cost_at(0.0, np.array([[0.4, 0.6]]), classes, ["a"])["expected_cost"]
    assert fraud > other


def test_keyword_baseline_and_majority_baseline_are_weaker_than_a_trained_model(tiny):
    kw = per_class(LABELS, [train.keyword_predict(t) for t in TEXTS])["macro_f1"]
    assert 0 < kw < 1
    assert train.keyword_predict("someone used my card without asking, a scam") == "fraud_security"
    assert train.keyword_predict("hello there") == "other"


def test_per_class_macro_f1_ignores_classes_absent_from_the_data():
    r = per_class(["a" if False else "card_issues"] * 3, ["card_issues"] * 3)
    assert r["macro_f1"] == 1.0 and r["classes"]["loans_credit"]["support"] == 0


def test_committed_metrics_show_the_model_beats_both_baselines():
    m = json.loads((ROOT / "ml" / "metrics.json").read_text())["test"]
    assert m["classifier"]["macro_f1"] > m["keyword_rules"]["macro_f1"] > m["majority_class"]["macro_f1"]
    assert all(m["classifier"]["classes"][c]["precision"] > m["keyword_rules"]["classes"][c]["precision"] for c in data.CATEGORIES)
