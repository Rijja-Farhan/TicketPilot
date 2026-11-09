"""Generate synthetic customer messages for the classifier (spec-a4 'Classifier').

Why: Banking77 has no loan or credit intents, and the CFPB narratives could not be obtained (the public API and the CSV
export no longer carry the complaint text). So `loans_credit` messages are written by the LLM from real CFPB topics
(product, issue, sub-issue), in the short informal style of Banking77, to keep style from being the label.
`fees_charges` gets a small LLM-written control set so that "LLM-written" is not a perfect proxy for `loans_credit`.
Output: data/synth/{loans_credit,fees_control}.jsonl, committed because LLM output is not reproducible from a seed.
Usage: python tools/gen/make_synth.py   (uses LLM_PROVIDER from .env; ~40 calls)
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ticketpilot import llm  # noqa: E402

CFPB = ROOT / "data" / "raw" / "cfpb" / "complaints_sample.jsonl"
OUT = ROOT / "data" / "synth"
LOAN_PRODUCTS = {"Mortgage", "Student loan", "Vehicle loan or lease",
                 "Payday loan, title loan, personal loan, or advance loan"}
N_LOAN_TOPICS = 30
PER_TOPIC = 15
FEE_TOPICS = ["an unexpected fee on a card payment", "a worse exchange rate than the app showed",
              "a fee for withdrawing cash at an ATM abroad", "a monthly account fee they did not expect",
              "a fee for an international transfer", "a fee charged twice on the statement",
              "interest or charges they did not understand", "a fee for ordering a replacement card"]

PROMPT = """Write {n} different short messages that a bank customer might send to customer support about: {topic}.
Style: informal, one or two sentences, like "I still haven't received my card" or "why was I charged a fee for this?".
Vary the wording and the situation. No names, no account or card numbers, no greetings or sign-offs. Return JSON: {{"messages": [...]}}"""


class Messages(BaseModel):
    messages: list[str]


def loan_topics() -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in CFPB.read_text().splitlines()]
    c = Counter((r["product"], r["issue"], r.get("sub_issue") or "") for r in rows if r["product"] in LOAN_PRODUCTS)
    return [(p, f"{p.split(',')[0].lower()}: {i}" + (f" ({s})" if s else "")) for (p, i, s), _ in c.most_common(N_LOAN_TOPICS)]


def generate(topic: str) -> list[str]:
    res = llm.complete([{"role": "user", "content": PROMPT.format(n=PER_TOPIC, topic=topic)}], schema=Messages)
    if not res.ok:
        raise SystemExit(f"generation failed for {topic!r}: {res.error.code} {res.error.message[:200]}")
    return [m.strip() for m in res.parsed.messages if 8 <= len(m.strip()) <= 300]


def write(path: Path, prefix: str, category: str, topics: list[str]) -> int:
    seen: set[str] = set()
    n = 0
    with path.open("w") as f:
        for t_idx, topic in enumerate(topics):
            for msg in generate(topic):
                key = " ".join(re.findall(r"[a-z0-9]+", msg.lower()))
                if key in seen:
                    continue
                seen.add(key)
                f.write(json.dumps({"id": f"{prefix}-{n}", "text": msg, "category": category,
                                    "topic": f"{prefix}-t{t_idx}", "topic_text": topic}) + "\n")
                n += 1
    return n


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loans_credit:", write(OUT / "loans_credit.jsonl", "synth-loans", "loans_credit", [t for _, t in loan_topics()]))
    print("fees control:", write(OUT / "fees_control.jsonl", "synth-fees", "fees_charges", FEE_TOPICS))


if __name__ == "__main__":
    main()
