"""Run the A1 eval questions through the assistant and print pass/fail per case.

Usage: LLM_PROVIDER=groq python evals/run_a1.py   (needs a key in .env; record results in docs/a1-eval.md)
Checks are substring based (case-insensitive); read the answers too, since substrings cannot judge grounding.
"""
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticketpilot.cli import dispatch  # noqa: E402


def main() -> int:
    failures = 0
    for line in (ROOT / "evals" / "a1_questions.jsonl").read_text().splitlines():
        case = json.loads(line)
        out = dispatch(case["q"])
        low = unicodedata.normalize("NFKC", out).lower()  # models emit no-break spaces/hyphens
        missing = [s for s in case["contains"] if s.lower() not in low]
        leaked = [s for s in case["absent"] if s.lower() in low]
        ok = not missing and not leaked
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] #{case['id']} ({case['type']}) {case['q']}")
        if not ok:
            print(f"   missing={missing} unexpected={leaked}")
        print("   " + out.replace("\n", "\n   ")[:600])
    print(f"\n{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
