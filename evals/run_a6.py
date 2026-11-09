"""A6 eval runner (spec-a6 'Runner').

  python evals/run_a6.py --label baseline                 full suite, live (needs LLM_PROVIDER key and the mock API on :8000)
  python evals/run_a6.py --label x --offline              only the cases that need no model
  python evals/run_a6.py --label x --only router-01,tools-02 --judge
  python evals/run_a6.py --rerun evals/results/FILE.jsonl --times 3      rerun the failing dev cases, flaky vs consistent
  python evals/run_a6.py --compare BEFORE.jsonl AFTER.jsonl [--out docs/FILE.md]

Held-out cases are scored and counted, but their individual outputs and failures are not printed unless
--reveal-held-out is given (spec-a6: only the aggregate is read until the final report).
"""
import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import graders, harness, judge as judge_mod  # noqa: E402
from evals.core import Case, load_cases  # noqa: E402
from ticketpilot import config  # noqa: E402
from ticketpilot.mcp_client import open_toolbox  # noqa: E402

RESULTS = ROOT / "evals" / "results"


def cost_usd(prompt: int, completion: int) -> float:
    p_in, p_out = config.TOKEN_PRICE_PER_M
    return (prompt * p_in + completion * p_out) / 1e6


INFRA_RETRIES, INFRA_WAIT_S = 3, 65.0   # the provider's limit is per minute: wait out a full window before retrying


def infra_failure(out) -> bool:
    """The run failed because the provider rate-limited it (a 429 that survived the harness's own retries), not because the
    agent did something wrong. Such a run says nothing about the agent, so the runner retries it after a pause."""
    if out.error and "rate_limited" in out.error:
        return True
    if "rate_limited" in out.text:
        return True
    # provider_error is what the SDK reports once Groq's per-day limit has cut the connection retries short
    return any(e.get("error") in ("rate_limited", "provider_error") and e.get("ok") is False for e in out.events) \
        and (out.report or {}).get("stop_reason") == "model_error"


async def run_one(case: Case, factory=open_toolbox, trace_dir: Path | None = None, with_judge: bool = False,
                  wait=time.sleep) -> dict:
    infra = 0
    out = await harness.run_case(case, factory, trace_dir)
    while case.live and infra < INFRA_RETRIES and infra_failure(out):
        infra += 1
        wait(INFRA_WAIT_S)
        out = await harness.run_case(case, factory, trace_dir)
    grades = graders.grade(out, case)
    rec = {"id": case.id, "split": case.split, "component": case.component, "severity": case.severity,
           "passed": all(g.passed for g in grades), "grades": [vars(g) for g in grades], "route": out.route,
           "text": out.text[:3000], "run_id": out.run_id, "error": out.error, "label": out.label,
           "prompt_tokens": out.prompt_tokens, "completion_tokens": out.completion_tokens, "model_calls": out.model_calls,
           "latency_s": out.latency_s, "cost_usd": round(cost_usd(out.prompt_tokens, out.completion_tokens), 5),
           "infra_retries": infra, "infra_failure": infra_failure(out),
           "report": out.report, "ledger": out.ledger, "tool_calls": [{k: v for k, v in c.items() if k != "result"} for c in out.tool_calls],
           "events": out.events, "budget": out.budget}
    if with_judge and out.report and out.ledger:
        v = judge_mod.judge(case.input, out.ledger, out.report)
        rec["judge"] = {"scores": v.scores, "total": v.total, "acceptable": v.acceptable, "error": v.error,
                        "reasoning": v.reasoning}
    return rec


async def run_suite(cases: list[Case], label: str, sleep: float = 3.0, with_judge: bool = False, factory=open_toolbox,
                    show_held_out: bool = False, out_dir: Path | None = None, echo=print) -> list[dict]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = out_dir or RESULTS
    trace_dir = out_dir / "traces" / f"{stamp}-{label}"
    records = []
    for case in cases:
        rec = await run_one(case, factory, trace_dir, with_judge)
        records.append(rec)
        hidden = rec["split"] == "held_out" and not show_held_out
        mark = "?" if hidden else ("PASS" if rec["passed"] else "FAIL")
        note = f"  (provider rate limit: {rec['infra_retries']} retries{', still failing' if rec['infra_failure'] else ''})" if rec["infra_retries"] else ""
        echo(f"[{mark:>4}] {rec['id']:<14} {rec['split']:<8} {rec['latency_s']:>6.1f}s {rec['prompt_tokens'] + rec['completion_tokens']:>6} tok{note}")
        if not rec["passed"] and not hidden:
            for g in rec["grades"]:
                if not g["passed"]:
                    echo(f"         - {g['name']}: {g['detail']}")
        if case.live and sleep:
            time.sleep(sleep)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stamp}-{label}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    echo(f"\nresults: {path}")
    return records


def summarize(records: list[dict]) -> dict:
    def rate(rows):
        return {"passed": sum(r["passed"] for r in rows), "total": len(rows)}

    by_component, by_split, by_grader = defaultdict(list), defaultdict(list), defaultdict(lambda: [0, 0])
    for r in records:
        by_component[r["component"]].append(r)
        by_split[r["split"]].append(r)
        for g in r["grades"]:
            by_grader[g["name"]][1] += 1
            by_grader[g["name"]][0] += g["passed"]
    return {"overall": rate(records), "by_component": {k: rate(v) for k, v in by_component.items()},
            "by_split": {k: rate(v) for k, v in by_split.items()},
            "by_grader": {k: {"passed": v[0], "total": v[1]} for k, v in sorted(by_grader.items())},
            "tokens": sum(r["prompt_tokens"] + r["completion_tokens"] for r in records),
            "cost_usd": round(sum(r["cost_usd"] for r in records), 4),
            "latency_s": round(sum(r["latency_s"] for r in records), 1),
            "infra_failures": [r["id"] for r in records if r.get("infra_failure")]}


def fmt(r: dict) -> str:
    return f"{r['passed']} of {r['total']} ({r['passed'] / r['total']:.0%})" if r["total"] else "0 of 0"


def summary_md(records: list[dict], title: str) -> str:
    s = summarize(records)
    lines = [f"### {title}", "", f"Overall: **{fmt(s['overall'])}**. Dev {fmt(s['by_split'].get('dev', {'passed': 0, 'total': 0}))}, "
             f"held-out {fmt(s['by_split'].get('held_out', {'passed': 0, 'total': 0}))}. "
             f"Tokens {s['tokens']:,}, estimated cost ${s['cost_usd']}, total latency {s['latency_s']} s.", "",
             "| component | passed |", "|---|---|"] + [f"| {k} | {fmt(v)} |" for k, v in s["by_component"].items()]
    if s["infra_failures"]:
        lines += ["", f"**{len(s['infra_failures'])} case(s) still failed on provider rate limits after retries, so they say nothing about the "
                      f"agent:** {s['infra_failures']}"]
    lines += ["", "| grader | passed |", "|---|---|"] + [f"| {k} | {v['passed']} of {v['total']} |" for k, v in s["by_grader"].items()]
    return "\n".join(lines)


def load_results(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


async def rerun(path: str, times: int, sleep: float, include_held_out: bool = False) -> dict:
    """Rerun each failing case `times` more times: all runs fail = consistent, mixed = flaky."""
    prior = load_results(path)
    cases = {c.id: c for c in load_cases()}
    failing = [r["id"] for r in prior if not r["passed"] and (include_held_out or r["split"] == "dev")]
    verdict = {}
    for cid in failing:
        runs = [False]  # the baseline run itself
        for _ in range(times):
            rec = await run_one(cases[cid])
            runs.append(rec["passed"])
            if cases[cid].live and sleep:
                time.sleep(sleep)
        verdict[cid] = {"runs": runs, "kind": "consistent" if not any(runs) else "flaky"}
        print(f"{cid}: {['pass' if x else 'FAIL' for x in runs]} -> {verdict[cid]['kind']}")
    return verdict


def compare(before: list[dict], after: list[dict], reveal: bool = False) -> str:
    b, a = {r["id"]: r for r in before}, {r["id"]: r for r in after}
    sb, sa = summarize(before), summarize(after)
    lines = ["| | before | after |", "|---|---|---|", f"| overall | {fmt(sb['overall'])} | {fmt(sa['overall'])} |"]
    for split in ("dev", "held_out"):
        lines.append(f"| {split} | {fmt(sb['by_split'][split])} | {fmt(sa['by_split'][split])} |")
    for comp in sb["by_component"]:
        lines.append(f"| {comp} | {fmt(sb['by_component'][comp])} | {fmt(sa['by_component'].get(comp, {'passed': 0, 'total': 0}))} |")
    shown = [i for i in b if i in a and (reveal or b[i]["split"] == "dev")]
    fixed = [i for i in shown if not b[i]["passed"] and a[i]["passed"]]
    broke = [i for i in shown if b[i]["passed"] and not a[i]["passed"]]
    hidden_diff = sum(1 for i in b if i in a and b[i]["split"] == "held_out" and b[i]["passed"] != a[i]["passed"]) if not reveal else 0
    lines += ["", f"Fixed (dev): {fixed or 'none'}", f"Regressions (dev): {broke or 'none'}"]
    if not reveal:
        lines.append(f"Held-out cases whose result changed: {hidden_diff} (IDs not shown; use --reveal-held-out in the final report)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--offline", action="store_true", help="only cases that need no model")
    ap.add_argument("--only", help="comma-separated case IDs")
    ap.add_argument("--component")
    ap.add_argument("--skip-component", help="comma-separated components to leave out (to run the suite in daily-quota slices)")
    ap.add_argument("--merge", nargs="+", metavar="RESULTS", help="combine result files (later files win per case) into --out")
    ap.add_argument("--judge", action="store_true", help="also run the LLM judge on coworker reports (advisory)")
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--rerun", metavar="RESULTS")
    ap.add_argument("--times", type=int, default=3)
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--out")
    ap.add_argument("--reveal-held-out", action="store_true")
    args = ap.parse_args(argv)
    if args.compare:
        text = compare(load_results(args.compare[0]), load_results(args.compare[1]), args.reveal_held_out)
        print(text)
        if args.out:
            Path(args.out).write_text(text + "\n")
        return 0
    if args.merge:
        merged = {}
        for f in args.merge:
            merged.update({r["id"]: r for r in load_results(f)})
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in merged.values()))
        print(f"{len(merged)} cases merged into {args.out}")
        return 0
    if args.rerun:
        print(json.dumps(asyncio.run(rerun(args.rerun, args.times, args.sleep, args.reveal_held_out)), indent=2))
        return 0
    cases = load_cases()
    if args.offline:
        cases = [c for c in cases if not c.live]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c.id in wanted]
    if args.component:
        cases = [c for c in cases if c.component == args.component]
    if args.skip_component:
        cases = [c for c in cases if c.component not in args.skip_component.split(",")]
    records = asyncio.run(run_suite(cases, args.label, args.sleep, args.judge, show_held_out=args.reveal_held_out))
    print("\n" + summary_md(records, args.label))
    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
