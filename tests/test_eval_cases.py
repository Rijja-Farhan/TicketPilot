"""The case file: schema, size, split checksum, and that everything a case refers to exists."""
import json
import re

from evals import build_cases
from evals.core import CASES_DIR, COMPONENTS, SPLIT_FILE, load_cases, split_digest
from evals.graders import GRADERS, TRAJECTORY_GRADERS
from ticketpilot import policies, tickets
from ticketpilot.config import CUSTOMERS_PATH

CASES = load_cases()


def test_at_least_30_unique_cases_across_all_components():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)) >= 30
    assert {c.component for c in CASES} == set(COMPONENTS)


def test_every_case_has_graders_that_exist():
    for c in CASES:
        assert c.graders and c.expected.strip(), c.id
        assert all(s["g"] in GRADERS for s in c.graders), c.id


def test_at_least_five_trajectory_checks_and_a_judge_exist():
    assert len(TRAJECTORY_GRADERS) >= 5
    coworker = [c for c in CASES if c.component == "coworker"]
    assert len(coworker) >= 8 and all(any(s["g"] == "trajectory" for s in c.graders) for c in coworker)


def test_at_least_eight_adversarial_cases_and_a_live_offline_mix():
    assert sum(c.adversarial for c in CASES) >= 8
    assert any(c.live for c in CASES) and any(not c.live for c in CASES)


def test_split_is_about_30_percent_held_out_in_every_big_component():
    held = [c for c in CASES if c.split == "held_out"]
    assert 0.25 <= len(held) / len(CASES) <= 0.35
    for comp in ("policy", "coworker", "tools", "router", "safety"):
        assert any(c.split == "held_out" for c in CASES if c.component == comp), comp


def test_split_checksum_matches_the_recorded_one():
    assert SPLIT_FILE.read_text().strip() == split_digest(CASES), "the dev/held-out split changed after it was fixed"


def test_rebuilding_gives_the_same_split():
    rows = build_cases.assign_split(build_cases.CASES)
    assert {r["id"]: r["split"] for r in rows} == {c.id: c.split for c in CASES}


def test_case_files_match_the_builder_exactly():
    for comp in COMPONENTS:
        want = [json.loads(line) for line in (CASES_DIR / f"{comp}.jsonl").read_text().splitlines()]
        assert [r["id"] for r in want] == [r["id"] for r in build_cases.assign_split(build_cases.CASES) if r["component"] == comp]


def test_imported_cases_come_from_their_source_files():
    a1 = {json.loads(line)["q"] for line in (CASES_DIR.parent / "a1_questions.jsonl").read_text().splitlines()}
    a4 = {json.loads(line)["q"] for line in (CASES_DIR.parent / "a4_retrieval.jsonl").read_text().splitlines()}
    for c in CASES:
        if c.source == "a1":
            assert c.input in a1, c.id
        if c.source == "a4" and c.component == "policy":
            assert c.input in a4, c.id


def test_every_referenced_ticket_customer_and_policy_exists():
    customers = {json.loads(line)["customer_id"] for line in CUSTOMERS_PATH.read_text().splitlines()}
    for c in CASES:
        for t in re.findall(r"\bT-\d+\b", c.input):
            if c.id in ("router-a1-03", "coworker-05"):   # deliberately unknown ticket IDs
                continue
            assert tickets.get_ticket(t) is not None, (c.id, t)
        for cust in re.findall(r"\bC-\d+\b", c.input):
            assert cust in customers or cust == "C-99", (c.id, cust)
    titles = {d.title for d in policies.load_docs()}
    for c in CASES:
        for s in c.graders:
            if s["g"] == "citation_valid":
                assert titles, "no policy docs"
