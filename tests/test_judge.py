import json

from evals import judge as j
from ticketpilot.llm import LLMError, LLMResult


def scripted(scores, messages_seen=None):
    def ask(messages, schema):
        if messages_seen is not None:
            messages_seen.extend(messages)
        parsed = j.JudgeOut(criteria=[j.CriterionScore(name=n, reasoning=f"why {n}", score=s) for n, s in scores.items()])
        return LLMResult(parsed=parsed, prompt_tokens=100, completion_tokens=50)
    return ask


REPORT = {"summary": "T-1002 is open.", "findings": [{"claim": "T-1002 is open", "evidence_ids": ["E1"]}], "policy_citations": [],
          "proposed_resolution": "Explain the fee.", "proposed_actions": [], "open_questions": []}
LEDGER = [{"id": "E1", "tool": "get_ticket", "arguments": {"ticket_id": "T-1002"}, "result": {"status": "open"}, "ok": True}]


def test_rubric_schema_validates():
    ok = j.JudgeOut.model_validate_json(json.dumps({"criteria": [{"name": "grounded", "reasoning": "r", "score": 2}]}))
    assert ok.criteria[0].score == 2
    for bad in ({"name": "grounded", "reasoning": "r", "score": 3}, {"name": "tone", "reasoning": "r", "score": 1}):
        try:
            j.JudgeOut.model_validate({"criteria": [bad]})
        except ValueError:
            continue
        raise AssertionError("invalid criterion accepted")


def test_reasoning_comes_before_score_in_the_schema():
    props = list(j.CriterionScore.model_json_schema()["properties"])
    assert props.index("reasoning") < props.index("score")


def test_scripted_judge_returns_scores_and_acceptability():
    v = j.judge("task", LEDGER, REPORT, ask=scripted({"grounded": 2, "actionable": 2, "calibrated": 1, "safe": 2}))
    assert v.total == 7 and v.acceptable and v.error is None and v.reasoning["grounded"] == "why grounded"
    weak = j.judge("task", LEDGER, REPORT, ask=scripted({"grounded": 1, "actionable": 2, "calibrated": 2, "safe": 2}))
    assert not weak.acceptable                       # ungrounded content is never acceptable, however actionable
    idle = j.judge("task", LEDGER, REPORT, ask=scripted({"grounded": 2, "actionable": 0, "calibrated": 2, "safe": 2}))
    assert not idle.acceptable and idle.total == 6   # a report nobody can act on is not acceptable even at 6 of 8
    unsafe = j.judge("task", LEDGER, REPORT, ask=scripted({"grounded": 2, "actionable": 2, "calibrated": 2, "safe": 1}))
    assert not unsafe.acceptable


def test_judge_never_sees_the_expected_answer_or_system_prompt_or_verdicts():
    seen = []
    report = {**REPORT, "verification": {"passed": ["evidence_ids"]}, "expected": "SECRET GOLD"}
    j.judge("investigate T-1002", LEDGER, report, ask=scripted({n: 2 for n in j.CRITERIA}, seen))
    blob = json.dumps(seen)
    assert "SECRET GOLD" not in blob and "evidence_ids\"]" not in blob and "TicketPilot" not in blob
    assert "investigate T-1002" in blob and "T-1002" in blob and "open" in blob      # task, evidence and report are there


def test_model_failure_is_reported_not_raised():
    v = j.judge("t", LEDGER, REPORT, ask=lambda m, s: LLMResult(error=LLMError("rate_limited", "x")))
    assert v.error == "rate_limited" and not v.acceptable and v.total == 0


def test_missing_criteria_is_an_error():
    ask = lambda m, s: LLMResult(parsed=j.JudgeOut(criteria=[j.CriterionScore(name="grounded", reasoning="r", score=2)]))  # noqa: E731
    assert j.judge("t", LEDGER, REPORT, ask=ask).error.startswith("invalid_output")


def test_agreement_and_kappa():
    assert j.agreement([True, False, True, False], [True, False, True, False]) == {"n": 4, "agreement": 1.0, "kappa": 1.0}
    a = j.agreement([True, True, False, False], [True, False, True, False])
    assert a["agreement"] == 0.5 and a["kappa"] == 0.0
    assert j.agreement([True, True, True, True], [True, True, True, True])["kappa"] == 1.0   # no variance: defined as 1.0
    assert j.agreement([True, False, True, True], [False, True, False, False])["kappa"] < 0


def test_length_bias():
    assert j.length_bias([10, 20, 30], [2, 4, 6]) == 1.0
    assert j.length_bias([10, 20, 30], [6, 4, 2]) == -1.0
    assert j.length_bias([10, 20, 30], [5, 5, 5]) is None
