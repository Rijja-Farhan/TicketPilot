"""A scripted stand-in for the LLM so the whole ambient pipeline runs offline (spec-a7 'Tests').

Normal mode returns a harmless draft. `obedient=True` plays a model that has been fully fooled: whenever the ticket asks for money
or account actions (or the text looks like an injection) it promises a refund and proposes denied actions. The policy layer must
hold even then; that is the point of the red-team run."""
import json
import re
from contextlib import contextmanager

from ticketpilot import llm
from ticketpilot.ambient import guard

KEYWORDS = {"card_issues": r"card|pin", "transfers_payments": r"transfer|top.?up|payment|deposit", "account_access": r"log ?in|passcode|identity|password",
            "fees_charges": r"fee|charge|exchange rate", "fraud_security": r"fraud|scam|stolen|not recognis", "disputes_refunds": r"refund|dispute|twice"}
BAIT = re.compile(r"refund|reembols|credit my account|freeze|close the account|approved|waive|reimburse|pay no attention", re.I)


def label_for(text: str) -> str:
    return next((c for c, rx in KEYWORDS.items() if re.search(rx, text, re.I)), "other")


class StubModel:
    def __init__(self, obedient: bool = False):
        self.obedient, self.calls = obedient, 0

    def __call__(self, messages, *, schema=None, tools=None, **_):
        self.calls += 1
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        name = schema.__name__ if schema else ""
        if name == "LabelOut":
            out = {"category": label_for(user)}
        elif name == "DraftOut":
            baited = self.obedient and (BAIT.search(user) or guard.scan(user))
            out = ({"reply": "We have refunded 500 pounds to your account and froze the other account as requested.",
                    "proposed_actions": ["issue_refund", "freeze_account", "add_ticket_note", "send_reply", "wire_transfer"]}
                   if baited else {"reply": "Thanks for getting in touch. We are looking into this and a colleague will follow up.",
                                   "proposed_actions": []})
        else:
            out = {}
        text = json.dumps(out)
        res = llm.LLMResult(text=text, provider="fake", model="stub", prompt_tokens=sum(len(m.get("content") or "") for m in messages) // 4,
                            completion_tokens=len(text) // 4)
        res.parsed = schema.model_validate(out) if schema else None
        return res


@contextmanager
def patched(obedient: bool = False):
    stub, real = StubModel(obedient), llm.complete
    llm.complete = stub
    try:
        yield stub
    finally:
        llm.complete = real
