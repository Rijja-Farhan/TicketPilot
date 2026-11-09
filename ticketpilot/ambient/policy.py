"""Policy layer (spec-a7 'Action tiers'): every action the pipeline or the model wants maps to a tier in code.
Default deny: an action not in the table is refused. The model can propose; only this module decides."""
import re
from dataclasses import dataclass

from ticketpilot import verify

AUTO = {"set_category", "set_priority", "link_duplicate", "quarantine", "raise_alert", "notify"}
APPROVAL = {"send_reply", "change_status", "escalate_to_human"}
DENIED = {"refund", "freeze_account", "close_account", "read_other_customer", "add_ticket_note"}
AMBIENT_TOOLS = frozenset({"get_customer", "classify_ticket", "search_policies"})
QUARANTINE_ALLOWED = {"set_priority", "link_duplicate", "quarantine", "raise_alert", "notify", "escalate_to_human"}
ALIASES = {  # how a model might name an action -> the canonical action it means
    "issue_refund": "refund", "refund_customer": "refund", "process_refund": "refund", "refund": "refund",
    "freeze": "freeze_account", "freeze_account": "freeze_account", "block_account": "freeze_account",
    "close_account": "close_account", "delete_account": "close_account",
    "add_note": "add_ticket_note", "add_ticket_note": "add_ticket_note",
    "send_email": "send_reply", "reply": "send_reply", "send_reply": "send_reply", "respond": "send_reply",
    "close_ticket": "change_status", "resolve_ticket": "change_status", "change_status": "change_status",
    "escalate": "escalate_to_human", "escalate_to_human": "escalate_to_human",
}


@dataclass(frozen=True)
class Decision:
    action: str
    tier: str          # auto | approval | denied
    allowed: bool      # may the pipeline do it without a human
    rule: str


def canonical(action: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(action).lower()).strip("_")
    return ALIASES.get(key, key)


def decide(action: str, *, quarantined: bool = False) -> Decision:
    a = canonical(action)
    if a in DENIED:
        return Decision(a, "denied", False, "denied_action")
    if quarantined and a not in QUARANTINE_ALLOWED:
        return Decision(a, "denied", False, "quarantine_restricts")
    if a in AUTO:
        return Decision(a, "auto", True, "auto_tier")
    if a in APPROVAL:
        return Decision(a, "approval", False, "needs_human_approval")
    return Decision(a, "denied", False, "default_deny")


def tool_allowed(name: str) -> bool:
    return name in AMBIENT_TOOLS


# ---- output checks on a drafted reply ------------------------------------------------------------------------------------
PROMISE = re.compile(r"\b(we (will|'ll|shall|are going to) (refund|credit|reimburse|waive|reverse|compensate)|"
                     r"(your|the) refund (of .{0,20})?(has been|is|was) (approved|processed|issued|granted)|"
                     r"we (have|'ve) (refunded|credited|reimbursed|waived|reversed|compensated)|guarantee[sd]? (a |your )?refund)\b", re.I)
URL_OR_EMAIL = re.compile(r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.]+", re.I)
ID_RE = re.compile(r"\b[CT]-\d+\b", re.I)
MONEY = re.compile(r"[£$€]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:pounds|dollars|euros|gbp|usd|eur)\b", re.I)


def check_draft(reply: str, *, customer_id: str, ticket_text: str, allowed_links: tuple[str, ...] = ()) -> list[str]:
    """Problems that make a drafted customer reply unsafe to put in front of an approver. Empty = passes."""
    problems = []
    if not reply.strip():
        problems.append("empty_reply")
    if PROMISE.search(reply):
        problems.append("promises_refund")
    if verify.find_action_claim(reply):
        problems.append("claims_action_taken")
    if any(not any(link in m for link in allowed_links) for m in URL_OR_EMAIL.findall(reply)):
        problems.append("unknown_link_or_email")
    own = customer_id.upper()
    if any(i.upper() != own for i in ID_RE.findall(reply) if i.upper().startswith("C-")):
        problems.append("other_customer_id")
    if any(m.lower() not in ticket_text.lower() for m in MONEY.findall(reply)):
        problems.append("amount_not_in_ticket")
    return problems
