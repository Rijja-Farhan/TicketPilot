"""Human-in-the-loop (spec-a7 'Action tiers'): approve, edit or reject a pending item; overdue items escalate.

The agent never sends. Approving a reply runs here, through a separate path that re-checks the policy and the output checks."""
import time

from ticketpilot.ambient import policy
from ticketpilot.ambient.events import Event

ESCALATION = ["", "support-lead", "head-of-support"]   # who an item moves to after each overdue period (index = level)
OVERDUE_S = 4 * 3600


class ApprovalError(Exception):
    pass


class Approvals:
    def __init__(self, store, audit, clock=time.time):
        self.store, self.audit, self.clock = store, audit, clock

    def _pending(self, approval_id: int) -> dict:
        item = self.store.approval(approval_id)
        if item is None:
            raise ApprovalError(f"no approval {approval_id}")
        if item["status"] != "pending":
            raise ApprovalError(f"approval {approval_id} is already {item['status']}")
        return item

    def _ticket_text(self, event_id: str) -> tuple[str, str]:
        row = self.store.event_row(event_id)
        ev = Event.model_validate_json(row["payload"])
        return ev.customer_id, ev.text

    def approve(self, approval_id: int, who: str) -> dict:
        item = self._pending(approval_id)
        d = policy.decide(item["kind"])
        if d.tier == "denied":                       # defence in depth: a denied action can never be approved into existence
            self.audit.append(item["event_id"], "approve", "refused", tier="denied", actor=who, reason=d.rule, approval_id=approval_id)
            raise ApprovalError(f"{item['kind']} is not an allowed action")
        if item["kind"] == "send_reply":
            customer, text = self._ticket_text(item["event_id"])
            problems = policy.check_draft(item["payload"]["reply"].split("\n\nPolicy reference:")[0], customer_id=customer, ticket_text=text)
            if problems:
                raise ApprovalError(f"the reply fails output checks: {problems}")
            self.store.send(approval_id, item["event_id"], item["payload"]["reply"], self.clock())
        self.store.update_approval(approval_id, status="approved", decided_by=who, decided_ts=self.clock())
        self.audit.append(item["event_id"], "approve", "approved", tier="approval", actor=who, approval_id=approval_id, kind=item["kind"])
        return self.store.approval(approval_id)

    def edit(self, approval_id: int, who: str, new_reply: str) -> dict:
        item = self._pending(approval_id)
        if item["kind"] != "send_reply":
            raise ApprovalError("only replies can be edited")
        customer, text = self._ticket_text(item["event_id"])
        problems = policy.check_draft(new_reply, customer_id=customer, ticket_text=text)
        if problems:
            self.audit.append(item["event_id"], "edit", "refused", tier="approval", actor=who, reason=",".join(problems), approval_id=approval_id)
            raise ApprovalError(f"the edited reply fails output checks: {problems}")
        self.store.update_approval(approval_id, payload={**item["payload"], "reply": new_reply, "edited_by": who})
        self.audit.append(item["event_id"], "edit", "edited", tier="approval", actor=who, approval_id=approval_id)
        return self.approve(approval_id, who)

    def reject(self, approval_id: int, who: str, reason: str) -> dict:
        item = self._pending(approval_id)
        if not reason.strip():
            raise ApprovalError("a rejection needs a reason")
        self.store.update_approval(approval_id, status="rejected", decided_by=who, decided_ts=self.clock(), reason=reason)
        self.audit.append(item["event_id"], "reject", "rejected", tier="approval", actor=who, reason=reason, approval_id=approval_id)
        return self.store.approval(approval_id)

    def escalate_overdue(self, overdue_s: float = OVERDUE_S) -> list[int]:
        """Move items nobody answered in `overdue_s` to the next person. Returns the IDs moved."""
        now, moved = self.clock(), []
        for item in self.store.approvals("pending"):
            if now - item["created_ts"] - item["level"] * overdue_s >= overdue_s and item["level"] < len(ESCALATION) - 1:
                level = item["level"] + 1
                self.store.update_approval(item["id"], level=level, recipient=ESCALATION[level], urgent=1)
                self.audit.append(item["event_id"], "escalate_overdue", "escalated", tier="auto", actor="scheduler",
                                  approval_id=item["id"], recipient=ESCALATION[level])
                moved.append(item["id"])
        return moved
