"""Admission control in front of the durable queue (spec-a7 'Pipeline' step 1): de-duplication, per-customer rate cap and
back-pressure. Every outcome is audited; nothing is dropped silently."""
import re
from dataclasses import dataclass

from ticketpilot.ambient import dedupe
from ticketpilot.ambient.events import Event
from ticketpilot.ambient.guard import normalise

URGENT = re.compile(r"\b(fraud|stolen|scam|unauthori[sz]ed|phishing|hacked|compromised|not me|did not make)\b", re.I)


@dataclass
class IntakeConfig:
    rate_cap: int = 5            # admitted events per customer per window
    rate_window_s: float = 600.0
    max_depth: int = 200         # past this, priority 1 (normal) events are shed
    hard_cap: int = 400          # past this, everything is shed


class Intake:
    def __init__(self, store, audit, config: IntakeConfig | None = None):
        self.store, self.audit, self.cfg = store, audit, config or IntakeConfig()

    def admit(self, event: Event) -> dict:
        s, eid = self.store, event.event_id
        if s.event_row(eid):
            self.audit.append(eid, "dedupe", "duplicate", tier="auto", actor="intake", reason="exact_event_id", duplicate_of=eid)
            return {"action": "duplicate", "kind": "exact", "of": eid}
        clean, _ = normalise(event.text)
        key = dedupe.norm(clean)
        if (orig := dedupe.find_near(s, event.customer_id, event.epoch, key)):
            s.insert_event(event, 1, "duplicate", key, duplicate_of=orig)
            self.audit.append(eid, "dedupe", "duplicate", tier="auto", actor="intake", reason="near_duplicate", duplicate_of=orig)
            return {"action": "duplicate", "kind": "near", "of": orig}
        recent = s.recent_for_customer(event.customer_id, event.epoch - self.cfg.rate_window_s, ("pending", "processing", "done"))
        if len(recent) >= self.cfg.rate_cap:
            return self._shed(event, key, "rate_limited", f"{len(recent)} events from {event.customer_id} in {self.cfg.rate_window_s:.0f}s")
        priority = 0 if URGENT.search(clean) else 1
        depth = s.depth()
        if depth >= self.cfg.hard_cap or (priority == 1 and depth >= self.cfg.max_depth):
            return self._shed(event, key, "backpressure", f"queue depth {depth}")
        s.insert_event(event, priority, "pending", key)
        self.audit.append(eid, "admit", "accepted", tier="auto", actor="intake", reason="queued", priority=priority, depth=depth)
        return {"action": "accepted", "priority": priority}

    def _shed(self, event: Event, key: str, reason: str, detail: str) -> dict:
        self.store.insert_event(event, 1, "shed", key)
        self.audit.append(event.event_id, "shed", "shed", tier="auto", actor="intake", reason=reason, detail_text=detail)
        return {"action": "shed", "reason": reason}
