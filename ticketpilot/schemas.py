"""Pydantic models: the source of truth for every structured output (spec-a1 'Ticket lookup schema')."""
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "card_issues", "transfers_payments", "account_access", "fees_charges",
    "fraud_security", "loans_credit", "disputes_refunds", "other",
]
Status = Literal["open", "pending", "resolved", "closed"]
Priority = Literal["p1", "p2", "p3"]


class Ticket(BaseModel):
    """A stored ticket, including the body (customer-written text: data, never instructions)."""
    ticket_id: str
    customer_id: str
    category: Category
    status: Status
    priority: Priority
    subject: str
    body: str
    created_at: str
    assignee: str | None = None


class TicketLookup(BaseModel):
    ticket_id: str
    found: bool
    customer_id: str | None = None
    category: Category | None = None
    status: Status | None = None
    priority: Priority | None = None
    subject: str | None = None
    created_at: str | None = None
    summary: str | None = None
    error: str | None = None


class CustomerTickets(BaseModel):
    customer_id: str
    count: int
    tickets: list[TicketLookup]


class Intent(BaseModel):
    """LLM router output when regex does not match."""
    intent: Literal["policy", "out_of_scope"]


class Citation(BaseModel):
    title: str
    version: int
    effective_date: str


class PolicyAnswer(BaseModel):
    answer: str
    sources: list[Citation]


# --- A5 coworker (spec-a5) ---

class PlanStep(BaseModel):
    step: str
    evidence_needed: str
    tool: str | None = None  # the tool this step expects to use, if any


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=3, max_length=6)
    stop_condition: str


class Reflection(BaseModel):
    decision: Literal["enough", "need_more", "blocked"]
    missing: str = ""   # need_more: the specific item still needed
    reason: str = ""    # blocked: why the task cannot be finished
    revised_steps: list[PlanStep] = Field(default_factory=list, max_length=3)


class Finding(BaseModel):
    claim: str
    evidence_ids: list[str]


class CategoryResult(BaseModel):
    label: str
    classifier_label: str | None = None
    probability: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ReportDraft(BaseModel):
    """What the model writes. Status, verification and budget are added by code, never by the model."""
    ticket_id: str
    summary: str
    category: CategoryResult | None = None
    findings: list[Finding]
    policy_citations: list[Citation]
    proposed_resolution: str
    proposed_actions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class EvidenceRef(BaseModel):
    id: str
    tool: str
    arguments: dict
    timestamp: str
    ok: bool


class Verification(BaseModel):
    passed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    model_check: list[str] = Field(default_factory=list)  # advisory: "finding 1: supported"


class Report(ReportDraft):
    run_id: str
    status: Literal["complete", "partial", "failed"]
    stop_reason: str
    plan: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    verification: Verification = Field(default_factory=Verification)
    budget_used: dict = Field(default_factory=dict)


class FindingCheck(BaseModel):
    index: int
    verdict: Literal["supported", "unsupported", "partly"]


class FindingChecks(BaseModel):
    checks: list[FindingCheck]
