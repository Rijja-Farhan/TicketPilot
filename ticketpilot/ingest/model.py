"""Plain data types shared by the ingestion stages."""
from dataclasses import dataclass, field


@dataclass
class RawFile:
    path: str   # file name, relative to the KB directory
    text: str


@dataclass
class Quarantined:
    path: str
    reason: str  # needs_ocr | empty | unsupported_type | pdftotext_missing


@dataclass
class Document:
    doc_id: str          # file stem of the kept copy; citation and index key
    path: str
    title: str
    version: int
    effective_date: str  # YYYY-MM-DD, or "" when unknown
    status: str          # current | superseded | draft
    body: str
    supersedes: str = ""   # doc_id of the previous version of the same title
    applies_to: str = ""
    inferred: list[str] = field(default_factory=list)  # metadata fields guessed, not declared

    @property
    def header(self) -> str:
        return f"[{self.title} | version {self.version} | effective {self.effective_date or 'unknown'} | {self.status}]"


@dataclass
class Dropped:
    path: str
    reason: str   # exact_duplicate | near_duplicate
    kept: str
    jaccard: float
