"""The event a webhook delivers (spec-a7 'Event'). Customer-written fields are data, never instructions."""
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

MAX_SUBJECT, MAX_BODY = 200, 4000


class Event(BaseModel):
    event_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,64}$")
    ts: str
    customer_id: str = Field(pattern=r"^C-\d{1,6}$")
    subject: str = Field(max_length=MAX_SUBJECT)
    body: str = Field(max_length=MAX_BODY)
    source: str = "webhook"

    @field_validator("ts")
    @classmethod
    def _ts(cls, v: str) -> str:
        parse_ts(v)
        return v

    @property
    def text(self) -> str:
        return f"{self.subject}\n{self.body}" if self.subject and self.subject not in self.body else self.body

    @property
    def epoch(self) -> float:
        return parse_ts(self.ts)


def parse_ts(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def load_stream(path: Path) -> list[Event]:
    return [Event.model_validate_json(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(obj) -> str:
    return json.dumps(obj, default=str, sort_keys=True)
