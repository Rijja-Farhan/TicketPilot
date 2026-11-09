import json
from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import BaseModel

from ticketpilot import fake_llm, llm
from ticketpilot.config import PROVIDERS, load_settings


class Answer(BaseModel):
    value: int


MSGS = [{"role": "user", "content": "hi"}]


def test_provider_switch_is_env_only(monkeypatch):
    for name in PROVIDERS:
        monkeypatch.setenv("LLM_PROVIDER", name)
        assert load_settings().provider == name


def test_unknown_provider_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        load_settings()


def test_model_override_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_MODEL", "custom-model")
    assert load_settings().model == "custom-model"


def test_fake_text():
    r = llm.complete(MSGS)
    assert r.ok and r.text == "FAKE ANSWER" and r.provider == "fake"


def test_structured_valid():
    fake_llm.queue_responses('{"value": 3}')
    r = llm.complete(MSGS, schema=Answer)
    assert r.ok and r.parsed == Answer(value=3) and r.attempts == 1


def test_invalid_json_retries_once_then_succeeds():
    fake_llm.queue_responses("not json", '{"value": 1}')
    r = llm.complete(MSGS, schema=Answer)
    assert r.ok and r.parsed.value == 1 and r.attempts == 2


def test_invalid_json_twice_returns_typed_error():
    fake_llm.queue_responses("nope", '{"value": "x"}')
    r = llm.complete(MSGS, schema=Answer)
    assert not r.ok and r.error.code == "invalid_output" and r.parsed is None and r.attempts == 2


def test_retry_message_contains_error():
    seen = []

    class C:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            seen.append(kw["messages"])
            return fake_llm.respond(kw["messages"], True)

    fake_llm.queue_responses("bad", '{"value": 2}')
    llm.complete(MSGS, schema=Answer, client=C())
    assert "failed validation" in seen[1][-1]["content"]


def _rate_limit():
    req = httpx2.Request("POST", "http://x")
    return openai.RateLimitError("slow down", response=httpx2.Response(429, request=req), body=None)


def test_backoff_on_429(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm, "_sleep", sleeps.append)
    calls = {"n": 0}

    class C:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _rate_limit()
            return fake_llm.respond(kw["messages"], False)

    r = llm.complete(MSGS, client=C())
    assert r.ok and calls["n"] == 3 and sleeps == [1, 2]


def test_429_exhausted_is_typed_error(monkeypatch):
    monkeypatch.setattr(llm, "_sleep", lambda s: None)

    class C:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kw):
            raise _rate_limit()

    r = llm.complete(MSGS, client=C())
    assert r.error.code == "rate_limited"


def test_missing_key_is_typed_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("ticketpilot.config.load_dotenv", lambda *a, **k: None)
    r = llm.complete(MSGS)
    assert r.error.code == "missing_api_key"


def test_trace_logs_tokens_and_latency(tmp_path):
    llm.complete(MSGS)
    rec = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[-1])
    assert rec["provider"] == "fake" and rec["latency_ms"] >= 0 and "prompt_tokens" in rec
