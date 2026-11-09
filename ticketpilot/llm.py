"""The only module that imports the LLM SDK. Everything else calls complete()."""
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import openai
from pydantic import BaseModel, ValidationError

from ticketpilot import fake_llm
from ticketpilot.config import ROOT, Settings, load_settings

T = TypeVar("T", bound=BaseModel)

MAX_429_RETRIES = 4
_sleep = time.sleep  # patched in tests


@dataclass
class LLMError:
    code: str  # missing_api_key | rate_limited | provider_error | invalid_output
    message: str


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as the model produced it; callers must validate before use


@dataclass
class LLMResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    parsed: BaseModel | None = None
    error: LLMError | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    provider: str = ""
    model: str = ""
    attempts: int = 0
    trace: dict = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def assistant_message(self) -> dict:
        """This turn as a chat message, ready to append to the history before the tool results."""
        msg: dict = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                                 for c in self.tool_calls]
        return msg


def _trace_path() -> Path:
    return Path(os.environ.get("TICKETPILOT_TRACE_PATH") or ROOT / "traces" / "llm_calls.jsonl")


def _log(result: LLMResult, schema) -> None:
    record = {
        "ts": time.time(), "provider": result.provider, "model": result.model,
        "schema": schema.__name__ if schema else None, "ok": result.ok,
        "error": result.error.code if result.error else None,
        "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens,
        "latency_ms": round(result.latency_ms, 1), "attempts": result.attempts,
        "tool_calls": [c.name for c in result.tool_calls],
    }
    result.trace = record
    try:
        path = _trace_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # tracing must never break a request


def _make_client(settings: Settings):
    return openai.OpenAI(base_url=settings.base_url, api_key=settings.api_key or "unused", max_retries=0)


class _FakeClient:
    """Duck-typed stand-in for the SDK client so the fake provider goes through the same code path."""

    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, *, messages, response_format=None, tools=None, **_):
        extra = {"tools": tools} if tools else {}
        return fake_llm.respond(messages, json_mode=response_format is not None, **extra)


def _call(client, settings: Settings, messages: list[dict], json_mode: bool, result: LLMResult,
          tools: list[dict] | None = None):
    """Return the provider's message object, or an LLMError."""
    kwargs = {"model": settings.model, "messages": messages, "temperature": 0}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if tools:
        kwargs["tools"] = tools
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            if attempt == MAX_429_RETRIES:
                return LLMError("rate_limited", str(e))
            _sleep(min(2 ** attempt, 30))
            continue
        except openai.OpenAIError as e:
            return LLMError("provider_error", f"{type(e).__name__}: {e}")
        usage = getattr(resp, "usage", None)
        result.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        result.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        return resp.choices[0].message
    return LLMError("rate_limited", "exhausted retries")  # unreachable


def complete(messages: list[dict], *, schema: type[T] | None = None, tools: list[dict] | None = None,
             settings: Settings | None = None, client=None) -> LLMResult:
    """Return text, or (with schema) a Pydantic-validated object, or (with tools) text and/or tool calls.
    Never raises for provider/format failures. `tools` uses the OpenAI function-tool format; schema and tools
    cannot be combined."""
    if schema and tools:
        raise ValueError("complete() takes either schema or tools, not both")
    settings = settings or load_settings()
    result = LLMResult(provider=settings.provider, model=settings.model)
    start = time.perf_counter()
    try:
        if client is None:
            if settings.provider == "fake":
                client = _FakeClient()
            else:
                if settings.provider != "ollama" and not settings.api_key:
                    result.error = LLMError("missing_api_key", f"No API key set for provider {settings.provider!r}")
                    return result
                client = _make_client(settings)

        msgs = list(messages)
        if schema:
            msgs.append({"role": "system", "content":
                         "Respond with a single JSON object matching this JSON schema, and nothing else:\n"
                         + json.dumps(schema.model_json_schema())})

        for attempt in (1, 2):  # one retry on invalid output, with the error fed back
            result.attempts = attempt
            msg = _call(client, settings, msgs, json_mode=schema is not None, result=result, tools=tools)
            if isinstance(msg, LLMError):
                result.error = msg
                return result
            out = msg.content or ""
            result.text = out
            result.tool_calls = [ToolCall(c.id, c.function.name, c.function.arguments or "{}")
                                 for c in getattr(msg, "tool_calls", None) or []]
            if schema is None:
                return result
            try:
                result.parsed = schema.model_validate_json(out)
                result.error = None
                return result
            except ValidationError as e:
                result.error = LLMError("invalid_output", str(e))
                msgs = msgs + [
                    {"role": "assistant", "content": out},
                    {"role": "user", "content": f"That output failed validation: {e}\nReturn corrected JSON only."},
                ]
        return result
    finally:
        result.latency_ms = (time.perf_counter() - start) * 1000
        _log(result, schema)
