"""Offline fake provider: canned, deterministic responses so tests run free and without a network.

Tests can queue exact responses with queue_responses(); otherwise a simple keyword heuristic is used.
A queued item is plain text, or a dict {"tool_calls": [{"name": ..., "arguments": {...}}], "content": "..."}
to script a tool-calling turn.
"""
import json
from itertools import count
from types import SimpleNamespace

_queue: list = []
_ids = count(1)

POLICY_WORDS = ("dispute", "refund", "fee", "fraud", "policy", "sla", "escalat", "privacy", "chargeback", "liability")


def queue_responses(*responses) -> None:
    _queue.clear()
    _queue.extend(responses)


def _default(messages: list[dict], json_mode: bool) -> str:
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "").lower()
    if json_mode:
        return json.dumps({"intent": "policy" if any(w in user for w in POLICY_WORDS) else "out_of_scope"})
    return "FAKE ANSWER"


def _tool_call(spec: dict):
    args = spec.get("arguments", {})
    return SimpleNamespace(
        id=f"call_{next(_ids)}",
        function=SimpleNamespace(name=spec["name"], arguments=args if isinstance(args, str) else json.dumps(args)))


def respond(messages: list[dict], json_mode: bool, tools: list[dict] | None = None):
    item = _queue.pop(0) if _queue else _default(messages, json_mode)
    spec = item if isinstance(item, dict) else {"content": item}
    text = spec.get("content") or ""
    calls = [_tool_call(c) for c in spec.get("tool_calls", [])]
    prompt_chars = sum(len(m.get("content") or "") for m in messages)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=calls or None))],
        usage=SimpleNamespace(prompt_tokens=prompt_chars // 4, completion_tokens=len(text) // 4),
    )
