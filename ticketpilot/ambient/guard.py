"""Input guard (spec-a7 'Pipeline' step 2): normalise ticket text and scan it for injection attempts.

A scan is a cheap first filter, not the defence: it misses new phrasings (docs/a7-redteam.md lists which). The defence is
that a flagged ticket never reaches the model and that policy.py bounds every action in code."""
import re
import unicodedata

HIDDEN = re.compile("[​‌‍⁠﻿‪-‮­]")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PATTERNS = {
    "override": r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your)\b.{0,30}\b(instruction|rule|prompt|polic|guideline)",
    "prompt_leak": r"\b(reveal|show|print|repeat|leak)\b.{0,40}\b(system prompt|your prompt|instructions|tools you)",
    "role_change": r"\byou are now\b|\bact as\b|\bpretend (to be|you)\b|\bdeveloper mode\b|\bDAN\b",
    "fake_markup": r"</?\s*(system|tool_result|assistant|memory|instructions?)\b|^\s*(system|assistant)\s*:|\bSYSTEM:",
    "authority": r"\b(admin|administrator|support lead|manager)\b.{0,30}\b(instruction|order|command|authori[sz]ed)|\byou are authori[sz]ed\b",
    "money_action": r"\b(issue|process|approve|send|credit)\b.{0,30}\b(refund|payment|compensation)\b.{0,40}\b(immediately|without|now)\b",
    "account_action": r"\b(freeze|close|delete|terminate|unlock)\b.{0,30}\b(account|customer)\b.{0,20}\bC-\d+",
    "tool_call": r"\bcall\b.{0,20}\b(add_ticket_note|freeze_account|get_customer|list_transactions|search_tickets|tool)\b",
    "exfiltration": r"\b(print|list|send|email|show)\b.{0,40}\b(history|address|details|data)\b.{0,30}\b(customer|C-\d+)",
    "base64_blob": r"[A-Za-z0-9+/]{40,}={0,2}",
}
COMPILED = {k: re.compile(v, re.I | re.S) for k, v in PATTERNS.items()}


def normalise(text: str) -> tuple[str, bool]:
    """NFKC, strip hidden and control characters, collapse whitespace. Returns (clean text, hidden characters were present)."""
    hidden = bool(HIDDEN.search(text))
    t = CONTROL.sub(" ", HIDDEN.sub("", unicodedata.normalize("NFKC", text)))
    return re.sub(r"\s+", " ", t).strip(), hidden


def scan(text: str) -> list[str]:
    """Names of the injection signals found in `text` (empty list = nothing recognised)."""
    clean, hidden = normalise(text)
    flags = [name for name, rx in COMPILED.items() if rx.search(clean) or (name == "fake_markup" and rx.search(text))]
    if hidden:
        flags.append("hidden_characters")
    return flags
