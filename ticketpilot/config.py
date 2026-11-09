"""Settings: reads .env; holds the provider table (the only place provider details live)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TICKETS_PATH = DATA_DIR / "tickets.jsonl"
CUSTOMERS_PATH = DATA_DIR / "customers.jsonl"
TRANSACTIONS_PATH = DATA_DIR / "transactions.jsonl"
NOTES_PATH = DATA_DIR / "notes.jsonl"  # runtime data written by the mock API; gitignored
KB_DIR = DATA_DIR / "kb"
INDEX_DIR = DATA_DIR / "index"
RUNS_DIR = ROOT / "runs"                 # coworker checkpoints; gitignored (env TICKETPILOT_RUNS_DIR overrides)
RUN_TRACE_DIR = ROOT / "traces" / "runs"  # one JSONL per coworker run (env TICKETPILOT_RUN_TRACE_DIR overrides)

# Coworker budgets (spec-a5 'Harness'); changeable per run by passing a Budget.
MAX_TOOL_CALLS = 12
MAX_REFLECT_ROUNDS = 4
MAX_TOKENS = 60_000
MAX_SECONDS = 240.0  # live runs take 30 to 60 s; headroom for two 20 s rate-limit waits
MAX_CALLS_PER_TOOL = 3
# Placeholder USD per 1M (input, output) tokens, only for the report's cost estimate: verify against the provider's
# pricing page before quoting it as a real figure.
TOKEN_PRICE_PER_M = (0.15, 0.60)

# Model IDs are placeholders to verify against each provider's current model list
# (Groq model must support JSON mode). Override per provider with <PROVIDER>_MODEL.
PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY", "model": "openai/gpt-oss-120b"},
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY", "model": "gemini-2.5-flash"},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None, "model": "llama3.1"},
    "fake": {"base_url": None, "key_env": None, "model": "fake"},
}


@dataclass(frozen=True)
class Settings:
    provider: str
    base_url: str | None
    api_key: str | None
    model: str


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; expected one of {sorted(PROVIDERS)}")
    p = PROVIDERS[provider]
    key = os.environ.get(p["key_env"]) if p["key_env"] else None
    model = os.environ.get(f"{provider.upper()}_MODEL") or p["model"]
    return Settings(provider=provider, base_url=p["base_url"], api_key=key, model=model)
