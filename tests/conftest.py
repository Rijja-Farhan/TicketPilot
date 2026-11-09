import pytest

from ticketpilot import fake_llm


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Every test runs against the fake provider with an isolated trace file."""
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("TICKETPILOT_TRACE_PATH", str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "hash")
    monkeypatch.setenv("TICKETPILOT_MEMORY_PATH", str(tmp_path / "memory.jsonl"))
    monkeypatch.setenv("TICKETPILOT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("TICKETPILOT_RUN_TRACE_DIR", str(tmp_path / "run-traces"))
    # No built index: policies fall back to A1 BM25 over data/kb/ unless a test builds its own index.
    monkeypatch.setenv("TICKETPILOT_INDEX_DIR", str(tmp_path / "no-index"))
    fake_llm.queue_responses()
    yield
    fake_llm.queue_responses()
