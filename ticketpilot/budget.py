"""Run budgets (spec-a5 'Harness'): checked in code before every model and tool call, never left to the prompt."""
import time
from dataclasses import dataclass, field
from typing import Callable

from ticketpilot import config


class StopRun(Exception):
    """Raised inside the coworker to end the loop with a named stop_reason."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail  # detail: e.g. the provider error code behind a model_error


@dataclass
class Budget:
    max_tool_calls: int = config.MAX_TOOL_CALLS
    max_reflect_rounds: int = config.MAX_REFLECT_ROUNDS
    max_tokens: int = config.MAX_TOKENS
    max_seconds: float = config.MAX_SECONDS
    max_calls_per_tool: int = config.MAX_CALLS_PER_TOOL
    tool_calls: int = 0
    reflect_rounds: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_before: float = 0.0  # seconds spent in earlier sessions of a resumed run
    calls_by_tool: dict[str, int] = field(default_factory=dict)
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _started: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self._started = self.clock()

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def elapsed(self) -> float:
        return self.elapsed_before + (self.clock() - self._started)

    def charge_tokens(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion

    def check_model(self) -> None:
        """Before any model call."""
        if self.tokens >= self.max_tokens:
            raise StopRun("token_budget")
        if self.elapsed >= self.max_seconds:
            raise StopRun("time_budget")

    def check_tool(self, name: str, retry: bool = False) -> str | None:
        """Before any tool attempt. Raises for run-wide limits; returns a message when only this tool is capped
        (a retry of a call already allowed is not a new call, but it still counts against the run-wide limit)."""
        self.check_model()
        if self.tool_calls >= self.max_tool_calls:
            raise StopRun("tool_budget")
        if not retry and self.calls_by_tool.get(name, 0) >= self.max_calls_per_tool:
            return f"{name} was already called {self.max_calls_per_tool} times in this run."
        return None

    def count_tool(self, name: str, retry: bool = False) -> None:
        self.tool_calls += 1
        if not retry:
            self.calls_by_tool[name] = self.calls_by_tool.get(name, 0) + 1

    def count_round(self) -> None:
        if self.reflect_rounds >= self.max_reflect_rounds:
            raise StopRun("round_budget")
        self.reflect_rounds += 1

    @property
    def cost_usd(self) -> float:
        p_in, p_out = config.TOKEN_PRICE_PER_M
        return (self.prompt_tokens * p_in + self.completion_tokens * p_out) / 1e6

    def used(self) -> dict:
        return {"tool_calls": self.tool_calls, "reflect_rounds": self.reflect_rounds, "tokens": self.tokens,
                "seconds": round(self.elapsed, 1), "cost_usd_estimate": round(self.cost_usd, 5)}

    def left(self) -> dict:
        return {"tool_calls": self.max_tool_calls - self.tool_calls, "tokens": self.max_tokens - self.tokens,
                "seconds": round(self.max_seconds - self.elapsed, 1)}

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("max_tool_calls", "max_reflect_rounds", "max_tokens", "max_seconds",
                                           "max_calls_per_tool", "tool_calls", "reflect_rounds", "prompt_tokens",
                                           "completion_tokens", "calls_by_tool")}
        d["elapsed_before"] = self.elapsed
        return d

    @classmethod
    def from_dict(cls, d: dict, clock: Callable[[], float] = time.monotonic) -> "Budget":
        return cls(**d, clock=clock)
