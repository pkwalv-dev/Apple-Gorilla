"""Run budgets — an explicit ceiling on what one task may spend.

An agent loop without a budget has exactly one stopping problem: it stops when the
model decides to stop. For a local model that means GPU-minutes, for an API model
it means money, and for an unattended autonomous run it means both, forever. A
Budget makes the ceiling a first-class value with a honest, declared outcome:

- every model call charges its tokens and increments the call count;
- every tool invocation charges one tool call;
- wall time is checked at each loop iteration;
- crossing ANY ceiling raises BudgetExceeded, and the loop degrades to the best
  answer it already has — it never dies mid-write, and the answer SAYS the budget
  ran out rather than presenting a truncated result as complete.

All limits default to 0 = unlimited, so existing behaviour is unchanged unless the
operator sets a ceiling. Enforcement lives in the reason loop (and any other loop
that accepts a budget), not in the model: a model cannot spend its way out of a
ceiling it does not control.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


class BudgetExceeded(Exception):
    """Raised by a charge method when a ceiling is crossed. Carries which one."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class Budget:
    max_model_calls: int = 0      # 0 = unlimited (all four)
    max_tool_calls: int = 0
    max_tokens: int = 0
    wall_s: float = 0.0
    model_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    started: float = field(default_factory=time.monotonic)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> Optional["Budget"]:
        """A Budget from the cfg.budget_* fields; None when everything is unlimited,
        so an unconfigured run pays zero bookkeeping cost."""
        limits = (
            int(getattr(cfg, "budget_model_calls", 0) or 0),
            int(getattr(cfg, "budget_tool_calls", 0) or 0),
            int(getattr(cfg, "budget_tokens", 0) or 0),
            float(getattr(cfg, "budget_wall_s", 0.0) or 0.0),
        )
        if not any(limits):
            return None
        return cls(*limits)

    # -- charging ------------------------------------------------------------
    def charge_model_call(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        self.check_wall()
        self.model_calls += 1
        self.tokens += max(0, int(input_tokens)) + max(0, int(output_tokens))
        if self.max_model_calls and self.model_calls > self.max_model_calls:
            raise BudgetExceeded(
                f"model-call budget exhausted ({self.max_model_calls})")
        if self.max_tokens and self.tokens > self.max_tokens:
            raise BudgetExceeded(f"token budget exhausted ({self.max_tokens})")

    def charge_tool_call(self, name: str = "") -> None:
        self.check_wall()
        self.tool_calls += 1
        if self.max_tool_calls and self.tool_calls > self.max_tool_calls:
            raise BudgetExceeded(
                f"tool-call budget exhausted ({self.max_tool_calls})"
                + (f" at '{name}'" if name else ""))

    def check_wall(self) -> None:
        if self.wall_s and (time.monotonic() - self.started) > self.wall_s:
            raise BudgetExceeded(f"wall-clock budget exhausted ({self.wall_s}s)")

    # -- reporting -----------------------------------------------------------
    def summary(self) -> str:
        parts = [f"{self.model_calls} model call(s)", f"{self.tool_calls} tool call(s)",
                 f"{self.tokens} token(s)"]
        return ", ".join(parts)
