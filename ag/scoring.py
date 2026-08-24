"""Directed-evolution scoring: rate every run on accuracy, quality, and speed.

Two of the three axes are *judged* by the critic model (it can assess whether an
answer is correct and whether it is good), but **speed cannot be judged by the
model** — a model has no view of its own wall-clock latency or token cost. So AG
splits the work:

  - accuracy, quality  -> scored by the critic (see prompts.CRITIC_SYSTEM)
  - speed              -> *measured* here from elapsed time + output tokens against
                          a configurable budget

The three combine into one weighted `overall`. The per-run iterate loop gates on
the answer-quality blend (accuracy+quality); speed is recorded and fed to the
evolve loop so self-improvement is *directed* at the weakest axis over time.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict


@dataclass
class Scorecard:
    """Per-run scores, all on a 0..10 scale (higher is better)."""

    accuracy: float = 0.0     # critic: factual correctness / verifiability
    quality: float = 0.0      # critic: fidelity + judgment + usefulness + form
    speed: float = 0.0        # measured: latency + token cost vs budget
    overall: float = 0.0      # weighted blend of the three
    # The blend of just the answer-quality axes, used to decide whether to iterate.
    answer_score: float = 0.0
    elapsed_s: float = 0.0
    output_tokens: int = 0
    weights: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def speed_score(elapsed_s: float, output_tokens: int, *, budget_s: float,
                token_budget: int = 1200) -> float:
    """Measure a 0..10 speed score from wall-clock and output size.

    Full marks at or under budget; degrades gracefully past it (never negative).
    Longer answers are given proportionally more time, so a thorough answer is not
    unfairly penalised for taking longer than a one-liner.
    """
    budget_s = max(1.0, float(budget_s))
    # Scale the time budget up for larger outputs (more tokens legitimately cost time).
    size_factor = 1.0 + max(0, output_tokens) / max(1, token_budget)
    allowed = budget_s * size_factor
    if elapsed_s <= allowed:
        return 10.0
    # Past budget: 10 * allowed/elapsed -> smooth decay toward 0, floored at 0.
    return round(max(0.0, 10.0 * allowed / max(elapsed_s, 0.001)), 2)


def default_weights() -> Dict[str, float]:
    return {"accuracy": 0.5, "quality": 0.3, "speed": 0.2}


def _normed(weights: Dict[str, float]) -> Dict[str, float]:
    w = {k: max(0.0, float(weights.get(k, 0.0))) for k in ("accuracy", "quality", "speed")}
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def build_scorecard(*, accuracy: float, quality: float, elapsed_s: float,
                    output_tokens: int, budget_s: float,
                    weights: Dict[str, float] | None = None) -> Scorecard:
    """Combine the judged axes (accuracy/quality) with the measured axis (speed)."""
    weights = weights or default_weights()
    w = _normed(weights)
    spd = speed_score(elapsed_s, output_tokens, budget_s=budget_s)
    accuracy = _clamp(accuracy)
    quality = _clamp(quality)
    overall = round(w["accuracy"] * accuracy + w["quality"] * quality
                    + w["speed"] * spd, 2)
    # answer_score ignores speed so iterating never looks "worse" just for being slow.
    aw = w["accuracy"] + w["quality"] or 1.0
    answer = round((w["accuracy"] * accuracy + w["quality"] * quality) / aw, 2)
    return Scorecard(
        accuracy=accuracy, quality=quality, speed=spd, overall=overall,
        answer_score=answer, elapsed_s=round(elapsed_s, 3),
        output_tokens=int(output_tokens), weights=w,
    )


def weakest_axis(cards: list) -> str:
    """Across recent scorecards, which axis is lowest on average.

    Used to *direct* evolution at the dimension that most needs improvement.
    """
    if not cards:
        return "accuracy"
    axes = ("accuracy", "quality", "speed")
    avg = {a: sum(float(c.get(a, 0.0)) for c in cards) / len(cards) for a in axes}
    return min(axes, key=lambda a: avg[a])


def _clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    try:
        return round(max(lo, min(hi, float(x))), 2)
    except (TypeError, ValueError):
        return 0.0
