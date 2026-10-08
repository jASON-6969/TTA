"""Central learning-rate arbitration for DLTTA and GraTa."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LearningRateDecision:
    target: str
    raw_hints: dict[str, float]
    final_lr: float
    skipped: bool


def combine_learning_rates(target: str, hints: dict[str, float], fallback: float) -> LearningRateDecision:
    """Combine positive hints geometrically; zero means skip that update group."""
    values = {name: float(value) for name, value in hints.items() if value is not None}
    if not values:
        return LearningRateDecision(target, {}, float(fallback), False)
    if any(value < 0 or not math.isfinite(value) for value in values.values()):
        raise ValueError(f"Learning-rate hints must be finite and non-negative: {values}")
    if any(value == 0 for value in values.values()):
        return LearningRateDecision(target, values, 0.0, True)
    final = math.exp(sum(math.log(value) for value in values.values()) / len(values))
    return LearningRateDecision(target, values, final, False)

