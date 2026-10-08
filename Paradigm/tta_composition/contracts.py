"""Shared contracts for the composable CXR test-time adaptation runner.

The contracts keep method implementations side-effect free during proposal
construction.  The engine is the only component that commits parameter
updates or persistent state transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor


METHODS = ("VPTTA", "DLTTA", "TestFit", "GraTa", "SmaRT")


@dataclass
class StepContext:
    """Inputs and pre-update model outputs for one target image."""

    step: int
    sample_id: str
    image: Tensor
    transformed_image: Tensor
    reference_logits: Tensor
    reference_features: Tensor
    student_logits: Tensor
    student_features: Tensor
    smart_logits: Tensor | None = None
    smart_features: Tensor | None = None
    ema_logits: Tensor | None = None
    ema_features: Tensor | None = None
    augmented_logits: Tensor | None = None
    augmented_features: Tensor | None = None
    rng_seed: int | None = None


@dataclass
class UpdateProposal:
    """A method's contribution to the central update transaction."""

    method: str
    gradients: dict[str, Tensor] = field(default_factory=dict)
    parameter_names: tuple[str, ...] = ()
    lr_hint: float | None = None
    loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"Unknown method in proposal: {self.method}")
        if set(self.gradients) != set(self.parameter_names):
            raise ValueError(
                f"{self.method} gradient keys and ownership differ: "
                f"{sorted(self.gradients)} != {sorted(self.parameter_names)}"
            )
        for name, gradient in self.gradients.items():
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite gradient from {self.method}: {name}")
        if self.loss is not None and not torch.isfinite(torch.tensor(self.loss)):
            raise FloatingPointError(f"Non-finite loss from {self.method}: {self.loss}")
        if self.lr_hint is not None and (self.lr_hint < 0 or not torch.isfinite(torch.tensor(self.lr_hint))):
            raise FloatingPointError(f"Invalid learning-rate hint from {self.method}: {self.lr_hint}")


@dataclass(frozen=True)
class UpdateRecord:
    """Serializable summary of one central update."""

    step: int
    losses: Mapping[str, float]
    lr_hints: Mapping[str, float]
    final_learning_rates: Mapping[str, float]
    gradient_norms: Mapping[str, float]
    updated_parameters: int
    elapsed_seconds: float
    iterations: tuple[UpdateRecord, ...] = ()


@dataclass
class MethodState:
    """Persistent state owned by one method inside an online stream."""

    values: dict[str, Any] = field(default_factory=dict)

