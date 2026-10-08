"""Small helpers shared by the five method modules."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn

from ..contracts import StepContext, UpdateProposal


def gradients_for(loss: Tensor, parameters: dict[str, nn.Parameter], names: Iterable[str]) -> dict[str, Tensor]:
    names = tuple(names)
    if not names:
        return {}
    values = [parameters[name] for name in names]
    # Several proposals share the same pre-update forward graph. The engine
    # releases it when the step context goes out of scope after commit.
    gradients = torch.autograd.grad(loss, values, allow_unused=True, retain_graph=True)
    return {
        name: (gradient.detach().clone() if gradient is not None else torch.zeros_like(parameter))
        for name, parameter, gradient in zip(names, values, gradients)
    }


class MethodModule:
    name = ""

    def parameter_names(self) -> tuple[str, ...]:
        return ()

    def prepare(self, image: Tensor, seed: int) -> Tensor:
        return image

    def propose(self, context: StepContext, parameters: dict[str, nn.Parameter]) -> UpdateProposal:
        raise NotImplementedError

    def after_commit(self, context: StepContext) -> None:
        return None

    def after_prediction(self, context: StepContext) -> None:
        return None

    def reset(self) -> None:
        return None

