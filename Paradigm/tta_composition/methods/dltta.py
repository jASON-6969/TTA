"""Distribution-memory learning-rate controller used by the scheduler."""

from __future__ import annotations

from collections import deque
import math

import torch
from torch import Tensor, nn

from ..contracts import StepContext, UpdateProposal
from .base import MethodModule, gradients_for


class DLTTA(MethodModule):
    name = "DLTTA"

    def __init__(self, base_lr: float = 5e-4, memory_size: int = 16):
        # 修复1: 学习率从1e-4提高到5e-4 (5倍)
        self.base_lr = base_lr
        self.memory_size = memory_size
        self.memory: deque[Tensor] = deque(maxlen=memory_size)
        self._host_names: tuple[str, ...] = ()

    def bind_host(self, names: tuple[str, ...]) -> None:
        self._host_names = names

    def parameter_names(self) -> tuple[str, ...]:
        return self._host_names

    def _hint(self, context: StepContext) -> float:
        feature = context.student_features.detach().mean(dim=(0, 2, 3))
        if not self.memory:
            return self.base_lr
        reference = torch.stack(list(self.memory)).mean(dim=0).to(feature.device)
        shift = float(torch.norm(feature - reference) / (torch.norm(reference) + 1e-6))
        return self.base_lr * max(0.25, min(4.0, 1.0 + shift))

    def propose(self, context: StepContext, parameters: dict[str, nn.Parameter]) -> UpdateProposal:
        hint = self._hint(context)
        gradients: dict[str, Tensor] = {}
        loss_value: float | None = None
        if self._host_names:
            probabilities = context.student_logits.softmax(dim=1)
            loss = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=1).mean()
            gradients = gradients_for(loss, parameters, self._host_names)
            loss_value = float(loss.detach().cpu())
        return UpdateProposal(
            method=self.name,
            gradients=gradients,
            parameter_names=self._host_names,
            lr_hint=hint,
            loss=loss_value,
            metrics={"distribution_shift": hint / max(self.base_lr, 1e-12)},
        )

    def after_commit(self, context: StepContext) -> None:
        self.memory.append(context.student_features.detach().mean(dim=(0, 2, 3)).clone())

    def reset(self) -> None:
        self.memory.clear()
        self._host_names = ()

