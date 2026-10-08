"""GraTa-style entropy, output-consistency and feature-alignment proposal."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..contracts import StepContext, UpdateProposal
from .base import MethodModule, gradients_for


class GraTa(MethodModule):
    name = "GraTa"

    def __init__(self, lr: float = 5e-4, *, consistency_weight: float = 1.0, feature_weight: float = 0.50):
        # 修复1: 学习率从1e-4提高到5e-4 (5倍)
        # 修复3: 特征权重从0.10提高到0.50 (5倍)
        self.lr = lr
        self.consistency_weight = consistency_weight
        self.feature_weight = feature_weight
        self._names: tuple[str, ...] = ()

    def bind_parameter_names(self, names: tuple[str, ...]) -> None:
        self._names = names

    def parameter_names(self) -> tuple[str, ...]:
        return self._names

    def propose(self, context: StepContext, parameters: dict[str, nn.Parameter]) -> UpdateProposal:
        probabilities = context.student_logits.softmax(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=1).mean()
        augmented_logits = context.augmented_logits if context.augmented_logits is not None else context.student_logits
        augmented_features = context.augmented_features if context.augmented_features is not None else context.student_features
        consistency = F.mse_loss(probabilities, augmented_logits.softmax(dim=1))
        feature_consistency = F.mse_loss(
            context.student_features.mean(dim=(2, 3)),
            augmented_features.mean(dim=(2, 3)),
        )
        loss = entropy + self.consistency_weight * consistency + self.feature_weight * feature_consistency
        gradients = gradients_for(loss, parameters, self._names)
        return UpdateProposal(
            method=self.name,
            gradients=gradients,
            parameter_names=self._names,
            lr_hint=self.lr,
            loss=float(loss.detach().cpu()),
            metrics={"entropy": float(entropy.detach().cpu()), "consistency": float(consistency.detach().cpu()), "feature_loss": float(feature_consistency.detach().cpu())},
        )

