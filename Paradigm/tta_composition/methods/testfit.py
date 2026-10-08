"""Central-update TestFit pseudo-label and feature alignment proposal."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..contracts import StepContext, UpdateProposal
from .base import MethodModule, gradients_for


class TestFit(MethodModule):
    name = "TestFit"

    def __init__(self, lr: float = 5e-4, confidence: float = 0.95, *, feature_weight: float = 0.50):
        # 修复1: 学习率从1e-5提高到5e-4 (50倍)
        # 提高門檻只保留高置信度偽標籤；低置信度像素不參與偽標籤損失。
        # 修复3: 特征权重从0.10提高到0.50 (5倍)
        self.lr = lr
        self.confidence = confidence
        self.feature_weight = feature_weight
        self._names: tuple[str, ...] = ()

    def bind_parameter_names(self, names: tuple[str, ...]) -> None:
        self._names = names

    def parameter_names(self) -> tuple[str, ...]:
        return self._names

    def propose(self, context: StepContext, parameters: dict[str, nn.Parameter]) -> UpdateProposal:
        with torch.no_grad():
            confidence, pseudo = context.reference_logits.softmax(dim=1).max(dim=1)
            valid = confidence > self.confidence
        pixel_loss = F.cross_entropy(context.student_logits, pseudo, reduction="none")
        pseudo_loss = pixel_loss[valid].mean() if bool(valid.any()) else pixel_loss.mean() * 0.0
        feature_loss = F.mse_loss(
            context.student_features.mean(dim=(2, 3)),
            context.reference_features.mean(dim=(2, 3)),
        )
        loss = pseudo_loss + self.feature_weight * feature_loss
        gradients = gradients_for(loss, parameters, self._names)
        return UpdateProposal(
            method=self.name,
            gradients=gradients,
            parameter_names=self._names,
            lr_hint=self.lr,
            loss=float(loss.detach().cpu()),
            metrics={"pseudo_valid_fraction": float(valid.float().mean().detach().cpu()), "feature_loss": float(feature_loss.detach().cpu())},
        )

