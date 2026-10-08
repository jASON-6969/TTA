"""2D CXR adaptation proposal with identity style and structural regularisation."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ..contracts import StepContext, UpdateProposal
from .base import MethodModule, gradients_for
from .structure import connected_foreground_regions, excess_component_suppression


def count_foreground_components(mask: Tensor) -> int:
    """Count 4-connected components in a 2D foreground mask."""
    while mask.ndim > 2:
        mask = mask[0]
    return len(connected_foreground_regions(mask))


def pixelwise_ema_kl(probabilities: Tensor, teacher_logits: Tensor) -> Tensor:
    """Mean class-summed KL per pixel, with the EMA teacher detached."""
    teacher_probabilities = teacher_logits.softmax(dim=1).detach()
    return F.kl_div(
        probabilities.clamp_min(1e-6).log(), teacher_probabilities, reduction="none",
    ).sum(dim=1).mean()


def two_lung_structure_loss(logits: Tensor) -> tuple[Tensor, dict[str, float]]:
    """Combine pixel-mean TV with suppression of regions beyond the two largest."""
    foreground = logits.softmax(dim=1)[:, 1:2]
    zero = foreground.sum() * 0.0
    horizontal = (foreground[..., :, 1:] - foreground[..., :, :-1]).abs().mean() if foreground.shape[-1] > 1 else zero
    vertical = (foreground[..., 1:, :] - foreground[..., :-1, :]).abs().mean() if foreground.shape[-2] > 1 else zero
    neighborhood = horizontal + vertical
    excess, metrics = excess_component_suppression(foreground)
    return neighborhood + excess, {**metrics, "neighborhood": float(neighborhood.detach().cpu())}


class SmaRT(MethodModule):
    name = "SmaRT"

    def __init__(self, smart_module: nn.Module, lr: float = 5e-4, *, structure_weight: float = 0.50, consistency_weight: float = 0.50):
        # 修复1: 学习率从1e-4提高到5e-4 (5倍)
        # 修复3: 结构权重从0.10提高到0.50 (5倍)
        # 修复3: 一致性权重从0.10提高到0.50 (5倍)
        self.smart_module = smart_module
        self.lr = lr
        self.structure_weight = structure_weight
        self.consistency_weight = consistency_weight
        self._names = tuple(f"smart.{name}" for name, _ in smart_module.named_parameters())

    def parameter_names(self) -> tuple[str, ...]:
        return self._names

    def propose(self, context: StepContext, parameters: dict[str, nn.Parameter]) -> UpdateProposal:
        logits = context.smart_logits if context.smart_logits is not None else context.student_logits
        probabilities = logits.softmax(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=1).mean()
        structure, metrics = two_lung_structure_loss(logits)
        consistency = logits.new_zeros(())
        if context.ema_logits is not None:
            consistency = pixelwise_ema_kl(probabilities, context.ema_logits)
        loss = entropy + self.structure_weight * structure + self.consistency_weight * consistency
        gradients = gradients_for(loss, parameters, self._names)
        return UpdateProposal(
            method=self.name,
            gradients=gradients,
            parameter_names=self._names,
            lr_hint=self.lr,
            loss=float(loss.detach().cpu()),
            metrics={"entropy": float(entropy.detach().cpu()), "ema_consistency": float(consistency.detach().cpu()), **metrics},
        )

    def reset(self) -> None:
        for parameter in self.smart_module.parameters():
            parameter.grad = None

