"""CXR-compatible low-frequency prompt and prompt-memory adapter."""

from __future__ import annotations

from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..contracts import StepContext, UpdateProposal
from .base import MethodModule, gradients_for


class VPTTA(MethodModule):
    name = "VPTTA"

    def __init__(self, device: torch.device, lr: float = 5e-3, memory_size: int = 16, *, prompt_size: int = 8, prompt_strength: float = 0.05):
        # 修复1: 学习率从1e-3提高到5e-3 (5倍)
        # 修复5: Prompt强度从0.02提高到0.05 (2.5倍) - 增加adaptation强度
        self.prompt = nn.Parameter(torch.zeros(1, 1, prompt_size, prompt_size, device=device))
        self.lr = lr
        self.prompt_size = prompt_size
        self.prompt_strength = prompt_strength
        self.memory_size = memory_size
        self.memory: deque[Tensor] = deque(maxlen=memory_size)

    def parameter_names(self) -> tuple[str, ...]:
        return ("vptta.prompt",)

    def prepare(self, image: Tensor, seed: int) -> Tensor:
        prompt = self.prompt
        if self.memory:
            retrieved = torch.stack(list(self.memory)).mean(dim=0).to(prompt.device)
            prompt = 0.5 * prompt + 0.5 * retrieved
        prompt = F.interpolate(prompt, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return image + self.prompt_strength * torch.tanh(prompt)

    def propose(self, context: StepContext, parameters: dict[str, nn.Parameter]) -> UpdateProposal:
        probabilities = context.student_logits.softmax(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-6).log()).sum(dim=1).mean()
        gradients = gradients_for(entropy, parameters, self.parameter_names())
        return UpdateProposal(
            method=self.name,
            gradients=gradients,
            parameter_names=self.parameter_names(),
            lr_hint=self.lr,
            loss=float(entropy.detach().cpu()),
            metrics={"prompt_memory": float(len(self.memory))},
        )

    def after_prediction(self, context: StepContext) -> None:
        """Publish this image's adapted prompt only after its prediction succeeds."""
        self.memory.append(self.prompt.detach().clone())

    def reset(self) -> None:
        with torch.no_grad():
            self.prompt.zero_()
        self.memory.clear()

