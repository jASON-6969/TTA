"""Bridge to the existing five-level UNet checkpoint and CXR data code."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import torch
from torch import Tensor, nn


def _load_pilot_modules():
    test_v1 = Path(__file__).resolve().parents[2]
    code_root = test_v1 / "code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from feature_alignment_2d_pilot.model import UNet2D, entropy_loss, freeze_bn_running_stats

    return UNet2D, entropy_loss, freeze_bn_running_stats


class SmartAdapter(nn.Module):
    """Small identity-initialised style/output adapter for 2D CXR inputs."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.style_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 2),
        )
        self.style_scale = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.style_shift = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.augmentation_logits = nn.Parameter(torch.zeros(3))
        self.augmentation_strength = nn.Parameter(torch.zeros(1))
        self.output_correction = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        nn.init.zeros_(self.output_correction.weight)
        nn.init.zeros_(self.output_correction.bias)
        # Keep hidden features image-dependent. A zero final projection starts
        # at identity, learns first, and then passes gradients to earlier layers.
        style_head = self.style_encoder[-1]
        assert isinstance(style_head, nn.Linear)
        nn.init.zeros_(style_head.weight)
        nn.init.zeros_(style_head.bias)

    def transform(self, image: Tensor) -> Tensor:
        # The encoder is intentionally bounded so its initial identity is stable.
        style = self.style_encoder(image)
        scale = self.style_scale * (1.0 + 0.05 * torch.tanh(style[:, :1]).unsqueeze(-1).unsqueeze(-1))
        shift = self.style_shift + 0.05 * torch.tanh(style[:, 1:2]).unsqueeze(-1).unsqueeze(-1)
        return image * scale + shift

    def correct(self, logits: Tensor) -> Tensor:
        return logits + self.output_correction(logits)

    def augment(self, image: Tensor, seed: int) -> Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype).to(image.device)
        center = image.mean(dim=(-2, -1), keepdim=True)
        candidates = torch.stack(
            (torch.zeros_like(image), noise * 0.03, (image - center) * 0.05),
            dim=0,
        )
        weights = self.augmentation_logits.softmax(dim=0).view(3, 1, 1, 1, 1)
        perturbation = (weights * candidates).sum(dim=0)
        return image + torch.tanh(self.augmentation_strength) * perturbation


@dataclass
class ModelBundle:
    reference: nn.Module
    student: nn.Module
    smart: SmartAdapter
    ema: nn.Module | None
    entropy_loss: object
    freeze_bn_running_stats: Callable[[nn.Module], None]


def build_model_bundle(
    checkpoint: Path,
    device: torch.device,
    *,
    image_channels: int = 1,
    num_classes: int = 2,
    base_width: int = 32,
    enable_smart: bool = False,
) -> ModelBundle:
    UNet2D, entropy_loss, freeze_bn_running_stats = _load_pilot_modules()
    reference = UNet2D(in_channels=image_channels, num_classes=num_classes, base=base_width).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    reference.load_state_dict(state_dict)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    student = deepcopy(reference).to(device)
    # SmaRT is optional. Keep its construction from advancing the caller's
    # RNG so a closed module cannot perturb another method's random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        smart = SmartAdapter(num_classes=num_classes).to(device)
    ema = deepcopy(student).to(device) if enable_smart else None
    if ema is not None:
        ema.eval()
        for parameter in ema.parameters():
            parameter.requires_grad = False
    return ModelBundle(reference, student, smart, ema, entropy_loss, freeze_bn_running_stats)


def model_forward(model: nn.Module, image: Tensor) -> tuple[Tensor, Tensor]:
    output = model(image, return_features=True)
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("The composition bridge requires UNet forward(..., return_features=True)")
    return output


def checked_probabilities(logits: Tensor, model_name: str) -> Tensor:
    """Reject invalid floating-point outputs before converting them to labels."""
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError(f"Non-finite {model_name} logits")
    probabilities = logits.softmax(dim=1)
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError(f"Non-finite {model_name} probabilities")
    return probabilities


def predict_source(model: nn.Module, image: Tensor, device: torch.device) -> Tensor:
    """Predict with the frozen source model without creating adaptation state."""
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        logits, _ = model_forward(model, image)
        return checked_probabilities(logits, "baseline").argmax(dim=1).cpu()


def bn_parameter_names(model: nn.Module) -> tuple[str, ...]:
    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            prefix = f"{name}." if name else ""
            names.extend((f"{prefix}weight", f"{prefix}bias"))
    return tuple(names)


def non_bn_parameter_names(model: nn.Module) -> tuple[str, ...]:
    bn_names = set(bn_parameter_names(model))
    return tuple(name for name, _ in model.named_parameters() if name not in bn_names)


def set_student_modes(bundle: ModelBundle) -> None:
    bundle.student.train()
    bundle.freeze_bn_running_stats(bundle.student)
    bundle.reference.eval()
    if bundle.ema is not None:
        bundle.ema.eval()


def source_buffers_snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.named_buffers()
        if "running_" in name or "num_batches_tracked" in name
    }

