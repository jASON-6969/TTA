from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """Five-level 2D UNet with a binary output."""
    def __init__(self, in_channels: int = 1, num_classes: int = 2, base: int = 32):
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc1 = ConvBlock(in_channels, widths[0])
        self.enc2 = ConvBlock(widths[0], widths[1])
        self.enc3 = ConvBlock(widths[1], widths[2])
        self.enc4 = ConvBlock(widths[2], widths[3])
        self.bottleneck = ConvBlock(widths[3], widths[4])
        self.up4 = nn.ConvTranspose2d(widths[4], widths[3], 2, stride=2)
        self.dec4 = ConvBlock(widths[4], widths[3])
        self.up3 = nn.ConvTranspose2d(widths[3], widths[2], 2, stride=2)
        self.dec3 = ConvBlock(widths[3], widths[2])
        self.up2 = nn.ConvTranspose2d(widths[2], widths[1], 2, stride=2)
        self.dec2 = ConvBlock(widths[2], widths[1])
        self.up1 = nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2)
        self.dec1 = ConvBlock(widths[1], widths[0])
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    def forward(self, x, return_features: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.head(d1)
        return (logits, b) if return_features else logits


def segmentation_loss(logits, target):
    ce = F.cross_entropy(logits, target)
    probs = logits.softmax(dim=1)[:, 1]
    target_f = target.float()
    inter = (probs * target_f).flatten(1).sum(1)
    denom = probs.flatten(1).sum(1) + target_f.flatten(1).sum(1)
    dice_loss = 1.0 - ((2.0 * inter + 1e-5) / (denom + 1e-5)).mean()
    return ce + dice_loss


def entropy_loss(logits):
    probs = logits.softmax(dim=1)
    return -(probs * (probs.clamp_min(1e-6)).log()).sum(dim=1).mean()


def configure_bn_adaptation(model: nn.Module):
    """Enable affine adaptation while preserving source BN running statistics."""
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    params = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            # BatchNorm train mode would update running_mean/running_var from
            # each target image and progressively overwrite source statistics.
            # Eval mode keeps those source statistics fixed while affine
            # weight/bias remain differentiable.
            module.eval()
            module.track_running_stats = True
            module.weight.requires_grad = True
            module.bias.requires_grad = True
            params.extend([module.weight, module.bias])
    return params


def freeze_bn_running_stats(model: nn.Module):
    """Keep BatchNorm statistics fixed after a model-wide train() call."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coefficient * grad_output, None


def gradient_reverse(x, coefficient: float):
    return GradientReverse.apply(x, coefficient)
