"""Prediction-only region selection and differentiable fragment suppression."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import Tensor


def connected_foreground_regions(mask: Tensor) -> list[list[int]]:
    """Return 4-connected regions as flat indices, discovered in row-major order."""
    if mask.ndim != 2:
        raise ValueError("Region discovery requires one 2D mask")
    values = mask.detach().bool().cpu().numpy()
    height, width = values.shape
    visited = np.zeros_like(values, dtype=bool)
    regions: list[list[int]] = []
    for row in range(height):
        for col in range(width):
            if not values[row, col] or visited[row, col]:
                continue
            region: list[int] = []
            pending = deque([(row, col)])
            visited[row, col] = True
            while pending:
                current_row, current_col = pending.popleft()
                region.append(current_row * width + current_col)
                for next_row, next_col in (
                    (current_row - 1, current_col), (current_row + 1, current_col),
                    (current_row, current_col - 1), (current_row, current_col + 1),
                ):
                    if 0 <= next_row < height and 0 <= next_col < width and values[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        pending.append((next_row, next_col))
            regions.append(region)
    return regions


def excess_component_suppression(foreground: Tensor) -> tuple[Tensor, dict[str, float]]:
    """Suppress regions beyond the two largest; only region selection is detached."""
    if foreground.ndim != 4 or foreground.shape[1] != 1 or foreground.numel() == 0:
        raise ValueError("Foreground probabilities require a non-empty [B, 1, H, W] tensor")
    losses: list[Tensor] = []
    component_counts: list[int] = []
    for sample in foreground[:, 0]:
        regions = connected_foreground_regions(sample.detach() > 0.5)
        component_counts.append(len(regions))
        # Stable area sorting keeps row-major discovery order when areas tie.
        ranked = sorted(regions, key=lambda region: -len(region))
        sample_loss = sample.sum() * 0.0
        extra_indices: list[int] = []
        region_weights: list[float] = []
        for region in ranked[2:]:
            extra_indices.extend(region)
            region_weights.extend([1.0 / len(region)] * len(region))
        if extra_indices:
            # One gather/reduction avoids a separate device operation per island.
            indices = torch.tensor(extra_indices, device=sample.device, dtype=torch.long)
            weights = sample.new_tensor(region_weights)
            sample_loss = sample_loss + 0.01 * (sample.flatten()[indices] * weights).sum()
        losses.append(sample_loss)
    loss = torch.stack(losses).mean()
    return loss, {
        "foreground_components": sum(component_counts) / len(component_counts),
        "extra_component_loss": float(loss.detach().cpu()),
    }
