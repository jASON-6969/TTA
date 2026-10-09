from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image


def hd95_all_foreground(pred: np.ndarray, target: np.ndarray) -> float:
    """Match the vendor 2D benchmark: nearest distances for all foreground pixels."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return 373.1287
    pred_to_target = distance_transform_edt(~target)[pred]
    target_to_pred = distance_transform_edt(~pred)[target]
    return float(np.percentile(np.concatenate([pred_to_target, target_to_pred]), 95))


def binary_metrics(pred: np.ndarray, target: np.ndarray, hd95_mode: str = "all_foreground") -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} != {target.shape}")
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    pred_n, target_n = pred.sum(), target.sum()
    dice = (2.0 * inter) / max(pred_n + target_n, 1)
    union = np.logical_or(pred, target).sum()
    iou = inter / max(union, 1)
    sensitivity = inter / max(target_n, 1)
    ppv = inter / max(pred_n, 1)
    if hd95_mode != "all_foreground":
        raise ValueError(f"Unsupported HD95 mode: {hd95_mode}")
    distance = hd95_all_foreground(pred, target)
    return {"Dice": float(dice), "HD95": distance, "JI": float(iou), "Sensitivity": float(sensitivity), "PPV": float(ppv)}


def aggregate(rows: list[dict]) -> dict:
    keys = ["Dice", "HD95", "JI", "Sensitivity", "PPV"]
    result = {"count": len(rows)}
    for key in keys:
        values = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        result[f"{key}_mean"] = float(values.mean()) if len(values) else None
        result[f"{key}_std"] = float(values.std(ddof=0)) if len(values) else 0.0
        result[f"{key}_median"] = float(np.median(values)) if len(values) else None
    return result


def save_prediction(mask: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)
