"""Compare TTA and source predictions on the same selected target cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .data_selection import TargetCase


QUALITY_METRICS = ("Dice", "HD95", "JI", "Sensitivity", "PPV")
COMPARISON_FIELDS = (
    "image_id", "evaluated", "changed_pixels", "total_pixels", "prediction_change_fraction",
    *(f"{prefix}_{metric}" for metric in QUALITY_METRICS for prefix in ("baseline", "tta", "delta")),
)


@dataclass(frozen=True)
class ComparisonResult:
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


def _index_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["image_id"])
        if sample_id in indexed:
            raise ValueError(f"Duplicate metric rows for {sample_id}")
        indexed[sample_id] = row
    return indexed


def _read_prediction(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def _aggregate_metrics(
    rows: list[dict[str, Any]],
    aggregate: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    metrics = dict(aggregate(rows))
    metrics.pop("count", None)
    return metrics


def compare_predictions(
    records: list[TargetCase],
    baseline_prediction_dir: Path,
    tta_prediction_dir: Path,
    baseline_rows: list[dict[str, Any]],
    tta_rows: list[dict[str, Any]],
    aggregate: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> ComparisonResult:
    """Use matched labels for quality deltas and all cases for prediction changes.

    Label pixels are not read here. Both metric lists come from evaluation after
    all adaptation steps, and the full baseline is restricted to this run's IDs.
    """
    baseline_by_id = _index_metrics(baseline_rows)
    tta_by_id = _index_metrics(tta_rows)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    paired_baseline: list[dict[str, Any]] = []
    paired_tta: list[dict[str, Any]] = []
    for record in records:
        sample_id = record.sample_id
        if sample_id in seen:
            raise ValueError(f"Duplicate comparison case {sample_id}")
        seen.add(sample_id)
        baseline = _read_prediction(baseline_prediction_dir / f"{sample_id}.png")
        prediction = _read_prediction(tta_prediction_dir / f"{sample_id}.png")
        if baseline.shape != prediction.shape:
            raise ValueError(f"Baseline/TTA prediction shape mismatch for {sample_id}")
        changed = int(np.count_nonzero(baseline != prediction))
        evaluated = sample_id in tta_by_id
        if evaluated != (sample_id in baseline_by_id):
            raise ValueError(f"Baseline/TTA label coverage mismatch for {sample_id}")
        row: dict[str, Any] = {
            "image_id": sample_id,
            "evaluated": evaluated,
            "changed_pixels": changed,
            "total_pixels": int(prediction.size),
            "prediction_change_fraction": changed / prediction.size,
        }
        for metric in QUALITY_METRICS:
            baseline_value = float(baseline_by_id[sample_id][metric]) if evaluated else None
            tta_value = float(tta_by_id[sample_id][metric]) if evaluated else None
            row[f"baseline_{metric}"] = baseline_value
            row[f"tta_{metric}"] = tta_value
            row[f"delta_{metric}"] = (
                tta_value - baseline_value
                if tta_value is not None and baseline_value is not None else None
            )
        if evaluated:
            paired_baseline.append(baseline_by_id[sample_id])
            paired_tta.append(tta_by_id[sample_id])
        rows.append(row)
    unexpected_ids = set(tta_by_id) - seen
    if unexpected_ids:
        raise ValueError(f"TTA metrics contain unselected cases: {sorted(unexpected_ids)}")
    baseline_metrics = _aggregate_metrics(paired_baseline, aggregate)
    tta_metrics = _aggregate_metrics(paired_tta, aggregate)
    delta: dict[str, float | None] = {}
    for metric in QUALITY_METRICS:
        key = f"{metric}_mean"
        delta[key] = (
            float(tta_metrics[key]) - float(baseline_metrics[key])
            if paired_tta else None
        )
    return ComparisonResult(
        summary={
            "cases": len(records),
            "evaluated_cases": len(paired_tta),
            "baseline_metrics": baseline_metrics,
            "tta_metrics": tta_metrics,
            "delta": delta,
            "prediction_changed_cases": sum(bool(row["changed_pixels"]) for row in rows),
            "prediction_change_fraction_mean": (
                sum(float(row["prediction_change_fraction"]) for row in rows) / len(rows)
                if rows else 0.0
            ),
            "delta_definition": "TTA minus baseline; Dice/IoU higher is better, HD95 lower is better",
        },
        rows=rows,
    )
