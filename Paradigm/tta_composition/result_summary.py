"""Human-readable final results shared by the desktop log and summary panel."""

from __future__ import annotations

import math
from typing import Any


def _metric(metrics: dict[str, Any], name: str, digits: int = 4) -> str:
    mean = metrics.get(f"{name}_mean")
    if mean is None:
        return "未計算"
    value = float(mean)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite result: {name}")
    result = f"{value:.{digits}f}"
    deviation = metrics.get(f"{name}_std")
    if deviation is not None:
        deviation = float(deviation)
        if not math.isfinite(deviation):
            raise ValueError(f"Non-finite result: {name} standard deviation")
        result += f" ± {deviation:.{digits}f}"
    return result


def _comparison_lines(summary: dict[str, Any]) -> list[str]:
    baseline = summary.get("baseline")
    comparison = summary.get("comparison")
    if not isinstance(baseline, dict) or not isinstance(comparison, dict):
        return []
    full_cases = int(baseline["cases"])
    selected_cases = int(comparison["cases"])
    paired = int(comparison["evaluated_cases"])
    cache_status = "重用" if baseline.get("reused") else "首次建立"
    lines = [f"Baseline：{cache_status}完整 {full_cases} 張記錄；本次比較相同 {selected_cases} 張影像。"]
    if paired:
        metrics = comparison["baseline_metrics"]
        lines.append(
            f"Base（同批 {paired} 張有 mask）：Dice {_metric(metrics, 'Dice')}    "
            f"HD95 {_metric(metrics, 'HD95', 2)} px    IoU {_metric(metrics, 'JI')}"
        )
        delta = comparison["delta"]
        values = [float(delta[f"{name}_mean"]) for name in ("Dice", "HD95", "JI")]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite baseline comparison")
        lines.append(
            f"Δ（TTA − Base）：Dice {values[0]:+.4f}    "
            f"HD95 {values[1]:+.2f} px    IoU {values[2]:+.4f}"
        )
        lines.append("Dice／IoU 越大越好；HD95 越小越好。")
    else:
        lines.append("沒有配對 mask，僅比較預測變化，未評估適配品質。")
    changed_fraction = float(comparison["prediction_change_fraction_mean"])
    if not math.isfinite(changed_fraction) or not 0 <= changed_fraction <= 1:
        raise ValueError("Invalid prediction change fraction")
    lines.append(f"相對 Base 的平均預測像素變更率：{changed_fraction:.2%}")
    lines.append(f"Baseline 記錄：{baseline['output']}")
    return lines


def format_result(summary: dict[str, Any]) -> str:
    cases = int(summary["cases"])
    evaluated = int(summary["evaluated_cases"])
    lines = [
        f"實驗完成｜預測 {cases} 張｜評估 {evaluated} 張",
    ]
    if evaluated:
        metrics = summary["metrics"]
        lines.append(
            f"Dice：{_metric(metrics, 'Dice')}    "
            f"HD95：{_metric(metrics, 'HD95', 2)} px    "
            f"IoU：{_metric(metrics, 'JI')}"
        )
        if evaluated < cases:
            lines.append(f"{cases - evaluated} 張影像沒有配對 mask，未納入評估。")
    else:
        lines.append("沒有配對的評估 mask，未計算 Dice／HD95／IoU。")
    lines.extend(_comparison_lines(summary))
    if summary.get("elapsed_seconds") is not None:
        lines.append(f"總耗時：{float(summary['elapsed_seconds']):.2f} 秒")
    lines.append(f"結果目錄：{summary['output']}")
    return "\n".join(lines)
