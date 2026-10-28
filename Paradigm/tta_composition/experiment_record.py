"""Hyperparameter snapshots shared by saved results and the desktop display."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import CompositionConfig
from .contracts import METHODS


def hyperparameter_record(config: CompositionConfig) -> dict[str, Any]:
    """Keep all configured values, including settings for inactive methods."""
    paths = {"checkpoint", "output", "image_dir", "mask_dir", "baseline_cache"}
    record = {key: value for key, value in config.to_dict().items() if key not in paths}
    record["methods"] = list(config.canonical_methods)
    record["effective_adaptation_steps"] = config.adaptation_steps if config.canonical_methods else 0
    return record


def format_hyperparameters(
    record: dict[str, Any], *, include_inactive: bool = False, compact: bool = False,
) -> str:
    """Show active settings in the GUI and every setting in the saved text."""
    selected = record["methods"]
    methods = ", ".join(selected) or "Source-only"
    limit = record["max_cases"] if record["max_cases"] is not None else "全部"
    steps = str(record["effective_adaptation_steps"])
    if record["effective_adaptation_steps"] != record["adaptation_steps"]:
        steps += f"（設定值 {record['adaptation_steps']}，本次不適配）"
    lines = [
        f"資料集：{record['dataset']}｜方法：{methods}",
        f"Seed：{record['seed']}｜影像上限：{limit}｜裝置設定：{record['device']}",
        f"前處理：{record['image_size']} × {record['image_size']}｜"
        f"正規化：{record['normalization']}｜每張更新次數：{steps}",
    ]
    if compact:
        if selected:
            lines.append("設定學習率：" + "，".join(f"{method}={record['lrs'][method]}" for method in selected))
        return "\n".join(lines)
    for method in METHODS:
        if method not in selected and not include_inactive:
            continue
        prefix = method.lower() + "_"
        parameters = {"learning_rate": record["lrs"][method]}
        parameters.update({key[len(prefix):]: value for key, value in record.items() if key.startswith(prefix)})
        state = "啟用" if method in selected else "未啟用"
        values = "，".join(f"{key}={value}" for key, value in parameters.items())
        lines.append(f"{method}（{state}）：{values}")
    return "\n".join(lines)


def write_hyperparameters(output_dir: Path, config: CompositionConfig) -> dict[str, Any]:
    """Snapshot parameters before inference so interrupted runs retain them."""
    record = hyperparameter_record(config)
    (output_dir / "hyperparameters.json").write_text(
        json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8",
    )
    (output_dir / "hyperparameters.txt").write_text(
        format_hyperparameters(record, include_inactive=True) + "\n", encoding="utf-8",
    )
    return record
