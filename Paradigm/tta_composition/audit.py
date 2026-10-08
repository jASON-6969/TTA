"""Reproducibility and per-step audit helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from .contracts import UpdateRecord


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def finite_gradient_norm(gradient: Tensor, name: str) -> float:
    """Accumulate float32 gradients in float64 before writing a numeric audit."""
    value = float(torch.linalg.vector_norm(gradient.detach().to(dtype=torch.float64)).cpu())
    if not math.isfinite(value):
        raise FloatingPointError(f"Non-finite gradient norm for {name}")
    return value


def record_to_dict(record: UpdateRecord, *, sample_id: str, method_metrics: dict[str, float]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "step": record.step,
        "image_id": sample_id,
        "losses": dict(record.losses),
        "lr_hints": dict(record.lr_hints),
        "final_learning_rates": dict(record.final_learning_rates),
        "gradient_norms": dict(record.gradient_norms),
        "updated_parameters": record.updated_parameters,
        "elapsed_seconds": record.elapsed_seconds,
        "method_metrics": method_metrics,
    }
    if record.iterations:
        value["iterations"] = [
            record_to_dict(iteration, sample_id=sample_id, method_metrics={})
            for iteration in record.iterations
        ]
    return value


def adaptation_provenance(adaptation_steps: int) -> dict[str, Any]:
    """Snapshot adaptation code separately from the source-only inference cache."""
    package = Path(__file__).resolve().parent
    sources = [package / name for name in (
        "run.py", "engine.py", "config.py", "contracts.py", "audit.py",
        "lr_policy.py", "parameter_registry.py", "model_bridge.py",
    )]
    sources.extend(sorted((package / "methods").glob("*.py")))
    return {
        "schema_version": 1,
        "workspace_root": str(package.parents[1]),
        "after_commit_policy": "once_per_update",
        "adaptation_steps": adaptation_steps,
        "update_log_version": 2,
        "adaptation_sources": {
            path.relative_to(package).as_posix(): sha256(path)
            for path in sources
        },
    }

