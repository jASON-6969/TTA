"""Complete, image-only source inference caches shared by TTA experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from PIL import Image, __version__ as pillow_version
import torch
from torch import Tensor

from .audit import sha256
from .data_selection import TargetCase


SCHEMA_VERSION = 1
INFERENCE_VERSION = "source-only-v1"
MANIFEST_FILENAME = "baseline_manifest.json"
READY_FILENAME = "READY.json"


@dataclass(frozen=True)
class BaselineResult:
    cache_key: str
    directory: Path
    prediction_dir: Path
    cases: int
    reused: bool
    elapsed_seconds: float


def _inference_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    pilot = package.parents[1] / "code" / "feature_alignment_2d_pilot"
    sources = {
        "baseline.py": package / "baseline.py",
        "model_bridge.py": package / "model_bridge.py",
        "pilot/model.py": pilot / "model.py",
        "pilot/data.py": pilot / "data.py",
    }
    return {name: sha256(path) for name, path in sources.items() if path.is_file()}


def _runtime_identity() -> dict[str, str | None]:
    return {
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pillow": pillow_version,
        "cuda": torch.version.cuda,
    }


def _prediction_identity(
    records: list[TargetCase], checkpoint: Path, image_size: int,
    normalization: str, device: str,
) -> tuple[dict[str, Any], list[TargetCase]]:
    if not records:
        raise ValueError("Baseline requires at least one target image")
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
        raise ValueError("Baseline image_size must be a positive integer")
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    image_only: list[TargetCase] = []
    for record in records:
        sample_id = record.sample_id
        if not sample_id or any(character in sample_id for character in '/\\:*?"<>|') or sample_id in (".", ".."):
            raise ValueError(f"Invalid baseline image ID: {sample_id!r}")
        image_path = Path(record.image_path).resolve(strict=True)
        if sample_id.casefold() in seen_ids or image_path in seen_paths:
            raise ValueError(f"Duplicate baseline image: {sample_id}")
        if not image_path.is_file():
            raise ValueError(f"Baseline image is not a file: {image_path}")
        seen_ids.add(sample_id.casefold())
        seen_paths.add(image_path)
        # Neither cached identity nor the inference callback receives labels.
        image_only.append(TargetCase(sample_id, image_path))
    image_only.sort(key=lambda record: record.sample_id.casefold())
    identity = {
        "schema_version": SCHEMA_VERSION,
        "inference_version": INFERENCE_VERSION,
        "checkpoint_sha256": sha256(Path(checkpoint).resolve(strict=True)),
        "image_size": image_size,
        "normalization": normalization,
        "device": str(device),
        "runtime": _runtime_identity(),
        "inference_sources": _inference_source_hashes(),
        "images": [
            {"image_id": record.sample_id, "image_path": str(record.image_path),
             "image_sha256": sha256(record.image_path)}
            for record in image_only
        ],
    }
    return identity, image_only


def _identity_key(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _binary_prediction(prediction: np.ndarray | Tensor, image_size: int) -> np.ndarray:
    if isinstance(prediction, Tensor):
        prediction = prediction.detach().float().cpu().numpy()
    values = np.asarray(prediction)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.shape != (image_size, image_size):
        raise ValueError(f"Baseline prediction shape {values.shape} != {(image_size, image_size)}")
    if not np.isfinite(values).all() or not np.isin(values, (0, 1, 255)).all():
        raise ValueError("Baseline predictions must be finite binary masks (0/1 or 0/255)")
    return (values > 0).astype(np.uint8) * 255


def _cache_is_ready(directory: Path, cache_key: str, identity: dict[str, Any]) -> bool:
    """Validate inference files; later evaluation reports do not affect readiness."""
    try:
        ready = json.loads((directory / READY_FILENAME).read_text(encoding="utf-8"))
        manifest_path = directory / MANIFEST_FILENAME
        if ready.get("cache_key") != cache_key or ready.get("schema_version") != SCHEMA_VERSION:
            return False
        if ready.get("manifest_sha256") != sha256(manifest_path):
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        images = identity["images"]
        if manifest.get("cache_key") != cache_key or manifest.get("identity") != identity or manifest.get("cases") != len(images):
            return False
        elapsed = manifest.get("build_elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            return False
        predictions = manifest.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(images):
            return False
        expected_names = {f"{record['image_id']}.png" for record in images}
        prediction_dir = directory / "predictions"
        if {path.name for path in prediction_dir.iterdir()} != expected_names:
            return False
        for record, prediction in zip(images, predictions):
            filename = f"{record['image_id']}.png"
            if prediction.get("image_id") != record["image_id"] or prediction.get("file") != f"predictions/{filename}":
                return False
            if prediction.get("shape") != [identity["image_size"], identity["image_size"]]:
                return False
            path = prediction_dir / filename
            if prediction.get("sha256") != sha256(path):
                return False
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "L" or image.size != (identity["image_size"], identity["image_size"]):
                    return False
                if not np.isin(np.asarray(image), (0, 255)).all():
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def _emit(progress: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    if progress is not None:
        progress(event)


def _publish(staging: Path, directory: Path, cache_key: str, identity: dict[str, Any]) -> None:
    """Publish in one rename, preserving damaged caches as quarantine directories."""
    for _attempt in range(3):
        if directory.exists():
            if _cache_is_ready(directory, cache_key, identity):
                # Another process completed this same source inference meanwhile.
                return
            quarantine = directory.with_name(f"{directory.name}.quarantine-{uuid4().hex}")
            try:
                directory.rename(quarantine)
            except FileNotFoundError:
                continue
        try:
            staging.rename(directory)
            return
        except OSError:
            if not directory.exists():
                raise
    raise RuntimeError(f"Could not publish baseline cache: {directory}")


def ensure_baseline(
    cache_root: Path,
    records: list[TargetCase],
    checkpoint: Path,
    image_size: int,
    normalization: str,
    device: str,
    predict: Callable[[TargetCase], np.ndarray | Tensor],
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> BaselineResult:
    """Build every supplied image once, or reuse a complete verified prediction cache.

    Callers supply the full dataset, before any TTA case limit. Label files and
    TTA methods, seed and learning rates are excluded from prediction identity.
    ``elapsed_seconds`` measures this call, including cache checks; the manifest
    separately records the original inference build duration.
    """
    started = time.perf_counter()
    identity, image_only = _prediction_identity(records, checkpoint, image_size, normalization, device)
    cache_key = _identity_key(identity)
    cache_root = Path(cache_root).resolve()
    directory = cache_root / cache_key
    event = {"event": "baseline", "total": len(image_only), "cache_key": cache_key, "directory": str(directory)}
    _emit(progress, {**event, "status": "checking"})
    if _cache_is_ready(directory, cache_key, identity):
        _emit(progress, {**event, "status": "reused"})
        return BaselineResult(cache_key, directory, directory / "predictions", len(image_only), True, time.perf_counter() - started)

    cache_root.mkdir(parents=True, exist_ok=True)
    _emit(progress, {**event, "status": "building"})
    build_started = time.perf_counter()
    # A failed/cancelled build is cleaned up; only a complete cache is published.
    with TemporaryDirectory(prefix=f".{cache_key}.partial-", dir=cache_root) as temporary:
        staging = Path(temporary)
        prediction_dir = staging / "predictions"
        prediction_dir.mkdir()
        predictions = []
        for completed, record in enumerate(image_only, start=1):
            values = _binary_prediction(predict(record), image_size)
            filename = f"{record.sample_id}.png"
            path = prediction_dir / filename
            Image.fromarray(values).save(path, format="PNG")
            predictions.append({"image_id": record.sample_id, "file": f"predictions/{filename}",
                                "shape": list(values.shape), "sha256": sha256(path)})
            _emit(progress, {"event": "baseline_progress", "completed": completed,
                             "total": len(image_only), "image_id": record.sample_id})

        # Do not publish predictions under an identity whose inputs changed mid-run.
        current_identity, _records = _prediction_identity(records, checkpoint, image_size, normalization, device)
        if current_identity != identity:
            raise RuntimeError("Baseline inputs changed while source inference was running; retry the experiment")
        manifest = {"cache_key": cache_key, "identity": identity, "cases": len(image_only),
                    "checkpoint": str(Path(checkpoint).resolve()), "predictions": predictions,
                    "build_elapsed_seconds": time.perf_counter() - build_started}
        _write_json(staging / MANIFEST_FILENAME, manifest)
        _write_json(staging / READY_FILENAME, {"cache_key": cache_key, "schema_version": SCHEMA_VERSION,
                                             "manifest_sha256": sha256(staging / MANIFEST_FILENAME)})
        if not _cache_is_ready(staging, cache_key, identity):
            raise RuntimeError("Baseline inference cache did not pass final validation")
        _publish(staging, directory, cache_key, identity)
    _emit(progress, {**event, "status": "ready"})
    return BaselineResult(cache_key, directory, directory / "predictions", len(image_only), False, time.perf_counter() - started)
