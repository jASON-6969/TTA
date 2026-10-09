from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .benchmark_config import (
    BASE_OUTPUT,
    BASE_WIDTH,
    DATA_ROOT,
    IMAGE_SIZE,
    INPUT_CHANNELS,
    METHOD_LRS,
    METHOD_OUTPUT,
    METHODS,
    NORMALIZATION,
    NUM_CLASSES,
    SOURCE_SEED,
    TARGET_BATCH_SIZE,
    protocol_audit,
)
from .data import (
    ImageOnlyDataset,
    Record,
    SegmentationDataset,
    audit_sz_cxr_pairs,
    build_montgomery_merged_masks,
    discover_montgomery,
    discover_sz_cxr,
    loader_audit,
    split_source,
)
from .metrics import aggregate
from .model import UNet2D
from .online import evaluate_saved_predictions, run_grata, run_source_only, run_tent, run_testfit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base_checkpoint(path: Path, device: torch.device) -> UNet2D:
    if not path.exists():
        raise FileNotFoundError(f"Base checkpoint is missing: {path}. Run run_base_model first.")
    model = UNet2D(in_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES, base=BASE_WIDTH).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)
    return model


def main() -> None:
    set_seed(SOURCE_SEED)
    METHOD_OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = METHOD_OUTPUT / "execution.log"
    log_path.write_text("", encoding="utf-8")

    def logger(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source_root = DATA_ROOT / "sz_cxr"
    target_root = DATA_ROOT / "montgomery"
    source_pair_audit = audit_sz_cxr_pairs(source_root)
    source = discover_sz_cxr(source_root)
    target_with_masks = build_montgomery_merged_masks(
        discover_montgomery(target_root), target_root / "merged_mask"
    )
    target_images = [Record(record.sample_id, record.image_path, None, record.domain) for record in target_with_masks]
    if not source or not target_with_masks or any(record.mask_path is None for record in source + target_with_masks):
        raise RuntimeError("CXR dataset audit failed")

    source_train, source_val = split_source(source, seed=SOURCE_SEED, val_fraction=0.20)
    audit = protocol_audit(source_pair_audit, source, source_train, source_val, target_with_masks)
    audit["source"] = loader_audit(source)
    audit["target"] = loader_audit(target_with_masks)
    audit["adaptation_records"] = {
        "count": len(target_images),
        "all_mask_path_none": all(record.mask_path is None for record in target_images),
        "domains": sorted({record.domain for record in target_images}),
    }
    audit["base_checkpoint"] = str(BASE_OUTPUT / "source_checkpoint.pth")
    audit["base_checkpoint_sha256"] = sha256(BASE_OUTPUT / "source_checkpoint.pth")
    write_json(METHOD_OUTPUT / "protocol_audit.json", audit)

    base_model = load_base_checkpoint(BASE_OUTPUT / "source_checkpoint.pth", device)
    prediction_root = METHOD_OUTPUT / "predictions"
    for method in ("Source-only", *METHODS):
        (prediction_root / method).mkdir(parents=True, exist_ok=True)

    logger(f"methods start seed={SOURCE_SEED} device={device} target={len(target_images)}")
    adaptation_rows: list[dict] = []
    adaptation_rows.extend(run_source_only(base_model, target_images, IMAGE_SIZE, device, prediction_root / "Source-only", logger, normalization=NORMALIZATION))

    set_seed(SOURCE_SEED)
    rows = run_tent(base_model, target_images, IMAGE_SIZE, device, prediction_root / "TENT", logger, lr=METHOD_LRS["TENT"], normalization=NORMALIZATION)
    adaptation_rows.extend(rows)
    set_seed(SOURCE_SEED)
    rows = run_testfit(base_model, target_images, IMAGE_SIZE, device, prediction_root / "TestFit", logger, lr=METHOD_LRS["TestFit"], normalization=NORMALIZATION)
    adaptation_rows.extend(rows)
    set_seed(SOURCE_SEED)
    rows = run_grata(base_model, target_images, IMAGE_SIZE, device, prediction_root / "GraTa", logger, lr=METHOD_LRS["GraTa"], normalization=NORMALIZATION)
    adaptation_rows.extend(rows)

    metric_rows = evaluate_saved_predictions(target_with_masks, prediction_root, ["Source-only", *METHODS], IMAGE_SIZE)
    write_rows(METHOD_OUTPUT / "adaptation_log.csv", adaptation_rows)
    write_rows(METHOD_OUTPUT / "metrics.csv", metric_rows)
    summary = {
        "experiment": "benchmark_cxr_selected_methods",
        "label": "target-only TTA after shared SZ-CXR source checkpoint",
        "seed": SOURCE_SEED,
        "device": str(device),
        "methods": {
            method: aggregate([row for row in metric_rows if row["method"] == method])
            for method in ("Source-only", *METHODS)
        },
        "adaptation_steps": {method: len([row for row in adaptation_rows if row["method"] == method]) for method in METHODS},
        "base_checkpoint": str(BASE_OUTPUT / "source_checkpoint.pth"),
        "base_checkpoint_sha256": sha256(BASE_OUTPUT / "source_checkpoint.pth"),
        "limitations": audit["fidelity_notes"],
    }
    write_json(METHOD_OUTPUT / "summary.json", summary)
    write_json(
        METHOD_OUTPUT / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "source_seed": SOURCE_SEED,
            "methods": list(METHODS),
        },
    )
    logger("methods completed")


if __name__ == "__main__":
    main()
