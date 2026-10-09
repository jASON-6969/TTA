from __future__ import annotations

import csv
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .benchmark_config import (
    BASE_OUTPUT,
    BASE_WIDTH,
    DATA_ROOT,
    IMAGE_SIZE,
    INPUT_CHANNELS,
    NUM_CLASSES,
    NORMALIZATION,
    SOURCE_AUGMENT,
    SOURCE_BATCH_SIZE,
    SOURCE_EPOCHS,
    SOURCE_LR,
    SOURCE_PATIENCE,
    SOURCE_SEED,
    SOURCE_VAL_FRACTION,
    TARGET_BATCH_SIZE,
    protocol_audit,
)
from .data import (
    SegmentationDataset,
    audit_sz_cxr_pairs,
    build_montgomery_merged_masks,
    discover_montgomery,
    discover_sz_cxr,
    loader_audit,
    split_source,
)
from .methods import evaluate, train_source
from .model import UNet2D
from .metrics import aggregate


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


def main() -> None:
    set_seed(SOURCE_SEED)
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = BASE_OUTPUT / "execution.log"
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
    target = build_montgomery_merged_masks(
        discover_montgomery(target_root), target_root / "merged_mask"
    )
    source_train, source_val = split_source(source, seed=SOURCE_SEED, val_fraction=SOURCE_VAL_FRACTION)

    if not source or not target or any(record.mask_path is None for record in source + target):
        raise RuntimeError("CXR dataset audit failed: paired source and target masks are required")

    audit = protocol_audit(source_pair_audit, source, source_train, source_val, target)
    audit["source"] = loader_audit(source)
    audit["source_train"] = loader_audit(source_train)
    audit["source_validation"] = loader_audit(source_val)
    audit["target"] = loader_audit(target)
    write_json(BASE_OUTPUT / "protocol_audit.json", audit)

    logger(f"base start seed={SOURCE_SEED} device={device} source_pairs={len(source)} target={len(target)}")
    train_loader = DataLoader(
        SegmentationDataset(source_train, IMAGE_SIZE, normalization=NORMALIZATION, augment=SOURCE_AUGMENT),
        batch_size=SOURCE_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        SegmentationDataset(source_val, IMAGE_SIZE, normalization=NORMALIZATION),
        batch_size=TARGET_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    target_loader = DataLoader(
        SegmentationDataset(target, IMAGE_SIZE, normalization=NORMALIZATION),
        batch_size=TARGET_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    checkpoint = BASE_OUTPUT / "source_checkpoint.pth"
    model = UNet2D(in_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES, base=BASE_WIDTH).to(device)
    source_training = train_source(
        model,
        train_loader,
        val_loader,
        device,
        epochs=SOURCE_EPOCHS,
        patience=SOURCE_PATIENCE,
        lr=SOURCE_LR,
        checkpoint_path=checkpoint,
        logger=logger,
    )
    audit["source_training"] = source_training
    write_json(BASE_OUTPUT / "protocol_audit.json", audit)

    target_rows = evaluate(model, target_loader, device, "Source-only", BASE_OUTPUT / "predictions", logger)
    write_rows(BASE_OUTPUT / "metrics.csv", target_rows)
    write_json(
        BASE_OUTPUT / "summary.json",
        {
            "experiment": "benchmark_cxr_base",
            "label": "source-only base model before TTA",
            "method": "Source-only",
            "dataset_direction": "SZ-CXR -> Montgomery",
            "source_training": source_training,
            "target_metrics": aggregate(target_rows),
            "target_count": len(target_rows),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "limitations": audit["fidelity_notes"],
        },
    )
    write_json(
        BASE_OUTPUT / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "source_seed": SOURCE_SEED,
        },
    )
    logger("base completed")


if __name__ == "__main__":
    main()
