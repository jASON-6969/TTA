from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random

import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import torch


@dataclass(frozen=True)
class Record:
    sample_id: str
    image_path: Path
    mask_path: Path | None
    domain: str


def _normalize_array(array: np.ndarray, normalization: str) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if normalization == "zscore":
        return (array - float(array.mean())) / max(float(array.std()), 1e-8)
    if normalization != "percentile":
        raise ValueError(f"Unsupported image normalization: {normalization}")
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        low, high = float(array.min()), float(array.max())
    return np.clip((array - low) / max(high - low, 1e-6), 0.0, 1.0)


class ImageOnlyDataset(Dataset):
    """Target adaptation dataset. It deliberately exposes no mask field."""
    def __init__(self, records: list[Record], image_size: int = 256, normalization: str = "percentile"):
        self.records = records
        self.image_size = image_size
        self.normalization = normalization

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        image = load_image(rec.image_path, self.image_size, self.normalization)
        return image, rec.sample_id


class SegmentationDataset(Dataset):
    def __init__(self, records: list[Record], image_size: int = 256, normalization: str = "percentile", augment: bool = False):
        if any(r.mask_path is None for r in records):
            raise ValueError("SegmentationDataset requires masks for every record")
        self.records = records
        self.image_size = image_size
        self.normalization = normalization
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        image_pil = Image.open(rec.image_path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask_pil = Image.open(rec.mask_path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        if self.augment:
            if random.random() < 0.5:
                image_pil = image_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            angle = random.uniform(-10.0, 10.0)
            image_pil = image_pil.rotate(angle, resample=Image.Resampling.BILINEAR)
            mask_pil = mask_pil.rotate(angle, resample=Image.Resampling.NEAREST)
            image_pil = ImageEnhance.Brightness(image_pil).enhance(random.uniform(0.8, 1.2))
            image_pil = ImageEnhance.Contrast(image_pil).enhance(random.uniform(0.8, 1.2))
        image_array = np.asarray(image_pil, dtype=np.float32)
        image = torch.from_numpy(_normalize_array(image_array, self.normalization)).unsqueeze(0)
        mask = torch.from_numpy((np.asarray(mask_pil, dtype=np.uint8) > 127).astype(np.int64))
        return image, mask, rec.sample_id


def load_image(path: Path, image_size: int, normalization: str = "percentile") -> torch.Tensor:
    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    array = _normalize_array(array, normalization)
    return torch.from_numpy(array).unsqueeze(0)


def load_mask(path: Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.int64)
    return torch.from_numpy(array)


def discover_jsrt(root: Path) -> list[Record]:
    image_dir = root / "extracted" / "content" / "jsrt" / "cxr"
    mask_dir = root / "extracted" / "content" / "jsrt" / "masks"
    records = []
    for image_path in sorted(image_dir.glob("*.png")):
        mask_path = mask_dir / image_path.name
        if mask_path.exists():
            records.append(Record(image_path.stem, image_path, mask_path, "JSRT"))
    return records


def discover_sz_cxr(root: Path) -> list[Record]:
    """Discover Shenzhen CXR images that have paired lung masks.

    The prepared local layout is ``images/CHNCXR_####_0.png`` and
    ``masks/CHNCXR_####_0_mask.png`` (and ``_1`` for the second cohort).
    """
    image_dir = root / "images"
    mask_dir = root / "masks"
    records = []
    for image_path in sorted(image_dir.glob("CHNCXR_*.png")):
        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        if mask_path.exists():
            records.append(Record(image_path.stem, image_path, mask_path, "SZ-CXR"))
    return records


def audit_sz_cxr_pairs(root: Path) -> dict:
    """Report source image/mask coverage without silently dropping examples."""
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_ids = {path.stem for path in image_dir.glob("CHNCXR_*.png")}
    mask_ids = {path.stem.removesuffix("_mask") for path in mask_dir.glob("CHNCXR_*_mask.png")}
    matched = image_ids & mask_ids
    return {
        "image_count": len(image_ids),
        "mask_count": len(mask_ids),
        "matched_count": len(matched),
        "images_without_masks": sorted(image_ids - mask_ids),
        "masks_without_images": sorted(mask_ids - image_ids),
    }


def discover_montgomery(root: Path) -> list[Record]:
    image_dir = root / "images"
    left_dir = root / "left_mask"
    right_dir = root / "right_mask"
    records = []
    for image_path in sorted(image_dir.glob("*.png")):
        left = left_dir / image_path.name
        right = right_dir / image_path.name
        if left.exists() and right.exists():
            records.append(Record(image_path.stem, image_path, None, "Montgomery"))
    return records


def build_montgomery_merged_masks(records: list[Record], output_dir: Path) -> list[Record]:
    """Merge the official left/right masks before evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.parent
    left_dir = root / "left_mask"
    right_dir = root / "right_mask"
    merged = []
    for rec in records:
        left = np.asarray(Image.open(left_dir / rec.image_path.name).convert("L"), dtype=np.uint8)
        right = np.asarray(Image.open(right_dir / rec.image_path.name).convert("L"), dtype=np.uint8)
        mask = np.where((left > 127) | (right > 127), 255, 0).astype(np.uint8)
        mask_path = output_dir / rec.image_path.name
        if not mask_path.exists():
            Image.fromarray(mask).save(mask_path)
        merged.append(Record(rec.sample_id, rec.image_path, mask_path, rec.domain))
    return merged


def split_source(records: list[Record], seed: int, val_fraction: float = 0.2):
    records = list(records)
    random.Random(seed).shuffle(records)
    cut = max(1, int(round(len(records) * (1.0 - val_fraction))))
    return records[:cut], records[cut:]


def loader_audit(records: Iterable[Record]) -> dict:
    records = list(records)
    missing = [r.sample_id for r in records if not r.image_path.exists() or (r.mask_path is not None and not r.mask_path.exists())]
    return {
        "count": len(records),
        "missing_files": missing,
        "domains": sorted({r.domain for r in records}),
        "sample_ids_unique": len({r.sample_id for r in records}) == len(records),
    }
