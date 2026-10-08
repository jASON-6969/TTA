"""Image-only target discovery; mask contents are read after adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATASETS = ("montgomery", "sz_cxr", "custom")
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"})


@dataclass(frozen=True)
class TargetCase:
    sample_id: str
    image_path: Path
    mask_paths: tuple[Path, ...] = ()


def _index_images(directory: Path, *, required: bool) -> dict[str, Path]:
    """Index one directory without decoding an image or following subfolders."""
    if not directory.is_dir():
        if required:
            raise FileNotFoundError(f"Image/mask directory does not exist: {directory}")
        return {}
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        key = path.stem.casefold()
        if key in indexed:
            raise ValueError(f"Duplicate image/mask stem '{path.stem}': {indexed[key]} and {path}")
        indexed[key] = path
    return indexed


def _match_mask(sample_id: str, indexed: dict[str, Path]) -> Path | None:
    key = sample_id.casefold()
    candidates = [indexed[name] for name in (key, f"{key}_mask") if name in indexed]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous masks for '{sample_id}': {candidates}")
    return candidates[0] if candidates else None


def discover_targets(
    test_v1: Path,
    dataset: str,
    image_dir: Path | None = None,
    mask_dir: Path | None = None,
) -> list[TargetCase]:
    """Discover all target images, retaining unlabelled cases for prediction.

    Custom datasets require an image directory and may omit masks. Montgomery
    uses both official lung masks unless a combined-mask directory is supplied.
    Missing individual masks never remove an image from the adaptation stream.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'; choices={DATASETS}")
    if dataset == "custom" and image_dir is None:
        raise ValueError("Custom dataset requires image_dir")
    dataset_root = test_v1 / "data" / "raw" / dataset
    resolved_images = Path(image_dir) if image_dir is not None else dataset_root / "images"
    if image_dir is not None and dataset != "custom":
        # An overridden built-in image folder must use its own sibling labels.
        dataset_root = resolved_images.parent
    images = _index_images(resolved_images, required=True)
    if not images:
        raise ValueError(f"No supported target images found in: {resolved_images}")

    if mask_dir is not None:
        mask_indexes = [_index_images(Path(mask_dir), required=True)]
    elif dataset == "montgomery":
        mask_indexes = [
            _index_images(dataset_root / "left_mask", required=False),
            _index_images(dataset_root / "right_mask", required=False),
        ]
    elif dataset == "sz_cxr":
        mask_indexes = [_index_images(dataset_root / "masks", required=False)]
    else:
        mask_indexes = []

    cases: list[TargetCase] = []
    for image_path in images.values():
        matched = tuple(_match_mask(image_path.stem, indexed) for indexed in mask_indexes)
        # One lung mask alone is an incomplete label; keep the image unlabelled.
        masks = tuple(path for path in matched if path is not None) if matched and all(matched) else ()
        cases.append(TargetCase(image_path.stem, image_path, masks))
    return cases
