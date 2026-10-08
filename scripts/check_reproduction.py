"""Check the clone's environment, benchmark data and optional source checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from collections.abc import Sequence


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "base_model" / "benchmark_cxr_base_no_source_aug" / "source_checkpoint.pth"


class ReproductionError(RuntimeError):
    """A required benchmark input does not match the documented protocol."""


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ReproductionError(f"{label}: expected {expected}, got {actual}")


def _require_materialized(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(64)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ReproductionError(f"Git LFS object not downloaded: {path}. Run git lfs install and git lfs pull")


def check_environment(root: Path) -> dict[str, object]:
    _require_equal("Python major/minor", sys.version_info[:2], (3, 12))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "code"))
    versions = {name: version(name) for name in ("torch", "numpy", "scipy", "Pillow")}
    for module_name in (
        "feature_alignment_2d_pilot.run_base_model",
        "feature_alignment_2d_pilot.run_selected_methods",
        "Paradigm.tta_composition.run",
        "verify_tta_improvements",
    ):
        module = importlib.import_module(module_name)
        if module.__file__ is None or not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise ReproductionError(f"{module_name} was imported from outside this clone")
    torch = importlib.import_module("torch")
    return {
        "python": sys.version.split()[0],
        "packages": versions,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "bundle_root": str(root.resolve()),
    }


def check_split(root: Path, train_ids: Sequence[str], validation_ids: Sequence[str]) -> None:
    split_dir = root / "data" / "splits"
    split = json.loads((split_dir / "source_split_seed42.json").read_text(encoding="utf-8"))
    train, validation = list(train_ids), list(validation_ids)
    _require_equal("Split seed", split["seed"], 42)
    _require_equal("Split validation fraction", split["validation_fraction"], 0.2)
    _require_equal("Split paired count", split["paired_source_count"], len(train) + len(validation))
    _require_equal("Split train count", split["train_count"], len(train))
    _require_equal("Split validation count", split["validation_count"], len(validation))
    if len(set(train + validation)) != len(train) + len(validation):
        raise ReproductionError("Source split contains duplicate or overlapping IDs")
    for key, filename, actual in (
        ("train_ids", "source_train.txt", train),
        ("validation_ids", "source_validation.txt", validation),
    ):
        saved_text = (split_dir / filename).read_text(encoding="utf-8").splitlines()
        if actual != split[key] or actual != saved_text:
            raise ReproductionError(f"Computed {key} or order differs from the saved split files")


def check_data(root: Path) -> dict[str, object]:
    for path in (root / "data" / "raw").rglob("*.png"):
        _require_materialized(path)
    sys.path.insert(0, str(root / "code"))
    data = importlib.import_module("feature_alignment_2d_pilot.data")
    source_root = root / "data" / "raw" / "sz_cxr"
    target_root = root / "data" / "raw" / "montgomery"
    audit = data.audit_sz_cxr_pairs(source_root)
    _require_equal(f"Source images in {source_root / 'images'}", audit["image_count"], 662)
    _require_equal("Source masks", audit["mask_count"], 566)
    _require_equal("Exact source image-mask pairs", audit["matched_count"], 566)
    source = data.discover_sz_cxr(source_root)
    train, validation = data.split_source(source, seed=42, val_fraction=0.2)
    _require_equal("Source training cases", len(train), 453)
    _require_equal("Source validation cases", len(validation), 113)
    check_split(root, [record.sample_id for record in train], [record.sample_id for record in validation])
    images = {path.name for path in (target_root / "images").glob("*.png")}
    _require_equal(f"Target images in {target_root / 'images'}", len(images), 138)
    for folder in ("left_mask", "right_mask"):
        masks = {path.name for path in (target_root / folder).glob("*.png")}
        if masks != images:
            raise ReproductionError(f"Montgomery {folder} filenames must match all 138 images")
    _require_equal("Paired target cases", len(data.discover_montgomery(target_root)), 138)
    return {"source_images": 662, "source_pairs": 566, "train": 453, "validation": 113, "target_pairs": 138}


def check_checkpoint(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ReproductionError(f"Missing checkpoint: {path}. Run git lfs pull or supply --checkpoint PATH")
    _require_materialized(path)
    torch = importlib.import_module("torch")
    model_module = importlib.import_module("feature_alignment_2d_pilot.model")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    model = model_module.UNet2D(in_channels=1, num_classes=2, base=32)
    model.load_state_dict(state, strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "architecture": "UNet2D/base32/1in/2out"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="store_true", help="Require the complete benchmark data and exact saved source split")
    parser.add_argument("--checkpoint", type=Path, nargs="?", const=CHECKPOINT, help="Validate source weights; without a path, use the default checkpoint")
    args = parser.parse_args(argv)
    try:
        report: dict[str, object] = {"status": "passed", "environment": check_environment(ROOT)}
        if args.data:
            report["data"] = check_data(ROOT)
        if args.checkpoint is not None:
            report["checkpoint"] = check_checkpoint(args.checkpoint)
    except (ImportError, OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
