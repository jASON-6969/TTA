from pathlib import Path

from feature_alignment_2d_pilot.benchmark_config import (
    BASE_WIDTH,
    ENCODER_BLOCKS,
    IMAGE_SIZE,
    NUM_CLASSES,
    PAPER_PROTOCOL,
    NORMALIZATION,
)
from feature_alignment_2d_pilot.data import (
    build_montgomery_merged_masks,
    discover_montgomery,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"


def test_benchmark_config_matches_paper_cxr_contract():
    assert IMAGE_SIZE == 256
    assert BASE_WIDTH == 32
    assert ENCODER_BLOCKS == 5
    assert NUM_CLASSES == 2
    assert PAPER_PROTOCOL["dataset_direction"] == "SZ-CXR -> Montgomery"
    assert PAPER_PROTOCOL["paper_source_images"] == 662
    assert PAPER_PROTOCOL["paper_target_images"] == 138


def test_montgomery_adaptation_records_can_be_mask_free():
    records = build_montgomery_merged_masks(
        discover_montgomery(DATA / "montgomery")[:2],
        DATA / "montgomery" / "merged_mask",
    )
    image_only = [type(record)(record.sample_id, record.image_path, None, record.domain) for record in records]
    assert len(image_only) == 2
    assert all(record.mask_path is None for record in image_only)


def test_zscore_normalization_is_finite():
    from feature_alignment_2d_pilot.data import load_image

    sample = next((DATA / "montgomery" / "images").glob("*.png"))
    image = load_image(sample, 64, NORMALIZATION)
    assert image.shape == (1, 64, 64)
    assert bool(image.isfinite().all())


def test_hd95_vendor_definition_is_finite():
    import numpy as np
    from feature_alignment_2d_pilot.metrics import binary_metrics

    pred = np.zeros((8, 8), dtype=bool)
    target = np.zeros((8, 8), dtype=bool)
    pred[2:6, 2:6] = True
    target[2:6, 3:7] = True
    values = binary_metrics(pred, target)
    assert values["HD95"] > 0.0
    assert np.isfinite(values["HD95"])
