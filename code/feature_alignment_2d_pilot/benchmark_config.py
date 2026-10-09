from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "raw"
BASE_OUTPUT = ROOT / "base_model" / "benchmark_cxr_base_no_source_aug"
METHOD_OUTPUT = ROOT / "selected_methods" / "benchmark_cxr_methods_no_source_aug"

# The paper specifies this architecture and input/output shape, but does not
# publish the optimizer or TTA learning-rate defaults in the main text.
IMAGE_SIZE = 256
INPUT_CHANNELS = 1
NUM_CLASSES = 2
BASE_WIDTH = 32
ENCODER_BLOCKS = 5
SOURCE_SEED = 42
SOURCE_VAL_FRACTION = 0.20
SOURCE_BATCH_SIZE = 2
TARGET_BATCH_SIZE = 1
SOURCE_EPOCHS = 25
SOURCE_PATIENCE = 5
SOURCE_LR = 1e-3
METHOD_LRS = {"TENT": 1e-4, "TestFit": 1e-5, "GraTa": 1e-4}
METHODS = ("TENT", "TestFit", "GraTa")
NORMALIZATION = "zscore"
SOURCE_AUGMENT = False

PAPER_PROTOCOL = {
    "dataset_direction": "SZ-CXR -> Montgomery",
    "paper_source_images": 662,
    "paper_target_images": 138,
    "input_size": "256x256",
    "input_channels": INPUT_CHANNELS,
    "output_channels": NUM_CLASSES,
    "encoder_blocks": ENCODER_BLOCKS,
    "base_width": BASE_WIDTH,
    "source_split": "8:2",
    "tta_source_access": "forbidden",
    "target_evaluation": "entire target domain",
    "metrics": ["Dice", "HD95", "JI", "Sensitivity", "PPV"],
}

LOCAL_TRAINING = {
    "source_seed": SOURCE_SEED,
    "source_val_fraction": SOURCE_VAL_FRACTION,
    "source_batch_size": SOURCE_BATCH_SIZE,
    "target_batch_size": TARGET_BATCH_SIZE,
    "source_epochs_max": SOURCE_EPOCHS,
    "source_early_stopping_patience": SOURCE_PATIENCE,
    "source_optimizer": "Adam",
    "source_learning_rate": SOURCE_LR,
    "source_weight_decay": 1e-5,
    "method_learning_rates": METHOD_LRS,
    "image_resize": "full image resized to 256x256; no sliding window",
    "normalization": NORMALIZATION,
    "source_augmentation": "none; source base training uses unaugmented images and masks",
    "tta_stream_policy": "persistent cumulative target order; one update and prediction per target image",
    "hd95_policy": "pixel units on 256x256 resized masks; no physical spacing available",
}


def protocol_audit(
    source_pair_audit: dict,
    source_records: list,
    source_train: list,
    source_val: list,
    target_records: list,
) -> dict:
    paired_count = len(source_records)
    image_count = int(source_pair_audit.get("image_count", paired_count))
    return {
        "status": "passed",
        "experiment_label": "benchmark CXR reproduction with local source-mask coverage",
        "paper_protocol": PAPER_PROTOCOL,
        "local_training": LOCAL_TRAINING,
        "dataset_direction": PAPER_PROTOCOL["dataset_direction"],
        "source_pair_audit": source_pair_audit,
        "source_paired_records": paired_count,
        "source_train_count": len(source_train),
        "source_validation_count": len(source_val),
        "target_count": len(target_records),
        "source_coverage_limitation": {
            "paper_source_images": PAPER_PROTOCOL["paper_source_images"],
            "local_source_images": image_count,
            "local_source_paired_masks": paired_count,
            "unpaired_source_images_excluded": max(image_count - paired_count, 0),
            "reason": "Only exact image-mask pairs are eligible for supervised source training.",
        },
        "provenance": {
            "source": "SZ-CXR images from the NLM/Open-i collection with locally available paired lung masks.",
            "target": "NIH/NLM Montgomery County CXR Set with official left/right masks merged by pixelwise union.",
        },
        "target_mask_isolation": "Adaptation receives ImageOnlyDataset records with mask_path=None; masks are loaded only during post-adaptation evaluation.",
        "method_scope": {
            "Source-only": "frozen source checkpoint; no target update",
            "TENT": "local target-only BN-affine entropy adapter",
            "TestFit": "local target-only teacher-student pseudo-label and bottleneck feature adapter",
            "GraTa": "local target-only noisy-view consistency and feature adapter",
        },
        "fidelity_notes": [
            "The paper does not publish all optimizer, epoch, batch, or TTA learning-rate defaults in the main text; local values are recorded separately.",
            "TestFit and GraTa use the existing local adapters and are not claimed as line-by-line official-wrapper reproductions.",
            "The local source set has fewer paired masks than the 662 images stated by the paper, so numerical equality with Table 9 is not expected.",
        ],
    }
