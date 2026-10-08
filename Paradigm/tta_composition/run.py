"""CLI for source-only or any non-empty subset of the five composition modules."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import csv
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

from .audit import adaptation_provenance, record_to_dict, sha256, write_json
from .baseline import BaselineResult, ensure_baseline
from .comparison import COMPARISON_FIELDS, QUALITY_METRICS, compare_predictions
from .config import CompositionConfig, parse_methods
from .data_selection import DATASETS, TargetCase, discover_targets
from .engine import CompositionEngine
from .model_bridge import ModelBundle, build_model_bundle, predict_source


def _load_pilot():
    test_v1 = Path(__file__).resolve().parents[2]
    code_root = test_v1 / "code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from feature_alignment_2d_pilot.data import load_image
    from feature_alignment_2d_pilot.metrics import aggregate, binary_metrics, save_prediction

    return test_v1, load_image, binary_metrics, save_prediction, aggregate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _default_config(methods: tuple[str, ...], args: argparse.Namespace) -> CompositionConfig:
    test_v1 = Path(__file__).resolve().parents[2]
    checkpoint = Path(args.checkpoint) if args.checkpoint else test_v1 / "base_model" / "benchmark_cxr_base_no_source_aug" / "source_checkpoint.pth"
    output = Path(args.output) if args.output else None
    return CompositionConfig(
        methods=methods,
        checkpoint=checkpoint,
        output=output,
        seed=args.seed if args.seed is not None else 42,
        max_cases=args.max_cases,
        device=args.device or "auto",
        dataset=args.dataset or "montgomery",
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        baseline_cache=args.baseline_cache,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default=None, help="Comma-separated subset: VPTTA,DLTTA,TestFit,GraTa,SmaRT; empty means Source-only")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--adaptation-steps", type=int, help="Parameter updates per target image")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--image-dir", type=Path, help="Single folder of target images")
    parser.add_argument("--mask-dir", type=Path, help="Optional combined binary mask folder; same stem or stem_mask")
    parser.add_argument("--baseline-cache", type=Path, help="Directory for complete reusable source-only predictions")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> CompositionConfig:
    methods = parse_methods(args.methods) if args.methods is not None else ()
    config = CompositionConfig.from_json(args.config) if args.config else _default_config(methods, args)
    if args.config:
        # None means omitted; an explicit empty methods value restores Source-only.
        if args.methods is not None:
            config.methods = methods
        for field in ("checkpoint", "output", "max_cases", "seed", "device", "dataset", "image_dir", "mask_dir", "baseline_cache", "adaptation_steps"):
            value = getattr(args, field)
            if value is not None:
                setattr(config, field, value)
    elif args.adaptation_steps is not None:
        config.adaptation_steps = args.adaptation_steps
    if config.checkpoint is None:
        test_v1 = Path(__file__).resolve().parents[2]
        config.checkpoint = test_v1 / "base_model" / "benchmark_cxr_base_no_source_aug" / "source_checkpoint.pth"
    config.__post_init__()
    return config


def _resolve_device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable; choose CPU or Auto")
    return torch.device("cuda:0" if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()) else "cpu")


def _output_dir(config: CompositionConfig, test_v1: Path) -> Path:
    if config.output is not None:
        return config.output
    label = "source_only" if not config.canonical_methods else "_".join(method.lower() for method in config.canonical_methods)
    return test_v1 / "composition_runs" / f"seed_{config.seed}_{label}"


def _load_target_mask(mask_paths: tuple[Path, ...], image_size: int) -> np.ndarray:
    """Read and union binary labels only during the post-adaptation evaluation."""
    from PIL import Image

    target: np.ndarray = np.zeros((image_size, image_size), dtype=bool)
    for mask_path in mask_paths:
        with Image.open(mask_path) as source:
            mask = source.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
            target |= np.asarray(mask) > 0
    return target


def _evaluate(
    prediction_dir: Path,
    records: list[TargetCase],
    image_size: int,
    binary_metrics: Callable[..., dict[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        if not record.mask_paths:
            continue
        from PIL import Image

        prediction_path = prediction_dir / f"{record.sample_id}.png"
        with Image.open(prediction_path) as source:
            prediction = np.asarray(source.convert("L"), dtype=np.uint8) > 127
        target = _load_target_mask(record.mask_paths, image_size)
        rows.append({"method": prediction_dir.name, "image_id": record.sample_id, **binary_metrics(prediction, target)})
    return rows


def _completion_summary(
    output_dir: Path,
    config: CompositionConfig,
    case_count: int,
    metrics: list[dict[str, object]],
    device: str,
    elapsed_seconds: float,
    aggregate: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    combined = aggregate(metrics)
    combined.pop("count", None)
    return {
        "event": "completed",
        "output": str(output_dir),
        "methods": list(config.canonical_methods),
        "cases": case_count,
        "evaluated_cases": len(metrics),
        "device": device,
        "dataset": config.dataset,
        "metrics": combined,
        "elapsed_seconds": elapsed_seconds,
    }


def _emit(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _record_baseline_evaluation(
    baseline: BaselineResult,
    records: list[TargetCase],
    metrics: list[dict[str, Any]],
    config: CompositionConfig,
    device: str,
    aggregate: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    """Record current labels only after TTA has finished, preserving isolation."""
    for row in metrics:
        row["method"] = "Source-only"
    inference_manifest = json.loads((baseline.directory / "baseline_manifest.json").read_text(encoding="utf-8"))
    build_elapsed = float(inference_manifest["build_elapsed_seconds"])
    summary = _completion_summary(
        baseline.directory, config, len(records), metrics,
        device, build_elapsed, aggregate,
    )
    summary["methods"] = []
    summary["event"] = "baseline_evaluated"
    summary["cache_key"] = baseline.cache_key
    summary["reused"] = baseline.reused
    summary["cache_prepare_seconds"] = baseline.elapsed_seconds
    label_manifest = [
        {
            "image_id": record.sample_id,
            "mask_paths": [
                {"path": str(path.resolve()), "sha256": sha256(path)}
                for path in record.mask_paths
            ],
        }
        for record in records
    ]
    metric_fields = ("method", "image_id", *QUALITY_METRICS)
    run_directory = config.output
    if run_directory is None:
        raise ValueError("Baseline evaluation requires a resolved run output")
    _write_csv(baseline.directory / "metrics.csv", metrics, metric_fields)
    write_json(baseline.directory / "label_manifest.json", {"cases": label_manifest})
    write_json(baseline.directory / "summary.json", summary)
    # Each experiment owns its label/evaluation snapshot even when the shared
    # inference cache is later evaluated using another set of target masks.
    _write_csv(run_directory / "baseline_metrics.csv", metrics, metric_fields)
    write_json(run_directory / "label_manifest.json", {"cases": label_manifest})
    write_json(run_directory / "baseline_summary.json", summary)
    return {
        "cache_key": baseline.cache_key,
        "output": str(baseline.directory),
        "reused": baseline.reused,
        "cases": len(records),
        "evaluated_cases": len(metrics),
        "metrics": summary["metrics"],
        "elapsed_seconds": build_elapsed,
        "cache_prepare_seconds": baseline.elapsed_seconds,
        "evaluation_record": str(run_directory / "baseline_summary.json"),
        "label_manifest": str(run_directory / "label_manifest.json"),
    }


def main() -> int:
    args = _parse_args()
    test_v1, load_image, binary_metrics, save_prediction, aggregate = _load_pilot()
    config = _config_from_args(args)
    methods = config.methods
    started = time.perf_counter()
    output_dir = _output_dir(config, test_v1).resolve()
    if any((output_dir / name).exists() for name in (
        "summary.json", "audit.json", "adaptation_log.jsonl", "predictions",
    )):
        raise FileExistsError(f"Run output already contains experiment artifacts: {output_dir}; choose a new --output")
    output_dir.mkdir(parents=True, exist_ok=True)
    config.output = output_dir
    config.baseline_cache = (config.baseline_cache or output_dir.parent / "baselines").resolve()
    write_json(output_dir / "config.json", config.to_dict())
    provenance = adaptation_provenance(config.adaptation_steps)
    write_json(output_dir / "provenance.json", provenance)
    set_seed(config.seed)
    all_targets = discover_targets(test_v1, config.dataset, config.image_dir, config.mask_dir)
    targets = all_targets[: config.max_cases] if config.max_cases else all_targets
    # Adaptation receives only image paths/IDs, never TargetCase or mask contents.
    target_images = [(record.sample_id, record.image_path) for record in targets]
    manifest = [
        {"image_id": record.sample_id, "image_path": str(record.image_path), "mask_paths": [str(path) for path in record.mask_paths]}
        for record in targets
    ]
    write_json(output_dir / "data_manifest.json", {"dataset": config.dataset, "cases": manifest})
    device = _resolve_device(config.device)
    checkpoint = config.checkpoint
    if checkpoint is None:
        raise ValueError("A checkpoint is required")
    bundle: ModelBundle | None = None

    def get_bundle() -> ModelBundle:
        nonlocal bundle
        if bundle is None:
            bundle = build_model_bundle(checkpoint, device, enable_smart="SmaRT" in config.canonical_methods)
        return bundle

    def baseline_prediction(record: TargetCase) -> torch.Tensor:
        image = load_image(record.image_path, config.image_size, config.normalization)
        return predict_source(get_bundle().reference, image, device)

    baseline = ensure_baseline(
        config.baseline_cache, all_targets, checkpoint,
        config.image_size, config.normalization, str(device), baseline_prediction, _emit,
    )
    # Resolve lazy model construction before resetting adaptation randomness.
    # Cache hits and misses must leave the same random state for the engine.
    engine: CompositionEngine | None = None
    if config.canonical_methods:
        adaptation_bundle = get_bundle()
        set_seed(config.seed)
        engine = CompositionEngine(adaptation_bundle, config)
    prediction_dir = output_dir / "predictions" / ("Source-only" if not config.canonical_methods else "+".join(config.canonical_methods))
    prediction_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "adaptation_log.jsonl").open("w", encoding="utf-8") as log_stream:
        for index, (sample_id, image_path) in enumerate(target_images, start=1):
            if engine is None:
                shutil.copyfile(baseline.prediction_dir / f"{sample_id}.png", prediction_dir / f"{sample_id}.png")
            else:
                image = load_image(image_path, config.image_size, config.normalization)
                prediction, update_record, method_metrics = engine.step(image, sample_id)
                save_prediction(prediction[0].numpy(), prediction_dir / f"{sample_id}.png")
                update = record_to_dict(update_record, sample_id=sample_id, method_metrics=method_metrics)
                log_stream.write(json.dumps(update, ensure_ascii=False, allow_nan=False) + "\n")
                log_stream.flush()
            _emit({
                "event": "progress", "phase": "tta" if engine is not None else "source_only",
                "completed": index, "total": len(target_images), "image_id": sample_id,
            })
    _emit({"event": "status", "message": "正在評估 Baseline 與本次結果…"})
    baseline_metrics = _evaluate(baseline.prediction_dir, all_targets, config.image_size, binary_metrics)
    metrics = _evaluate(prediction_dir, targets, config.image_size, binary_metrics)
    baseline_summary = _record_baseline_evaluation(baseline, all_targets, baseline_metrics, config, str(device), aggregate)
    comparison = compare_predictions(targets, baseline.prediction_dir, prediction_dir, baseline_metrics, metrics, aggregate)
    _write_csv(output_dir / "comparison.csv", comparison.rows, COMPARISON_FIELDS)
    write_json(output_dir / "comparison.json", {"baseline": baseline_summary, "comparison": comparison.summary})
    elapsed_seconds = time.perf_counter() - started
    write_json(
        output_dir / "audit.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "provenance": provenance,
            "methods_input_order": list(methods),
            "methods_execution_order": list(config.canonical_methods),
            "dataset": config.dataset,
            "data_manifest": "data_manifest.json",
            "target_count": len(target_images),
            "evaluated_cases": len(metrics),
            "target_mask_isolation": True,
            "mask_read_phase": "evaluation_after_all_adaptation",
            "binary_mask_threshold": "> 0",
            "engine": engine.audit_state() if engine is not None else {"mode": "source_only", "adaptation_steps": 0},
            "baseline": baseline_summary,
            "baseline_evaluation": "baseline_summary.json",
            "label_manifest": "label_manifest.json",
            "comparison": "comparison.json",
            "elapsed_seconds": elapsed_seconds,
            "limitations": [
                "Composition modules are explicit local adaptations of the benchmark mechanisms.",
                "Eight-case runs are functional validation and are not a performance claim.",
                "Target masks are loaded only after all target adaptation steps complete; unlabelled cases are prediction-only.",
            ],
        },
    )
    _write_csv(output_dir / "metrics.csv", metrics, ("method", "image_id", *QUALITY_METRICS))
    summary = _completion_summary(output_dir, config, len(target_images), metrics, str(device), elapsed_seconds, aggregate)
    summary["baseline"] = baseline_summary
    summary["comparison"] = comparison.summary
    write_json(output_dir / "summary.json", summary)
    _emit(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

