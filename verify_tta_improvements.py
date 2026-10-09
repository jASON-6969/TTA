#!/usr/bin/env python3
"""驗證 TTA 改善效果的測試腳本"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from tempfile import mkdtemp

from Paradigm.tta_composition.config import parse_methods
from Paradigm.tta_composition.contracts import METHODS


BUNDLE_ROOT = Path(__file__).resolve().parent


def _new_output_directory(output_root: Path, label: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return Path(mkdtemp(prefix=f"{label}_{timestamp}_", dir=output_root)).resolve()


def run_tta_experiment(methods, seed=42, max_cases=None, output_suffix="", output_root=None):
    """執行單個 TTA 實驗"""
    cmd = [
        sys.executable,
        "-m", "Paradigm.tta_composition.run",
        "--methods", methods,
        "--seed", str(seed),
    ]

    if max_cases:
        cmd.extend(["--max-cases", str(max_cases)])

    output_root = Path(output_root) if output_root else BUNDLE_ROOT / "composition_runs"
    output_dir = _new_output_directory(output_root, f"verify_{output_suffix}")
    cmd.extend(["--output", str(output_dir)])

    print(f"\n{'='*80}")
    print(f"執行實驗：{methods}")
    print(f"輸出目錄：{output_dir}")
    print(f"命令：{' '.join(cmd)}")
    print(f"{'='*80}\n")

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONPATH"] = str(BUNDLE_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    log_path = output_dir / "execution.log"
    log_path.write_text("$ " + subprocess.list2cmdline(cmd) + "\n", encoding="utf-8")
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, encoding="utf-8",
            cwd=BUNDLE_ROOT, env=environment,
        )
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(result.stdout or "")
            stream.write(result.stderr or "")
        print(result.stdout)
        return output_dir, True
    except (subprocess.CalledProcessError, OSError) as error:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(getattr(error, "stdout", None) or "")
            stream.write(getattr(error, "stderr", None) or "")
            stream.write(str(error) + "\n")
        print(f"實驗失敗：{error}")
        print(f"錯誤日誌：{log_path}")
        return output_dir, False


def read_comparison_results(output_dir):
    """讀取並解析比較結果"""
    comparison_file = output_dir / "comparison.json"
    with open(comparison_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("event") != "completed":
        raise ValueError("Result summary does not describe a completed experiment")
    methods = parse_methods(summary["methods"])
    if data["comparison"].get("evaluated_cases", 0) <= 0:
        raise ValueError("Experiment has no evaluated target cases")
    result = {
        "methods": "+".join(methods) or "Source-only",
        "cases": data["comparison"]["cases"],
        "baseline_dice": data["comparison"]["baseline_metrics"]["Dice_mean"],
        "tta_dice": data["comparison"]["tta_metrics"]["Dice_mean"],
        "dice_delta": data["comparison"]["delta"]["Dice_mean"],
        "hd95_delta": data["comparison"]["delta"]["HD95_mean"],
        "iou_delta": data["comparison"]["delta"]["JI_mean"],
        "changed_cases": data["comparison"]["prediction_changed_cases"],
        "change_fraction": data["comparison"]["prediction_change_fraction_mean"],
    }
    for name in ("baseline_dice", "tta_dice", "dice_delta", "hd95_delta", "iou_delta", "change_fraction"):
        if not math.isfinite(result[name]):
            raise ValueError(f"Result contains non-finite {name}")
    return result


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", default=[*METHODS, ",".join(METHODS)],
        help="Method sets separated by spaces; use commas within each combination",
    )
    parser.add_argument("--max-cases", type=int, default=138)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=BUNDLE_ROOT / "composition_runs")
    args = parser.parse_args(argv)
    if args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if not 0 <= args.seed <= 4294967295:
        parser.error("--seed must be in [0, 4294967295]")
    try:
        args.methods = [
            tuple(method for method in METHODS if method in parse_methods(methods))
            for methods in args.methods
        ]
        if any(not methods for methods in args.methods):
            raise ValueError("Method sets must not be empty")
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    """主測試流程"""
    print("\n" + "="*80)
    print("TTA 改善效果驗證腳本")
    print("使用目前配置，比較各方法與同批 Source-only baseline。")
    print("="*80)

    args = _parse_args(argv)
    batch_dir = _new_output_directory(args.output_root, "verify_batch")
    experiments = [
        {"name": "+".join(methods), "methods": ",".join(methods), "max_cases": args.max_cases}
        for methods in args.methods
    ]

    results = []
    runs = []

    for exp in experiments:
        print(f"\n{'#'*80}")
        print(f"# {exp['name']}")
        print(f"{'#'*80}")

        output_dir, success = run_tta_experiment(
            exp["methods"],
            seed=args.seed,
            max_cases=exp["max_cases"],
            output_suffix=exp["methods"].replace(",", "_"),
            output_root=batch_dir,
        )

        run = {"methods": exp["name"], "output": str(output_dir), "status": "failed"}
        if success:
            try:
                result = read_comparison_results(output_dir)
                if result["methods"] != exp["name"]:
                    raise ValueError("Saved method names do not match the requested experiment")
                run.update(status="completed", result=result)
                results.append(result)
                print(f"\n實驗完成：")
                print(f"   Baseline Dice: {result['baseline_dice']:.6f}")
                print(f"   TTA Dice:      {result['tta_dice']:.6f}")
                print(f"   ΔDice:         {result['dice_delta']:+.6f} ({result['dice_delta']*100:+.4f} 百分點)")
                print(f"   ΔHD95:         {result['hd95_delta']:+.6f}")
                print(f"   ΔIoU:          {result['iou_delta']:+.6f}")
                print(f"   改變 cases:    {result['changed_cases']}/{result['cases']}")
            except (OSError, ValueError, TypeError, KeyError) as error:
                run["error"] = str(error)
                print(f"結果不完整：{error}")
        else:
            run["error"] = "Experiment process failed; see execution.log"
            print(f"\n實驗失敗，繼續下一組")
        runs.append(run)

    # 輸出總結
    print(f"\n\n{'='*80}")
    print("實驗總結")
    print(f"{'='*80}\n")

    if results:
        print(f"{'方法':<40} {'ΔDice':<12} {'ΔHD95':<12} {'ΔIoU':<12} {'變化cases'}")
        print("-" * 95)
        for r in results:
            methods_name = r['methods']
            print(f"{methods_name:<40} {r['dice_delta']:+.6f}   {r['hd95_delta']:+.6f}   {r['iou_delta']:+.6f}   {r['changed_cases']}/{r['cases']}")

        # 找出最佳方法
        best = max(results, key=lambda x: x['dice_delta'])
        print(f"\n本批最高 ΔDice：{best['methods']}")
        print(f"   ΔDice：{best['dice_delta']:+.6f} ({best['dice_delta']*100:+.4f} 百分點)")

    else:
        print("沒有成功完成的實驗")

    failed = sum(run["status"] != "completed" for run in runs)
    batch_summary = {
        "status": "failed" if failed else "completed", "seed": args.seed,
        "max_cases": args.max_cases, "failed_runs": failed, "runs": runs,
    }
    summary_path = batch_dir / "verification_summary.json"
    summary_path.write_text(json.dumps(batch_summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"批次摘要：{summary_path}")
    print(f"\n{'='*80}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
