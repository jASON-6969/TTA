"""Classify saved runs without rewriting historical experiment artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .audit import adaptation_provenance, sha256, write_json


def classify_run(directory: Path, source_runs: Path, current_sources: dict[str, str]) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    original = source_runs / directory.name
    compared_files = [name for name in ("summary.json", "config.json", "comparison.json")
                      if (directory / name).is_file() and (original / name).is_file()]
    copied = "summary.json" in compared_files and all(
        sha256(directory / name) == sha256(original / name) for name in compared_files
    )
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    if copied:
        origin = "copied_from_test_v1"
    elif provenance.get("adaptation_sources") == current_sources:
        origin = "current_adaptation_code"
    elif provenance.get("adaptation_sources"):
        origin = "different_adaptation_code"
    else:
        origin = "unversioned_run"
    return {
        "run": directory.name,
        "path": str(directory.resolve()),
        "origin": origin,
        "completed": summary.get("event") == "completed",
        "source_run": str(original) if copied else None,
        "recorded_output": summary.get("output", config.get("output")),
        "methods": summary.get("methods", config.get("methods", [])),
        "cases": summary.get("cases"),
        "adaptation_steps": config.get("adaptation_steps"),
        "metrics": summary.get("metrics", {}),
    }


def build_catalog(runs: Path, source_runs: Path) -> dict[str, Any]:
    current_sources = adaptation_provenance(1)["adaptation_sources"]
    directories = list(runs.iterdir())
    for directory in tuple(directories):
        if directory.is_dir() and directory.name.startswith("verify_batch_"):
            directories.extend(directory.iterdir())
    entries = [classify_run(directory, source_runs, current_sources)
               for directory in sorted(directories)
               if directory.is_dir() and (directory / "config.json").is_file()]
    return {
        "schema_version": 1,
        "runs_directory": str(runs.resolve()),
        "source_runs_directory": str(source_runs.resolve()),
        "counts": dict(Counter(entry["origin"] for entry in entries)),
        "runs": entries,
        "limitations": [
            "A missing adaptation-code snapshot cannot be reconstructed from today's files.",
            "Copied or differently versioned results are preserved and are not current-code validation.",
        ],
    }


def main() -> int:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=workspace / "composition_runs")
    parser.add_argument("--source-runs", type=Path, default=workspace.parent / "test_v1" / "composition_runs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    catalog = build_catalog(args.runs, args.source_runs)
    output = args.output or args.runs / "run_catalog.json"
    write_json(output, catalog)
    print(json.dumps({"output": str(output), "counts": catalog["counts"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
