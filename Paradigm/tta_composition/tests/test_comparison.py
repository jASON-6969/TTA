"""Matched-case baseline comparisons and their user-facing result summaries."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from Paradigm.tta_composition.comparison import QUALITY_METRICS, compare_predictions
from Paradigm.tta_composition.data_selection import TargetCase
from Paradigm.tta_composition.result_summary import format_result


def aggregate(rows):
    result = {"count": len(rows)}
    for metric in QUALITY_METRICS:
        values = [float(row[metric]) for row in rows]
        result[f"{metric}_mean"] = float(np.mean(values)) if values else None
        result[f"{metric}_std"] = float(np.std(values)) if values else 0.0
    return result


def metric_row(sample_id, dice, hd95=2.0):
    return {"image_id": sample_id, "Dice": dice, "HD95": hd95,
            "JI": dice, "Sensitivity": dice, "PPV": dice}


class ComparisonChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tta matched comparisons ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.baseline_dir = self.root / "baseline"
        self.tta_dir = self.root / "tta"
        self.baseline_dir.mkdir()
        self.tta_dir.mkdir()
        self.records = [TargetCase(sample_id, self.root / f"{sample_id}.png") for sample_id in ("a", "b")]
        for record in self.records:
            Image.new("L", (16, 16), color=0).save(self.baseline_dir / f"{record.sample_id}.png")
            Image.new("L", (16, 16), color=255 if record.sample_id == "a" else 0).save(
                self.tta_dir / f"{record.sample_id}.png"
            )

    def compare(self, baseline_rows, tta_rows, records=None):
        return compare_predictions(
            self.records if records is None else records,
            self.baseline_dir, self.tta_dir, baseline_rows, tta_rows, aggregate,
        )

    def test_subset_uses_same_case_baseline_not_full_dataset_mean(self):
        result = self.compare([metric_row("a", 0.2, 5.0), metric_row("b", 1.0)],
                              [metric_row("a", 0.6, 3.0)], self.records[:1])
        self.assertEqual(result.summary["cases"], 1)
        self.assertEqual(result.summary["baseline_metrics"]["Dice_mean"], 0.2)
        self.assertAlmostEqual(result.summary["delta"]["Dice_mean"], 0.4)
        self.assertEqual(result.summary["delta"]["HD95_mean"], -2.0)
        self.assertEqual(result.rows[0]["changed_pixels"], 256)
        self.assertEqual(result.rows[0]["prediction_change_fraction"], 1.0)

    def test_partial_labels_compare_only_paired_cases_but_all_predictions(self):
        result = self.compare([metric_row("a", 0.4), metric_row("unselected", 0.9)],
                              [metric_row("a", 0.5)])
        self.assertEqual(result.summary["cases"], 2)
        self.assertEqual(result.summary["evaluated_cases"], 1)
        self.assertAlmostEqual(result.summary["delta"]["Dice_mean"], 0.1)
        self.assertEqual(result.summary["prediction_change_fraction_mean"], 0.5)
        self.assertEqual(result.summary["prediction_changed_cases"], 1)
        self.assertIsNone(result.rows[1]["baseline_Dice"])
        self.assertFalse(result.rows[1]["evaluated"])

    def test_unlabelled_comparison_does_not_report_quality_improvement(self):
        result = self.compare([], [])
        self.assertEqual(result.summary["evaluated_cases"], 0)
        self.assertIsNone(result.summary["delta"]["Dice_mean"])
        self.assertEqual(result.summary["prediction_change_fraction_mean"], 0.5)

    def test_different_label_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "label coverage mismatch"):
            self.compare([], [metric_row("a", 0.5)])

    def test_unselected_tta_metric_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unselected"):
            self.compare([metric_row("c", 0.5)], [metric_row("c", 0.6)])

    def test_duplicate_metric_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.compare([metric_row("a", 0.5), metric_row("a", 0.4)], [])

    def test_shape_mismatch_is_rejected(self):
        Image.new("L", (8, 8), color=255).save(self.tta_dir / "a.png")
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            self.compare([], [])

    def test_result_panel_shows_matched_base_tta_and_signed_deltas(self):
        result = self.compare([metric_row("a", 0.2, 5.0), metric_row("b", 1.0)],
                              [metric_row("a", 0.6, 3.0)], self.records[:1])
        summary = {
            "cases": 1, "evaluated_cases": 1, "metrics": result.summary["tta_metrics"],
            "output": str(self.tta_dir),
            "baseline": {"cases": 2, "reused": True, "output": str(self.baseline_dir)},
            "comparison": result.summary,
        }
        text = format_result(summary)
        self.assertIn("重用完整 2 張", text)
        self.assertIn("本次比較相同 1 張", text)
        self.assertIn("0.2000", text)
        self.assertIn("0.6000", text)
        self.assertIn("+0.4000", text)
        self.assertIn("-2.00", text)

    def test_unlabelled_result_panel_separates_change_from_quality(self):
        result = self.compare([], [])
        summary = {
            "cases": 2, "evaluated_cases": 0, "metrics": {}, "output": str(self.tta_dir),
            "baseline": {"cases": 2, "reused": False, "output": str(self.baseline_dir)},
            "comparison": result.summary,
        }
        text = format_result(summary)
        self.assertIn("未評估適配品質", text)
        self.assertIn("50.00%", text)
        self.assertNotIn("Δ（TTA", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
