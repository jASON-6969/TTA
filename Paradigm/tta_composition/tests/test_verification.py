"""Batch verification regressions with controlled experiment processes."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TEST_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(TEST_ROOT))

import verify_tta_improvements as verification


class VerificationChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tta verification ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def completed_result(self, directory, methods="GraTa"):
        directory.mkdir(parents=True, exist_ok=True)
        comparison = {
            "cases": 8, "evaluated_cases": 8,
            "baseline_metrics": {"Dice_mean": 0.94},
            "tta_metrics": {"Dice_mean": 0.941},
            "delta": {"Dice_mean": 0.001, "HD95_mean": -0.1, "JI_mean": 0.002},
            "prediction_changed_cases": 3, "prediction_change_fraction_mean": 0.01,
        }
        (directory / "comparison.json").write_text(
            json.dumps({"comparison": comparison}), encoding="utf-8",
        )
        (directory / "summary.json").write_text(
            json.dumps({"event": "completed", "methods": methods.split(",")}), encoding="utf-8",
        )
        return directory

    def test_defaults_cover_five_methods_and_combination(self):
        args = verification._parse_args([])
        self.assertEqual(args.methods, [(method,) for method in verification.METHODS] + [verification.METHODS])
        self.assertEqual(args.max_cases, 138)
        selected = verification._parse_args(["--methods", "TestFit,DLTTA", "grata", "--max-cases", "8"])
        self.assertEqual(selected.methods, [("DLTTA", "TestFit"), ("GraTa",)])
        self.assertEqual(selected.max_cases, 8)

    def test_result_identity_comes_from_summary_and_invalid_results_fail(self):
        output = self.completed_result(self.directory / "verify_any_timestamp", "DLTTA,TestFit")
        result = verification.read_comparison_results(output)
        self.assertEqual(result["methods"], "DLTTA+TestFit")
        self.assertAlmostEqual(result["dice_delta"], 0.001)
        (output / "summary.json").unlink()
        with self.assertRaises(FileNotFoundError):
            verification.read_comparison_results(output)

    def test_process_uses_bundle_cwd_utf8_and_unique_outputs(self):
        successful = SimpleNamespace(stdout="completed\n", stderr="")
        with patch.object(verification, "datetime") as clock, \
                patch.object(verification.subprocess, "run", return_value=successful) as process, redirect_stdout(io.StringIO()):
            clock.now.return_value.strftime.return_value = "fixed_timestamp"
            first, first_success = verification.run_tta_experiment(
                "DLTTA", max_cases=8, output_suffix="DLTTA", output_root=self.directory,
            )
            second, second_success = verification.run_tta_experiment(
                "DLTTA", max_cases=8, output_suffix="DLTTA", output_root=self.directory,
            )
        self.assertTrue(first_success and second_success)
        self.assertNotEqual(first, second)
        self.assertEqual(process.call_args.kwargs["cwd"], verification.BUNDLE_ROOT)
        self.assertEqual(process.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")
        self.assertIn("completed", (first / "execution.log").read_text(encoding="utf-8"))
        self.assertIn("--max-cases", process.call_args.args[0])

    def test_process_failure_keeps_error_log(self):
        failure = subprocess.CalledProcessError(2, ["python"], output="before failure\n", stderr="controlled error\n")
        with patch.object(verification.subprocess, "run", side_effect=failure), redirect_stdout(io.StringIO()):
            output, success = verification.run_tta_experiment("GraTa", output_root=self.directory)
        self.assertFalse(success)
        text = (output / "execution.log").read_text(encoding="utf-8")
        self.assertIn("controlled error", text)
        self.assertIn("before failure", text)

    def test_partial_failure_returns_nonzero_and_retains_other_results(self):
        def experiment(methods, **kwargs):
            output = kwargs["output_root"] / methods
            if methods == "GraTa":
                self.completed_result(output, methods)
                return output, True
            return output, False

        captured = io.StringIO()
        with patch.object(verification, "run_tta_experiment", side_effect=experiment) as runner, redirect_stdout(captured):
            code = verification.main([
                "--methods", "GraTa", "TestFit", "--max-cases", "8", "--seed", "7",
                "--output-root", str(self.directory),
            ])
        self.assertEqual(code, 1)
        self.assertEqual(runner.call_args.kwargs["seed"], 7)
        summary = json.loads(next(self.directory.glob("*/verification_summary.json")).read_text(encoding="utf-8"))
        self.assertEqual(summary["failed_runs"], 1)
        self.assertEqual([run["status"] for run in summary["runs"]], ["completed", "failed"])
        self.assertIn("GraTa", captured.getvalue())
        self.assertNotIn("修正前", captured.getvalue())

    def test_missing_comparison_and_all_process_failures_return_nonzero(self):
        for subprocess_success in (True, False):
            with self.subTest(subprocess_success=subprocess_success):
                output_root = self.directory / str(subprocess_success)
                with patch.object(verification, "run_tta_experiment", return_value=(output_root / "missing", subprocess_success)), \
                        redirect_stdout(io.StringIO()):
                    code = verification.main(["--methods", "VPTTA", "--output-root", str(output_root)])
                self.assertEqual(code, 1)
                summary = json.loads(next(output_root.glob("*/verification_summary.json")).read_text(encoding="utf-8"))
                self.assertEqual(summary["failed_runs"], 1)
                self.assertEqual(summary["runs"][0]["status"], "failed")

    def test_saved_methods_mismatch_is_rejected(self):
        output = self.completed_result(self.directory / "wrong methods", "TestFit")
        with patch.object(verification, "run_tta_experiment", return_value=(output, True)), redirect_stdout(io.StringIO()):
            code = verification.main(["--methods", "GraTa", "--output-root", str(self.directory / "batch")])
        self.assertEqual(code, 1)

    def test_all_successes_return_zero(self):
        output = self.completed_result(self.directory / "completed", "DLTTA")
        with patch.object(verification, "run_tta_experiment", return_value=(output, True)), redirect_stdout(io.StringIO()):
            code = verification.main(["--methods", "DLTTA", "--output-root", str(self.directory / "batch")])
        self.assertEqual(code, 0)

    def test_batches_remain_unique_when_clock_does_not_advance(self):
        output = self.completed_result(self.directory / "completed", "DLTTA")
        output_root = self.directory / "batches"
        with patch.object(verification, "datetime") as clock, \
                patch.object(verification, "run_tta_experiment", return_value=(output, True)), redirect_stdout(io.StringIO()):
            clock.now.return_value.strftime.return_value = "fixed_timestamp"
            for _ in range(2):
                self.assertEqual(verification.main(["--methods", "DLTTA", "--output-root", str(output_root)]), 0)
        summaries = list(output_root.glob("*/verification_summary.json"))
        self.assertEqual(len(summaries), 2)
        self.assertTrue(all(json.loads(path.read_text(encoding="utf-8"))["status"] == "completed" for path in summaries))


if __name__ == "__main__":
    unittest.main()
