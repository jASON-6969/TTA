"""Runner regressions for complete baselines and matched TTA comparisons."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from Paradigm.tta_composition import run
from Paradigm.tta_composition.config import CompositionConfig
from Paradigm.tta_composition.contracts import UpdateRecord


class RunnerBaselineChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tta baseline runner ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.masks = self.root / "masks"
        self.images.mkdir()
        self.masks.mkdir()
        for sample_id in ("a", "b", "c"):
            Image.new("L", (16, 16), color=127).save(self.images / f"{sample_id}.png")
        Image.new("L", (16, 16), color=255).save(self.masks / "a_mask.png")
        Image.new("L", (16, 16), color=0).save(self.masks / "c_mask.png")
        self.checkpoint = self.root / "checkpoint.pth"
        self.checkpoint.write_bytes(b"controlled source weights")
        self.source_calls = 0
        self.adapted: list[str] = []
        self.fail_adaptation = False
        self.engine_initial_random: list[float] = []
        self.last_events: list[dict] = []
        owner = self

        class Reference(torch.nn.Module):
            def forward(self, image, return_features=False):
                owner.source_calls += 1
                logits = torch.cat((torch.zeros_like(image), torch.ones_like(image)), dim=1)
                return (logits, image) if return_features else logits

        class Adapter:
            def __init__(self, bundle, config):
                self.config = config
                owner.adapted = []
                owner.engine_initial_random.append(float(torch.rand(())))

            def step(self, image, sample_id):
                if owner.fail_adaptation:
                    raise RuntimeError("controlled adaptation failure")
                owner.adapted.append(sample_id)
                foreground = self.config.lrs["GraTa"] <= 1e-4
                prediction = torch.full((1, 16, 16), int(foreground), dtype=torch.long)
                record = UpdateRecord(len(owner.adapted), {}, {}, {}, {}, 0, 0.0)
                return prediction, record, {}

            def audit_state(self):
                return {"fixture": True}

        self.reference_type = Reference
        self.adapter_type = Adapter

    def config(self, name, **overrides):
        values = {
            "methods": ("GraTa",), "checkpoint": self.checkpoint,
            "output": self.root / name, "baseline_cache": self.root / "baselines",
            "image_size": 16, "dataset": "custom", "image_dir": self.images,
            "mask_dir": self.masks, "max_cases": 1, "device": "cpu",
            "lrs": {"GraTa": 1e-4},
        }
        values.update(overrides)
        return CompositionConfig(**values)

    def execute(self, config, forbid_model_build=False):
        config_path = self.root / "run_config.json"
        config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        captured = io.StringIO()
        def build_bundle(*args, **kwargs):
            if forbid_model_build:
                raise AssertionError("Cached Source-only rebuilt a model")
            # Construction of a real UNet also advances the global Torch RNG.
            torch.rand(100)
            return SimpleNamespace(reference=self.reference_type())

        model_patch = patch.object(run, "build_model_bundle", side_effect=build_bundle)
        with patch.object(sys, "argv", ["run", "--config", str(config_path)]), model_patch, \
                patch.object(run, "CompositionEngine", self.adapter_type), redirect_stdout(captured):
            self.assertEqual(run.main(), 0)
        self.last_events = [json.loads(line) for line in captured.getvalue().splitlines()]
        return json.loads((config.output / "summary.json").read_text(encoding="utf-8"))

    def test_full_baseline_precedes_limited_tta_and_comparison_is_matched(self):
        original_open = Image.open

        def guard_label_reads(path, *args, **kwargs):
            if Path(path).parent == self.masks:
                self.assertEqual(self.adapted, ["a"], "Labels opened before selected TTA finished")
            return original_open(path, *args, **kwargs)

        config = self.config("first")
        with patch.object(Image, "open", guard_label_reads):
            summary = self.execute(config)
        self.assertEqual(self.source_calls, 3)
        self.assertEqual(summary["cases"], 1)
        self.assertEqual(summary["baseline"]["cases"], 3)
        self.assertEqual(summary["baseline"]["evaluated_cases"], 2)
        self.assertEqual(summary["baseline"]["metrics"]["Dice_mean"], 0.5)
        self.assertEqual(summary["comparison"]["baseline_metrics"]["Dice_mean"], 1.0)
        self.assertEqual(summary["comparison"]["delta"]["Dice_mean"], 0.0)
        baseline_progress = [event for event in self.last_events if event["event"] == "baseline_progress"]
        self.assertEqual([event["image_id"] for event in baseline_progress], ["a", "b", "c"])
        self.assertTrue((config.output / "comparison.csv").is_file())
        self.assertTrue((Path(summary["baseline"]["output"]) / "summary.json").is_file())

    def test_changed_tta_learning_rate_reuses_baseline_and_reports_delta(self):
        first = self.execute(self.config("first"))
        second = self.execute(self.config("changed_lr", lrs={"GraTa": 2e-4}))
        self.assertTrue(second["baseline"]["reused"])
        self.assertEqual(first["baseline"]["cache_key"], second["baseline"]["cache_key"])
        self.assertEqual(self.source_calls, 3)
        self.assertEqual(second["comparison"]["delta"]["Dice_mean"], -1.0)
        self.assertEqual(second["comparison"]["prediction_change_fraction_mean"], 1.0)
        self.assertFalse(any(event["event"] == "baseline_progress" for event in self.last_events))

    def test_completed_output_is_not_overwritten_by_another_run(self):
        config = self.config("preserved")
        self.execute(config)
        summary_path = config.output / "summary.json"
        before = summary_path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.execute(config)
        self.assertEqual(summary_path.read_bytes(), before)

    def test_run_saves_the_actual_adaptation_code_snapshot(self):
        config = self.config("versioned")
        self.execute(config)
        provenance = json.loads((config.output / "provenance.json").read_text(encoding="utf-8"))
        audit = json.loads((config.output / "audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["provenance"], provenance)
        self.assertEqual(provenance["adaptation_steps"], config.adaptation_steps)
        self.assertIn("engine.py", provenance["adaptation_sources"])

    def test_cached_source_only_uses_saved_predictions_without_model_build(self):
        first = self.execute(self.config("first"))
        second = self.execute(self.config("source_only", methods=(), max_cases=2), forbid_model_build=True)
        self.assertTrue(second["baseline"]["reused"])
        self.assertEqual(second["baseline"]["cache_key"], first["baseline"]["cache_key"])
        self.assertEqual(self.source_calls, 3)
        self.assertEqual(second["comparison"]["prediction_change_fraction_mean"], 0.0)
        self.assertTrue(all(event["phase"] == "source_only" for event in self.last_events if event["event"] == "progress"))

    def test_current_labels_are_reevaluated_without_rebuilding_predictions(self):
        first = self.execute(self.config("first"))
        first_manifest_path = self.root / "first" / "label_manifest.json"
        original_manifest = first_manifest_path.read_bytes()
        first_evaluation = (self.root / "first" / "baseline_summary.json").read_bytes()
        Image.new("L", (16, 16), color=0).save(self.masks / "a_mask.png")
        second = self.execute(self.config("updated_labels"))
        self.assertTrue(second["baseline"]["reused"])
        self.assertEqual(self.source_calls, 3)
        self.assertEqual(first["comparison"]["baseline_metrics"]["Dice_mean"], 1.0)
        self.assertEqual(second["comparison"]["baseline_metrics"]["Dice_mean"], 0.0)
        self.assertEqual(first_manifest_path.read_bytes(), original_manifest)
        self.assertEqual((self.root / "first" / "baseline_summary.json").read_bytes(), first_evaluation)
        self.assertNotEqual((self.root / "updated_labels" / "label_manifest.json").read_bytes(), original_manifest)

    def test_cache_hit_and_miss_have_identical_adaptation_random_state(self):
        self.execute(self.config("first"))
        self.execute(self.config("reused"))
        self.assertEqual(len(self.engine_initial_random), 2)
        self.assertEqual(self.engine_initial_random[0], self.engine_initial_random[1])

    def test_missing_labels_record_prediction_changes_without_quality_scores(self):
        summary = self.execute(self.config("unlabelled", mask_dir=None))
        self.assertEqual(summary["baseline"]["cases"], 3)
        self.assertEqual(summary["evaluated_cases"], 0)
        self.assertIsNone(summary["comparison"]["delta"]["Dice_mean"])
        self.assertTrue((self.root / "unlabelled" / "comparison.csv").is_file())

    def test_failed_tta_keeps_complete_baseline_for_retry(self):
        self.fail_adaptation = True
        config = self.config("failed")
        with self.assertRaisesRegex(RuntimeError, "controlled adaptation failure"):
            self.execute(config)
        self.assertFalse((config.output / "summary.json").exists())
        self.assertEqual(self.source_calls, 3)
        self.assertEqual(len(list((self.root / "baselines").glob("*/READY.json"))), 1)
        self.fail_adaptation = False
        summary = self.execute(self.config("retry"))
        self.assertTrue(summary["baseline"]["reused"])
        self.assertEqual(self.source_calls, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
