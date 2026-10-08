"""Source inference cache regressions using temporary images and CPU fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from Paradigm.tta_composition.audit import sha256
from Paradigm.tta_composition.baseline import ensure_baseline
from Paradigm.tta_composition.data_selection import TargetCase


class BaselineChecks(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="tta baseline cache with spaces ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache_root = self.root / "baselines"
        self.checkpoint = self.root / "source checkpoint.pth"
        self.checkpoint.write_bytes(b"source weights fixture")
        self.size = 8
        self.records: list[TargetCase] = []
        for sample_id in ("a", "b", "c"):
            image_path = self.root / "images" / f"{sample_id}.png"
            image_path.parent.mkdir(exist_ok=True)
            Image.new("L", (self.size, self.size), color=100).save(image_path)
            self.records.append(TargetCase(sample_id, image_path))
        self.predicted: list[str] = []
        self.events: list[dict] = []

    def predict(self, case: TargetCase) -> np.ndarray:
        self.assertEqual(case.mask_paths, (), "Baseline callback received labels")
        self.predicted.append(case.sample_id)
        values = np.zeros((self.size, self.size), dtype=np.uint8)
        values[:, : self.size // 2] = 1
        return values

    def ensure(self, **overrides):
        arguments = {
            "cache_root": self.cache_root, "records": self.records,
            "checkpoint": self.checkpoint, "image_size": self.size,
            "normalization": "zscore", "device": "cpu", "predict": self.predict,
            "progress": self.events.append,
        }
        arguments.update(overrides)
        return ensure_baseline(**arguments)

    def test_builds_all_images_then_reuses_without_predicting(self) -> None:
        first = self.ensure()
        self.assertFalse(first.reused)
        self.assertEqual(first.cases, 3)
        self.assertEqual(self.predicted, ["a", "b", "c"])
        self.assertEqual(len(list(first.prediction_dir.glob("*.png"))), 3)
        self.assertTrue((first.directory / "READY.json").is_file())
        manifest = json.loads((first.directory / "baseline_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["cases"], 3)
        self.assertEqual(len(manifest["identity"]["images"]), 3)
        self.assertEqual(len(manifest["predictions"]), 3)
        self.assertTrue(manifest["identity"]["inference_sources"])
        with Image.open(first.prediction_dir / "a.png") as prediction:
            self.assertEqual(prediction.mode, "L")
            self.assertEqual(prediction.size, (self.size, self.size))
            self.assertEqual(set(np.unique(np.asarray(prediction))), {0, 255})
        second = self.ensure(predict=lambda _case: self.fail("A complete cache called predict again"))
        self.assertTrue(second.reused)
        self.assertEqual(first.cache_key, second.cache_key)
        self.assertEqual(first.directory, second.directory)
        self.assertEqual(self.predicted, ["a", "b", "c"])
        self.assertEqual([event["status"] for event in self.events if event["event"] == "baseline"],
                         ["checking", "building", "ready", "checking", "reused"])
        self.assertEqual([event["completed"] for event in self.events if event["event"] == "baseline_progress"], [1, 2, 3])

    def test_mask_changes_and_evaluation_reports_do_not_invalidate_inference(self) -> None:
        mask = self.root / "label.png"
        Image.new("L", (self.size, self.size), color=0).save(mask)
        labelled = [TargetCase(record.sample_id, record.image_path, (mask,)) for record in self.records]
        original_open = Image.open

        def guard_mask_reads(path, *args, **kwargs):
            self.assertNotEqual(Path(path).resolve(), mask.resolve(), "Baseline opened target labels")
            return original_open(path, *args, **kwargs)

        with patch.object(Image, "open", guard_mask_reads):
            first = self.ensure(records=labelled)
            mask.write_bytes(b"changed label contents; not an image")
            (first.directory / "summary.json").write_text("invalid evaluation JSON", encoding="utf-8")
            (first.directory / "metrics.csv").write_text("later evaluation", encoding="utf-8")
            second = self.ensure(records=self.records, predict=lambda _case: self.fail("Label changes rebuilt inference"))
        self.assertTrue(second.reused)
        self.assertEqual(first.cache_key, second.cache_key)

    def test_image_bytes_invalidate_cache_even_when_file_size_and_timestamp_match(self) -> None:
        first = self.ensure()
        image_path = self.records[0].image_path
        old_stat = image_path.stat()
        Image.new("L", (self.size, self.size), color=101).save(image_path)
        self.assertEqual(image_path.stat().st_size, old_stat.st_size)
        import os
        os.utime(image_path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        second = self.ensure()
        self.assertFalse(second.reused)
        self.assertNotEqual(first.cache_key, second.cache_key)
        self.assertTrue(first.directory.is_dir())

    def test_checkpoint_bytes_invalidate_cache(self) -> None:
        first = self.ensure()
        self.checkpoint.write_bytes(b"updated source weights fixture")
        second = self.ensure()
        self.assertFalse(second.reused)
        self.assertNotEqual(first.cache_key, second.cache_key)

    def test_preprocessing_and_actual_device_each_invalidate_cache(self) -> None:
        first = self.ensure()
        for overrides in ({"normalization": "minmax"}, {"device": "cuda:0"},
                          {"image_size": 4, "predict": lambda _case: np.zeros((4, 4), dtype=bool)}):
            with self.subTest(overrides=list(overrides)):
                changed = self.ensure(**overrides)
                self.assertFalse(changed.reused)
                self.assertNotEqual(first.cache_key, changed.cache_key)

    def test_inference_code_changes_invalidate_cache(self) -> None:
        first = self.ensure()
        with patch("Paradigm.tta_composition.baseline._inference_source_hashes", return_value={"model.py": "changed"}):
            second = self.ensure()
        self.assertFalse(second.reused)
        self.assertNotEqual(first.cache_key, second.cache_key)

    def test_runtime_version_changes_invalidate_cache(self) -> None:
        first = self.ensure()
        manifest = json.loads((first.directory / "baseline_manifest.json").read_text(encoding="utf-8"))
        runtime = manifest["identity"]["runtime"]
        self.assertEqual(set(runtime), {"torch", "numpy", "pillow", "cuda"})
        for dependency in runtime:
            changed = {**runtime, dependency: "updated inference dependency"}
            with self.subTest(dependency=dependency), patch(
                "Paradigm.tta_composition.baseline._runtime_identity", return_value=changed,
            ):
                updated = self.ensure()
            self.assertFalse(updated.reused)
            self.assertNotEqual(first.cache_key, updated.cache_key)

    def test_record_order_and_callback_settings_do_not_change_prediction_identity(self) -> None:
        first = self.ensure()
        tta_settings = {"lr": 0.1, "seed": 1, "max_cases": 1, "methods": ["GraTa"]}

        def another_callback(_case):
            self.fail(f"TTA settings unexpectedly caused source inference: {tta_settings}")

        second = self.ensure(records=list(reversed(self.records)), predict=another_callback)
        self.assertTrue(second.reused)
        self.assertEqual(first.cache_key, second.cache_key)

    def test_partial_prediction_failure_is_not_ready_and_retry_builds_all(self) -> None:
        def failing_predict(case):
            if case.sample_id == "b":
                raise RuntimeError("controlled inference failure")
            return self.predict(case)

        with self.assertRaisesRegex(RuntimeError, "controlled inference failure"):
            self.ensure(predict=failing_predict)
        self.assertEqual(list(self.cache_root.iterdir()), [])
        self.predicted.clear()
        recovered = self.ensure()
        self.assertFalse(recovered.reused)
        self.assertEqual(self.predicted, ["a", "b", "c"])
        self.assertTrue((recovered.directory / "READY.json").is_file())

    def test_cancellation_during_progress_does_not_publish_ready_cache(self) -> None:
        def cancel(event):
            if event["event"] == "baseline_progress":
                raise KeyboardInterrupt("controlled cancellation")

        with self.assertRaises(KeyboardInterrupt):
            self.ensure(progress=cancel)
        self.assertEqual(list(self.cache_root.iterdir()), [])
        recovered = self.ensure()
        self.assertFalse(recovered.reused)
        self.assertEqual(recovered.cases, len(self.records))

    def test_corrupt_png_is_quarantined_and_rebuilt(self) -> None:
        first = self.ensure()
        (first.prediction_dir / "a.png").write_bytes(b"damaged PNG")
        self.predicted.clear()
        rebuilt = self.ensure()
        self.assertFalse(rebuilt.reused)
        self.assertEqual(first.cache_key, rebuilt.cache_key)
        self.assertEqual(self.predicted, ["a", "b", "c"])
        quarantines = list(self.cache_root.glob(f"{first.cache_key}.quarantine-*"))
        self.assertEqual(len(quarantines), 1)
        self.assertEqual((quarantines[0] / "predictions" / "a.png").read_bytes(), b"damaged PNG")
        with Image.open(rebuilt.prediction_dir / "a.png") as prediction:
            self.assertEqual(prediction.size, (self.size, self.size))

    def test_missing_ready_marker_and_extra_prediction_are_not_reused(self) -> None:
        first = self.ensure()
        (first.directory / "READY.json").unlink()
        rebuilt = self.ensure()
        self.assertFalse(rebuilt.reused)
        Image.new("L", (self.size, self.size)).save(rebuilt.prediction_dir / "unexpected.png")
        again = self.ensure()
        self.assertFalse(again.reused)
        self.assertEqual(len(list(self.cache_root.glob(f"{first.cache_key}.quarantine-*"))), 2)

    def test_prediction_shape_is_checked_even_when_manifest_and_hashes_agree(self) -> None:
        first = self.ensure()
        path = first.prediction_dir / "a.png"
        Image.new("L", (2, 2)).save(path)
        manifest_path = first.directory / "baseline_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["predictions"][0]["sha256"] = sha256(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        marker_path = first.directory / "READY.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["manifest_sha256"] = sha256(manifest_path)
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
        self.assertFalse(self.ensure().reused)

    def test_inputs_changed_during_build_do_not_publish_old_identity(self) -> None:
        def mutating_predict(case):
            if case.sample_id == "b":
                self.checkpoint.write_bytes(b"changed while predicting")
            return self.predict(case)

        with self.assertRaisesRegex(RuntimeError, "inputs changed"):
            self.ensure(predict=mutating_predict)
        self.assertEqual(list(self.cache_root.iterdir()), [])

    def test_tensor_batch_predictions_are_saved_as_binary_png(self) -> None:
        result = self.ensure(predict=lambda _case: torch.ones((1, self.size, self.size), dtype=torch.bfloat16))
        with Image.open(result.prediction_dir / "a.png") as prediction:
            self.assertEqual(set(np.unique(np.asarray(prediction))), {255})

    def test_invalid_predictions_never_publish_ready_cache(self) -> None:
        invalid_predictions = [np.zeros((2, 2)), np.full((self.size, self.size), np.nan),
                               np.full((self.size, self.size), 0.5)]
        for prediction in invalid_predictions:
            with self.subTest(shape=prediction.shape), self.assertRaises(ValueError):
                self.ensure(predict=lambda _case: prediction)
            self.assertEqual(list(self.cache_root.iterdir()), [])

    def test_empty_duplicate_and_unsafe_records_are_rejected(self) -> None:
        invalid_records = [[], [self.records[0], self.records[0]],
                           [self.records[0], TargetCase("A", self.records[1].image_path)],
                           [self.records[0], TargetCase("different_id", self.records[0].image_path)],
                           [TargetCase("../escape", self.records[0].image_path)]]
        for records in invalid_records:
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.ensure(records=records)
        self.assertFalse(self.cache_root.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
