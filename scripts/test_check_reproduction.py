"""Reproduction checks using synthetic split metadata and controlled failures."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import check_reproduction as checks


class ReproductionChecks(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="reproduction with spaces ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.split_dir = self.root / "data" / "splits"
        self.split_dir.mkdir(parents=True)
        self.train = ["a", "b", "c", "d"]
        self.validation = ["e"]
        self.write_split()

    def write_split(self) -> None:
        metadata = {
            "seed": 42, "validation_fraction": 0.2,
            "paired_source_count": len(self.train) + len(self.validation),
            "train_count": len(self.train), "validation_count": len(self.validation),
            "train_ids": self.train, "validation_ids": self.validation,
        }
        (self.split_dir / "source_split_seed42.json").write_text(json.dumps(metadata), encoding="utf-8")
        (self.split_dir / "source_train.txt").write_text("\n".join(self.train) + "\n", encoding="utf-8")
        (self.split_dir / "source_validation.txt").write_text("\n".join(self.validation) + "\n", encoding="utf-8")

    def test_matching_split_passes(self) -> None:
        checks.check_split(self.root, self.train, self.validation)

    def test_same_ids_in_different_order_fail(self) -> None:
        with self.assertRaisesRegex(checks.ReproductionError, "order differs"):
            checks.check_split(self.root, list(reversed(self.train)), self.validation)

    def test_stale_text_split_fails(self) -> None:
        (self.split_dir / "source_train.txt").write_text("a\nb\nc\nx\n", encoding="utf-8")
        with self.assertRaisesRegex(checks.ReproductionError, "saved split"):
            checks.check_split(self.root, self.train, self.validation)

    def test_overlapping_or_duplicate_ids_fail(self) -> None:
        self.validation = ["a"]
        self.write_split()
        with self.assertRaisesRegex(checks.ReproductionError, "duplicate or overlapping"):
            checks.check_split(self.root, self.train, self.validation)

    def test_missing_split_fails(self) -> None:
        (self.split_dir / "source_split_seed42.json").unlink()
        with self.assertRaises(FileNotFoundError):
            checks.check_split(self.root, self.train, self.validation)

    def test_missing_checkpoint_fails_before_loading(self) -> None:
        with self.assertRaisesRegex(checks.ReproductionError, "Missing checkpoint"):
            checks.check_checkpoint(self.root / "missing.pth")

    def test_lfs_checkpoint_pointer_reports_how_to_restore_weights(self) -> None:
        checkpoint = self.root / "source_checkpoint.pth"
        checkpoint.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:unused\nsize 123\n", encoding="utf-8")
        with patch.object(checks, "check_environment", return_value={}):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(checks.main(["--checkpoint", str(checkpoint)]), 1)
        self.assertIn("git lfs pull", json.loads(output.getvalue())["error"])

    def test_lfs_image_pointer_is_rejected_before_dataset_import(self) -> None:
        image = self.root / "data" / "raw" / "sz_cxr" / "images" / "CHNCXR_0001_0.png"
        image.parent.mkdir(parents=True)
        image.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:unused\nsize 123\n", encoding="utf-8")
        with patch.object(checks, "check_environment", return_value={}), patch.object(checks, "ROOT", self.root):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(checks.main(["--data"]), 1)
        self.assertIn("git lfs pull", json.loads(output.getvalue())["error"])

    def test_default_cli_checks_environment_without_data(self) -> None:
        with patch.object(checks, "check_environment", return_value={"fixture": True}), patch.object(checks, "check_data") as data_check:
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(checks.main([]), 0)
            data_check.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["status"], "passed")

    def test_cli_failure_returns_nonzero_and_json(self) -> None:
        with patch.object(checks, "check_environment", return_value={}), patch.object(checks, "check_data", side_effect=checks.ReproductionError("Missing benchmark data")):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(checks.main(["--data"]), 1)
        self.assertEqual(json.loads(output.getvalue()), {"status": "failed", "error": "Missing benchmark data"})


if __name__ == "__main__":
    unittest.main()
