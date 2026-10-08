"""Historical runs must not be relabelled as current-code validation."""

import json
from pathlib import Path
import tempfile
import unittest

from Paradigm.tta_composition.audit import adaptation_provenance, write_json
from Paradigm.tta_composition.run_catalog import build_catalog, classify_run


class RunCatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "v2" / "run"
        self.source_runs = self.root / "v1"
        self.sources = {"engine.py": "recorded-engine-hash"}
        write_json(self.run / "config.json", {"methods": ["DLTTA"], "adaptation_steps": 3})
        write_json(self.run / "summary.json", {"event": "completed", "cases": 138})

    def classify(self):
        return classify_run(self.run, self.source_runs, self.sources)

    def test_identical_copied_results_keep_the_original_origin(self):
        for name in ("config.json", "summary.json"):
            data = json.loads((self.run / name).read_text(encoding="utf-8"))
            write_json(self.source_runs / "run" / name, data)
        before = (self.run / "summary.json").read_bytes()
        self.assertEqual(self.classify()["origin"], "copied_from_test_v1")
        self.assertEqual((self.run / "summary.json").read_bytes(), before)

    def test_unversioned_runs_are_not_claimed_as_current(self):
        self.assertEqual(self.classify()["origin"], "unversioned_run")

    def test_current_and_different_source_snapshots_are_distinct(self):
        write_json(self.run / "provenance.json", {"adaptation_sources": self.sources})
        self.assertEqual(self.classify()["origin"], "current_adaptation_code")
        write_json(self.run / "provenance.json", {"adaptation_sources": {"engine.py": "older-hash"}})
        self.assertEqual(self.classify()["origin"], "different_adaptation_code")

    def test_source_snapshot_includes_update_and_method_implementations(self):
        provenance = adaptation_provenance(3)
        self.assertEqual(provenance["adaptation_steps"], 3)
        self.assertEqual(provenance["after_commit_policy"], "once_per_update")
        self.assertIn("engine.py", provenance["adaptation_sources"])
        self.assertIn("methods/dltta.py", provenance["adaptation_sources"])
        self.assertTrue(all(len(value) == 64 for value in provenance["adaptation_sources"].values()))

    def test_catalog_includes_runs_inside_a_verification_batch(self):
        batch_run = self.root / "v2" / "verify_batch_fixture" / "method_run"
        write_json(batch_run / "config.json", {"methods": ["SmaRT"], "adaptation_steps": 3})
        write_json(batch_run / "summary.json", {"event": "completed", "cases": 138})
        catalog = build_catalog(self.root / "v2", self.source_runs)
        self.assertEqual({entry["run"] for entry in catalog["runs"]}, {"run", "method_run"})
        self.assertEqual(catalog["counts"]["unversioned_run"], 2)
