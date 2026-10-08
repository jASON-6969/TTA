"""GUI launch gating without opening a window or requiring real data."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import launch_gui


class LaunchChecks(unittest.TestCase):
    def test_launch_checks_bundled_inputs_before_opening_gui(self) -> None:
        with patch.object(launch_gui, "check_reproduction", return_value=0) as checks, \
                patch("launch_gui.importlib.import_module") as importer:
            importer.return_value.main.return_value = 0
            self.assertEqual(launch_gui.main(), 0)
        checks.assert_called_once_with(["--data", "--checkpoint"])
        importer.assert_called_once_with("Paradigm.tta_composition.gui")
        importer.return_value.main.assert_called_once_with()

    def test_failed_input_check_prevents_gui_launch(self) -> None:
        with patch.object(launch_gui, "check_reproduction", return_value=1), \
                patch("launch_gui.importlib.import_module") as importer:
            self.assertEqual(launch_gui.main(), 1)
        importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
