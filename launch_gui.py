"""Validate the bundled data/weights, then open the TTA desktop interface."""

from __future__ import annotations

import importlib

from scripts.check_reproduction import main as check_reproduction


def main() -> int:
    status = check_reproduction(["--data", "--checkpoint"])
    if status:
        return status
    gui = importlib.import_module("Paradigm.tta_composition.gui")
    return int(gui.main())


if __name__ == "__main__":
    raise SystemExit(main())
