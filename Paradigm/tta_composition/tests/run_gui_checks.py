"""Run focused GUI/data regressions without pytest or a real TTA experiment."""

from pathlib import Path
import sys
import unittest


def main() -> int:
    directory = Path(__file__).resolve().parent
    sys.path.insert(0, str(directory.parents[2]))
    suite = unittest.defaultTestLoader.discover(str(directory), pattern="test_gui_features.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
