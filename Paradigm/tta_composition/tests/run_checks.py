"""Run unittest cases and the existing function-based composition checks."""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import tempfile
from typing import Callable
import unittest

import torch


def _run_function(function: Callable[..., None]) -> None:
    parameters = inspect.signature(function).parameters
    if not parameters:
        function()
    elif tuple(parameters) == ("tmp_path",):
        with tempfile.TemporaryDirectory(prefix="tta composition checks ") as directory:
            function(Path(directory))
    else:
        raise TypeError(f"Unsupported test fixture for {function.__name__}: {tuple(parameters)}")


def _function_case(function: Callable[..., None]) -> unittest.FunctionTestCase:
    def invoke() -> None:
        _run_function(function)
    return unittest.FunctionTestCase(invoke, description=function.__name__)


def main() -> int:
    directory = Path(__file__).resolve().parent
    sys.path.insert(0, str(directory.parents[2]))
    from Paradigm.tta_composition.tests import test_composition

    torch.set_num_threads(1)
    suite = unittest.defaultTestLoader.discover(str(directory), pattern="test_*.py")
    for name, function in sorted(vars(test_composition).items()):
        if name.startswith("test_") and inspect.isfunction(function):
            suite.addTest(_function_case(function))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
