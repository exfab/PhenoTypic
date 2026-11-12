"""The ``>>>`` examples in the ABC docstrings run, and at least one exists."""

from __future__ import annotations

import doctest
import importlib

import pytest

_MODULES = [
    "phenotypic.abc_._image_operation",
    "phenotypic.abc_._threshold_detector",
]


@pytest.mark.parametrize("module_name", _MODULES)
def test_docstring_examples_run(module_name: str) -> None:
    module = importlib.import_module(module_name)
    finder = doctest.DocTestFinder(exclude_empty=True)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    for test in finder.find(module, module_name):
        runner.run(test)
    results = runner.summarize(verbose=False)
    assert results.attempted > 0, f"{module_name} has no runnable examples"
    assert results.failed == 0, f"{results.failed} doctest examples failed"
