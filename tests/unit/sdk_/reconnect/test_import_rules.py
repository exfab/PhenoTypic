"""Import-boundary tests for :mod:`phenotypic.sdk_.reconnect`."""

from __future__ import annotations

import json
import subprocess
import sys


def test_reconnect_import_does_not_load_optional_or_ui_modules():
    # A dotted import necessarily initializes the existing eager ``phenotypic`` and
    # ``phenotypic.sdk_`` packages first. Measure the reconnect package's incremental
    # imports so this test does not attribute that established parent behavior to the
    # new pure-helper package.
    script = """
import json
import sys
import phenotypic.sdk_

blocked = (
    "astropy",
    "fil_finder",
    "filterpy",
    "gudhi",
    "matplotlib",
    "napari",
    "plotly",
    "PyQt5",
    "PySide6",
)
before = set(sys.modules)
import phenotypic.sdk_.reconnect
added = set(sys.modules) - before
print(json.dumps(sorted(name for name in added if name.split(".")[0] in blocked)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
