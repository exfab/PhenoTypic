"""Layer 2 — runtime resource-access tests.

These resolve GUI assets through ``importlib.resources`` and exercise the
import-time read that broke ``pip``-installed deployments. From a source
checkout they resolve into ``src/`` and pass trivially; their real value is
when run **against the installed wheel** in a fresh venv (the CI
``package.yml`` install step), where a missing asset fails loudly.

Layer 1 (``test_package_contents.py``) is the source-tree authority that
inspects the built artifact directly.
"""
from __future__ import annotations

import importlib
import importlib.resources as resources


def test_shell_css_resource_readable():
    """``shell.css`` resolves as package data and is non-empty."""
    ref = resources.files("phenotypic.gui.shell") / "_assets" / "shell.css"
    assert ref.is_file(), "shell.css not found in installed package"
    assert ref.read_text(encoding="utf-8").strip(), "shell.css is empty"


def test_builder_js_resource_readable():
    ref = resources.files("phenotypic.gui.builder") / "assets" / "builder.js"
    assert ref.is_file(), "builder.js not found in installed package"
    assert ref.read_text(encoding="utf-8").strip(), "builder.js is empty"


def test_openseadragon_icon_resource_readable():
    ref = (
        resources.files("phenotypic.gui.results_viewer")
        / "_assets"
        / "openseadragon"
        / "images"
        / "zoomin_rest.png"
    )
    assert ref.is_file(), "OpenSeadragon control icon not packaged"
    assert ref.read_bytes(), "OpenSeadragon control icon is empty"


def test_shell_layout_imports_and_inlines_css():
    """The exact import that crashes ``phenotypic-gui`` on a broken wheel.

    ``gui/shell/_layout.py`` reads ``shell.css`` at import time, so a wheel
    missing that asset raises ``FileNotFoundError`` here — before argparse
    ever runs.
    """
    layout = importlib.import_module("phenotypic.gui.shell._layout")
    assert isinstance(layout._SHELL_CSS, str)
    assert layout._SHELL_CSS.strip(), "_SHELL_CSS inlined as empty string"
