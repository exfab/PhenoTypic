"""The GUI ships as the private ``phenotypic._gui`` package.

Users start the hub only through the ``phenotypic-gui`` console script. The
public ``phenotypic.gui`` import path is gone; the five sub-app debug launchers
remain for contributors.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

SUB_APP_DEBUG_LAUNCHERS = ("analysis", "browse", "builder", "results_viewer", "run_console")


def test_public_gui_import_path_is_gone() -> None:
    """``phenotypic.gui`` must not resolve -- not even as a namespace package."""
    assert importlib.util.find_spec("phenotypic.gui") is None


def test_private_gui_package_resolves() -> None:
    assert importlib.util.find_spec("phenotypic._gui") is not None


@pytest.mark.parametrize("sub_app", SUB_APP_DEBUG_LAUNCHERS)
def test_sub_app_debug_launchers_remain(sub_app: str) -> None:
    assert importlib.util.find_spec(f"phenotypic._gui.{sub_app}.__main__") is not None


def test_console_script_targets_private_launcher() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'phenotypic-gui = "phenotypic._gui.shell._launcher:main"' in pyproject
