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
    spec = importlib.util.find_spec("phenotypic.gui")
    assert spec is None, (
        f"phenotypic.gui still resolves ({spec.submodule_search_locations}); a checkout "
        "that predates the move keeps src/phenotypic/gui/__pycache__ -- delete src/phenotypic/gui/"
    )


def test_private_gui_package_resolves() -> None:
    assert importlib.util.find_spec("phenotypic._gui") is not None


@pytest.mark.parametrize("sub_app", SUB_APP_DEBUG_LAUNCHERS)
def test_sub_app_debug_launchers_remain(sub_app: str) -> None:
    assert importlib.util.find_spec(f"phenotypic._gui.{sub_app}.__main__") is not None


def test_console_script_targets_private_launcher() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'phenotypic-gui = "phenotypic._gui.shell._launcher:main"' in pyproject


def test_hub_has_no_module_entry() -> None:
    """``python -m phenotypic._gui`` is not a way to start the hub."""
    assert importlib.util.find_spec("phenotypic._gui.__main__") is None
