"""Unit tests for the empty-state hand-off banner gate.

Pins the Open-button ``disabled`` gate of the results-viewer and analysis
sub-apps' selection-store callbacks (the pure ``_handoff_banner_state``
helper each callback delegates to). A standalone deliverables bundle is
viewer-openable, so its Open button must be ENABLED even though it carries
no ``is_cli_output`` capability.
"""
from __future__ import annotations

import pytest

from phenotypic.gui.analysis._app import (
    _handoff_banner_state as analysis_banner_state,
)
from phenotypic.gui.results_viewer._app import (
    _handoff_banner_state as viewer_banner_state,
)

_BANNERS = pytest.mark.parametrize(
    "banner_state",
    [viewer_banner_state, analysis_banner_state],
    ids=["results_viewer", "analysis"],
)


@_BANNERS
def test_open_enabled_for_cli_output(banner_state) -> None:
    """A full CLI output enables the Open button (regression guard)."""
    style, label, disabled = banner_state(
        {"path": "run", "capabilities": {"is_cli_output": True}}
    )
    assert label == "run"
    assert style.get("display") == "flex"
    assert disabled is False


@_BANNERS
def test_open_enabled_for_deliverables_bundle(banner_state) -> None:
    """A standalone deliverables bundle (no ``is_cli_output``) is openable."""
    style, label, disabled = banner_state(
        {
            "path": "bundle/deliverables",
            "capabilities": {
                "is_cli_output": False,
                "is_deliverables_bundle": True,
            },
        }
    )
    assert label == "bundle/deliverables"
    assert style.get("display") == "flex"
    assert disabled is False


@_BANNERS
def test_open_disabled_for_plain_directory(banner_state) -> None:
    """A directory that is neither a run nor a bundle keeps Open disabled."""
    _, _, disabled = banner_state(
        {
            "path": "images",
            "capabilities": {
                "is_cli_output": False,
                "is_deliverables_bundle": False,
                "is_image_dir": True,
            },
        }
    )
    assert disabled is True


@_BANNERS
def test_hidden_and_disabled_without_selection(banner_state) -> None:
    """No selection (or no path) hides the banner and disables Open."""
    style, label, disabled = banner_state(None)
    assert style == {"display": "none"}
    assert label == "(none)"
    assert disabled is True

    style2, label2, disabled2 = banner_state({"path": "", "capabilities": {}})
    assert style2 == {"display": "none"}
    assert disabled2 is True
