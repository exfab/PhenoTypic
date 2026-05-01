"""Unit tests for ``phenotypic.gui.shell._release_button``.

The Release button is scaffolding for Phase 5 (the chrome wraps each tool
with one). Phase 3 doesn't render it anywhere, so without these tests it
could silently rot before Phase 5 wires it in.
"""
from __future__ import annotations

from dash import html

from phenotypic.gui.shell._ids import release_button_id, release_status_id
from phenotypic.gui.shell._release_button import build_release_button


def test_button_returns_div_with_pattern_matching_id() -> None:
    component = build_release_button("viewer")
    assert isinstance(component, html.Div)

    children = component.children
    assert isinstance(children, list)

    # First child is the dbc.Button with the pattern-matching ID.
    btn = children[0]
    assert btn.id == release_button_id("viewer")

    # Last child is the status span with the matching pattern ID.
    status = children[-1]
    assert isinstance(status, html.Span)
    assert status.id == release_status_id("viewer")


def test_button_label_matches_honest_ux_copy() -> None:
    """Plan reviewer pinned the wording ('Release loaded data') because it
    is honest about RSS retention. Lock the contract."""
    component = build_release_button("viewer")
    btn = component.children[0]
    assert btn.children == "Release loaded data"


def test_button_tooltip_warns_about_rss_retention() -> None:
    """Tooltip must surface the 'RSS may stay elevated' caveat."""
    component = build_release_button("viewer")
    tooltip = component.children[1]
    assert "RSS" in tooltip.children
    # Phrase from the spec; rooted in the plan-reviewer feedback.
    assert "stay elevated" in tooltip.children


def test_button_pattern_id_is_keyed_by_tool() -> None:
    """Different tools get distinct IDs so one chrome callback can dispatch."""
    a = build_release_button("viewer")
    b = build_release_button("builder")
    assert a.children[0].id != b.children[0].id
    assert a.children[0].id["tool"] == "viewer"
    assert b.children[0].id["tool"] == "builder"
