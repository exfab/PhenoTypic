"""Verify palette buttons advertise the point-picker capability via badge + accent."""

from __future__ import annotations

from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui.builder._layout import _palette_for_categories
from phenotypic.gui.builder import _ids as ids


def _walk(node):
    """Yield every Dash component in a tree (best-effort recursion on .children)."""
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for c in children:
            if c is not None:
                yield from _walk(c)
    else:
        yield from _walk(children)


def test_badge_renders_for_pickable_ops():
    """ManualPointDetector and ManualRefine get a PICK badge; OtsuDetector does not."""
    reg = OperationRegistry()
    reg.discover()

    palette = _palette_for_categories(
            reg,
            accordion_id=ids.PALETTE_CONTAINER,
            category_filter={"Detector", "Refiner", "Enhancer", "Corrector"},
    )

    # Find the buttons keyed by op name (id has the form palette-button-<name>).
    pickable_button_classnames = {}
    badge_owners = set()

    from dash import html
    import dash_bootstrap_components as dbc

    for node in _walk(palette):
        if isinstance(node, dbc.Button):
            # The button's id is a string of the form returned by palette_button_id;
            # we don't need to parse it — read its first text child instead.
            name_span = next(
                    (c for c in (node.children or []) if isinstance(c, html.Span)),
                    None,
            )
            if name_span is None:
                continue
            op_name = name_span.children
            cls = node.className or ""
            if "builder-op-pickable" in cls:
                pickable_button_classnames[op_name] = cls
                if any(isinstance(c, dbc.Badge) for c in (node.children or [])):
                    badge_owners.add(op_name)

    assert "ManualPointDetector" in pickable_button_classnames
    assert "ManualRefine" in pickable_button_classnames
    assert "OtsuDetector" not in pickable_button_classnames
    assert "BlurGauss" not in pickable_button_classnames

    assert "ManualPointDetector" in badge_owners
    assert "ManualRefine" in badge_owners


def test_non_pickable_buttons_have_no_badge():
    """Buttons for non-mixin ops have no PICK badge child and no pickable class."""
    reg = OperationRegistry()
    reg.discover()

    palette = _palette_for_categories(
            reg,
            accordion_id=ids.PALETTE_CONTAINER,
            category_filter={"Detector"},
    )

    from dash import html
    import dash_bootstrap_components as dbc

    for node in _walk(palette):
        if isinstance(node, dbc.Button):
            name_span = next(
                    (c for c in (node.children or []) if isinstance(c, html.Span)),
                    None,
            )
            if name_span is None:
                continue
            op_name = name_span.children
            if op_name == "OtsuDetector":
                # No badge; no pickable class.
                assert not any(isinstance(c, dbc.Badge) for c in (node.children or []))
                assert "builder-op-pickable" not in (node.className or "")
                return
    raise AssertionError("OtsuDetector button not found in detector palette")
