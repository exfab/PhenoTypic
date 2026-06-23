"""Browse Compare-strip layout + id (Phase 4 Task 5).

The synced Compare strip is JS-owned (``timeline.js``); the Dash layout supplies
only the "Compare selected" button and the static ``data-compare-cap`` on the
grid container (so ``timeline.js`` reads the cap off the DOM, exactly like it
already reads ``data-focus-margin``/``data-mount-cap``/``data-warm-concurrency``).
The button carries the SURFACE-AGNOSTIC class ``.timeline-compare-btn`` so the
vendored controller finds it identically on Browse + Results.
"""
from __future__ import annotations

from typing import Any


def _walk_ids(component: Any) -> set[str]:
    found: set[str] = set()
    stack = [component]
    while stack:
        node = stack.pop()
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            found.add(cid)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return found


def _walk_classes(component: Any) -> set[str]:
    found: set[str] = set()
    stack = [component]
    while stack:
        node = stack.pop()
        class_name = getattr(node, "className", None)
        if isinstance(class_name, str):
            found.update(class_name.split())
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)
        elif children is not None:
            stack.append(children)
    return found


def _find(node: Any, target_id: str) -> Any:
    if getattr(node, "id", None) == target_id:
        return node
    children = getattr(node, "children", None)
    seq = (
        children
        if isinstance(children, (list, tuple))
        else ([children] if children is not None else [])
    )
    for child in seq:
        hit = _find(child, target_id)
        if hit is not None:
            return hit
    return None


def test_compare_button_id_is_a_nonempty_str() -> None:
    from phenotypic.gui.browse import _ids

    assert isinstance(_ids.BROWSE_TL_COMPARE_BTN, str)
    assert _ids.BROWSE_TL_COMPARE_BTN


def test_compare_button_id_is_unique_and_exported() -> None:
    from phenotypic.gui.browse import _ids

    assert "BROWSE_TL_COMPARE_BTN" in _ids.__all__
    # Distinct from every other Browse id (no accidental clobber).
    all_ids = [getattr(_ids, name) for name in _ids.__all__]
    assert len(all_ids) == len(set(all_ids))


def test_timeline_body_has_compare_button() -> None:
    from phenotypic.gui.browse import _ids
    from phenotypic.gui.browse._layout import build_browse_layout

    ids = _walk_ids(build_browse_layout())
    assert _ids.BROWSE_TL_COMPARE_BTN in ids


def test_compare_button_carries_surface_agnostic_class() -> None:
    # timeline.js (Task 6) finds the button by the surface-agnostic class
    # `.timeline-compare-btn` (NOT the Browse-specific id), so the vendored
    # controller is portable across Browse + Results.
    from phenotypic.gui.browse import _ids
    from phenotypic.gui.browse._layout import build_browse_layout

    button = _find(build_browse_layout(), _ids.BROWSE_TL_COMPARE_BTN)
    assert button is not None
    classes = getattr(button, "className", "").split()
    assert "timeline-compare-btn" in classes


def test_grid_container_exposes_compare_cap_dataattr() -> None:
    # timeline.js reads the cap off the DOM (like data-focus-margin), so the
    # static data-compare-cap must equal TIMELINE_COMPARE_CAP.
    from phenotypic.gui._config import TIMELINE_COMPARE_CAP
    from phenotypic.gui.browse import _ids
    from phenotypic.gui.browse._layout import build_browse_layout

    grid = _find(build_browse_layout(), _ids.BROWSE_TL_GRID)
    assert grid is not None
    props = grid.to_plotly_json().get("props", {})
    assert props.get("data-compare-cap") == str(TIMELINE_COMPARE_CAP)


def test_compare_button_is_a_real_button_with_no_dash_callback_shape() -> None:
    # The button is a DOM target for timeline.js, not a Dash-callback input;
    # it should still be a proper <button type=button> for accessibility.
    from phenotypic.gui.browse import _ids
    from phenotypic.gui.browse._layout import build_browse_layout

    button = _find(build_browse_layout(), _ids.BROWSE_TL_COMPARE_BTN)
    assert button is not None
    props = button.to_plotly_json().get("props", {})
    assert props.get("type") == "button"
