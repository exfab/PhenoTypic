from pathlib import Path

from phenotypic.gui.browse import _ids as ids
from phenotypic.gui.browse._layout import build_browse_layout


def _ids_in_tree(node, found):
    cid = getattr(node, "id", None)
    if cid:
        found.add(cid)
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for c in children:
            _ids_in_tree(c, found)
    elif children is not None:
        _ids_in_tree(children, found)


def _component_with_id(node, component_id):
    if getattr(node, "id", None) == component_id:
        return node
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            found = _component_with_id(child, component_id)
            if found is not None:
                return found
    elif children is not None:
        return _component_with_id(children, component_id)
    return None


def test_layout_contains_core_ids():
    found: set[str] = set()
    _ids_in_tree(build_browse_layout(), found)
    for required in (
        ids.BROWSE_DATASET_PICKER,
        ids.BROWSE_IMAGE_PICKER,
        ids.BROWSE_PREV_BTN,
        ids.BROWSE_NEXT_BTN,
        ids.BROWSE_OSD_DIV,
        ids.BROWSE_OSD_LOADING,
        ids.BROWSE_LOADING_TEXT,
        ids.BROWSE_CURRENT_IMAGE_STORE,
        ids.BROWSE_DATASETS_STORE,
        ids.BROWSE_OSD_SYNC,
        ids.BROWSE_META_DIMS,
        ids.BROWSE_CSV_METADATA_PANEL,
        ids.BROWSE_EMPTY_HINT,
    ):
        assert required in found


def test_prev_next_buttons_use_wider_stable_hit_target():
    layout = build_browse_layout()
    prev_button = _component_with_id(layout, ids.BROWSE_PREV_BTN)
    next_button = _component_with_id(layout, ids.BROWSE_NEXT_BTN)

    assert "browse-step-button" in getattr(prev_button, "className", "")
    assert "browse-step-button" in getattr(next_button, "className", "")

    css = (
        Path(__file__).parents[3] / "src/phenotypic/gui/browse/_assets/browse.css"
    ).read_text(encoding="utf-8")
    assert ".browse-step-button" in css
    assert "min-width: 3rem;" in css


def _walk_ids(component) -> set[str]:
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


def _walk_classes(component) -> set[str]:
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


def test_layout_has_view_mode_toggle_and_both_bodies() -> None:
    from phenotypic.gui.browse._layout import build_browse_layout
    from phenotypic.gui.browse import _ids

    ids_found = _walk_ids(build_browse_layout())
    assert _ids.BROWSE_VIEW_MODE_TOGGLE in ids_found
    assert _ids.BROWSE_SINGLE_BODY in ids_found
    assert _ids.BROWSE_TIMELINE_BODY in ids_found
    assert _ids.BROWSE_TL_GRID in ids_found
    assert _ids.BROWSE_TL_PATTERN_INPUT in ids_found
    assert _ids.BROWSE_TL_TILE_SIZE_READOUT in ids_found
    # Focus-and-navigate chrome (spec §16): four edge buttons + position readout.
    assert _ids.BROWSE_TL_NAV_UP in ids_found
    assert _ids.BROWSE_TL_NAV_DOWN in ids_found
    assert _ids.BROWSE_TL_NAV_LEFT in ids_found
    assert _ids.BROWSE_TL_NAV_RIGHT in ids_found
    assert _ids.BROWSE_TL_POSITION in ids_found
    # Stores + revision-bound pop-out event.
    assert _ids.BROWSE_TL_STORE_TILE_SIZE in ids_found
    assert _ids.BROWSE_TL_STORE_WARNINGS in ids_found
    # The warnings store now has a UI sink (M3 — was a dead store).
    assert _ids.BROWSE_TL_WARNINGS_ALERT in ids_found
    assert _ids.BROWSE_TL_POPOUT_MODAL in ids_found
    assert _ids.BROWSE_TL_POPOUT_TITLE in ids_found
    assert _ids.BROWSE_TL_POPOUT_EVENT in ids_found
    assert _ids.BROWSE_TL_SOURCE_REVISION in ids_found


def test_timeline_body_carries_surface_agnostic_controller_classes() -> None:
    # timeline.js (P2-C) locates DOM by these stable classes, NOT by Dash id,
    # so the vendored controller is portable across Browse + Results.
    from phenotypic.gui.browse._layout import build_browse_layout

    classes = _walk_classes(build_browse_layout())
    for required in (
        "timeline-body",
        "timeline-viewport",
        "timeline-grid-container",
        "timeline-nav-up",
        "timeline-nav-down",
        "timeline-nav-left",
        "timeline-nav-right",
        "timeline-position",
    ):
        assert required in classes, f"missing controller class: {required}"


def test_timeline_grid_carries_static_focus_navigate_data_attrs() -> None:
    # The focus-navigate constants ride as STATIC data-* on the grid container
    # (the render callback replaces only its children). data-focus-margin
    # REPLACES the scroll-era data-margin-screens (spec §16.7).
    from phenotypic.gui.browse._layout import build_browse_layout
    from phenotypic.gui.browse import _ids
    from phenotypic.gui._config import (
        TIMELINE_FOCUS_MARGIN,
        TIMELINE_MOUNT_CAP,
        TIMELINE_WARM_CONCURRENCY,
    )

    grid = _component_with_id(build_browse_layout(), _ids.BROWSE_TL_GRID)
    assert grid is not None
    assert getattr(grid, "data-focus-margin") == str(TIMELINE_FOCUS_MARGIN)
    assert getattr(grid, "data-mount-cap") == str(TIMELINE_MOUNT_CAP)
    assert getattr(grid, "data-warm-concurrency") == str(TIMELINE_WARM_CONCURRENCY)
    assert getattr(grid, "data-grid-revision") == ""
    assert not hasattr(grid, "data-margin-screens")
