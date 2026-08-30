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
        ids.BROWSE_POSITION,
        ids.BROWSE_NAV_EVENT_STORE,
        ids.BROWSE_KEEP_POSITION,
        ids.BROWSE_OSD_DIV,
        ids.BROWSE_PREVIEW_IMG,
        ids.BROWSE_OSD_LOADING,
        ids.BROWSE_LOADING_TEXT,
        ids.BROWSE_FILMSTRIP,
        ids.BROWSE_PREPARE_BTN,
        ids.BROWSE_STOP_PREPARE_BTN,
        ids.BROWSE_CLEAR_CACHE_BTN,
        ids.BROWSE_PREPARATION_STATUS,
        ids.BROWSE_PREPARATION_PROGRESS,
        ids.BROWSE_CACHE_USAGE,
        ids.BROWSE_BACKEND_DETAILS,
        ids.BROWSE_PREPARATION_STATUS_STORE,
        ids.BROWSE_PREPARATION_POLL,
        ids.BROWSE_CURRENT_IMAGE_STORE,
        ids.BROWSE_DATASETS_STORE,
        ids.BROWSE_OSD_SYNC,
        ids.BROWSE_META_IMAGE_NAME,
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


def test_single_view_has_accessible_navigation_and_preparation_chrome():
    layout = build_browse_layout()
    filmstrip = _component_with_id(layout, ids.BROWSE_FILMSTRIP)
    preview = _component_with_id(layout, ids.BROWSE_PREVIEW_IMG)
    keep_position = _component_with_id(layout, ids.BROWSE_KEEP_POSITION)
    stop = _component_with_id(layout, ids.BROWSE_STOP_PREPARE_BTN)
    progress = _component_with_id(layout, ids.BROWSE_PREPARATION_PROGRESS)

    assert getattr(filmstrip, "role") == "list"
    assert getattr(filmstrip, "aria-label") == "Nearby images in this plate series"
    assert getattr(preview, "aria-hidden") == "true"
    assert keep_position.persistence is True
    assert keep_position.persistence_type == "local"
    assert stop.disabled is True
    assert getattr(progress, "aria-label") == "Dataset preparation progress"
