"""Unit tests for the QC tab layout builders (Configure | Review).

These walk the component trees the layout factories return so a latent
``NameError`` / missing-mount in a layout branch (e.g. the
``review_ids``-referencing toggle builder) is caught without booting a
Dash app — the app-build tests render the tree but don't assert that
every id the sub-view-switch + Review callbacks bind is actually
mounted, nor that the summary header lays out horizontally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from dash import html

from phenotypic.gui.results_viewer._qc_tab._layout import build_qc_tab_body
from phenotypic.gui.results_viewer._qc_tab.review import _ids as rids
from phenotypic.gui.results_viewer._qc_tab.review._callbacks import (
    _render_summary_header,
    sidebar_layout_state,
)
from phenotypic.gui.results_viewer._qc_tab.review._layout import (
    SIDEBAR_DEFAULT_WIDTH_PX,
    SIDEBAR_MAX_WIDTH_PX,
    SIDEBAR_MIN_WIDTH_PX,
    build_review_view,
    clamp_sidebar_width,
)
from phenotypic.qc import QcRecipe


def _walk(component: object) -> Iterator[object]:
    """Yield ``component`` and every descendant, depth-first."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def _ids_in(component: object) -> set:
    """Collect every static (string) component id in the tree."""
    found = set()
    for node in _walk(component):
        cid = getattr(node, "id", None)
        if isinstance(cid, str):
            found.add(cid)
    return found


def _empty_recipe() -> QcRecipe:
    """A QcRecipe backed by a nonexistent file (no entries, no warnings)."""
    return QcRecipe(path=Path("/nonexistent/pipeline.json"))


def test_qc_tab_body_mounts_toggle_and_both_subviews() -> None:
    """``build_qc_tab_body`` mounts the toggle + Configure + Review containers.

    Regression guard for a latent ``NameError`` in the toggle builder (it
    references the ``review_ids`` module): if that branch raised, this
    build would too. Also pins the exact ids the sub-view-switch callback
    (Callback G) drives, so a rename can't silently break the toggle.
    """
    body = build_qc_tab_body(_empty_recipe())
    ids = _ids_in(body)

    # Toggle + its mirror store.
    assert rids.QC_SUBVIEW_TOGGLE_ID in ids
    assert rids.STORE_QC_SUBVIEW in ids
    # Both sub-view wrappers (the switch callback flips their display).
    assert rids.QC_CONFIGURE_VIEW_ID in ids
    assert rids.QC_REVIEW_VIEW_ID in ids


def test_review_view_mounts_every_callback_target() -> None:
    """``build_review_view`` mounts every id the Review callbacks bind.

    Mirrors the Dash callback-graph validation but as a fast unit test:
    if any of these containers/stores were renamed or dropped, the
    Review callbacks would reference a nonexistent component at app boot.
    """
    ids = _ids_in(build_review_view())
    expected = {
        rids.QC_REVIEW_MODULE_PICKER_ID,
        rids.QC_REVIEW_MODULE_CHIPS_ID,
        rids.QC_REVIEW_SHOW_FILTER_ID,
        rids.QC_REVIEW_RESORT_BTN_ID,
        rids.QC_REVIEW_SUMMARY_HEADER_ID,
        rids.QC_REVIEW_SIDEBAR_ID,
        rids.QC_REVIEW_WORKLIST_ID,
        rids.QC_REVIEW_SPLITTER_ID,
        rids.QC_REVIEW_SIDEBAR_TOGGLE_ID,
        rids.QC_REVIEW_DETAIL_HEADER_ID,
        rids.QC_REVIEW_GALLERY_ID,
        rids.QC_REVIEW_MARK_REVIEWED_BTN_ID,
        rids.QC_REVIEW_NEXT_BTN_ID,
        rids.QC_REVIEW_BULK_REMOVE_BTN_ID,
        rids.QC_REVIEW_BULK_RESTORE_BTN_ID,
        rids.QC_REVIEW_EMPTY_STATE_ID,
        rids.STORE_QC_WORKLIST_ORDER,
        rids.STORE_QC_SELECTED_GROUP,
        rids.STORE_QC_RECOMPUTE_DELTAS,
        rids.STORE_QC_SIDEBAR_COLLAPSED,
        rids.STORE_QC_SIDEBAR_WIDTH,
    }
    missing = expected - ids
    assert not missing, f"Review layout missing callback targets: {missing}"


def test_summary_header_lays_out_horizontally() -> None:
    """The summary header is a horizontal flex row, not a vertical stack.

    Regression guard for the reported bug where the stat tiles stacked
    vertically and consumed the whole column.
    """
    stats = {
        "total": 8,
        "fail": 2,
        "warn": 5,
        "pass": 1,
        "insufficient": 0,
        "median_metric": 0.149,
    }
    header = _render_summary_header(stats, reviewed=0, colonies_removed=0)
    assert isinstance(header, html.Div)
    style = header.style or {}
    assert style.get("display") == "flex"
    assert style.get("flexDirection") == "row"
    # Eight tiles laid out as direct children of the flex row.
    assert isinstance(header.children, list)
    assert len(header.children) == 8


def test_summary_header_distinguishes_insufficient_from_pass() -> None:
    """An insufficient (NaN-metric) tile is rendered separately from pass."""
    stats = {
        "total": 3,
        "fail": 0,
        "warn": 0,
        "pass": 1,
        "insufficient": 2,
        "median_metric": None,
    }
    header = _render_summary_header(stats, reviewed=0, colonies_removed=0)
    labels = [
        node.children
        for tile in header.children
        for node in _walk(tile)
        if isinstance(getattr(node, "children", None), str)
    ]
    assert "Insufficient" in labels
    assert "Pass" in labels
    # No finite metric → median renders as an em dash, not "0.000".
    assert any(node == "—" for node in labels)


# ---------------------------------------------------------------------------
# Sidebar: narrow default + resizable + collapse chevron
# ---------------------------------------------------------------------------


def _find_by_id(component: object, target_id: str) -> object | None:
    """Return the first descendant whose ``id`` equals ``target_id``."""
    for node in _walk(component):
        if getattr(node, "id", None) == target_id:
            return node
    return None


def test_worklist_has_narrow_default_width() -> None:
    """The worklist starts at the narrow default width (was 280px)."""
    worklist = _find_by_id(build_review_view(), rids.QC_REVIEW_WORKLIST_ID)
    assert worklist is not None
    assert (worklist.style or {}).get("width") == f"{SIDEBAR_DEFAULT_WIDTH_PX}px"
    assert SIDEBAR_DEFAULT_WIDTH_PX == 180


def test_review_mounts_drag_splitter_and_width_store() -> None:
    """The Review view mounts the resize splitter handle + its width store."""
    view = build_review_view()
    splitter = _find_by_id(view, rids.QC_REVIEW_SPLITTER_ID)
    assert splitter is not None
    assert (splitter.style or {}).get("cursor") == "col-resize"
    # The width store seeds the default; the JS splitter writes the dragged px.
    store = _find_by_id(view, rids.STORE_QC_SIDEBAR_WIDTH)
    assert store is not None
    assert store.data == SIDEBAR_DEFAULT_WIDTH_PX


def test_clamp_sidebar_width_bounds_and_fallback() -> None:
    """``clamp_sidebar_width`` mirrors the JS clamp: bound + default-on-garbage.

    This is the automatable proof of the resizer's width logic (the JS
    splitter applies the identical clamp), so the drag's effect is covered
    without actuating a real pointer drag.
    """
    # In-range passes through (rounded).
    assert clamp_sidebar_width(250) == 250
    assert clamp_sidebar_width(250.4) == 250
    # Below min / above max clamp to the bounds.
    assert clamp_sidebar_width(50) == SIDEBAR_MIN_WIDTH_PX == 140
    assert clamp_sidebar_width(9999) == SIDEBAR_MAX_WIDTH_PX == 380
    # Garbage / None falls back to the default.
    assert clamp_sidebar_width(None) == SIDEBAR_DEFAULT_WIDTH_PX
    assert clamp_sidebar_width("nope") == SIDEBAR_DEFAULT_WIDTH_PX


def test_sidebar_has_collapse_chevron() -> None:
    """The sidebar mounts a chevron toggle button initialised to collapse (◀)."""
    toggle = _find_by_id(build_review_view(), rids.QC_REVIEW_SIDEBAR_TOGGLE_ID)
    assert toggle is not None
    assert toggle.children == "◀"


def test_sidebar_layout_state_combines_collapse_and_width() -> None:
    """``sidebar_layout_state`` flips visibility/glyph AND applies the dragged width.

    Regression guard for the single sidebar callback: collapsed hides the
    worklist + shows ▶; expanded shows it at the (clamped) dragged width +
    ◀. The persisted width is applied even when collapsed, so expanding
    restores the user's width rather than the default.
    """
    # Expanded at a user-dragged 260px.
    exp_sidebar, exp_worklist, exp_glyph = sidebar_layout_state(False, 260)
    assert exp_worklist["display"] == "block"
    assert exp_worklist["width"] == "260px"
    assert exp_glyph == "◀"
    assert exp_sidebar.get("position") == "sticky"

    # Collapsed: hidden + ▶, but the 260px width is retained for re-expand.
    col_sidebar, col_worklist, col_glyph = sidebar_layout_state(True, 260)
    assert col_worklist["display"] == "none"
    assert col_worklist["width"] == "260px"
    assert col_glyph == "▶"
    assert col_sidebar.get("position") == "sticky"

    # An out-of-range / garbage width is clamped/defaulted, never raw.
    _, clamped_worklist, _ = sidebar_layout_state(False, 9999)
    assert clamped_worklist["width"] == f"{SIDEBAR_MAX_WIDTH_PX}px"
