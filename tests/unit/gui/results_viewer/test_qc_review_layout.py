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
    sidebar_collapse_state,
)
from phenotypic.gui.results_viewer._qc_tab.review._layout import build_review_view
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


def test_worklist_is_narrow_default_and_resizable() -> None:
    """The worklist has a narrow default width and a native resize grip."""
    worklist = _find_by_id(build_review_view(), rids.QC_REVIEW_WORKLIST_ID)
    assert worklist is not None
    style = worklist.style or {}
    # Narrower than the old 280px default.
    assert style.get("width") == "180px"
    # Native, dependency-free horizontal resize (requires overflow != visible).
    assert style.get("resize") == "horizontal"
    assert style.get("overflow") == "auto"
    assert style.get("minWidth") == "140px"
    assert style.get("maxWidth") == "380px"


def test_sidebar_has_collapse_chevron() -> None:
    """The sidebar mounts a chevron toggle button initialised to collapse (◀)."""
    toggle = _find_by_id(build_review_view(), rids.QC_REVIEW_SIDEBAR_TOGGLE_ID)
    assert toggle is not None
    assert toggle.children == "◀"


def test_sidebar_collapse_state_flips_visibility_and_glyph() -> None:
    """``sidebar_collapse_state`` flips worklist visibility + chevron glyph.

    Regression guard for the collapse callback: collapsed hides the
    worklist and shows the ▶ expand glyph; expanded shows the worklist and
    the ◀ collapse glyph. The detail pane reclaims width via flex (no
    explicit gallery output), so this is the full behavioural contract.
    """
    expanded_style, expanded_vis, expanded_glyph = sidebar_collapse_state(False)
    assert expanded_vis == {"display": "block"}
    assert expanded_glyph == "◀"
    assert expanded_style.get("position") == "sticky"

    collapsed_style, collapsed_vis, collapsed_glyph = sidebar_collapse_state(True)
    assert collapsed_vis == {"display": "none"}
    assert collapsed_glyph == "▶"
    # Collapsed wrapper stays sticky (the chevron rail remains in view).
    assert collapsed_style.get("position") == "sticky"
