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
from phenotypic.gui.results_viewer._qc_tab import _ids as qc_ids
from phenotypic.gui.results_viewer._qc_tab.review import _ids as rids
from phenotypic.gui.results_viewer._qc_tab.review._callbacks import (
    _previous_group,
    _render_summary_header,
    _row_metric_status,
    render_worklist_row_metric_cell,
    sidebar_layout_state,
    worklist_row_metric_update,
)
from phenotypic.gui.results_viewer._qc_tab.review._layout import (
    SIDEBAR_DEFAULT_WIDTH_PX,
    SIDEBAR_MAX_WIDTH_PX,
    SIDEBAR_MIN_WIDTH_PX,
    build_review_view,
    clamp_sidebar_width,
)
from phenotypic.sdk_._qc_recipe import QcRecipe


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
    assert qc_ids.QC_MIGRATE_RECIPE_BTN_ID in ids
    assert qc_ids.QC_REBUILD_DATABASE_BTN_ID in ids
    assert qc_ids.QC_MIGRATE_CONFIRM_ID in ids
    assert qc_ids.QC_REBUILD_CONFIRM_ID in ids


def test_blocked_recipe_disables_qc_mutation_controls(tmp_path: Path) -> None:
    """An unreadable existing recipe exposes actions but refuses config writes."""
    pipeline = tmp_path / "pipeline.json.pht-pipe"
    pipeline.write_text('{"qc": [broken', encoding="utf-8")
    body = build_qc_tab_body(
        QcRecipe._load_from_paths(pipeline, pipeline)
    )
    nodes = {
        getattr(node, "id", None): node
        for node in _walk(body)
        if isinstance(getattr(node, "id", None), str)
    }

    assert nodes[qc_ids.QC_ADD_CHECK_BTN_ID].disabled is True
    assert nodes[qc_ids.QC_MIGRATE_RECIPE_BTN_ID].disabled is True
    assert nodes[qc_ids.QC_REBUILD_DATABASE_BTN_ID].disabled is True


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
        rids.QC_REVIEW_PREV_BTN_ID,
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


def test_previous_group_wraps_in_frozen_order() -> None:
    order = ["a", "b", "c"]
    assert _previous_group(order, "b") == "a"
    assert _previous_group(order, "a") == "c"
    assert _previous_group(order, "missing") == "missing"


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
    # No finite metric → median renders as "N/A" (no em dashes per DESIGN.md),
    # not "0.000".
    assert any(node == "N/A" for node in labels)


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


def test_splitter_handle_declares_its_target_and_store() -> None:
    """The handle carries the ids the generalized JS splitter reads.

    ``results_viewer.js`` no longer spells ``qc-review-worklist`` or
    ``store-qc-sidebar-width``: any element carrying
    ``data-splitter-target`` is a drag handle, and the two attributes name
    the pane to resize and the store to persist to. Python is now the only
    place those ids appear, so this is where the wiring is pinned.
    """
    splitter = _find_by_id(build_review_view(), rids.QC_REVIEW_SPLITTER_ID)
    assert splitter is not None
    props = splitter.to_plotly_json()["props"]
    assert props["data-splitter-target"] == rids.QC_REVIEW_WORKLIST_ID
    assert props["data-splitter-store"] == rids.STORE_QC_SIDEBAR_WIDTH

    # What this handle must NOT carry, which is as load-bearing as what
    # it must. The worklist is a left pane and its handle follows it, so
    # it rides the pane's RIGHT edge and wants the controller's default
    # sign; the Scatter inspector is right-docked and declares
    # `edge="left"` for the opposite one. Copy that declaration here --
    # a plausible merge, since the two call sites are otherwise twins --
    # and the worklist resizes backwards, which is precisely the bug that
    # motivated the edge attribute, reintroduced in the surface it was
    # fixed from. Nothing raises; a mutation adding these three kwargs to
    # this call site passed the whole GUI suite.
    assert "data-splitter-edge" not in props
    assert "data-splitter-min" not in props
    assert "data-splitter-max" not in props


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


# ---------------------------------------------------------------------------
# Worklist row metric/badge in-place update after recompute (spec §D.5)
# ---------------------------------------------------------------------------


def _metric_text_of(cell_children: list) -> str:
    """Extract the metric text from a worklist-row metric cell's children."""
    return cell_children[0].children


def _badge_of(cell_children: list):
    """Extract the status badge from a worklist-row metric cell's children."""
    return cell_children[1]


def test_metric_cell_renders_metric_and_status_badge() -> None:
    """The metric cell pairs the formatted metric with a status-coloured badge."""
    cell = render_worklist_row_metric_cell(0.281, "fail")
    assert _metric_text_of(cell).strip() == "0.281"
    badge = _badge_of(cell)
    assert badge.children == "fail"
    assert badge.color == "danger"
    # NaN / None metric renders the insufficient sentinel.
    insuf = render_worklist_row_metric_cell(None, "insufficient")
    assert _metric_text_of(insuf).strip() == "insuf."
    assert _badge_of(insuf).color == "secondary"


def test_metric_cell_appends_moved_hint() -> None:
    """``moved=True`` appends the ⤳ changed-after-recompute hint to the cell."""
    cell = render_worklist_row_metric_cell(0.226, "warn", moved=True)
    assert len(cell) == 3
    assert cell[2].children == " ⤳"
    # No hint when the metric was unchanged.
    assert len(render_worklist_row_metric_cell(0.226, "warn", moved=False)) == 2


def test_worklist_row_metric_update_flips_value_and_badge_in_place() -> None:
    """A recompute delta rewrites the row cell's metric AND badge in place.

    This is the regression that would have caught the stale-worklist-row
    bug: before the fix the worklist row kept the frozen-frame metric/badge
    after a curate→recompute. The delta carries the recomputed ``after``
    metric + ``status_after`` straight from the rewritten qc_summary
    artifact, so the cell flips from (0.281, fail) to (0.226, warn) — the
    badge colour changes too, not just the number.
    """
    delta = {
        "before": 0.281,
        "after": 0.226,
        "status_after": "warn",
        "moved": True,
    }
    cell = worklist_row_metric_update(delta)
    assert _metric_text_of(cell).strip() == "0.226"
    badge = _badge_of(cell)
    assert badge.children == "warn"
    assert badge.color == "warning"  # flipped from danger
    # The ⤳ moved hint is present because the metric changed.
    assert cell[2].children == " ⤳"


def test_worklist_row_metric_update_is_noop_without_delta() -> None:
    """A row with no recompute delta is left untouched (no cross-row repaint).

    The per-row MATCH callback fires for *every* row when the deltas store
    changes; rows that were not recomputed must short-circuit to
    ``dash.no_update`` so recomputing group A never repaints group B.
    """
    from dash import no_update

    assert worklist_row_metric_update(None) is no_update
    assert worklist_row_metric_update({}) is no_update


def test_worklist_row_metric_update_falls_back_when_status_missing() -> None:
    """A delta without ``status_after`` keeps a non-blank badge (neutral)."""
    cell = worklist_row_metric_update({"after": 0.5, "moved": True})
    badge = _badge_of(cell)
    assert badge.children == "insufficient"
    assert badge.color == "secondary"
    # An explicit fallback status is honoured over the neutral default.
    cell2 = worklist_row_metric_update(
        {"after": 0.5, "moved": False}, fallback_status="pass"
    )
    assert _badge_of(cell2).children == "pass"


def test_encoded_key_recovered_from_match_output_shapes() -> None:
    """The MATCH callback recovers the row's encoded key from its output id.

    Pins both shapes Dash may hand the single-output MATCH callback (a bare
    dict, or a one-element list), and the defensive ``None`` on a malformed
    id — so the in-place update targets the right row and degrades to a
    no-op rather than raising.
    """
    from phenotypic.gui.results_viewer._qc_tab.review._callbacks import (
        _encoded_key_from_output,
    )

    resolved_id = {
        "type": "qc-worklist-row-metric",
        "instance": "qc-SE-1a2b",
        "key": '["plate_002"]',
    }
    bare = {"id": resolved_id, "property": "children"}
    assert _encoded_key_from_output(bare) == '["plate_002"]'
    assert _encoded_key_from_output([bare]) == '["plate_002"]'
    # Malformed shapes degrade to None (callback no-ops, never raises).
    assert _encoded_key_from_output(None) is None
    assert _encoded_key_from_output({"property": "children"}) is None
    assert _encoded_key_from_output([]) is None


def test_row_metric_status_prefers_recompute_delta() -> None:
    """A full re-render (module switch / re-sort) carries the recompute value.

    ``_row_metric_status`` resolves a row's display metric/status, and a
    recompute delta's ``after``/``status_after`` must win over the frozen
    summary row so a re-render does not regress to the pre-recompute value.
    """
    frozen = {"metric": 0.281, "status": "fail"}
    # No delta → frozen values.
    assert _row_metric_status(frozen, {}) == (0.281, "fail")
    # With a delta → after values win.
    metric, status = _row_metric_status(
        frozen, {"after": 0.226, "status_after": "warn", "moved": True}
    )
    assert metric == 0.226
    assert status == "warn"
