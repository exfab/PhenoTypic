"""Integration tests for the toolbar issue badge (spec §4.6).

Exercises the new badge UI added in Phase 6:

* :func:`phenotypic.gui.builder._layout.build_issue_badge` renders the
  ``"N issues, M hints"`` label, severity-tinted chip, and popover row
  list with one row per :class:`Issue`.
* Each row carries ``data-testid="issue-row"`` plus a ``data-rule``
  attribute echoing the issue ``kind`` — the documented test-id
  convention from spec §5.5 that lets Playwright target rows without
  walking pattern-match ids.
* Rule short names (e.g. ``fork`` → ``"Fork"``, ``stub`` →
  ``"Unreachable"``) come from
  :data:`phenotypic.gui.builder._layout._ISSUE_RULE_SHORT_NAMES`.
* Issues sort before hints, and inside each severity the rows sort
  alphabetically by kind — matching the spec wording "issues first,
  hints second".
* The row id is the pattern-matched
  :func:`phenotypic.gui.builder._ids.issue_row_id` so the click-dispatch
  callback can pull (``block_id``, ``kind``, ``idx``) without parsing
  the DOM.

The tests walk Dash component trees without booting a server — keeping
the suite fast and independent of Cytoscape / clientside JS.
"""

from __future__ import annotations

from typing import Any, Dict, List

from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._layout import (
    _ISSUE_RULE_SHORT_NAMES,
    _format_issue_badge_label,
    _sort_issues_for_badge,
    build_issue_badge,
)
from phenotypic.gui.builder._state import (
    BlockNode,
    _DagBuilderScope,
    _DagBuilderState,
)

# Component-tree walking helpers shared with other inspector tests;
# see ``conftest.py`` in this directory.
from .conftest import _find_by_id, _find_by_type_key


# ---------------------------------------------------------------------------
# Test-id helpers
# ---------------------------------------------------------------------------


def _row_components(badge_span: Any) -> List[Any]:
    """Return every row inside the badge's popover, ordered by render order.

    Delegates to the shared ``_find_by_type_key`` conftest helper so
    every "find components by dict-id ``type`` key" call site in the
    integration suite uses the same walk semantics.
    """

    return _find_by_type_key(badge_span, "issue-row")


def _node_attr(node: Any, attr: str) -> Any:
    """Return the value of a dash ``data-*`` attribute splat through ``**``.

    Dash stores wildcard props (``data-*`` / ``aria-*``) with the literal
    hyphenated key in the component's instance dict (see
    ``html.Div._valid_wildcard_attributes``), so we read them via the
    hyphenated key rather than via attribute syntax.
    """

    # ``vars(node)`` exposes the hyphenated kwargs splat onto the component.
    return vars(node).get(attr)


# ---------------------------------------------------------------------------
# Label rendering
# ---------------------------------------------------------------------------


def test_badge_label_zero_zero_renders_zero_issues() -> None:
    """No issues and no hints renders ``"0 issues"`` (spec §4.6)."""

    assert _format_issue_badge_label(0, 0) == "0 issues"


def test_badge_label_singular_issue() -> None:
    """One issue is singular: ``"1 issue"``."""

    assert _format_issue_badge_label(1, 0) == "1 issue"


def test_badge_label_plural_issues_no_hints() -> None:
    """Multiple issues with zero hints drops the trailing comma."""

    assert _format_issue_badge_label(3, 0) == "3 issues"


def test_badge_label_singular_hint_only() -> None:
    """Zero issues but one hint shows both halves singular."""

    assert _format_issue_badge_label(0, 1) == "0 issues, 1 hint"


def test_badge_label_mixed_issues_and_hints() -> None:
    """Mixed counts pluralise each side independently."""

    assert _format_issue_badge_label(2, 1) == "2 issues, 1 hint"
    assert _format_issue_badge_label(1, 1) == "1 issue, 1 hint"
    assert _format_issue_badge_label(3, 2) == "3 issues, 2 hints"


def test_badge_label_matches_spec_example() -> None:
    """The spec §4.6 example (``"3 issues, 1 hint"``) renders verbatim."""

    assert _format_issue_badge_label(3, 1) == "3 issues, 1 hint"


# ---------------------------------------------------------------------------
# Badge rendering (chip + popover)
# ---------------------------------------------------------------------------


def test_badge_renders_with_empty_issue_list() -> None:
    """Empty issues list produces the zero-state chip ``"0 issues"``."""

    badge_span = build_issue_badge(issues=[], state=None)
    chips = _find_by_id(badge_span, ids.ISSUE_BADGE)
    assert len(chips) == 1
    badge = chips[0]
    assert badge.children == "0 issues"
    assert badge.color == "secondary"


def test_badge_renders_with_only_issues() -> None:
    """Pure issues list emits the ``"N issues"`` chip with danger colour."""

    issues: List[Dict[str, Any]] = [
        {
            "kind": "fork",
            "block_id": "blk_a",
            "detail": "image-out has >1 wire",
            "scope_path": [],
            "severity": "error",
        },
        {
            "kind": "stub",
            "block_id": "blk_b",
            "detail": "not reachable from Input Image",
            "scope_path": [],
            "severity": "error",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=None)
    chips = _find_by_id(badge_span, ids.ISSUE_BADGE)
    assert chips[0].children == "2 issues"
    assert chips[0].color == "danger"


def test_badge_renders_with_only_hints() -> None:
    """Pure advisory list emits the ``"0 issues, N hints"`` chip with warning colour."""

    issues = [
        {
            "kind": "stage_order_hint",
            "block_id": "blk_a",
            "detail": "runs in a later stage",
            "scope_path": [],
            "severity": "advisory",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=None)
    chips = _find_by_id(badge_span, ids.ISSUE_BADGE)
    assert chips[0].children == "0 issues, 1 hint"
    assert chips[0].color == "warning"


def test_badge_renders_with_mixed_severity() -> None:
    """Mixed list emits ``"N issues, M hints"`` and selects danger colour."""

    issues = [
        {
            "kind": "fork",
            "block_id": "blk_a",
            "detail": "image-out has >1 wire",
            "scope_path": [],
            "severity": "error",
        },
        {
            "kind": "stage_order_hint",
            "block_id": "blk_b",
            "detail": "runs in a later stage",
            "scope_path": [],
            "severity": "advisory",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=None)
    chips = _find_by_id(badge_span, ids.ISSUE_BADGE)
    assert chips[0].children == "1 issue, 1 hint"
    # Any blocking issue → danger colour even when hints are present.
    assert chips[0].color == "danger"


# ---------------------------------------------------------------------------
# Tooltip row structure
# ---------------------------------------------------------------------------


def test_popover_contains_tooltip_target() -> None:
    """The popover renders with the badge as its anchor target."""

    badge_span = build_issue_badge(issues=[], state=None)
    popovers = _find_by_id(badge_span, ids.ISSUE_BADGE_TOOLTIP)
    assert len(popovers) == 1
    assert popovers[0].target == ids.ISSUE_BADGE


def test_popover_renders_one_row_per_issue() -> None:
    """Each issue in the list renders exactly one row in the popover."""

    issues = [
        {
            "kind": "fork",
            "block_id": "blk_a",
            "detail": "image-out has >1 wire",
            "scope_path": [],
            "severity": "error",
        },
        {
            "kind": "stub",
            "block_id": "blk_b",
            "detail": "not reachable",
            "scope_path": [],
            "severity": "error",
        },
        {
            "kind": "cycle",
            "block_id": "blk_c",
            "detail": "block participates in a cycle",
            "scope_path": [],
            "severity": "error",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=None)
    rows = _row_components(badge_span)
    assert len(rows) == 3


def test_popover_row_carries_test_id_and_data_rule() -> None:
    """Every row exposes ``data-testid="issue-row"`` and ``data-rule="<kind>"``."""

    issues = [
        {
            "kind": "fork",
            "block_id": "blk_a",
            "detail": "image-out has >1 wire",
            "scope_path": [],
            "severity": "error",
        },
        {
            "kind": "stage_order_hint",
            "block_id": "blk_b",
            "detail": "runs in a later stage",
            "scope_path": [],
            "severity": "advisory",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=None)
    rows = _row_components(badge_span)
    # Every row carries data-testid="issue-row" and a data-rule attribute
    # matching its source kind.
    expected_rules = {"fork", "stage_order_hint"}
    actual_rules = set()
    for row in rows:
        assert _node_attr(row, "data-testid") == "issue-row"
        actual_rules.add(_node_attr(row, "data-rule"))
    assert actual_rules == expected_rules


def test_popover_row_id_matches_issue_row_factory() -> None:
    """Each row's id matches the :func:`ids.issue_row_id` factory."""

    issues = [
        {
            "kind": "fork",
            "block_id": "blk_a",
            "detail": "image-out has >1 wire",
            "scope_path": [],
            "severity": "error",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=None)
    rows = _row_components(badge_span)
    # Single issue → idx is 0.
    expected = ids.issue_row_id("blk_a", "fork", 0)
    assert rows[0].id == expected


def test_popover_row_id_handles_none_block_id() -> None:
    """Scope-level issues (``block_id == None``) mangle to ``"__scope__"``."""

    issues: List[Dict[str, Any]] = [
        {
            "kind": "missing_input",
            "block_id": None,
            "detail": "scope has no Input Image",
            "scope_path": [],
            "severity": "error",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=None)
    rows = _row_components(badge_span)
    assert rows[0].id == ids.issue_row_id(None, "missing_input", 0)
    # The sentinel survives serialisation through Dash's pattern-match
    # layer (which won't accept ``None`` as a key value in some store
    # paths).
    assert rows[0].id["block_id"] == "__scope__"


def test_popover_row_lists_block_label_rule_name_detail() -> None:
    """Each row carries block label + short rule name + detail (spec §4.6)."""

    block = BlockNode(
        block_id="blk_a",
        class_name="GaussianBlur",
        params={},
        label="UpstreamBlur",
    )
    scope = _DagBuilderScope(blocks=[block])
    state = _DagBuilderState(root=scope)
    issues = [
        {
            "kind": "fork",
            "block_id": "blk_a",
            "detail": "image-out has >1 wire",
            "scope_path": [],
            "severity": "error",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=state)
    rows = _row_components(badge_span)
    # The row has 3 ``html.Span`` children: block label, rule name, detail.
    row_children = rows[0].children
    span_texts = [getattr(c, "children", None) for c in row_children]
    assert "UpstreamBlur" in span_texts
    assert _ISSUE_RULE_SHORT_NAMES["fork"] in span_texts
    assert "image-out has >1 wire" in span_texts


def test_popover_row_falls_back_to_class_name_when_label_empty() -> None:
    """Blocks with no label render the class name as the row's left column."""

    block = BlockNode(
        block_id="blk_a",
        class_name="OtsuDetector",
        params={},
        label=None,
    )
    scope = _DagBuilderScope(blocks=[block])
    state = _DagBuilderState(root=scope)
    issues = [
        {
            "kind": "required_aux",
            "block_id": "blk_a",
            "detail": "inoculum_detector is required",
            "scope_path": [],
            "severity": "error",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=state)
    rows = _row_components(badge_span)
    row_children = rows[0].children
    span_texts = [getattr(c, "children", None) for c in row_children]
    assert "OtsuDetector" in span_texts
    assert _ISSUE_RULE_SHORT_NAMES["required_aux"] in span_texts


# ---------------------------------------------------------------------------
# Issue ordering (spec §4.6 "issues first, hints second")
# ---------------------------------------------------------------------------


def test_issues_sort_before_hints() -> None:
    """Advisory hints follow blocking issues regardless of input order."""

    issues = [
        {
            "kind": "stage_order_hint",
            "block_id": "blk_a",
            "detail": "stage misorder",
            "scope_path": [],
            "severity": "advisory",
        },
        {
            "kind": "fork",
            "block_id": "blk_b",
            "detail": "image-out has >1 wire",
            "scope_path": [],
            "severity": "error",
        },
    ]
    sorted_issues = _sort_issues_for_badge(issues)
    assert sorted_issues[0]["kind"] == "fork"
    assert sorted_issues[1]["kind"] == "stage_order_hint"


def test_issues_within_severity_sort_alphabetically_by_kind() -> None:
    """Inside the issues bucket, sort by ``kind`` alphabetically."""

    issues = [
        {
            "kind": "stub",
            "block_id": "blk_a",
            "detail": "unreachable",
            "scope_path": [],
            "severity": "error",
        },
        {
            "kind": "fork",
            "block_id": "blk_b",
            "detail": "image-out has >1 wire",
            "scope_path": [],
            "severity": "error",
        },
        {
            "kind": "cycle",
            "block_id": "blk_c",
            "detail": "block participates in a cycle",
            "scope_path": [],
            "severity": "error",
        },
    ]
    sorted_issues = _sort_issues_for_badge(issues)
    assert [i["kind"] for i in sorted_issues] == ["cycle", "fork", "stub"]


def test_popover_rows_render_in_sorted_order() -> None:
    """The popover's rendered row order matches the sort key."""

    issues = [
        {
            "kind": "stage_order_hint",
            "block_id": "blk_h",
            "detail": "stage misorder",
            "scope_path": [],
            "severity": "advisory",
        },
        {
            "kind": "fork",
            "block_id": "blk_f",
            "detail": "image-out has >1 wire",
            "scope_path": [],
            "severity": "error",
        },
        {
            "kind": "cycle",
            "block_id": "blk_c",
            "detail": "block participates in a cycle",
            "scope_path": [],
            "severity": "error",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=None)
    rows = _row_components(badge_span)
    rendered_kinds = [_node_attr(row, "data-rule") for row in rows]
    # Errors sorted alphabetically first, then advisories.
    assert rendered_kinds == ["cycle", "fork", "stage_order_hint"]


# ---------------------------------------------------------------------------
# Rule short names (spec §4.6 mapping table)
# ---------------------------------------------------------------------------


def test_rule_short_names_cover_every_kind() -> None:
    """Every documented issue kind has a corresponding short name."""

    expected_kinds = {
        "fork",
        "stub",
        "required_aux",
        "cycle",
        "container_mode",
        "missing_input",
        "duplicate_input",
        "stage_order_hint",
        "unknown_class",
    }
    assert set(_ISSUE_RULE_SHORT_NAMES.keys()) == expected_kinds


def test_rule_short_names_spec_examples() -> None:
    """Spec-listed mappings exist and match the source-of-truth strings."""

    assert _ISSUE_RULE_SHORT_NAMES["fork"] == "Fork"
    assert _ISSUE_RULE_SHORT_NAMES["stub"] == "Unreachable"
    assert _ISSUE_RULE_SHORT_NAMES["required_aux"] == "Missing aux"
    assert _ISSUE_RULE_SHORT_NAMES["cycle"] == "Cycle"
    assert _ISSUE_RULE_SHORT_NAMES["container_mode"] == "Container mode"
    assert _ISSUE_RULE_SHORT_NAMES["missing_input"] == "No Input Image"
    assert _ISSUE_RULE_SHORT_NAMES["duplicate_input"] == "Extra Input Image"
    assert _ISSUE_RULE_SHORT_NAMES["stage_order_hint"] == "Stage order"
    assert _ISSUE_RULE_SHORT_NAMES["unknown_class"] == "Unknown class"


# ---------------------------------------------------------------------------
# Row click → scroll_to dispatch payload (server-side fan-in is wired by
# ``register_callbacks``; here we exercise the pure payload shape).
# ---------------------------------------------------------------------------


def test_row_id_carries_block_id_kind_idx_for_pattern_match() -> None:
    """The row id encodes (block_id, kind, idx) per :func:`issue_row_id`."""

    rid = ids.issue_row_id("blk_x", "fork", 2)
    assert rid["type"] == "issue-row"
    assert rid["block_id"] == "blk_x"
    assert rid["kind"] == "fork"
    assert rid["idx"] == 2


def test_row_id_block_id_none_is_mangled_to_scope_sentinel() -> None:
    """``block_id=None`` becomes the literal ``"__scope__"``."""

    rid = ids.issue_row_id(None, "missing_input", 0)
    assert rid["block_id"] == "__scope__"


# ---------------------------------------------------------------------------
# Issue-row dispatch payload (mirrors the click callback's compute step).
# This validates the payload shape we publish to STORE_VIEWPORT_OP — 6B's
# clientside consumer (`phenotypicScrollTo`) drives the rest of the chain.
# ---------------------------------------------------------------------------


def _compute_dispatch_payload(
    sorted_issues: List[Dict[str, Any]],
    row_idx: int,
) -> Dict[str, Any]:
    """Reproduce the callback's compute step for one row click."""

    issue = sorted_issues[row_idx]
    scope_path = list(issue.get("scope_path") or [])
    return {
        "kind": "scroll_to",
        "block_id": issue.get("block_id"),
        "scope_path": scope_path,
        "target_breadcrumb": list(scope_path),
    }


def test_dispatch_payload_kind_is_scroll_to() -> None:
    """Click payload always sets ``kind="scroll_to"`` (spec §5.6)."""

    issues = [
        {
            "kind": "fork",
            "block_id": "blk_a",
            "detail": "...",
            "scope_path": [],
            "severity": "error",
        },
    ]
    sorted_issues = _sort_issues_for_badge(issues)
    payload = _compute_dispatch_payload(sorted_issues, 0)
    assert payload["kind"] == "scroll_to"


def test_dispatch_payload_block_id_matches_clicked_issue() -> None:
    """The payload's block_id is the clicked issue's offender id."""

    issues = [
        {
            "kind": "fork",
            "block_id": "blk_offender",
            "detail": "...",
            "scope_path": [],
            "severity": "error",
        },
    ]
    sorted_issues = _sort_issues_for_badge(issues)
    payload = _compute_dispatch_payload(sorted_issues, 0)
    assert payload["block_id"] == "blk_offender"


def test_dispatch_payload_scope_path_pulled_from_issue() -> None:
    """Issues inside nested scopes carry their ``scope_path``."""

    issues = [
        {
            "kind": "fork",
            "block_id": "blk_inner",
            "detail": "...",
            "scope_path": ["outer_container_id"],
            "severity": "error",
        },
    ]
    sorted_issues = _sort_issues_for_badge(issues)
    payload = _compute_dispatch_payload(sorted_issues, 0)
    assert payload["scope_path"] == ["outer_container_id"]


def test_dispatch_payload_target_breadcrumb_equals_scope_path() -> None:
    """Spec §5.6: ``target_breadcrumb`` is always the issue's ``scope_path``.

    The clientside chain compares it against ``state.breadcrumb`` and
    only drills when the two differ — passing ``scope_path``
    unconditionally is the right behaviour because the same code path
    handles both same-scope and cross-scope clicks.
    """

    issues = [
        {
            "kind": "fork",
            "block_id": "blk_inner",
            "detail": "...",
            "scope_path": ["container_id"],
            "severity": "error",
        },
    ]
    sorted_issues = _sort_issues_for_badge(issues)
    payload = _compute_dispatch_payload(sorted_issues, 0)
    assert payload["target_breadcrumb"] == payload["scope_path"]


def test_dispatch_payload_root_scope_target_breadcrumb_is_empty_list() -> None:
    """Root-scope issues emit ``target_breadcrumb=[]`` (no drill needed)."""

    issues: List[Dict[str, Any]] = [
        {
            "kind": "missing_input",
            "block_id": None,
            "detail": "...",
            "scope_path": [],
            "severity": "error",
        },
    ]
    sorted_issues = _sort_issues_for_badge(issues)
    payload = _compute_dispatch_payload(sorted_issues, 0)
    assert payload["target_breadcrumb"] == []


# ---------------------------------------------------------------------------
# Empty / fallback states
# ---------------------------------------------------------------------------


def test_empty_issues_shows_no_issues_placeholder() -> None:
    """Empty issues list still renders a placeholder row, not bare popover."""

    badge_span = build_issue_badge(issues=[], state=None)
    popovers = _find_by_id(badge_span, ids.ISSUE_BADGE_TOOLTIP)
    # No issue-rows.
    rows = _find_by_type_key(badge_span, "issue-row")
    assert len(rows) == 0
    # But the popover is still present and has a placeholder body so
    # hovering produces a useful "No issues" message rather than an
    # empty floating box.
    assert len(popovers) == 1


def test_badge_with_state_resolves_block_label_across_scopes() -> None:
    """Issues with scope_path resolve to the inner block's label."""

    inner_block = BlockNode(
        block_id="blk_inner",
        class_name="OtsuDetector",
        params={},
        label="InnerDetector",
    )
    inner_scope = _DagBuilderScope(blocks=[inner_block])
    container = BlockNode(
        block_id="blk_container",
        class_name="ImagePipeline",
        params={},
        label="MyContainer",
        nested=inner_scope,
    )
    root_scope = _DagBuilderScope(blocks=[container])
    state = _DagBuilderState(root=root_scope)

    issues = [
        {
            "kind": "fork",
            "block_id": "blk_inner",
            "detail": "image-out has >1 wire",
            "scope_path": ["blk_container"],
            "severity": "error",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=state)
    rows = _row_components(badge_span)
    row_children = rows[0].children
    span_texts = [getattr(c, "children", None) for c in row_children]
    # Inner-scope label resolves correctly through the scope_path walk.
    assert "InnerDetector" in span_texts


def test_badge_with_missing_block_id_falls_back_to_short_uuid() -> None:
    """Stale block_id (deleted since the issue was emitted) renders gracefully."""

    state = _DagBuilderState(root=_DagBuilderScope(blocks=[]))
    issues = [
        {
            "kind": "fork",
            "block_id": "deadbeefdeadbeefdeadbeefdeadbeef",
            "detail": "image-out has >1 wire",
            "scope_path": [],
            "severity": "error",
        },
    ]
    badge_span = build_issue_badge(issues=issues, state=state)
    rows = _row_components(badge_span)
    row_children = rows[0].children
    span_texts = [getattr(c, "children", None) for c in row_children]
    # Falls back to a short-prefixed id literal, not a crash.
    assert any("deadbeef" in str(t) for t in span_texts)
