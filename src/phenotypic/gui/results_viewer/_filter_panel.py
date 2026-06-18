"""Filter sidebar layout and callbacks for the results viewer.

The sidebar is the user's entry point for narrowing the master measurements
table down to a list of overlay images worth inspecting. Each *filter row*
binds a single column to a multi-value selection; rows AND together while
each row's values OR together (see
:class:`phenotypic.gui.results_viewer._filter_state.FilterSpec`).

This module owns two responsibilities and nothing else:

1. **Layout** (`layout`) — the static skeleton of the sidebar (header, the
   pattern-matching row container, an "+ Add filter" button, and a
   read-only chip showing how many image pairs survive). The dynamic rows
   themselves are rendered into the container by a callback so the
   sidebar can be re-hydrated from ``STORE_FILTER_SPEC`` after a page
   reload (session storage) or programmatic update.
2. **Callbacks** (`register_callbacks`) — every callback whose Output is
   confined to the filter sidebar or the spec / image-pairs stores.

Cards, headers, and Lock-views toggling live in ``_layout.py`` /
``_viewer_card.py``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Iterable

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import polars as pl
from dash import ALL, Input, Output, State, dcc, html, no_update
from dash.development.base_component import Component

from phenotypic.gui._design import (
    COLOR_BLUE,
    COLOR_NAVY,
    COLOR_SURFACE,
    FONT_FAMILY_MONO,
    FONT_SIZE_CAPTION,
    OI_VERMILION_TEXT,
)
from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._filter_state import (
    COMPARE_OPS,
    FilterSpec,
    METHOD_COMPARE,
    METHOD_CONTAINS,
    METHOD_IS_ANY_OF,
    METHOD_IS_NONE_OF,
    METHOD_RANGE,
    VALID_METHODS,
    _coerce_float,
)
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


# Maximum number of preview chips rendered inside a bulk-paste popover.
# Anything beyond this collapses to a single trailing "… and N more" chip
# so the popover doesn't blow up on a 10k-row paste.
_MAX_PASTE_PREVIEW_CHIPS = 200

# Bulk-paste design choice
# ------------------------
# Single-stage commit: every click of "Apply" inside the popover commits the
# *matched* tokens into the row's ``values`` (overwriting, not merging).
# Unmatched tokens are surfaced as red-outline chips next to the matched
# ones — the user can immediately see which entries were dropped and adjust
# the textarea before clicking Apply again. This is simpler than a
# Preview→Apply two-stage and matches the standard "submit-with-feedback"
# pattern used elsewhere in the project's Dash UIs.


# Canonical empty payload for every method field, used by _blank_row and
# the per-method setters to reset a row when its method or column changes.
_EMPTY_PAYLOAD: dict[str, Any] = {
    "values": [],
    "range_min": None,
    "range_max": None,
    "compare_op": None,
    "compare_value": None,
    "text_pattern": "",
    "text_regex": False,
    "text_case_sensitive": False,
}


def _blank_row(column: str = "") -> dict[str, Any]:
    """Return a fresh, fully-defaulted row dict with a new uuid id."""
    return {
        "id": uuid.uuid4().hex,
        "column": column,
        "method": METHOD_IS_ANY_OF,
        **{k: (list(v) if isinstance(v, list) else v) for k, v in _EMPTY_PAYLOAD.items()},
    }


def _reset_payload(row: dict[str, Any]) -> None:
    """Clear every method-specific field on ``row`` in place."""
    for key, empty in _EMPTY_PAYLOAD.items():
        row[key] = list(empty) if isinstance(empty, list) else empty


def _find(rows: list[dict[str, Any]], idx: str) -> dict[str, Any] | None:
    return next((r for r in rows if r.get("id") == idx), None)


def set_row_method(
    rows: list[dict[str, Any]], idx: str, method: str
) -> list[dict[str, Any]]:
    """Set a row's method and reset its payload (cached values are stale)."""
    if method not in VALID_METHODS:
        method = METHOD_IS_ANY_OF
    row = _find(rows, idx)
    if row is not None:
        row["method"] = method
        _reset_payload(row)
    return rows


def set_row_range(
    rows: list[dict[str, Any]], idx: str, lo: Any, hi: Any
) -> list[dict[str, Any]]:
    row = _find(rows, idx)
    if row is not None:
        row["range_min"] = _coerce_float(lo)
        row["range_max"] = _coerce_float(hi)
    return rows


def set_row_compare(
    rows: list[dict[str, Any]], idx: str, op: Any, value: Any
) -> list[dict[str, Any]]:
    row = _find(rows, idx)
    if row is not None:
        row["compare_op"] = op if op in COMPARE_OPS else None
        row["compare_value"] = _coerce_float(value)
    return rows


def set_row_text(
    rows: list[dict[str, Any]], idx: str, pattern: Any, *, regex: Any, case: Any
) -> list[dict[str, Any]]:
    row = _find(rows, idx)
    if row is not None:
        row["text_pattern"] = str(pattern or "")
        row["text_regex"] = bool(regex)
        row["text_case_sensitive"] = bool(case)
    return rows


# Human-readable labels for each filter method, in dropdown display order.
_METHOD_LABELS: list[tuple[str, str]] = [
    (METHOD_IS_ANY_OF, "Is any of"),
    (METHOD_IS_NONE_OF, "Is none of"),
    (METHOD_RANGE, "Range (between)"),
    (METHOD_COMPARE, "Compare"),
    (METHOD_CONTAINS, "Contains"),
]
# Methods that only make sense on a numeric column; disabled otherwise.
_NUMERIC_ONLY_METHODS = {METHOD_RANGE, METHOD_COMPARE}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def layout(output_root: OutputRoot) -> Component:
    """Return the filter sidebar component tree.

    The sidebar is intentionally static at boot: the dynamic filter rows
    are rendered into ``FILTER_ROWS_CONTAINER_ID`` by a callback that
    listens to ``STORE_FILTER_SPEC``. This keeps re-hydration after a
    page reload (session-stored spec) and add/remove events on the
    same code path.

    Args:
        output_root: Validated handle on the CLI output directory. Used
            here only to size the eventual options list — the actual
            column / value option lists are populated reactively by
            callbacks so the layout stays cheap.

    Returns:
        A :class:`dbc.Card` (typed as
        :class:`dash.development.base_component.Component`) suitable for
        dropping into the layout's left column.
    """
    del output_root  # currently unused; kept for symmetry with future helpers

    header = html.Div(
        [
            html.H6(
                "Filter",
                className="mb-0",
                style={"color": COLOR_NAVY},
            ),
            html.Span(
                "0 images match",
                id=ids.FILTER_MATCH_COUNT_ID,
                className="text-muted small ms-2",
            ),
        ],
        className="d-flex justify-content-between align-items-center mb-2",
    )

    rows_container = html.Div(
        children=[],
        id=ids.FILTER_ROWS_CONTAINER_ID,
        className="filter-rows",
        style={
            "maxHeight": "70vh",
            "overflowY": "auto",
            "paddingRight": "0.25rem",
        },
    )

    add_button = dbc.Button(
        "+ Add filter",
        id=ids.BTN_ADD_FILTER_ROW,
        color="primary",
        outline=True,
        size="sm",
        n_clicks=0,
        className="w-100 mt-2",
    )

    return dbc.Card(
        dbc.CardBody(
            [header, html.Hr(className="my-2"), rows_container, add_button],
            className="py-2",
        ),
        className="filter-panel",
        style={"backgroundColor": COLOR_SURFACE},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_spec(stored: Any) -> list[dict[str, Any]]:
    """Coerce the store payload into a list of full, defaulted row dicts.

    Each row gains ``id``, ``column``, ``method`` (default is_any_of), and
    every method payload field. Malformed entries are dropped; legacy rows
    (``{column, values}`` with no ``method``) are backfilled.
    """
    if not isinstance(stored, list):
        return []
    rows: list[dict[str, Any]] = []
    for entry in stored:
        if not isinstance(entry, dict):
            continue
        row = _blank_row(str(entry.get("column", "") or ""))
        row["id"] = entry.get("id") or row["id"]
        method = entry.get("method") or METHOD_IS_ANY_OF
        row["method"] = method if method in VALID_METHODS else METHOD_IS_ANY_OF
        raw_values = entry.get("values") or []
        row["values"] = [str(v) for v in raw_values] if isinstance(raw_values, list) else []
        row["range_min"] = _coerce_float(entry.get("range_min"))
        row["range_max"] = _coerce_float(entry.get("range_max"))
        op = entry.get("compare_op")
        row["compare_op"] = op if op in COMPARE_OPS else None
        row["compare_value"] = _coerce_float(entry.get("compare_value"))
        row["text_pattern"] = str(entry.get("text_pattern", "") or "")
        row["text_regex"] = bool(entry.get("text_regex", False))
        row["text_case_sensitive"] = bool(entry.get("text_case_sensitive", False))
        rows.append(row)
    return rows


def _column_options(df: pl.DataFrame) -> list[dict[str, str]]:
    """Return the column-dropdown options (every master-frame column)."""
    return [{"label": col, "value": col} for col in df.columns]


def _split_paste_text(text: str) -> list[str]:
    """Split bulk-paste text into stripped, deduplicated tokens.

    Splits on newlines, commas, and tabs (any combination); strips
    whitespace per token; drops empty tokens; preserves first-seen
    order for deterministic chip rendering.
    """
    if not text:
        return []
    # Replace separator characters with newlines, then split.
    text = text.replace(",", "\n").replace("\t", "\n")
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in text.splitlines():
        token = raw.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _classify_paste_tokens(
    tokens: Iterable[str], allowed: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Partition tokens into (matched, unmatched) against ``allowed``."""
    allowed_set = set(allowed)
    matched: list[str] = []
    unmatched: list[str] = []
    for token in tokens:
        if token in allowed_set:
            matched.append(token)
        else:
            unmatched.append(token)
    return matched, unmatched


def _render_chip(label: str, *, matched: bool) -> Component:
    """Render a single chip (matched = navy outline, unmatched = vermilion)."""
    if matched:
        bg = "rgba(27,117,188,0.08)"
        border = "rgba(27,117,188,0.20)"
        color = COLOR_BLUE
    else:
        bg = "rgba(213,94,0,0.08)"
        border = "rgba(213,94,0,0.20)"
        color = OI_VERMILION_TEXT
    return html.Span(
        label,
        className="me-1 mb-1 d-inline-block",
        style={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_CAPTION,
            "padding": "0.15rem 0.5rem",
            "border": f"1px solid {border}",
            "background": bg,
            "color": color,
            "borderRadius": "var(--radius-sm)",
        },
    )


def _render_paste_chips(matched: list[str], unmatched: list[str]) -> list[Component]:
    """Render the chip list for the popover preview (capped, with overflow)."""
    chips: list[Component] = []
    visible = matched + unmatched
    if not visible:
        return [
            html.Div(
                "Paste values, then click Apply.",
                className="text-muted small",
            )
        ]
    overflow = 0
    if len(visible) > _MAX_PASTE_PREVIEW_CHIPS:
        overflow = len(visible) - _MAX_PASTE_PREVIEW_CHIPS
        # Trim, preferring to keep matched ones visible.
        head_matched = matched[:_MAX_PASTE_PREVIEW_CHIPS]
        remaining_slots = _MAX_PASTE_PREVIEW_CHIPS - len(head_matched)
        head_unmatched = unmatched[: max(remaining_slots, 0)]
    else:
        head_matched = matched
        head_unmatched = unmatched
    chips.extend(_render_chip(t, matched=True) for t in head_matched)
    chips.extend(_render_chip(t, matched=False) for t in head_unmatched)
    if overflow > 0:
        chips.append(
            html.Span(
                f"… and {overflow} more",
                className="me-1 mb-1 d-inline-block text-muted small",
                style={"fontFamily": FONT_FAMILY_MONO},
            )
        )
    return chips


def _render_filter_rows(
    rows: list[dict[str, Any]], df: pl.DataFrame, output_root: OutputRoot
) -> list[Component]:
    """Render the dynamic filter rows for the rows-container."""
    column_options = _column_options(df)
    children: list[Component] = []
    for row in rows:
        column = row["column"]
        is_numeric = bool(column) and output_root.is_numeric_column(column)
        children.append(
            _render_filter_row(row["id"], row, column_options, is_numeric=is_numeric)
        )
    return children


def _build_method_dropdown(
    idx: str, method: str, *, is_numeric: bool
) -> Component:
    """Method selector; range/compare options disabled for non-numeric cols."""
    options = [
        {
            "label": label,
            "value": value,
            "disabled": (value in _NUMERIC_ONLY_METHODS and not is_numeric),
        }
        for value, label in _METHOD_LABELS
    ]
    return dcc.Dropdown(
        id=ids.filter_row_method_id(idx),
        options=options,
        value=method or METHOD_IS_ANY_OF,
        clearable=False,
        searchable=False,
        className="mb-2",
    )


def _build_list_controls(idx: str, values: list[str]) -> list[Component]:
    """The shared multi-select + bulk-paste controls (is_any_of / is_none_of)."""
    values_dropdown = dcc.Dropdown(
        id=ids.filter_row_values_id(idx),
        options=[{"label": v, "value": v} for v in values],
        value=values,
        multi=True,
        placeholder="values",
    )
    paste_button = dbc.Button(
        "Paste",
        id=ids.filter_row_paste_btn_id(idx),
        color="secondary",
        outline=True,
        size="sm",
        n_clicks=0,
    )
    paste_popover = dbc.Popover(
        dbc.PopoverBody(
            [
                html.Div(
                    "Paste values separated by newline, comma, or tab.",
                    className="text-muted small mb-2",
                ),
                dbc.Textarea(
                    id=ids.filter_row_paste_textarea_id(idx),
                    placeholder="value1\nvalue2\nvalue3",
                    style={"width": "100%", "minHeight": "8rem"},
                ),
                dbc.Button(
                    "Apply",
                    id=ids.filter_row_paste_apply_id(idx),
                    color="primary",
                    size="sm",
                    className="mt-2",
                    n_clicks=0,
                ),
                html.Div(
                    id=ids.filter_row_paste_chips_id(idx),
                    className="mt-2",
                    children=_render_paste_chips([], []),
                ),
            ]
        ),
        id=ids.filter_row_paste_popover_id(idx),
        target=ids.filter_row_paste_btn_id(idx),
        is_open=False,
        # Open leftward so the popover stays on-screen inside the
        # right-docked filter offcanvas (placement="end").
        placement="left",
        trigger=None,
        style={"minWidth": "20rem", "maxWidth": "28rem"},
    )
    return [
        html.Div(values_dropdown, className="mb-2"),
        html.Div(paste_button, className="d-flex gap-1"),
        paste_popover,
    ]


def _build_range_controls(idx: str, row: dict[str, Any]) -> list[Component]:
    return [
        html.Div(
            [
                dcc.Input(
                    id=ids.filter_row_range_min_id(idx),
                    type="number",
                    value=row["range_min"],
                    placeholder="min",
                    className="form-control form-control-sm",
                    style={"width": "45%"},
                ),
                html.Span("–", className="mx-1"),
                dcc.Input(
                    id=ids.filter_row_range_max_id(idx),
                    type="number",
                    value=row["range_max"],
                    placeholder="max",
                    className="form-control form-control-sm",
                    style={"width": "45%"},
                ),
            ],
            className="d-flex align-items-center mb-2",
        )
    ]


def _build_compare_controls(idx: str, row: dict[str, Any]) -> list[Component]:
    return [
        html.Div(
            [
                dcc.Dropdown(
                    id=ids.filter_row_compare_op_id(idx),
                    options=[{"label": op, "value": op} for op in (">", ">=", "<", "<=")],
                    value=row["compare_op"],
                    clearable=False,
                    searchable=False,
                    placeholder="op",
                    style={"width": "40%"},
                ),
                dcc.Input(
                    id=ids.filter_row_compare_value_id(idx),
                    type="number",
                    value=row["compare_value"],
                    placeholder="value",
                    className="form-control form-control-sm ms-1",
                    style={"width": "55%"},
                ),
            ],
            className="d-flex align-items-center mb-2",
        )
    ]


def _build_contains_controls(idx: str, row: dict[str, Any]) -> list[Component]:
    return [
        dbc.Input(
            id=ids.filter_row_text_pattern_id(idx),
            type="text",
            value=row["text_pattern"],
            placeholder="contains…",
            size="sm",
            className="mb-2",
        ),
        html.Div(
            [
                dbc.Checkbox(
                    id=ids.filter_row_text_regex_id(idx),
                    label="regex",
                    value=row["text_regex"],
                    className="me-3",
                ),
                dbc.Checkbox(
                    id=ids.filter_row_text_case_id(idx),
                    label="case-sensitive",
                    value=row["text_case_sensitive"],
                ),
            ],
            className="d-flex small mb-2",
        ),
    ]


def _render_filter_row(
    idx: str,
    row: dict[str, Any],
    column_options: list[dict[str, str]],
    *,
    is_numeric: bool,
) -> Component:
    """Build a single filter-row component tree for the row's active method."""
    column = row["column"]
    method = row["method"]

    column_dropdown = dcc.Dropdown(
        id=ids.filter_row_column_id(idx),
        options=column_options,
        value=column or None,
        searchable=True,
        clearable=False,
        placeholder="column",
        className="mb-2",
    )
    method_dropdown = _build_method_dropdown(idx, method, is_numeric=is_numeric)

    if method == METHOD_RANGE:
        method_controls = _build_range_controls(idx, row)
    elif method == METHOD_COMPARE:
        method_controls = _build_compare_controls(idx, row)
    elif method == METHOD_CONTAINS:
        method_controls = _build_contains_controls(idx, row)
    else:  # is_any_of / is_none_of
        method_controls = _build_list_controls(idx, row["values"])

    remove_button = dbc.Button(
        "✕",
        id=ids.filter_row_remove_id(idx),
        color="danger",
        outline=True,
        size="sm",
        n_clicks=0,
        title="Remove this filter",
    )

    return html.Div(
        [
            html.Div(column_dropdown),
            method_dropdown,
            *method_controls,
            html.Div(remove_button, className="d-flex justify-content-end"),
        ],
        id=ids.filter_row_id(idx),
        className="filter-row mb-2",
        style={
            "borderLeft": f"2px solid {COLOR_BLUE}",
            "paddingLeft": "0.5rem",
            "paddingTop": "0.25rem",
            "paddingBottom": "0.25rem",
        },
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_callbacks(
    app: dash.Dash,
    output_root: OutputRoot,
    filtered_state: CurationLabels,
) -> None:
    """Register every callback owned by the filter sidebar.

    Callbacks registered here:

    1. Add a fresh row when the static "+ Add filter" button is clicked.
    2. Render the dynamic rows into the container whenever the spec
       store changes.
    3. Sync per-row column-dropdown values back into the spec.
    4. Sync per-row multi-value dropdown values back into the spec.
    5. Populate the values dropdown options from
       :attr:`OutputRoot.column_value_sets` reactively.
    6. Remove a row when its remove button is clicked.
    7. Toggle the bulk-paste popover ``is_open`` on its trigger button.
    8. Validate the paste textarea against the row's column value-set
       and commit matches into the row's values on Apply.
    9. Derive ``STORE_IMAGE_PAIRS`` (and the count chip) from the
       current spec, with the chip's ``(− K removed)`` suffix sourced
       from the curation backend.

    Args:
        app: The Dash application to attach the callbacks to.
        output_root: Validated handle on the CLI output directory; used
            both as a polars-frame source for filter application and to
            populate value-set options.
    """

    df = output_root.master_df
    column_value_sets = output_root.column_value_sets

    # --- 1. Add filter row ------------------------------------------------

    @app.callback(
        Output(ids.STORE_FILTER_SPEC, "data", allow_duplicate=True),
        Input(ids.BTN_ADD_FILTER_ROW, "n_clicks"),
        State(ids.STORE_FILTER_SPEC, "data"),
        prevent_initial_call=True,
    )
    def _add_filter_row(n_clicks: int | None, stored: Any) -> list[dict[str, Any]]:
        """Append an empty row with a fresh uuid index.

        Triggered by ``BTN_ADD_FILTER_ROW``. The renderer callback (#2)
        picks up the new row from the store and emits the matching DOM.
        """
        del n_clicks
        rows = _normalise_spec(stored)
        rows.append({"id": uuid.uuid4().hex, "column": "", "values": []})
        return rows

    # --- 2. Render filter rows -------------------------------------------

    @app.callback(
        Output(ids.FILTER_ROWS_CONTAINER_ID, "children"),
        Input(ids.STORE_FILTER_SPEC, "data"),
    )
    def _render_rows(stored: Any) -> list[Component]:
        """Render one component tree per row in the spec store."""
        rows = _normalise_spec(stored)
        return _render_filter_rows(rows, df, output_root)

    # --- 3. Column dropdown → spec ---------------------------------------

    @app.callback(
        Output(ids.STORE_FILTER_SPEC, "data", allow_duplicate=True),
        Input({"type": "filter-row-column", "index": ALL}, "value"),
        State({"type": "filter-row-column", "index": ALL}, "id"),
        State(ids.STORE_FILTER_SPEC, "data"),
        prevent_initial_call=True,
    )
    def _update_columns(
        values: list[Any],
        component_ids: list[dict[str, str]],
        stored: Any,
    ) -> Any:
        """Write column changes back to the spec store.

        Clears the row's ``values`` whenever its ``column`` changes,
        because cached values are no longer valid for the new column.
        """
        rows = _normalise_spec(stored)
        # Map idx -> new column for fast lookup.
        new_by_id: dict[str, str] = {
            comp_id["index"]: str(value or "")
            for comp_id, value in zip(component_ids, values, strict=False)
        }
        changed = False
        for row in rows:
            new_column = new_by_id.get(row["id"], row["column"])
            if new_column != row["column"]:
                row["column"] = new_column
                row["values"] = []
                changed = True
        if not changed:
            return no_update
        return rows

    # --- 4. Values dropdown → spec ---------------------------------------

    @app.callback(
        Output(ids.STORE_FILTER_SPEC, "data", allow_duplicate=True),
        Input({"type": "filter-row-values", "index": ALL}, "value"),
        State({"type": "filter-row-values", "index": ALL}, "id"),
        State(ids.STORE_FILTER_SPEC, "data"),
        prevent_initial_call=True,
    )
    def _update_values(
        values: list[Any],
        component_ids: list[dict[str, str]],
        stored: Any,
    ) -> Any:
        """Write multi-select changes back to the spec store."""
        rows = _normalise_spec(stored)
        new_by_id: dict[str, list[str]] = {}
        for comp_id, value in zip(component_ids, values, strict=False):
            if value is None:
                new_by_id[comp_id["index"]] = []
            elif isinstance(value, list):
                new_by_id[comp_id["index"]] = [str(v) for v in value]
            else:
                new_by_id[comp_id["index"]] = [str(value)]
        changed = False
        for row in rows:
            new_values = new_by_id.get(row["id"], row["values"])
            if new_values != row["values"]:
                row["values"] = new_values
                changed = True
        if not changed:
            return no_update
        return rows

    # --- 5. Populate values options reactively ---------------------------

    @app.callback(
        Output({"type": "filter-row-values", "index": ALL}, "options"),
        Input({"type": "filter-row-column", "index": ALL}, "value"),
    )
    def _populate_value_options(columns: list[Any]) -> list[list[dict[str, str]]]:
        """Map each row's selected column to its value-set options."""
        out: list[list[dict[str, str]]] = []
        for col in columns:
            if not col:
                out.append([])
                continue
            allowed = column_value_sets.get(str(col), [])
            out.append([{"label": v, "value": v} for v in allowed])
        return out

    # --- 6. Remove filter row --------------------------------------------

    @app.callback(
        Output(ids.STORE_FILTER_SPEC, "data", allow_duplicate=True),
        Input({"type": "filter-row-remove", "index": ALL}, "n_clicks"),
        State(ids.STORE_FILTER_SPEC, "data"),
        prevent_initial_call=True,
    )
    def _remove_filter_row(n_clicks_list: list[Any], stored: Any) -> Any:
        """Pop the row whose remove-button just fired.

        Uses ``ctx.triggered_id`` to pick the right index; ignores the
        spurious initial fire when a new row mounts (n_clicks=None).
        """
        triggered = dash.callback_context.triggered_id
        if not triggered or not isinstance(triggered, dict):
            return no_update
        # Confirm an actual click — avoids removing on initial mount.
        if not any(n for n in n_clicks_list if n):
            return no_update
        target_idx = triggered.get("index")
        rows = _normalise_spec(stored)
        new_rows = [r for r in rows if r["id"] != target_idx]
        if len(new_rows) == len(rows):
            return no_update
        return new_rows

    # --- 7. Toggle paste popover -----------------------------------------

    @app.callback(
        Output({"type": "filter-row-paste-popover", "index": ALL}, "is_open"),
        Input({"type": "filter-row-paste-btn", "index": ALL}, "n_clicks"),
        State({"type": "filter-row-paste-popover", "index": ALL}, "is_open"),
        State({"type": "filter-row-paste-btn", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def _toggle_paste_popover(
        click_counts: list[Any],
        is_open_states: list[bool],
        component_ids: list[dict[str, str]],
    ) -> list[bool]:
        """Toggle the popover for the button that just fired."""
        triggered = dash.callback_context.triggered_id
        if not triggered or not isinstance(triggered, dict):
            return [bool(s) for s in is_open_states]
        target_idx = triggered.get("index")
        out: list[bool] = []
        for comp_id, was_open in zip(component_ids, is_open_states, strict=False):
            if comp_id["index"] == target_idx:
                out.append(not bool(was_open))
            else:
                out.append(bool(was_open))
        return out

    # --- 8. Apply bulk paste ---------------------------------------------

    @app.callback(
        Output(ids.STORE_FILTER_SPEC, "data", allow_duplicate=True),
        Output({"type": "filter-row-paste-chips", "index": ALL}, "children"),
        Input({"type": "filter-row-paste-apply", "index": ALL}, "n_clicks"),
        State({"type": "filter-row-paste-textarea", "index": ALL}, "value"),
        State({"type": "filter-row-paste-apply", "index": ALL}, "id"),
        State(ids.STORE_FILTER_SPEC, "data"),
        prevent_initial_call=True,
    )
    def _apply_paste(
        click_counts: list[Any],
        textarea_values: list[Any],
        component_ids: list[dict[str, str]],
        stored: Any,
    ) -> tuple[Any, list[Any]]:
        """Validate paste tokens and commit matches into the row.

        Single-stage commit (see module docstring): each Apply click
        replaces the row's ``values`` with the matched-and-allowed
        tokens, and surfaces unmatched tokens as red chips.
        """
        triggered = dash.callback_context.triggered_id
        n_chip_outputs = len(component_ids)
        # Default chip outputs: pass-through (no_update keeps existing
        # children for every row except the one we touched).
        chip_outputs: list[Any] = [no_update] * n_chip_outputs

        if not triggered or not isinstance(triggered, dict):
            return no_update, chip_outputs
        if not any(n for n in click_counts if n):
            return no_update, chip_outputs

        target_idx = triggered.get("index")
        # Look up textarea value for the firing row.
        textarea_text = ""
        target_pos = -1
        for pos, comp_id in enumerate(component_ids):
            if comp_id["index"] == target_idx:
                target_pos = pos
                if pos < len(textarea_values):
                    textarea_text = str(textarea_values[pos] or "")
                break
        if target_pos < 0:
            return no_update, chip_outputs

        rows = _normalise_spec(stored)
        target_row = next((r for r in rows if r["id"] == target_idx), None)
        if target_row is None:
            return no_update, chip_outputs

        column = target_row["column"]
        if not column:
            chip_outputs[target_pos] = [
                html.Div(
                    "Select a column first.",
                    className="text-muted small",
                )
            ]
            return no_update, chip_outputs

        allowed = column_value_sets.get(column, [])
        tokens = _split_paste_text(textarea_text)
        matched, unmatched = _classify_paste_tokens(tokens, allowed)
        chip_outputs[target_pos] = _render_paste_chips(matched, unmatched)

        # Overwrite (not merge) the row's values with matched tokens.
        target_row["values"] = matched
        return rows, chip_outputs

    # --- 9. Filter apply → image pairs + count chip ----------------------

    @app.callback(
        Output(ids.STORE_IMAGE_PAIRS, "data"),
        Output(ids.FILTER_MATCH_COUNT_ID, "children"),
        Input(ids.STORE_FILTER_SPEC, "data"),
        Input(ids.STORE_REMOVED_KEYS, "data"),
        State(ids.STORE_IMAGE_PAIRS, "data"),
    )
    def _derive_image_pairs(
        stored: Any, removed_keys: Any, current_pairs: Any
    ) -> tuple[Any, Any]:
        """Re-derive the filtered (dataset, stem) pairs and update the chip.

        Returns ``no_update`` for the pairs output when the filter result
        is unchanged from the existing store payload — otherwise every
        viewer card's picker options refresh and the OSD clientside
        callback re-runs even when nothing actually moved. The chip text
        always re-emits because the removed-count suffix can change even
        when the surviving image pairs do not.
        """
        spec = FilterSpec.from_store(_normalise_spec(stored))
        try:
            filtered = spec.apply_to(df)
        except Exception:
            logger.exception("FilterSpec.apply_to failed; passing through unfiltered.")
            filtered = df
        pairs = output_root.image_pairs(filtered)
        payload = [{"dataset": dataset, "stem": stem} for dataset, stem in pairs]

        # ``removed_keys`` is plumbed through as an Input only to retrigger
        # this callback on curation changes; the actual count comes from
        # the curation backend so it stays in sync with the lock-guarded
        # source of truth.
        del removed_keys
        removed_in_filtered = filtered_state.removed_count_in(filtered)

        chip_text = f"{len(pairs)} images match"
        if removed_in_filtered > 0:
            chip_text += f" (− {removed_in_filtered} removed)"

        if isinstance(current_pairs, list) and current_pairs == payload:
            return no_update, chip_text
        return payload, chip_text


__all__ = ["layout", "register_callbacks"]
