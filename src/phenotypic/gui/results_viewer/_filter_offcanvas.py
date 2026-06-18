"""Right-docked filter offcanvas: toggle + active-filter count badge.

The filter panel itself (rows, match-count chip, bulk-paste) lives in
:mod:`._filter_panel` and is mounted *inside* a top-level
``dbc.Offcanvas`` by :mod:`._layout`. This module owns only the two
behaviors that surround that panel:

1. **Toggle** — the top-bar "Filters" button flips the offcanvas
   ``is_open``. dbc's own backdrop / ✕ close the offcanvas internally, so
   the toggle reads the current ``is_open`` as State and inverts it.
2. **Count badge** — a small badge on the toggle button shows how many
   *configured* filter rows are active (rows with a column chosen), so the
   user sees filtering state without opening the panel. The panel keeps its
   own "N images match" chip (result size); this badge is the applied-count.

The logic is split into pure, importable helpers so it is unit-testable
without booting Dash (mirrors the smart-QC ``worklist_row_metric_update``
pattern).
"""

from __future__ import annotations

from typing import Any

import dash
from dash import Input, Output, State

from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._filter_state import FilterRow


def next_offcanvas_state(n_clicks: int | None, is_open: bool | None) -> bool:
    """Return the offcanvas ``is_open`` after a toggle-button click.

    A real click (truthy ``n_clicks``) inverts the current state; a falsy
    ``n_clicks`` (initial mount / no click) leaves it unchanged.
    """
    if not n_clicks:
        return bool(is_open)
    return not bool(is_open)


def row_is_active(row: Any) -> bool:
    """Return True if a spec row contributes a real constraint.

    Mirrors the "unset = skip" rule in
    :meth:`phenotypic.gui.results_viewer._filter_state.FilterRow.to_expr`:
    a row needs a column AND a usable payload for its method. The
    per-method payload check is delegated to ``to_expr`` (the single
    source of truth) so the two never drift; this helper adds only the
    dict guard and the blank/whitespace-column reject that ``to_expr``'s
    callers don't need.
    """
    if not isinstance(row, dict):
        return False
    if not str(row.get("column", "") or "").strip():
        return False
    return FilterRow.from_dict(row).to_expr() is not None


def active_filter_count(spec: Any) -> int:
    """Count rows that contribute a real constraint (see :func:`row_is_active`)."""
    if not isinstance(spec, list):
        return 0
    return sum(1 for row in spec if row_is_active(row))


def badge_children(count: int) -> str:
    """Badge text: blank at 0 (so an empty badge can be hidden), else the count."""
    return "" if count <= 0 else str(count)


def badge_style(count: int) -> dict[str, str]:
    """Badge style: hidden at 0, inline otherwise (no stray empty pill)."""
    return {"display": "none"} if count <= 0 else {"display": "inline-block"}


def register_filter_offcanvas_callbacks(app: dash.Dash) -> None:
    """Wire the Filters toggle and the active-filter count badge."""

    @app.callback(
        Output(ids.OFFCANVAS_FILTER_ID, "is_open"),
        Input(ids.BTN_FILTERS_TOGGLE, "n_clicks"),
        State(ids.OFFCANVAS_FILTER_ID, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_filter_offcanvas(n_clicks: int | None, is_open: bool | None) -> bool:
        """Flip the offcanvas open/closed on a toggle-button click."""
        return next_offcanvas_state(n_clicks, is_open)

    @app.callback(
        Output(ids.FILTER_TOGGLE_BADGE_ID, "children"),
        Output(ids.FILTER_TOGGLE_BADGE_ID, "style"),
        Input(ids.STORE_FILTER_SPEC, "data"),
    )
    def _update_filter_badge(spec: Any) -> tuple[str, dict[str, str]]:
        """Reflect the active-filter count on the toggle button badge."""
        count = active_filter_count(spec)
        return badge_children(count), badge_style(count)


__all__ = [
    "next_offcanvas_state",
    "active_filter_count",
    "row_is_active",
    "badge_children",
    "badge_style",
    "register_filter_offcanvas_callbacks",
]
