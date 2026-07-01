"""Results-viewer Timeline tab body + the pure grid-build helper it shares.

The Timeline tab is the 6th results-viewer tab. It renders the SAME Phase 1
shared engine (``build_matrix`` → ``build_timeline_grid``) and the SAME Phase 2
focus-and-navigate ``timeline.js`` controller as Browse, but over **overlay**
PNGs with axes drawn from ``OutputRoot.master_df`` (the post-applied mirror).

The controller (``timeline.js``) is vendored byte-for-byte into the viewer's
assets folder and is fully surface-agnostic: it finds every sibling control
**by class scoped to** ``.timeline-body`` (``.timeline-viewport``,
``.timeline-grid-container``, ``.timeline-nav-{up,down,left,right}``,
``.timeline-position``, ``.timeline-popout-bridge``) and re-attaches via the
grid's own id. This layout therefore carries each ``timeline-*`` class
**alongside** its Dash id (the same dual id+class pattern Browse uses) so the
byte-identical controller "just works" once a clientside callback calls
``window.__phenotypicTimeline.attach("timeline-grid")``.
"""
from __future__ import annotations

from typing import Any, cast

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import polars as pl
from dash import dcc, html
from dash.development.base_component import Component

from phenotypic.gui._config import (
    TIMELINE_FOCUS_MARGIN,
    TIMELINE_MOUNT_CAP,
    TIMELINE_TILE_SIZE_DEFAULT,
    TIMELINE_WARM_CONCURRENCY,
    VIEWER_THUMB_URL_SEGMENT,
    snap_thumb_bucket,
)
from phenotypic.gui._design import (
    COLOR_BG,
    COLOR_BLUE,
    COLOR_NAVY,
    COLOR_SURFACE,
    FONT_FAMILY_MONO,
    FONT_SIZE_LABEL,
)
from phenotypic.gui._shared.timeline import build_matrix, build_timeline_grid
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.timeline_view import _ids as ids
from phenotypic.gui.results_viewer.timeline_view._grid import (
    build_timeline_records,
    has_eligible_time_axis,
)
from phenotypic.gui.results_viewer.timeline_view._thumb_routes import encode_cell_ref

__all__ = ["layout", "build_timeline_grid_component"]

_NAVY = COLOR_NAVY
_BLUE = COLOR_BLUE
_BG = COLOR_BG

#: Empty-state guidance shown when no eligible time column exists (D9 / §16.6).
_EMPTY_STATE_TEXT = (
    "The Timeline needs a time field. Re-run with `--metadata <csv>` (or add a "
    "post step like ExpandMetadata) so a column such as MetadataCulture_Time "
    "or a monotonic image-number column is available. Pick a monotonic column "
    "(e.g. image number) — a time-of-day column mis-orders across days."
)


def build_timeline_grid_component(
    output_root: OutputRoot,
    df: pl.DataFrame,
    *,
    row_col: str | None,
    time_col: str | None,
    tile_size: int,
    url_prefix: str = "/",
) -> tuple[Component, bool, int]:
    """Build the Timeline grid (or the empty state) for a filtered slice.

    Pure (no Dash callback machinery), so the render path is unit-testable.

    Args:
        output_root: The viewer's output handle (overlay membership + paths).
        df: The active (filtered) master mirror slice.
        row_col: Y-axis column name (``None`` while the dropdown is unset).
        time_col: X (time)-axis column name (``None`` while unset / ineligible).
        tile_size: Rendered placeholder size in px (the stepper value).
        url_prefix: Mount-point prefix prepended to each thumbnail ``data-src``
            (e.g. ``"/results/"`` under the hub; ``"/"`` standalone).

    Returns:
        ``(grid_children, show_empty_state, n_time_values)`` — the grid
        component (or an empty children list when no eligible/selected time
        axis exists, so the separately-mounted ``TIMELINE_EMPTY_STATE`` alert
        is the only empty-state node — no duplicate id), whether the empty
        state is shown, and the distinct-time-value count (drives the
        large-axis warning; ``0`` for the empty state).
    """
    if (
        time_col is None
        or row_col is None
        or not has_eligible_time_axis(df, output_root.column_value_sets)
    ):
        # The grid container empties; the static TIMELINE_EMPTY_STATE alert
        # (mounted once in layout()) is toggled visible by the render callback.
        return html.Div(), True, 0

    records = build_timeline_records(
        output_root, df, row_col=row_col, time_col=time_col
    )
    matrix = build_matrix(records)
    fetch_size = snap_thumb_bucket(tile_size)

    def _url_builder(cell_ref: object, fetch: int) -> str:
        identity = encode_cell_ref(*cast(tuple[str, str], cell_ref))
        return f"{url_prefix}{VIEWER_THUMB_URL_SEGMENT}/{identity}?size={fetch}"

    def _ref_builder(cell_ref: object) -> str:
        return encode_cell_ref(*cast(tuple[str, str], cell_ref))

    grid, _grid_order = build_timeline_grid(
        matrix,
        url_builder=_url_builder,
        display_size=tile_size,
        fetch_size=fetch_size,
        ref_builder=_ref_builder,
    )
    return grid, False, len(matrix.columns)


def build_empty_state() -> Component:
    """Guided empty state shown when no eligible time column exists (D9)."""
    return dbc.Alert(
        _EMPTY_STATE_TEXT,
        id=ids.TIMELINE_EMPTY_STATE,
        color="secondary",
        className="timeline-empty-state",
        style={"display": "none"},  # toggled visible by the render callback
    )


# ---------------------------------------------------------------------------
# Sub-builders
# ---------------------------------------------------------------------------


def _build_toolbar() -> Component:
    """Build the toolbar: Y (row) dropdown, X (time) dropdown, tile stepper."""
    y_label = html.Span(
        "Group (Y)",
        className="me-1",
        style={
            "color": _NAVY,
            "fontWeight": 500,
            "fontSize": FONT_SIZE_LABEL,
            "whiteSpace": "nowrap",
        },
    )
    y_dropdown = dcc.Dropdown(
        id=ids.TIMELINE_Y_DROPDOWN,
        options=[],
        value=None,
        placeholder="Group by…",
        clearable=False,
        style={"minWidth": "200px"},
    )
    x_label = html.Span(
        "Time (X)",
        className="me-1",
        style={
            "color": _NAVY,
            "fontWeight": 500,
            "fontSize": FONT_SIZE_LABEL,
            "whiteSpace": "nowrap",
        },
    )
    x_dropdown = dcc.Dropdown(
        id=ids.TIMELINE_X_DROPDOWN,
        options=[],
        value=None,
        placeholder="Time axis…",
        clearable=False,
        style={"minWidth": "200px"},
    )

    tile_stepper = html.Div(
        [
            html.Span(
                "Tile size",
                className="me-1",
                style={
                    "color": _NAVY,
                    "fontWeight": 500,
                    "fontSize": FONT_SIZE_LABEL,
                    "whiteSpace": "nowrap",
                },
            ),
            html.Button(
                "−",
                id=ids.TIMELINE_TILE_SIZE_MINUS,
                n_clicks=0,
                title="Decrease tile size",
                className="btn btn-outline-secondary btn-sm",
                type="button",
                **cast(Any, {"aria-label": "Decrease tile size"}),
            ),
            html.Span(
                f"{TIMELINE_TILE_SIZE_DEFAULT} px",
                id=ids.TIMELINE_TILE_SIZE_READOUT,
                style={
                    "fontFamily": FONT_FAMILY_MONO,
                    "fontSize": FONT_SIZE_LABEL,
                    "minWidth": "3.75rem",
                    "textAlign": "center",
                    "whiteSpace": "nowrap",
                },
            ),
            html.Button(
                "+",
                id=ids.TIMELINE_TILE_SIZE_PLUS,
                n_clicks=0,
                title="Increase tile size",
                className="btn btn-outline-secondary btn-sm",
                type="button",
                **cast(Any, {"aria-label": "Increase tile size"}),
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "0.5rem",
            "flex": "0 0 auto",
        },
    )

    return html.Div(
        [
            html.Div(
                [y_label, y_dropdown],
                style={"display": "flex", "alignItems": "center", "gap": "0.25rem"},
            ),
            html.Div(
                [x_label, x_dropdown],
                style={"display": "flex", "alignItems": "center", "gap": "0.25rem"},
            ),
            tile_stepper,
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "flexWrap": "wrap",
            "rowGap": "0.5rem",
            "columnGap": "1rem",
            "padding": "0.75rem 1rem",
            "borderBottom": f"1px solid {_BLUE}22",
            "background": COLOR_SURFACE,
        },
    )


def _build_viewport() -> Component:
    """Build the no-scroll focus-window viewport (grid + edge buttons + readout).

    Mirrors Browse's ``build_timeline_body`` structure EXACTLY for the
    surface-agnostic ``timeline-*`` classes + the static ``data-*`` attrs that
    the byte-identical vendored controller reads.
    """
    # The focus-navigate constants ride as STATIC data-* on the grid container
    # (they never change at runtime; the render callback only swaps CHILDREN).
    grid = html.Div(
        id=ids.TIMELINE_GRID,
        className="timeline-grid-container",
        **cast(
            Any,
            {
                "data-mount-cap": str(TIMELINE_MOUNT_CAP),
                "data-focus-margin": str(TIMELINE_FOCUS_MARGIN),
                "data-warm-concurrency": str(TIMELINE_WARM_CONCURRENCY),
            },
        ),
    )
    nav_up = html.Button(
        "▲",
        id=ids.TIMELINE_NAV_UP,
        type="button",
        n_clicks=0,
        className="timeline-nav-up",
        **cast(Any, {"aria-label": "Move focus up one row"}),
    )
    nav_down = html.Button(
        "▼",
        id=ids.TIMELINE_NAV_DOWN,
        type="button",
        n_clicks=0,
        className="timeline-nav-down",
        **cast(Any, {"aria-label": "Move focus down one row"}),
    )
    nav_left = html.Button(
        "◀",
        id=ids.TIMELINE_NAV_LEFT,
        type="button",
        n_clicks=0,
        className="timeline-nav-left",
        **cast(Any, {"aria-label": "Move focus back one time step"}),
    )
    nav_right = html.Button(
        "▶",
        id=ids.TIMELINE_NAV_RIGHT,
        type="button",
        n_clicks=0,
        className="timeline-nav-right",
        **cast(Any, {"aria-label": "Move focus forward one time step"}),
    )
    position = html.Div(id=ids.TIMELINE_POSITION, className="timeline-position")
    return html.Div(
        [grid, nav_up, nav_down, nav_left, nav_right, position],
        className="timeline-viewport",
        tabIndex=0,
        style={
            "overflow": "hidden",
            "position": "relative",
            "height": "75vh",
            "border": "1px solid var(--color-border)",
        },
    )


def _build_popout() -> Component:
    """Build the deep-zoom pop-out modal (reuses the /tiles DZI route + OSD)."""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("")),
            dbc.ModalBody(
                html.Div(id=ids.TIMELINE_POPOUT_OSD, style={"height": "70vh"})
            ),
        ],
        id=ids.TIMELINE_POPOUT_MODAL,
        is_open=False,
        size="xl",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def layout(output_root: OutputRoot) -> Component:
    """Build the static Timeline tab body.

    Vertical stack: toolbar → large-axis warning (hidden) → empty state
    (hidden) → no-scroll focus-window viewport → pop-out modal + stores +
    the hidden JS→Dash pop-out bridge.

    Args:
        output_root: Validated handle on the CLI output directory. Currently
            unused by the static layout (the render callback consumes it), but
            kept in the signature for API symmetry with sibling tab factories.

    Returns:
        A :class:`dash.html.Div` carrying every controller-required
        ``timeline-*`` class, ready to drop into a :class:`dbc.Tab`.
    """
    del output_root  # the render callback consumes it; the static layout is fixed

    toolbar = _build_toolbar()
    warning = dbc.Alert(
        "",
        id=ids.TIMELINE_LARGE_AXIS_WARNING,
        color="warning",
        is_open=False,
        dismissable=True,
        className="py-1 px-2 timeline-large-axis-warning",
    )
    empty_state = build_empty_state()
    viewport = _build_viewport()
    popout = _build_popout()

    return html.Div(
        [
            toolbar,
            warning,
            empty_state,
            viewport,
            popout,
            # Hidden JS→Dash bridge: timeline.js sets .value to the
            # clicked/focused cell's data-ref token (+ a `#<nonce>` suffix).
            # SURFACE-AGNOSTIC class `.timeline-popout-bridge`: the controller
            # finds this input by class (scoped to the timeline body); the id
            # stays for the Dash server callback's Input target.
            dcc.Input(
                id=ids.TIMELINE_POPOUT_INPUT,
                value="",
                type="text",
                className="timeline-popout-bridge",
                style={"display": "none"},
            ),
            dcc.Store(
                id=ids.TIMELINE_STORE_TILE_SIZE, data=TIMELINE_TILE_SIZE_DEFAULT
            ),
            dcc.Store(id=ids.TIMELINE_POPOUT_STORE, data=None),
            dcc.Store(id=ids.TIMELINE_POPOUT_OSD_SYNC, data=None),
        ],
        id=ids.TIMELINE_BODY,
        # SURFACE-AGNOSTIC class `.timeline-body`: the controller scopes its
        # sibling-control queries (nav buttons, readout, bridge input) to the
        # enclosing element carrying this class.
        className="timeline-body",
        style={
            "padding": "1rem",
            "maxHeight": "calc(100vh - 8rem)",
            "overflow": "auto",
            "background": _BG,
        },
    )
