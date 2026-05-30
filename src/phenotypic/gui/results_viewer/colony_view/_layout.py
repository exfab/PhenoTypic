"""Colony-view tab body for the results viewer.

The colony view is the second tab in the right-hand column (the first is
the existing per-image ``Plate`` cards view). It exposes a 2D grid of
per-colony crop thumbnails so the user can visually triage many colonies
at once and remove or restore them in bulk via a multi-select.

This module owns only the *static* skeleton of the tab body — a toolbar
(axis dropdowns, refresh button, crop-size info chip), a hidden-by-default
bulk-action alert bar, and an empty grid container. The grid itself is
rendered by callbacks in :mod:`._callbacks`; the per-cell
HTML is built by :mod:`._grid`.

DESIGN.md tokens are honoured throughout: navy ``#003660`` for primary
text, blue ``#1b75bc`` for accents, gold ``#febc11`` for highlights, and
``#f5f7fa`` for the page background.
"""

from __future__ import annotations

import logging

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html
from dash.development.base_component import Component

from phenotypic.gui._config import TILE_DIM_DEFAULT
from phenotypic.gui._design import (
    COLOR_BG,
    COLOR_BLUE,
    COLOR_NAVY,
    COLOR_SURFACE,
    FONT_FAMILY_MONO,
    FONT_SIZE_LABEL,
)
from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style tokens (mirrors DESIGN.md / FRONTEND_STYLE_GUIDE.md)
# ---------------------------------------------------------------------------

_NAVY = COLOR_NAVY
_BLUE = COLOR_BLUE
_BG = COLOR_BG


# ---------------------------------------------------------------------------
# Sub-builders
# ---------------------------------------------------------------------------


def _build_dim_stepper(
    *,
    minus_id: str,
    plus_id: str,
    readout_id: str,
) -> Component:
    """Build the ``[ − ]  dim 0.60  [ + ]`` tile-spotlight stepper.

    A compact control wiring the shared
    :data:`phenotypic.gui.results_viewer._ids.STORE_TILE_DIM_ALPHA`
    strength: the ``−``/``+`` buttons step it one click (a callback
    rebuilds the tile grid), and the readout span shows the current
    strength (synced from the store by the shared readout callback). The
    readout text is seeded from :data:`TILE_DIM_DEFAULT` so it reads
    correctly before the first store echo.

    Args:
        minus_id: Component id for the ``−`` (step-down) button.
        plus_id: Component id for the ``+`` (step-up) button.
        readout_id: Component id for the ``dim 0.60`` readout span.

    Returns:
        A flex ``html.Div`` matching the toolbar's widget styling.
    """
    minus_btn = dbc.Button(
        "−",
        id=minus_id,
        n_clicks=0,
        color="secondary",
        outline=True,
        size="sm",
        title="Soften the colony-spotlight dimming",
        style={"padding": "0 0.5rem", "lineHeight": "1.2"},
    )
    readout = html.Span(
        f"dim {TILE_DIM_DEFAULT:.2f}",
        id=readout_id,
        style={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_LABEL,
            "color": _NAVY,
            "minWidth": "4.5rem",
            "textAlign": "center",
            "whiteSpace": "nowrap",
        },
    )
    plus_btn = dbc.Button(
        "+",
        id=plus_id,
        n_clicks=0,
        color="secondary",
        outline=True,
        size="sm",
        title="Strengthen the colony-spotlight dimming",
        style={"padding": "0 0.5rem", "lineHeight": "1.2"},
    )
    return html.Div(
        [minus_btn, readout, plus_btn],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "0.35rem",
            "flex": "0 0 auto",
        },
    )


def _build_toolbar() -> Component:
    """Build the colony-view toolbar (axis dropdowns + refresh + info chip).

    The toolbar is a horizontal flex row that hosts the X / Y axis
    dropdowns the grid sorts by, a read-only crop-size info chip
    populated by a callback (e.g. ``"crop size: 320 px"``), and a manual
    refresh button. The dropdown ``options`` and ``value`` lists are
    intentionally empty here — they are populated reactively from the
    master measurements once the user opens the tab.

    Returns:
        An :class:`dash.html.Div` styled as a navy-tinted top bar.
    """
    x_label = html.Span(
        "X axis",
        className="me-1",
        style={
            "color": _NAVY,
            "fontWeight": 500,
            "fontSize": FONT_SIZE_LABEL,
            "whiteSpace": "nowrap",
        },
    )
    x_dropdown = dcc.Dropdown(
        id=ids.COLONY_X_AXIS_DROPDOWN_ID,
        options=[],
        value=None,
        placeholder="X axis…",
        clearable=False,
        style={"minWidth": "200px"},
    )
    y_label = html.Span(
        "Y axis",
        className="me-1",
        style={
            "color": _NAVY,
            "fontWeight": 500,
            "fontSize": FONT_SIZE_LABEL,
            "whiteSpace": "nowrap",
        },
    )
    y_dropdown = dcc.Dropdown(
        id=ids.COLONY_Y_AXIS_DROPDOWN_ID,
        options=[],
        value=None,
        placeholder="Y axis…",
        clearable=False,
        style={"minWidth": "200px"},
    )

    crop_size_info = html.Span(
        id=ids.COLONY_CROP_SIZE_INFO_ID,
        children="",
        style={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_LABEL,
            "color": _NAVY,
        },
    )

    refresh_button = dbc.Button(
        "⟳ Refresh",
        id=ids.COLONY_BTN_REFRESH_ID,
        n_clicks=0,
        color="secondary",
        size="sm",
        # `marginLeft: auto` right-aligns the button within its flex row
        # in place of a dedicated spacer item. This survives wrapping —
        # when the toolbar wraps onto multiple rows the auto margin
        # still pushes refresh to the right end of whichever row it
        # ends up on.
        style={"marginLeft": "auto"},
    )

    tile_size_slider = html.Div(
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
            dcc.Slider(
                id=ids.COLONY_TILE_SIZE_SLIDER_ID,
                min=64,
                max=400,
                step=16,
                value=150,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "0.5rem",
            "minWidth": "180px",
            "flex": "0 0 220px",
        },
    )

    dim_stepper = _build_dim_stepper(
        minus_id=ids.COLONY_DIM_MINUS,
        plus_id=ids.COLONY_DIM_PLUS,
        readout_id=ids.COLONY_DIM_READOUT,
    )

    return html.Div(
        [
            html.Div(
                [x_label, x_dropdown],
                style={"display": "flex", "alignItems": "center", "gap": "0.25rem"},
            ),
            html.Div(
                [y_label, y_dropdown],
                style={"display": "flex", "alignItems": "center", "gap": "0.25rem"},
            ),
            tile_size_slider,
            dim_stepper,
            crop_size_info,
            refresh_button,
        ],
        id=ids.COLONY_TOOLBAR_ID,
        style={
            "display": "flex",
            "alignItems": "center",
            # Wrap section blocks onto a new row when the viewport is too
            # narrow to fit them all. The bar's background grows
            # vertically because flex-wrap'd rows extend the container's
            # block-axis size; rowGap keeps wrapped rows visually
            # separated from the row above.
            "flexWrap": "wrap",
            "rowGap": "0.5rem",
            "columnGap": "1rem",
            "padding": "0.75rem 1rem",
            "borderBottom": f"1px solid {_BLUE}22",
            "background": COLOR_SURFACE,
        },
    )


def _build_bulk_bar() -> Component:
    """Build the bulk-action alert bar (count label + remove/restore/clear).

    The bar is hidden by default (``display: none``); a callback in
    :mod:`._callbacks` flips it to ``flex`` whenever the colony selection
    store is non-empty. The count label sits on the left, the action
    buttons on the right.

    Returns:
        A :class:`dbc.Alert` styled as a subtle navy-tinted info bar.
    """
    count_label = html.Span(
        "0 selected",
        id=ids.COLONY_BULK_COUNT_LABEL_ID,
        style={"fontWeight": 600, "color": _NAVY},
    )
    remove_btn = dbc.Button(
        "Remove",
        id=ids.COLONY_BULK_REMOVE_BTN_ID,
        color="danger",
        size="sm",
        n_clicks=0,
    )
    restore_btn = dbc.Button(
        "Restore",
        id=ids.COLONY_BULK_RESTORE_BTN_ID,
        color="success",
        size="sm",
        n_clicks=0,
    )
    clear_btn = dbc.Button(
        "Clear",
        id=ids.COLONY_BULK_CLEAR_BTN_ID,
        color="secondary",
        size="sm",
        outline=True,
        n_clicks=0,
    )

    return dbc.Alert(
        [
            count_label,
            html.Div(style={"flex": "1 1 auto"}),  # spacer
            html.Div(
                [remove_btn, restore_btn, clear_btn],
                style={"display": "flex", "gap": "0.5rem"},
            ),
        ],
        id=ids.COLONY_BULK_BAR_ID,
        color="info",
        className="mb-2",
        style={
            "display": "none",  # toggled to "flex" by callback when selection > 0
            "alignItems": "center",
        },
    )


def _build_grid_container() -> Component:
    """Build the (initially empty) colony-grid container.

    The grid is populated by a callback in :mod:`._callbacks`.
    Styling is intentionally minimal here — the grid component manages
    its own CSS via :mod:`._assets/results_viewer.css`.

    Returns:
        An :class:`dash.html.Div` with id
        :data:`ids.COLONY_GRID_CONTAINER_ID` and class
        ``colony-grid-container``.
    """
    return html.Div(
        children=[],
        id=ids.COLONY_GRID_CONTAINER_ID,
        className="colony-grid-container",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def layout(output_root: OutputRoot) -> Component:
    """Build the colony-view tab body.

    Vertical stack (top → bottom):

    1. **Toolbar** (:data:`ids.COLONY_TOOLBAR_ID`) — X / Y axis dropdowns,
       crop-size info chip, refresh button.
    2. **Bulk action bar** (:data:`ids.COLONY_BULK_BAR_ID`) — count
       label plus Remove / Restore / Clear; hidden by default.
    3. **Grid container** (:data:`ids.COLONY_GRID_CONTAINER_ID`) — the
       2D grid of per-colony crops; populated by callbacks in
       :mod:`._callbacks`.

    The whole tab body is wrapped in a scrollable container that mirrors
    the cards column's max-height so vertical sizing matches the Plate
    tab.

    Args:
        output_root: Validated handle on the CLI output directory.
            Currently unused (the layout itself depends only on static
            id constants), but kept in the signature for API uniformity
            with sibling layout factories — Wave 4 may extend this to
            seed default axis options from on-disk measurements.

    Returns:
        A :class:`dash.html.Div` with class ``colony-view-root`` ready
        to drop into a :class:`dbc.Tab`.
    """
    del output_root  # currently unused; kept for API symmetry

    toolbar = _build_toolbar()
    bulk_bar = _build_bulk_bar()
    grid_container = _build_grid_container()

    return html.Div(
        [toolbar, bulk_bar, grid_container],
        className="colony-view-root",
        style={
            "padding": "1rem",
            "maxHeight": "calc(100vh - 8rem)",
            "overflow": "auto",
            "background": _BG,
        },
    )


__all__ = ["layout"]
