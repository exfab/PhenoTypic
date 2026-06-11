"""Static layout for the Error-analysis tab.

Top-level shape (vertical stack):

1. Control strip — category-chip container, good-baseline toggle
   (All unlabeled / Verified only), verified-good count badge, and the
   "Save analysis report" button.
2. Stale banner — hidden until the recompute callback surfaces a
   re-key/stale state.
3. Content block (``ERROR_CONTENT_ID``) — the ranked cutoff
   ``DataTable`` beside the distribution ``dcc.Graph`` (editable cutoff
   line), with the numeric cutoff input, recall/specificity readout, and
   the copy-able filter-spec ``dcc.Textarea`` + ``dcc.Clipboard`` beneath.
4. Empty-state card (``ERROR_EMPTY_STATE_ID``) — the "need more labels"
   explanation, shown in place of the content block when the engine has
   insufficient data.

The containers ship empty; the recompute callback (Task 6) fills the
chips / table / figure / badges. No data reads happen at build time.
"""
from __future__ import annotations

import logging

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dash_table, dcc, html
from dash.development.base_component import Component

from phenotypic.gui._design import (
    COLOR_BG,
    COLOR_MUTED,
    COLOR_NAVY,
    COLOR_SURFACE,
    FONT_FAMILY_MONO,
    FONT_SIZE_BODY_SM,
    FONT_SIZE_CAPTION,
)
from phenotypic.gui._schema_cache import MeasurementSchema
from phenotypic.gui.results_viewer._error_tab import _ids as ids
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-builders
# ---------------------------------------------------------------------------


def _build_control_strip() -> Component:
    """Build the control strip: chips, baseline toggle, badge, save button."""
    good_mode_toggle = dbc.RadioItems(
        id=ids.ERROR_GOOD_MODE_TOGGLE_ID,
        options=[
            {"label": "All unlabeled", "value": "all_unlabeled"},
            {"label": "Verified only", "value": "verified"},
        ],
        value="all_unlabeled",
        inline=True,
        className="error-good-mode-toggle",
    )
    verified_badge = dbc.Badge(
        "",
        id=ids.ERROR_VERIFIED_COUNT_ID,
        color="info",
        className="error-verified-badge",
        style={"display": "none"},
    )
    save_button = dbc.Button(
        "Save analysis report",
        id=ids.ERROR_SAVE_REPORT_BTN_ID,
        color="secondary",
        size="sm",
        outline=True,
        n_clicks=0,
    )
    chips = html.Div(
        id=ids.ERROR_CATEGORY_CHIPS_ID,
        className="error-category-chips",
    )
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "Category",
                        style={
                            "color": COLOR_NAVY,
                            "fontWeight": 500,
                            "fontSize": FONT_SIZE_CAPTION,
                            "marginRight": "0.5rem",
                        },
                    ),
                    chips,
                ],
                className="error-chip-row",
            ),
            html.Div(
                [
                    html.Span(
                        "Good baseline",
                        style={
                            "color": COLOR_NAVY,
                            "fontWeight": 500,
                            "fontSize": FONT_SIZE_CAPTION,
                            "marginRight": "0.5rem",
                        },
                    ),
                    good_mode_toggle,
                    verified_badge,
                    html.Div(save_button, style={"marginLeft": "auto"}),
                ],
                className="error-baseline-row",
            ),
        ],
        className="error-control-strip",
        style={
            "padding": "0.75rem 1rem",
            "background": COLOR_SURFACE,
        },
    )


def _build_stale_banner() -> Component:
    """Build the (hidden) re-key / stale banner."""
    return dbc.Alert(
        "",
        id=ids.ERROR_STALE_BANNER_ID,
        color="warning",
        className="error-stale-banner",
        is_open=False,
        style={"margin": "0.5rem 1rem 0"},
    )


def _build_table() -> Component:
    """Build the ranked cutoff ``DataTable`` (columns filled by callback)."""
    return dash_table.DataTable(  # type: ignore[attr-defined]
        id=ids.ERROR_TABLE_ID,
        columns=[],
        data=[],
        page_size=15,
        sort_action="native",
        row_selectable="single",
        cell_selectable=True,
        editable=False,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_BODY_SM,
            "padding": "4px 8px",
            "textAlign": "right",
        },
        style_cell_conditional=[
            {"if": {"column_id": "measurement"}, "textAlign": "left"},
        ],
        style_header={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_CAPTION,
            "fontWeight": "500",
            "textTransform": "uppercase",
            "letterSpacing": "0.08em",
            "color": COLOR_MUTED,
            "borderBottom": f"2px solid {COLOR_NAVY}",
        },
    )


def _build_figure_block() -> Component:
    """Build the distribution graph + cutoff input + readout + filter spec."""
    figure = dcc.Graph(
        id=ids.ERROR_FIGURE_ID,
        config={"editable": True, "edits": {"shapePosition": True}, "responsive": True},
        style={"width": "100%", "height": "55vh"},
    )
    cutoff_input = html.Div(
        [
            html.Span(
                "Cutoff",
                style={
                    "color": COLOR_NAVY,
                    "fontWeight": 500,
                    "fontSize": FONT_SIZE_CAPTION,
                    "marginRight": "0.5rem",
                },
            ),
            dcc.Input(
                id=ids.ERROR_CUTOFF_INPUT_ID,
                type="number",
                debounce=True,
                className="error-cutoff-input",
                style={"width": "10rem"},
            ),
            html.Div(
                id=ids.ERROR_READOUT_ID,
                className="error-readout",
                style={"marginLeft": "1rem"},
            ),
        ],
        className="error-cutoff-row",
        style={"display": "flex", "alignItems": "center", "marginTop": "0.5rem"},
    )
    filter_spec = html.Div(
        [
            html.Span(
                "Filter spec",
                style={
                    "color": COLOR_NAVY,
                    "fontWeight": 500,
                    "fontSize": FONT_SIZE_CAPTION,
                    "marginRight": "0.5rem",
                },
            ),
            dcc.Textarea(
                id=ids.ERROR_FILTER_SPEC_ID,
                value="",
                readOnly=True,
                className="error-filter-spec",
                style={
                    "width": "100%",
                    "minHeight": "5rem",
                    "fontFamily": FONT_FAMILY_MONO,
                    "fontSize": FONT_SIZE_BODY_SM,
                },
            ),
            dcc.Clipboard(
                target_id=ids.ERROR_FILTER_SPEC_ID,
                className="error-filter-spec-clipboard",
                style={"marginTop": "0.25rem"},
            ),
        ],
        className="error-filter-spec-row",
        style={"marginTop": "0.5rem"},
    )
    return html.Div([figure, cutoff_input, filter_spec])


def _build_content_block() -> Component:
    """Build the table + figure content block (hidden in the empty state)."""
    return html.Div(
        dbc.Row(
            [
                dbc.Col(_build_table(), md=5),
                dbc.Col(_build_figure_block(), md=7),
            ],
            className="g-3",
        ),
        id=ids.ERROR_CONTENT_ID,
        className="error-content",
        style={"padding": "0.5rem 1rem"},
    )


def _build_empty_state_card() -> Component:
    """Build the "need more labels" empty-state card (hidden by default)."""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6("Need more labels", style={"color": COLOR_NAVY}),
                html.P(
                    "Label more objects in this category (and a good baseline) "
                    "before the cutoff finder can rank measurements reliably.",
                    className="mb-0",
                    style={"color": COLOR_MUTED, "fontSize": FONT_SIZE_BODY_SM},
                ),
            ]
        ),
        id=ids.ERROR_EMPTY_STATE_ID,
        className="error-empty-state",
        style={"display": "none", "margin": "1rem"},
    )


def _build_save_toast() -> Component:
    """Build the (hidden) save-confirmation toast."""
    return dbc.Toast(
        "Saved error_analysis.html to deliverables/.",
        id=ids.ERROR_SAVE_TOAST_ID,
        header="Report saved",
        icon="success",
        duration=4000,
        is_open=False,
        dismissable=True,
        style={"position": "fixed", "top": "1rem", "right": "1rem", "zIndex": 1080},
    )


def _build_stores() -> Component:
    """Build the per-tab ``dcc.Store``s for the focus context + modes."""
    return html.Div(
        [
            dcc.Store(id=ids.STORE_ERROR_FOCUS_ID),
            dcc.Store(id=ids.STORE_ERROR_GOOD_MODE_ID, data="all_unlabeled"),
            dcc.Store(id=ids.STORE_ERROR_CATEGORY_ID),
        ]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_error_tab_body(
    output_root: OutputRoot,
    schema: MeasurementSchema,
) -> Component:
    """Build the Error-analysis tab body.

    Ships every container empty; the recompute callback (Task 6) fills the
    chips, table, figure, and badges. Import-light and side-effect-free —
    no ``OutputRoot``/``schema`` reads happen at build time (the arguments
    mirror the other tab factories' signature so the layout module can
    construct it uniformly).

    Args:
        output_root: Validated handle on the CLI output directory.
        schema: Measurement schema cache (unused at build time; reserved
            for parity with sibling tab factories).

    Returns:
        A :class:`dash.html.Div` ready to drop into a :class:`dbc.Tab`.
    """
    del output_root, schema  # not read at build time
    return html.Div(
        [
            _build_stores(),
            _build_control_strip(),
            _build_stale_banner(),
            _build_content_block(),
            _build_empty_state_card(),
            _build_save_toast(),
        ],
        className="error-tab-root",
        style={
            "padding": "0",
            "maxHeight": "calc(100vh - 8rem)",
            "overflow": "auto",
            "background": COLOR_BG,
        },
    )


__all__ = ["build_error_tab_body"]
