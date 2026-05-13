"""Static layout for the Heatmap tab.

Top-level shape (vertical stack):

1. Picker strip - one row per control: Color column, Aggregator, Image,
   Time slider. Each row uses a label span + a single Dash control so
   the picker strip wraps cleanly on narrow viewports.
2. Figure container - a ``dcc.Graph`` that the render callback fills
   with the figure built by :func:`._figure.build_heatmap_figure`.
3. Optional empty-state card - rendered alongside the figure (hidden
   when the figure is non-empty); explanation copy lives in the
   figure builder's annotation, so this slot is reserved for richer
   chrome a future wave might add.

Population of dropdown options and time-slider marks lives in
:mod:`._callbacks` so this module stays free of ``OutputRoot`` /
``MeasurementSchema`` imports.
"""
from __future__ import annotations

import logging

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html
from dash.development.base_component import Component

from phenotypic.gui._design import (
    COLOR_BG,
    COLOR_BLUE,
    COLOR_MUTED,
    COLOR_NAVY,
    COLOR_SURFACE,
    FONT_SIZE_CAPTION,
    FONT_SIZE_LABEL,
)
from phenotypic.gui._schema_cache import MeasurementSchema
from phenotypic.gui.results_viewer._heatmap_tab import _ids as ids
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style tokens
# ---------------------------------------------------------------------------

_NAVY = COLOR_NAVY
_BLUE = COLOR_BLUE
_BG = COLOR_BG

#: Default aggregator presented to the user. Matches the spec's
#: "common case is one row per well, aggregator is a no-op" baseline.
_DEFAULT_AGGREGATOR: str = "mean"

#: Aggregator option list - matches the closed value-set on
#: :data:`._figure.AggregatorName`.
_AGGREGATOR_OPTIONS: list[dict[str, str]] = [
    {"label": "Mean", "value": "mean"},
    {"label": "Median", "value": "median"},
    {"label": "Max", "value": "max"},
    {"label": "Min", "value": "min"},
]


# ---------------------------------------------------------------------------
# Sub-builders
# ---------------------------------------------------------------------------


def _picker_row(label: str, control: Component) -> Component:
    """Build one label + control row for the picker strip."""
    return html.Div(
        [
            html.Span(
                label,
                style={
                    "color": _NAVY,
                    "fontWeight": 500,
                    "fontSize": FONT_SIZE_LABEL,
                    "minWidth": "120px",
                    "display": "inline-block",
                },
            ),
            html.Div(control, style={"flex": "1 1 auto"}),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "0.5rem",
            "marginBottom": "0.5rem",
        },
    )


def _build_picker_strip(
    color_options: list[dict[str, str]],
    image_options: list[dict[str, str]],
) -> Component:
    """Build the picker strip (color, aggregator, image, time)."""
    color_picker = dcc.Dropdown(
        id=ids.HEATMAP_COLOR_PICKER_ID,
        options=color_options,
        value=color_options[0]["value"] if color_options else None,
        placeholder="Pick a measurement column...",
        clearable=False,
    )
    aggregator_picker = dcc.Dropdown(
        id=ids.HEATMAP_AGGREGATOR_PICKER_ID,
        options=_AGGREGATOR_OPTIONS,
        value=_DEFAULT_AGGREGATOR,
        clearable=False,
    )
    image_picker = dcc.Dropdown(
        id=ids.HEATMAP_IMAGE_PICKER_ID,
        options=image_options,
        value=image_options[0]["value"] if image_options else None,
        placeholder="Pick an image...",
        clearable=False,
    )

    # Time slider lives inside a wrapper div so its visibility (hide /
    # show) is toggled cleanly by the controls-refresh callback without
    # rebuilding the slider component.
    time_slider = dcc.Slider(
        id=ids.HEATMAP_TIME_SLIDER_ID,
        min=0,
        max=1,
        step=None,
        marks={},
        value=None,
        tooltip={"placement": "bottom", "always_visible": False},
    )
    non_numeric_caption = html.Div(
        id=ids.HEATMAP_TIME_NON_NUMERIC_CAPTION_ID,
        children="",
        style={
            "color": COLOR_MUTED,
            "fontSize": FONT_SIZE_CAPTION,
            "marginTop": "0.25rem",
        },
    )
    time_slider_wrapper = html.Div(
        [time_slider, non_numeric_caption],
        id=ids.HEATMAP_TIME_SLIDER_WRAPPER_ID,
        style={"display": "none"},  # hidden until a numeric time column shows up
    )

    return html.Div(
        [
            _picker_row("Color column", color_picker),
            _picker_row("Aggregator", aggregator_picker),
            _picker_row("Image", image_picker),
            _picker_row("Time", time_slider_wrapper),
        ],
        style={
            "padding": "0.75rem 1rem",
            "borderBottom": f"1px solid {_BLUE}22",
            "background": COLOR_SURFACE,
        },
    )


def _build_figure_container() -> Component:
    """Build the ``dcc.Graph`` that hosts the heatmap.

    The mode-bar is disabled so the picker strip remains the canonical
    interaction surface; users that need pan/zoom can re-enable it via
    a future toolbar control if scale demands it.
    """
    return dcc.Graph(
        id=ids.HEATMAP_FIGURE_ID,
        config={"displayModeBar": False, "responsive": True},
        style={"width": "100%", "height": "70vh"},
    )


def _build_empty_state_card() -> Component:
    """Build the reserved empty-state slot.

    The figure builder paints its own annotation when grid columns are
    missing, so this slot stays mounted as an empty hook for callbacks
    a future wave might attach (e.g. a recipe-import suggestion). It
    starts hidden so the layout is visually clean at boot.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6("Heatmap unavailable", style={"color": _NAVY}),
                html.P(
                    "The active filter does not yield grid coordinates.",
                    className="mb-0",
                    style={"color": COLOR_MUTED, "fontSize": FONT_SIZE_LABEL},
                ),
            ]
        ),
        id=ids.HEATMAP_EMPTY_STATE_ID,
        style={"display": "none", "margin": "1rem"},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_heatmap_tab_body(
    output_root: OutputRoot,
    schema: MeasurementSchema,
) -> Component:
    """Build the Heatmap tab body.

    Args:
        output_root: Validated handle on the CLI output directory; used
            to seed the image picker's option list from the master
            frame's unique ``Metadata_ImageFile`` values.
        schema: Measurement schema cache; used to seed the color picker
            options at boot. The callbacks refresh both lists when QC
            recipe revisions or curation events fire.

    Returns:
        A :class:`dash.html.Div` ready to drop into a :class:`dbc.Tab`.
    """
    # Seed initial options from on-disk schema + master frame; the
    # callbacks repopulate on revision changes.
    column_names = schema.columns_for("measurements")
    color_options = [{"label": c, "value": c} for c in column_names]

    image_files: list[str] = []
    if "Metadata_ImageFile" in output_root.master_df.columns:
        image_files = sorted(
            v
            for v in output_root.master_df["Metadata_ImageFile"].unique().to_list()
            if v is not None
        )
    image_options = [{"label": f, "value": f} for f in image_files]

    picker_strip = _build_picker_strip(color_options, image_options)
    figure_container = _build_figure_container()
    empty_state = _build_empty_state_card()

    return html.Div(
        [picker_strip, figure_container, empty_state],
        className="heatmap-tab-root",
        style={
            "padding": "0",
            "maxHeight": "calc(100vh - 8rem)",
            "overflow": "auto",
            "background": _BG,
        },
    )


__all__ = ["build_heatmap_tab_body"]
