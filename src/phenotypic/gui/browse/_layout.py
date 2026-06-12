"""Single-pane Browse layout: dataset + image pickers, OSD canvas, metadata."""
from __future__ import annotations

from typing import Any, cast

from dash import dcc, html

from phenotypic.gui._design import COLOR_MUTED, FONT_SIZE_CAPTION
from phenotypic.gui.browse import _ids as ids

__all__ = ["DATASET_ROW_STYLE", "build_browse_layout"]

_OSD_STYLE = {"height": "70vh", "width": "100%"}

#: Base style for the dataset-row wrapper. Single-sourced so callbacks toggle
#: visibility without clobbering the row's spacing (Dash ``style`` replaces,
#: it never merges).
DATASET_ROW_STYLE = {"marginRight": "0.75rem", "flex": "0 0 auto"}


def _meta_chip(label: str, value_id: str) -> Any:
    return html.Div(
        [
            html.Span(
                label,
                style={
                    "color": COLOR_MUTED,
                    "fontSize": FONT_SIZE_CAPTION,
                    "textTransform": "uppercase",
                    "letterSpacing": "0.06em",
                    "marginRight": "0.4rem",
                },
            ),
            html.Span("—", id=value_id),
        ],
        className="browse-meta-chip",
        style={"marginRight": "1.25rem"},
    )


def build_browse_layout() -> Any:
    """Build the Browse page body (chrome is applied by ``wrap_in_chrome``)."""
    dataset_picker = dcc.Dropdown(
        id=ids.BROWSE_DATASET_PICKER,
        options=[],
        value=None,
        placeholder="Dataset…",
        clearable=False,
        searchable=True,
        style={"minWidth": "12rem"},
    )
    dataset_row = html.Div(
        dataset_picker,
        id=ids.BROWSE_DATASET_ROW,
        style=dict(DATASET_ROW_STYLE),
    )

    image_picker = dcc.Dropdown(
        id=ids.BROWSE_IMAGE_PICKER,
        options=[],
        value=None,
        placeholder="Select image…",
        clearable=False,
        searchable=True,
        style={"flex": "1 1 auto", "minWidth": "12rem"},
    )
    picker_group = html.Div(
        [
            html.Button(
                "‹",
                id=ids.BROWSE_PREV_BTN,
                n_clicks=0,
                title="Previous image",
                className="btn btn-outline-secondary btn-sm",
                type="button",
                **cast(Any, {"aria-label": "Previous image"}),
            ),
            html.Div(image_picker, style={"flex": "1 1 auto"}),
            html.Button(
                "›",
                id=ids.BROWSE_NEXT_BTN,
                n_clicks=0,
                title="Next image",
                className="btn btn-outline-secondary btn-sm",
                type="button",
                **cast(Any, {"aria-label": "Next image"}),
            ),
        ],
        className="d-flex align-items-center",
        style={"gap": "0.35rem", "flex": "1 1 auto"},
    )

    header = html.Div(
        [dataset_row, picker_group],
        className="d-flex align-items-center",
        style={"gap": "0.5rem", "flexWrap": "wrap", "marginBottom": "0.75rem"},
    )

    empty_hint = html.Div(
        "No source image root selected. Pick one from the top bar "
        "(“source:”) to browse its images.",
        id=ids.BROWSE_EMPTY_HINT,
        className="text-muted",
        style={"display": "none", "padding": "2rem 0"},
    )

    osd_div = html.Div(
        id=ids.BROWSE_OSD_DIV,
        className="browse-osd-canvas",
        style=_OSD_STYLE,
    )

    metadata_panel = html.Div(
        [
            _meta_chip("Dimensions", ids.BROWSE_META_DIMS),
            _meta_chip("Size", ids.BROWSE_META_SIZE),
            _meta_chip("Captured", ids.BROWSE_META_CAPTURED),
            _meta_chip("Camera", ids.BROWSE_META_CAMERA),
        ],
        className="browse-meta-panel d-flex flex-wrap",
        style={"marginTop": "0.75rem"},
    )

    return html.Div(
        [
            header,
            empty_hint,
            osd_div,
            metadata_panel,
            dcc.Store(id=ids.BROWSE_DATASETS_STORE, data={}),
            dcc.Store(id=ids.BROWSE_CURRENT_IMAGE_STORE, data=None),
            dcc.Store(id=ids.BROWSE_OSD_SYNC, data=None),
        ],
        className="browse-page",
        style={"padding": "1rem"},
    )
