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
    # Both steppers sit together on the left (a ‹ › pair via Bootstrap's
    # btn-group) so the user can toggle prev/next without crossing the wide
    # image dropdown, which fills the remaining width to its right.
    stepper_pair = html.Div(
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
        className="btn-group",
        role="group",
        **cast(Any, {"aria-label": "Step through images"}),
    )
    picker_group = html.Div(
        [
            stepper_pair,
            html.Div(image_picker, style={"flex": "1 1 auto"}),
        ],
        className="d-flex align-items-center",
        style={"gap": "0.5rem", "flex": "1 1 auto"},
    )

    header = html.Div(
        [dataset_row, picker_group],
        className="d-flex align-items-center",
        style={"gap": "0.5rem", "flexWrap": "wrap", "marginBottom": "0.75rem"},
    )

    empty_hint = html.Div(
        "No source image root selected. Pick an input folder from Settings "
        "to browse its images.",
        id=ids.BROWSE_EMPTY_HINT,
        className="text-muted",
        style={"display": "none", "padding": "2rem 0"},
    )

    osd_div = html.Div(
        id=ids.BROWSE_OSD_DIV,
        className="browse-osd-canvas",
        style=_OSD_STYLE,
    )

    # Spinner + caption overlay, hidden until browse.js shows it while the
    # source image is normalized + tiled into DZI, then hides it on OSD's
    # ``open`` event (or swaps to an error caption on ``open-failed``). It is
    # a sibling of the OSD div inside a position:relative stage so OSD never
    # clobbers it when it mounts its canvas into BROWSE_OSD_DIV.
    loading_overlay = html.Div(
        [
            html.Div(className="browse-spinner"),
            html.Div(
                "Loading image…",
                id=ids.BROWSE_LOADING_TEXT,
                className="browse-loading-text",
            ),
        ],
        id=ids.BROWSE_OSD_LOADING,
        className="browse-loading-overlay",
        **cast(Any, {"role": "status", "aria-live": "polite"}),
    )
    osd_stage = html.Div(
        [osd_div, loading_overlay],
        className="browse-osd-stage",
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
    csv_metadata_panel = html.Div(
        "No metadata CSV selected",
        id=ids.BROWSE_CSV_METADATA_PANEL,
        className="browse-csv-metadata-panel",
        style={"marginTop": "0.5rem"},
    )

    return html.Div(
        [
            header,
            empty_hint,
            osd_stage,
            metadata_panel,
            csv_metadata_panel,
            dcc.Store(id=ids.BROWSE_DATASETS_STORE, data={}),
            dcc.Store(id=ids.BROWSE_CURRENT_IMAGE_STORE, data=None),
            dcc.Store(id=ids.BROWSE_OSD_SYNC, data=None),
        ],
        className="browse-page",
        style={"padding": "1rem"},
    )
