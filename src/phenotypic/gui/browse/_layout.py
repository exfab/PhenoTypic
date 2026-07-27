"""Single-pane Browse layout: dataset + image pickers, OSD canvas, metadata."""
from __future__ import annotations

from typing import Any, cast

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui._config import (
    TIMELINE_COMPARE_CAP,
    TIMELINE_FOCUS_MARGIN,
    TIMELINE_MOUNT_CAP,
    TIMELINE_TILE_SIZE_DEFAULT,
    TIMELINE_WARM_CONCURRENCY,
)
from phenotypic.gui._design import COLOR_MUTED, FONT_SIZE_CAPTION
from phenotypic.gui.browse import _ids as ids

__all__ = ["DATASET_ROW_STYLE", "build_browse_layout", "build_timeline_body"]

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
                className="btn btn-outline-secondary btn-sm browse-step-button",
                type="button",
                **cast(Any, {"aria-label": "Previous image"}),
            ),
            html.Button(
                "›",
                id=ids.BROWSE_NEXT_BTN,
                n_clicks=0,
                title="Next image",
                className="btn btn-outline-secondary btn-sm browse-step-button",
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

    view_toggle = dcc.RadioItems(
        id=ids.BROWSE_VIEW_MODE_TOGGLE,
        options=[
            {"label": "Single", "value": "single"},
            {"label": "Timeline", "value": "timeline"},
        ],
        value="single",
        inline=True,
        className="browse-view-mode",
    )
    header.children.append(view_toggle)  # header.children is [dataset_row, picker_group]

    single_body = html.Div(
        [empty_hint, osd_stage, metadata_panel, csv_metadata_panel],
        id=ids.BROWSE_SINGLE_BODY,
    )

    return html.Div(
        [
            header,
            single_body,
            build_timeline_body(),
            dcc.Store(id=ids.BROWSE_DATASETS_STORE, data={}),
            dcc.Store(id=ids.BROWSE_CURRENT_IMAGE_STORE, data=None),
            dcc.Store(id=ids.BROWSE_OSD_SYNC, data=None),
        ],
        className="browse-page",
        style={"padding": "1rem"},
    )


def build_timeline_body() -> Any:
    """Build the Timeline matrix body (hidden until the view toggle selects it)."""
    row_source = dcc.Dropdown(
        id=ids.BROWSE_TL_ROW_SOURCE,
        options=[
            {"label": "Folder", "value": "folder"},
            {"label": "Filename pattern", "value": "pattern"},
            {"label": "CSV column", "value": "csv"},
        ],
        value="folder",
        clearable=False,
        style={"minWidth": "10rem"},
    )
    time_source = dcc.Dropdown(
        id=ids.BROWSE_TL_TIME_SOURCE,
        options=[
            {"label": "EXIF capture time", "value": "exif"},
            {"label": "Folder", "value": "folder"},
            {"label": "Filename pattern", "value": "pattern"},
            {"label": "CSV column", "value": "csv"},
        ],
        value="exif",
        clearable=False,
        style={"minWidth": "10rem"},
    )
    csv_cols = [
        dcc.Dropdown(id=ids.BROWSE_TL_ROW_CSV_COL, options=[], placeholder="Row column…"),
        dcc.Dropdown(id=ids.BROWSE_TL_TIME_CSV_COL, options=[], placeholder="Time column…"),
        dcc.Dropdown(
            id=ids.BROWSE_TL_CSV_IMAGE_COL, options=[], placeholder="Image-name column…"
        ),
    ]
    pattern_controls = html.Div(
        [
            dcc.Input(
                id=ids.BROWSE_TL_PATTERN_INPUT,
                type="text",
                placeholder="{plate}_t{time}",
                debounce=True,
                style={"minWidth": "14rem"},
            ),
            dcc.Checklist(
                id=ids.BROWSE_TL_PATTERN_ADVANCED,
                options=[{"label": "regex", "value": "advanced"}],
                value=[],
            ),
            html.Div(
                id=ids.BROWSE_TL_PATTERN_PREVIEW, className="browse-tl-pattern-preview"
            ),
        ],
        className="d-flex align-items-center",
        style={"gap": "0.5rem", "flexWrap": "wrap"},
    )
    tile_stepper = html.Div(
        [
            html.Button(
                "−",
                id=ids.BROWSE_TL_TILE_SIZE_MINUS,
                n_clicks=0,
                className="btn btn-outline-secondary btn-sm",
                type="button",
            ),
            html.Span(
                f"{TIMELINE_TILE_SIZE_DEFAULT} px", id=ids.BROWSE_TL_TILE_SIZE_READOUT
            ),
            html.Button(
                "+",
                id=ids.BROWSE_TL_TILE_SIZE_PLUS,
                n_clicks=0,
                className="btn btn-outline-secondary btn-sm",
                type="button",
            ),
        ],
        className="d-flex align-items-center",
        style={"gap": "0.25rem"},
    )
    # "Compare selected" opens the synced Compare strip (spec §7) for the cells
    # the user multi-selected (shift/ctrl-click). It is a DOM target driven by
    # `timeline.js` — NO Dash callback. The SURFACE-AGNOSTIC class
    # `.timeline-compare-btn` is what the vendored controller binds (scoped to
    # the timeline body), so the same `timeline.js` wires this on Browse +
    # Results; `browse-tl-compare` stays for Browse-only CSS.
    compare_button = html.Button(
        "Compare selected",
        id=ids.BROWSE_TL_COMPARE_BTN,
        n_clicks=0,
        type="button",
        className="timeline-compare-btn browse-tl-compare btn btn-outline-secondary btn-sm",
    )
    nudge = html.Div(
        "Add a metadata CSV (Settings → Metadata) for richer time × group axes.",
        id=ids.BROWSE_TL_NUDGE,
        className="alert alert-info py-1 px-2",
        style={"display": "none"},
    )
    # Surfaces the CSV-join warnings (e.g. cross-folder stem collisions) the
    # render callback writes to BROWSE_TL_STORE_WARNINGS. Hidden (is_open=False)
    # when the warning list is empty; opened with the warning text otherwise.
    warnings_alert = dbc.Alert(
        id=ids.BROWSE_TL_WARNINGS_ALERT,
        color="warning",
        is_open=False,
        dismissable=True,
        className="py-1 px-2",
    )
    controls = html.Div(
        [
            html.Span("Rows"),
            row_source,
            html.Span("Time"),
            time_source,
            *csv_cols,
            pattern_controls,
            tile_stepper,
            compare_button,
        ],
        className="d-flex align-items-center",
        style={"gap": "0.5rem", "flexWrap": "wrap", "marginBottom": "0.5rem"},
    )
    # The focus-navigate constants ride as STATIC data-* on the container div
    # (they never change at runtime). timeline.js reads them off the grid
    # container; the render callback only replaces this div's *children*, so it
    # cannot set the container's own attributes — they must live here.
    # NOTE (spec §16.7): data-focus-margin REPLACES the scroll-era
    # data-margin-screens. The inner grid is positioned by the JS via a CSS
    # transform to centre the focused cell.
    # SURFACE-AGNOSTIC class `.timeline-grid-container`: `timeline.js` is
    # vendored byte-for-byte into both Browse and Results, so the controller
    # locates the grid container (and re-attaches to it) by this stable class,
    # NOT by the surface-specific id — the id stays only for the Dash render
    # callback's Output target. Each surface's clientside callback passes its
    # own container id into attach(containerId).
    grid = html.Div(
        id=ids.BROWSE_TL_GRID,
        className="timeline-grid-container",
        **cast(
            Any,
            {
                "data-mount-cap": str(TIMELINE_MOUNT_CAP),
                "data-focus-margin": str(TIMELINE_FOCUS_MARGIN),
                "data-warm-concurrency": str(TIMELINE_WARM_CONCURRENCY),
                # Compare-strip cap (spec §7): timeline.js reads it off the DOM
                # like the other static focus-navigate constants above.
                "data-compare-cap": str(TIMELINE_COMPARE_CAP),
                # Replaced on every render. Browse's delegated popout events
                # echo this value and the server rejects retired revisions.
                "data-grid-revision": "",
                "data-revision-generation": "",
                "data-session-id": "",
                "data-authorized-revision": "",
            },
        ),
    )
    # No-scroll focus-window VIEWPORT (spec §16.1): overflow hidden (no
    # scrollbar), bounded height, position:relative so the four edge buttons +
    # the focus position readout anchor to its edges. timeline.js centres the
    # inner grid on the focused cell via a CSS transform.
    # SURFACE-AGNOSTIC classes: the four nav buttons + readout carry stable
    # `timeline-*` classes that `timeline.js` queries (scoped to the timeline
    # body). The Dash ids stay (server callbacks + Browse e2e selectors target
    # them); the controller never reads the ids. The `browse-tl-*` classes
    # remain for Browse-only CSS styling.
    nav_up = html.Button(
        "▲",
        id=ids.BROWSE_TL_NAV_UP,
        type="button",
        n_clicks=0,
        className="timeline-nav-up browse-tl-nav browse-tl-nav--up",
        **cast(Any, {"aria-label": "Move focus up one row"}),
    )
    nav_down = html.Button(
        "▼",
        id=ids.BROWSE_TL_NAV_DOWN,
        type="button",
        n_clicks=0,
        className="timeline-nav-down browse-tl-nav browse-tl-nav--down",
        **cast(Any, {"aria-label": "Move focus down one row"}),
    )
    nav_left = html.Button(
        "◀",
        id=ids.BROWSE_TL_NAV_LEFT,
        type="button",
        n_clicks=0,
        className="timeline-nav-left browse-tl-nav browse-tl-nav--left",
        **cast(Any, {"aria-label": "Move focus back one time step"}),
    )
    nav_right = html.Button(
        "▶",
        id=ids.BROWSE_TL_NAV_RIGHT,
        type="button",
        n_clicks=0,
        className="timeline-nav-right browse-tl-nav browse-tl-nav--right",
        **cast(Any, {"aria-label": "Move focus forward one time step"}),
    )
    # timeline.js sets this readout's text (e.g. "row 1/74 · time 1/24").
    position = html.Div(
        id=ids.BROWSE_TL_POSITION, className="timeline-position browse-tl-position"
    )
    grid_viewport = html.Div(
        [grid, nav_up, nav_down, nav_left, nav_right, position],
        # `.timeline-viewport` is the surface-agnostic anchor the controller
        # walks to from the grid container (`.closest(".timeline-viewport")`);
        # `.browse-tl-viewport` stays for Browse-only CSS.
        className="timeline-viewport browse-tl-viewport",
        # tabIndex makes the viewport focusable so its scoped keyboard handler
        # (arrow keys / Enter / Space) receives events; overflow:hidden = no
        # scrollbar (focus-and-navigate, not scroll).
        tabIndex=0,
        style={
            "overflow": "hidden",
            "position": "relative",
            "height": "75vh",
            "border": "1px solid var(--color-border)",
        },
    )
    popout = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("", id=ids.BROWSE_TL_POPOUT_TITLE)),
            dbc.ModalBody(html.Div(id=ids.BROWSE_TL_POPOUT_OSD, style={"height": "70vh"})),
        ],
        id=ids.BROWSE_TL_POPOUT_MODAL,
        is_open=False,
        size="xl",
    )
    return html.Div(
        [
            nudge,
            warnings_alert,
            controls,
            grid_viewport,
            popout,
            # Browse uses a delegated, revision-bound ``set_props`` event
            # published by browse.js. Unlike a synthetic event on a controlled
            # dcc.Input, the store remains connected across grid remounts.
            dcc.Store(id=ids.BROWSE_TL_POPOUT_EVENT, data=None),
            dcc.Store(id=ids.BROWSE_TL_POPOUT_APPROVED, data=None),
            dcc.Store(id=ids.BROWSE_TL_SOURCE_REVISION, data=None),
            dcc.Store(
                id=ids.BROWSE_TL_SESSION,
                storage_type="memory",
                data=None,
            ),
            dcc.Store(id=ids.BROWSE_TL_REVISION_CANDIDATE, data=None),
            dcc.Store(id=ids.BROWSE_TL_REVISION_AUTHORIZED, data=None),
            dcc.Store(id=ids.BROWSE_TL_STORE_TILE_SIZE, data=TIMELINE_TILE_SIZE_DEFAULT),
            dcc.Store(id=ids.BROWSE_TL_STORE_WARNINGS, data=[]),
            dcc.Store(id=ids.BROWSE_TL_POPOUT_STORE, data=None),
        ],
        id=ids.BROWSE_TIMELINE_BODY,
        # SURFACE-AGNOSTIC class `.timeline-body`: the controller scopes its
        # sibling-control queries (nav buttons and readout) to the enclosing
        # element carrying this class, so the vendored timeline.js never reads
        # a surface-specific id.
        className="timeline-body",
        style={"display": "none"},  # toggled on by the view-mode callback
    )
