"""Static layout for the QC Review sub-view (master–detail walkthrough).

This module builds the *shells* the Review callbacks populate — it is
layout-only and Dash-state-free so it stays importable from tests. The
shape (spec §D.2):

```
toolbar:   [Module ▾] [on/groupby chips]  [Show: …]   [↻ Re-sort]
header:    (total) (fail) (warn) (pass) (insufficient) (reviewed) (removed) (median)
body:      ┌ worklist sidebar ┐ ┌ detail pane ──────────────────────┐
           │ group rows       │ │ group header (metric before→after) │
           │ (worst-first,    │ │ faceted tile gallery               │
           │  frozen order)   │ │ [mark reviewed] [next] [bulk …]    │
           └──────────────────┘ └────────────────────────────────────┘
```

Per-row worklist content, the summary tiles, the detail header, and the
gallery are all rebuilt by callbacks (see
:mod:`._callbacks`) — this module mounts their stable container ids plus
the Review-scoped ``dcc.Store``s.
"""

from __future__ import annotations

from dash import dcc, html
from dash.development.base_component import Component

from phenotypic.gui._config import TILE_DIM_DEFAULT
from phenotypic.gui._design import (
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_NAVY,
    COLOR_SURFACE,
    FONT_FAMILY_MONO,
    FONT_SIZE_CAPTION,
    FONT_SIZE_LABEL,
)
from phenotypic.gui.results_viewer._qc_tab.review import _ids as rids

# dbc is import-untyped in this project (see colony_view/_grid.py).
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

#: Approximate rendered height of the sticky summary-stat header strip.
#: Used as the sticky ``top`` offset for the worklist sidebar so it pins
#: just beneath the header (which is itself ``position: sticky; top: 0``)
#: as the page scrolls. A small over-estimate is harmless (a few px gap);
#: under-estimating would let the sidebar slide under the header.
_SUMMARY_HEADER_HEIGHT: str = "64px"


def _build_dim_stepper() -> Component:
    """Build the ``[ − ]  dim 0.60  [ + ]`` tile-spotlight stepper.

    Mirrors the colony-view stepper but with the Review toolbar's
    ``dbc`` button styling. Wires the shared
    :data:`phenotypic.gui.results_viewer._ids.STORE_TILE_DIM_ALPHA`
    strength via the ``−``/``+`` buttons (a callback rebuilds the
    gallery); the readout span is synced from the store by the shared
    readout callback and seeded from :data:`TILE_DIM_DEFAULT`.

    Returns:
        A flex ``html.Div`` matching the Review toolbar's widget styling.
    """
    minus_btn = dbc.Button(
        "−",
        id=rids.QC_REVIEW_DIM_MINUS,
        n_clicks=0,
        color="secondary",
        outline=True,
        size="sm",
        title="Soften the colony-spotlight dimming",
        style={"padding": "0 0.5rem", "lineHeight": "1.2"},
    )
    readout = html.Span(
        f"dim {TILE_DIM_DEFAULT:.2f}",
        id=rids.QC_REVIEW_DIM_READOUT,
        style={
            "fontFamily": FONT_FAMILY_MONO,
            "fontSize": FONT_SIZE_LABEL,
            "color": COLOR_NAVY,
            "minWidth": "4.5rem",
            "textAlign": "center",
            "whiteSpace": "nowrap",
        },
    )
    plus_btn = dbc.Button(
        "+",
        id=rids.QC_REVIEW_DIM_PLUS,
        n_clicks=0,
        color="secondary",
        outline=True,
        size="sm",
        title="Strengthen the colony-spotlight dimming",
        style={"padding": "0 0.5rem", "lineHeight": "1.2"},
    )
    return html.Div(
        [minus_btn, readout, plus_btn],
        className="d-flex align-items-center",
        style={"gap": "0.35rem", "flex": "0 0 auto"},
    )


def _build_toolbar() -> Component:
    """Build the Review toolbar: module picker, chips, show-filter, re-sort."""
    module_picker = dcc.Dropdown(
        id=rids.QC_REVIEW_MODULE_PICKER_ID,
        options=[],
        value=None,
        placeholder="Select a QC module…",
        clearable=False,
        style={"minWidth": "260px"},
    )
    chips = html.Span(
        id=rids.QC_REVIEW_MODULE_CHIPS_ID,
        children=[],
        style={"color": COLOR_MUTED, "fontSize": FONT_SIZE_CAPTION},
    )
    show_filter = dbc.RadioItems(
        id=rids.QC_REVIEW_SHOW_FILTER_ID,
        options=[
            {"label": "Unreviewed", "value": rids.QC_SHOW_UNREVIEWED},
            {"label": "All", "value": rids.QC_SHOW_ALL},
            {"label": "Fail+Warn", "value": rids.QC_SHOW_FAIL_WARN},
        ],
        value=rids.QC_SHOW_UNREVIEWED,
        inline=True,
        labelStyle={"marginRight": "0.75rem", "fontSize": FONT_SIZE_LABEL},
    )
    resort_btn = dbc.Button(
        "↻ Re-sort queue",
        id=rids.QC_REVIEW_RESORT_BTN_ID,
        color="secondary",
        outline=True,
        size="sm",
        n_clicks=0,
        title="Re-apply worst-first order (the only action that reorders the queue)",
    )
    return html.Div(
        [
            html.Div(
                [html.Span("Module:", className="fw-semibold me-2"), module_picker],
                className="d-flex align-items-center",
                style={"gap": "0.5rem"},
            ),
            chips,
            html.Div(style={"flex": "1 1 auto"}),  # spacer
            _build_dim_stepper(),
            show_filter,
            resort_btn,
        ],
        className="d-flex align-items-center flex-wrap",
        style={
            # Fixed/intrinsic height inside the Review column flex parent
            # so the toolbar never grows and steals the body's height.
            "flex": "0 0 auto",
            "gap": "0.75rem",
            "padding": "0.75rem 1rem",
            "borderBottom": f"1px solid {COLOR_BORDER}",
            "background": COLOR_SURFACE,
        },
    )


#: Worklist sidebar width bounds (px), the single source of truth shared by
#: the default store value, the width-apply callback's clamp, and the JS
#: drag-splitter. Narrower default than the old 280px per user feedback.
SIDEBAR_DEFAULT_WIDTH_PX: int = 180
SIDEBAR_MIN_WIDTH_PX: int = 140
SIDEBAR_MAX_WIDTH_PX: int = 380


def clamp_sidebar_width(px: object) -> int:
    """Clamp a candidate sidebar width to ``[MIN, MAX]`` px (falls back to default).

    Single source of truth for the width clamp — the JS drag-splitter
    mirrors these same bounds, and the width-apply callback routes through
    here so a malformed / out-of-range store value can never produce an
    invalid sidebar width. Non-numeric input falls back to the default.

    Args:
        px: Candidate width (the store value; may be ``None`` / a string /
            out of range after a JS write or a stale store).

    Returns:
        An ``int`` width within ``[SIDEBAR_MIN_WIDTH_PX,
        SIDEBAR_MAX_WIDTH_PX]``.
    """
    try:
        value = int(round(float(px)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return SIDEBAR_DEFAULT_WIDTH_PX
    return max(SIDEBAR_MIN_WIDTH_PX, min(SIDEBAR_MAX_WIDTH_PX, value))


def expanded_sidebar_style() -> dict[str, str]:
    """Sidebar wrapper style when expanded (sticky, intrinsic width).

    Sticky just below the summary header so it — and the header — stay in
    view while the gallery scrolls. ``flex: 0 0 auto`` lets the inner
    worklist's (resizable) width drive the wrapper; the detail pane
    (``flex: 1 1 auto``) reclaims whatever the sidebar doesn't take.
    """
    return {
        "flex": "0 0 auto",
        "alignSelf": "flex-start",
        "position": "sticky",
        "top": _SUMMARY_HEADER_HEIGHT,
        "maxHeight": f"calc(100vh - {_SUMMARY_HEADER_HEIGHT})",
        "borderRight": f"1px solid {COLOR_BORDER}",
        "display": "flex",
        "flexDirection": "column",
        "minHeight": "0",
    }


def collapsed_sidebar_style() -> dict[str, str]:
    """Sidebar wrapper style when collapsed to a thin chevron rail.

    The worklist itself is hidden by the callback; only the expand
    chevron shows. The detail/gallery pane reclaims the freed width.
    """
    return {
        "flex": "0 0 auto",
        "alignSelf": "flex-start",
        "position": "sticky",
        "top": _SUMMARY_HEADER_HEIGHT,
        "borderRight": f"1px solid {COLOR_BORDER}",
        "display": "flex",
        "flexDirection": "column",
        "padding": "0.25rem",
    }


def _build_sidebar() -> Component:
    """Build the collapsible worklist sidebar: chevron toggle + resizable list."""
    toggle = dbc.Button(
        "◀",
        id=rids.QC_REVIEW_SIDEBAR_TOGGLE_ID,
        color="link",
        size="sm",
        n_clicks=0,
        title="Collapse / expand the worklist",
        style={
            "alignSelf": "flex-end",
            "padding": "0 0.35rem",
            "textDecoration": "none",
            "color": COLOR_NAVY,
        },
    )
    worklist = html.Div(
        id=rids.QC_REVIEW_WORKLIST_ID,
        children=[],
        style={
            # Width is driven by STORE_QC_SIDEBAR_WIDTH (the JS drag-
            # splitter writes the dragged px; a callback applies it here)
            # so it survives re-renders + collapse. Starts at the default.
            "width": f"{SIDEBAR_DEFAULT_WIDTH_PX}px",
            "overflow": "auto",
            # Bounded height so a long worklist scrolls internally without
            # ever pushing the gallery off-screen.
            "maxHeight": f"calc(100vh - {_SUMMARY_HEADER_HEIGHT} - 2rem)",
            "padding": "0.5rem",
        },
    )
    return html.Div(
        [toggle, worklist],
        id=rids.QC_REVIEW_SIDEBAR_ID,
        style=expanded_sidebar_style(),
    )


def _build_splitter() -> Component:
    """Build the thin draggable splitter between the sidebar and the gallery.

    A visible 6px grab handle; the clientside drag logic in
    ``results_viewer.js`` widens/narrows the worklist live as the user
    drags and persists the final width (clamped) to
    :data:`STORE_QC_SIDEBAR_WIDTH` on mouse-up. ``flex: 0 0 auto`` keeps
    it from stretching; ``cursor: col-resize`` signals the affordance.
    """
    return html.Div(
        id=rids.QC_REVIEW_SPLITTER_ID,
        children=[],
        title="Drag to resize the worklist",
        style={
            "flex": "0 0 auto",
            "alignSelf": "stretch",
            "width": "6px",
            "minHeight": f"calc(100vh - {_SUMMARY_HEADER_HEIGHT})",
            "cursor": "col-resize",
            "background": COLOR_BORDER,
            "position": "sticky",
            "top": _SUMMARY_HEADER_HEIGHT,
        },
    )


def _build_body() -> Component:
    """Build the master–detail body: collapsible sticky sidebar + page-scroll detail.

    The body is a flex *row* aligned to the top. The whole page scrolls
    (the Review view no longer caps its height), so the detail/gallery
    pane grows with its content and tiles flow down the page. The
    worklist sidebar is collapsible (chevron) and resizable (a draggable
    splitter handle sits between it and the gallery), and stays
    ``position: sticky`` so it and the summary header remain in view while
    the gallery scrolls; the detail pane reclaims any width the sidebar
    gives up.
    """
    detail = html.Div(
        [
            html.Div(id=rids.QC_REVIEW_DETAIL_HEADER_ID, children=[]),
            html.Div(id=rids.QC_REVIEW_GALLERY_ID, children=[]),
            _build_detail_action_bar(),
        ],
        id=rids.QC_REVIEW_DETAIL_ID,
        # Grows with its content (no fixed height, no internal scroll) so
        # the full-width gallery flows down the page and the page scrolls.
        style={
            "flex": "1 1 auto",
            "minWidth": "0",
            "padding": "0.75rem 1rem",
        },
    )
    return html.Div(
        [_build_sidebar(), _build_splitter(), detail],
        className="d-flex",
        style={"alignItems": "flex-start"},
    )


def _build_detail_action_bar() -> Component:
    """Build the detail action bar: mark-reviewed / next / bulk remove-restore."""
    return html.Div(
        [
            dbc.Button(
                "✓ Mark reviewed",
                id=rids.QC_REVIEW_MARK_REVIEWED_BTN_ID,
                color="primary",
                size="sm",
                n_clicks=0,
                style={"background": COLOR_NAVY, "borderColor": COLOR_NAVY},
            ),
            dbc.Button(
                "Next group →",
                id=rids.QC_REVIEW_NEXT_BTN_ID,
                color="secondary",
                outline=True,
                size="sm",
                n_clicks=0,
            ),
            html.Div(style={"flex": "1 1 auto"}),
            dbc.Button(
                "Remove selected",
                id=rids.QC_REVIEW_BULK_REMOVE_BTN_ID,
                color="danger",
                outline=True,
                size="sm",
                n_clicks=0,
            ),
            dbc.Button(
                "Restore selected",
                id=rids.QC_REVIEW_BULK_RESTORE_BTN_ID,
                color="secondary",
                outline=True,
                size="sm",
                n_clicks=0,
            ),
        ],
        className="d-flex align-items-center",
        style={
            "gap": "0.5rem",
            "padding": "0.5rem 0",
            "borderTop": f"1px solid {COLOR_BORDER}",
            "marginTop": "0.5rem",
        },
    )


def build_review_view() -> Component:
    """Build the Review sub-view body (toolbar + summary header + master-detail).

    Returns a single ``html.Div`` ready to slot into the QC tab's Review
    container (:data:`._ids.QC_REVIEW_VIEW_ID`). All dynamic content is
    rendered by callbacks into the stable container ids mounted here; the
    Review-scoped stores travel with the view so they reset when the
    viewer reloads.

    Returns:
        The Review sub-view component tree.
    """
    return html.Div(
        [
            _build_toolbar(),
            html.Div(
                id=rids.QC_REVIEW_SUMMARY_HEADER_ID,
                children=[],
                style={
                    # Compact horizontal strip pinned to the top of the
                    # scroll container: stays in view as the gallery
                    # scrolls down the page (the stat tiles are a flex row
                    # inside; scrolls sideways on a narrow viewport rather
                    # than stacking). ``zIndex`` keeps it above the tiles.
                    "position": "sticky",
                    "top": "0",
                    "zIndex": "3",
                    "flex": "0 0 auto",
                    "overflowX": "auto",
                    "padding": "0.5rem 1rem",
                    "borderBottom": f"1px solid {COLOR_BORDER}",
                    "background": COLOR_SURFACE,
                },
            ),
            html.Div(
                id=rids.QC_REVIEW_EMPTY_STATE_ID,
                children=_default_empty_state(),
                style={"flex": "0 0 auto", "padding": "2rem", "textAlign": "center"},
            ),
            _build_body(),
            # Review-scoped stores.
            dcc.Store(id=rids.STORE_QC_WORKLIST_ORDER, data=[]),
            dcc.Store(id=rids.STORE_QC_SELECTED_GROUP, data=None),
            dcc.Store(id=rids.STORE_QC_RECOMPUTE_DELTAS, data={}),
            dcc.Store(id=rids.STORE_QC_SIDEBAR_COLLAPSED, data=False),
            dcc.Store(
                id=rids.STORE_QC_SIDEBAR_WIDTH, data=SIDEBAR_DEFAULT_WIDTH_PX
            ),
        ],
        className="qc-review-root d-flex flex-column",
        # No height cap / overflow lock: the view sizes to its content so
        # the gallery flows down and the tab's own scroll container (the
        # ``qc-tab-root`` wrapper) scrolls the whole page. Sticky header +
        # sidebar keep the nav in view throughout.
        style={},
    )


def _default_empty_state() -> Component:
    """Build the empty-state placeholder shown before a module is picked."""
    return html.Div(
        [
            html.Div("No QC review queue yet.", className="fw-semibold"),
            html.Div(
                "Configure a quality check, then re-run "
                "`python -m phenotypic --recompile <output>` (or pick a "
                "module above if a qc/ artifact already exists).",
                style={"color": COLOR_MUTED, "fontSize": FONT_SIZE_CAPTION},
            ),
        ]
    )


__all__ = ["build_review_view"]
