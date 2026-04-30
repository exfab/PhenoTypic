"""Top-level layout shell for the results viewer.

The shell composes (left → right):

* a header bar with the app title, a one-line ``pipeline.json`` chip, a
  ``Lock views`` switch, and a monospace subtitle showing the absolute
  output-root path;
* a two-column body: the filter sidebar (left) and a scrollable cards
  column (right) with an ``+ Add card`` button below;
* every shared ``dcc.Store`` instance the rest of the viewer reads
  (filter spec, image pairs, card list, lock-views).

Layout-level callbacks are minimal — only the ``Lock views`` switch
mirror — because the heavy callbacks live in the modules that own each
sub-tree (``_filter_panel`` for the sidebar, ``_viewer_card`` for the
cards). This keeps each module self-contained and Wave 4 ends up as a
thin ``register_callbacks(app, output_root)`` orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import Input, Output, dcc, html
from dash.development.base_component import Component

from phenotypic.gui.results_viewer import _filter_panel, _ids as ids
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style tokens (mirrors DESIGN.md / FRONTEND_STYLE_GUIDE.md)
# ---------------------------------------------------------------------------

_NAVY = "#003660"
_BLUE = "#1b75bc"
_GOLD = "#febc11"
_BG = "#f5f7fa"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _build_header(output_root: OutputRoot) -> Component:
    """Build the top header bar.

    Args:
        output_root: Validated handle on the CLI output directory; the
            pipeline summary and root path are surfaced as info chips.

    Returns:
        A header :class:`dash.html.Div` styled as a navy-on-white bar.
    """
    pipeline_label = output_root.pipeline_summary or "unknown"

    pipeline_chip = html.Span(
        [
            html.Span(
                "Pipeline:",
                className="me-1",
                style={"color": "#8892a4"},
            ),
            html.Span(
                pipeline_label,
                id=ids.HEADER_PIPELINE_CHIP_ID,
                style={"color": _NAVY, "fontWeight": 500},
            ),
        ],
        className="me-3",
        style={
            "fontFamily": "'DM Mono', monospace",
            "fontSize": "0.75rem",
            "padding": "0.25rem 0.6rem",
            "border": f"1px solid {_BLUE}33",
            "borderRadius": "9999px",
            "background": "#ffffff",
        },
    )

    lock_switch = dbc.Switch(
        id=ids.BTN_LOCK_VIEWS_TOGGLE,
        label="Lock views",
        value=False,
        className="ms-2 mb-0",
    )

    title = html.H4(
        "Results Viewer",
        className="mb-0 me-3",
        style={
            "color": _NAVY,
            "fontFamily": "'DM Serif Display', Georgia, serif",
        },
    )

    subtitle = html.Div(
        str(output_root.root),
        className="text-muted small",
        style={
            "fontFamily": "'DM Mono', monospace",
            "marginTop": "0.1rem",
        },
    )

    top_row = html.Div(
        [
            title,
            pipeline_chip,
            html.Div(style={"flex": "1 1 auto"}),  # spacer
            lock_switch,
        ],
        className="d-flex align-items-center",
    )

    return html.Div(
        [top_row, subtitle],
        className="results-viewer-header px-3 py-2",
        style={
            "background": "#ffffff",
            "borderBottom": f"1px solid {_BLUE}22",
        },
    )


# ---------------------------------------------------------------------------
# Body
# ---------------------------------------------------------------------------


def _build_cards_column() -> Component:
    """Build the right-side cards column (container + add-card button).

    Wave 3B (``_viewer_card``) is responsible for the actual card
    component trees rendered into ``CARDS_CONTAINER_ID``; this module
    only wires the static skeleton.
    """
    cards_container = html.Div(
        children=[],
        id=ids.CARDS_CONTAINER_ID,
        className="cards-container",
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "1rem",
        },
    )
    add_card_button = dbc.Button(
        "+ Add card",
        id=ids.BTN_ADD_CARD,
        color="primary",
        n_clicks=0,
        className="mt-3",
        style={"background": _NAVY, "borderColor": _NAVY},
    )
    return html.Div(
        [cards_container, add_card_button],
        className="results-viewer-cards-col",
        style={
            "padding": "1rem",
            "overflowY": "auto",
            "maxHeight": "calc(100vh - 8rem)",
        },
    )


def _build_startup_banner(output_root: OutputRoot) -> Component:
    """Build the dismissable startup banner (SSH-forward + cache nuke hint)."""
    cache_dir = output_root.cache_dir
    return dbc.Alert(
        [
            html.Span(
                "Tip:",
                className="fw-bold me-1",
                style={"color": _NAVY},
            ),
            html.Span(
                "Forward this port over SSH with "
                "`ssh -L 8050:localhost:8050 cluster` and open the URL in a "
                "local browser. Stale tiles? Nuke the DZI cache with: ",
            ),
            html.Code(
                f"rm -rf {cache_dir}",
                style={
                    "background": "#edf2f7",
                    "color": _NAVY,
                    "padding": "1px 5px",
                    "borderRadius": "3px",
                },
            ),
        ],
        id="results-viewer-startup-banner",
        color="info",
        dismissable=True,
        is_open=True,
        className="mx-3 mt-2 mb-0 small",
        style={
            "borderLeft": "4px solid #56B4E9",
            "background": "rgba(86,180,233,0.08)",
            "color": "#0B5E87",
        },
    )


def _build_stores() -> Component:
    """Mount every shared ``dcc.Store`` the viewer reads.

    In addition to the four session-storage stores backing the filter spec,
    image pair list, card list, and lock-views toggle, this also mounts two
    hidden trigger stores (:data:`ids.OSD_MOUNT_TRIGGER_ID` and
    :data:`ids.LOCK_VIEWS_EFFECT_ID`) used by Wave 4's clientside callbacks
    to bridge Dash state changes into the OpenSeadragon JS lifecycle.
    """
    return html.Div(
        [
            dcc.Store(
                id=ids.STORE_FILTER_SPEC,
                data=[],
                storage_type="session",
            ),
            dcc.Store(
                id=ids.STORE_IMAGE_PAIRS,
                data=[],
                storage_type="session",
            ),
            dcc.Store(
                id=ids.STORE_CARD_LIST,
                data=[],
                storage_type="session",
            ),
            dcc.Store(
                id=ids.STORE_LOCK_VIEWS,
                data=False,
                storage_type="session",
            ),
            # Clientside-callback effect targets — the data itself is a
            # millisecond timestamp used purely as a change-trigger; the
            # Python side never reads it.
            dcc.Store(id=ids.OSD_MOUNT_TRIGGER_ID, data=0),
            dcc.Store(id=ids.LOCK_VIEWS_EFFECT_ID, data=0),
        ]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_app_layout(output_root: OutputRoot) -> Component:
    """Compose the top-level Dash component tree for the results viewer.

    Mounts every shared ``dcc.Store``, the header bar, the dismissable
    startup banner, and the two-column body. Sub-trees defer to their
    owning modules (``_filter_panel`` for the sidebar; ``_viewer_card``
    for cards in Wave 3B).

    Args:
        output_root: Validated handle on the CLI output directory.

    Returns:
        A :class:`dash.html.Div` ready to assign to ``app.layout``.
    """
    header = _build_header(output_root)
    banner = _build_startup_banner(output_root)
    sidebar = _filter_panel.layout(output_root)
    cards_column = _build_cards_column()
    stores = _build_stores()

    body = dbc.Row(
        [
            dbc.Col(
                sidebar,
                width=12,
                lg=3,
                className="px-3 py-3",
                style={"background": _BG},
            ),
            dbc.Col(
                cards_column,
                width=12,
                lg=9,
                className="px-0",
                style={"background": _BG},
            ),
        ],
        className="g-0",
        style={"minHeight": "calc(100vh - 7rem)", "alignItems": "stretch"},
    )

    return html.Div(
        [stores, header, banner, body],
        id="results-viewer-root",
        style={"background": _BG, "minHeight": "100vh"},
    )


def register_callbacks(app: dash.Dash, output_root: OutputRoot) -> None:
    """Register layout-owned callbacks.

    Currently a single mirror callback that copies the ``Lock views``
    switch value into ``STORE_LOCK_VIEWS`` so downstream consumers
    (clientside OSD JS in Wave 3C) can subscribe to the store rather
    than the switch.

    Other layout-area callbacks (filter rows, card spawning, etc.)
    belong to the modules that own those component trees.

    Args:
        app: The Dash application to attach the callbacks to.
        output_root: Validated handle on the CLI output directory.
            Currently unused at this level — kept in the signature so
            Wave 4's orchestrator can call every module's
            ``register_callbacks`` with a uniform shape.
    """
    del output_root  # currently unused at the layout level

    @app.callback(
        Output(ids.STORE_LOCK_VIEWS, "data"),
        Input(ids.BTN_LOCK_VIEWS_TOGGLE, "value"),
    )
    def _mirror_lock_views(value: Any) -> bool:
        """Pass-through: switch value → store."""
        return bool(value)


__all__ = ["build_app_layout", "register_callbacks"]
