"""Top-level layout shell for the results viewer.

The shell composes (left → right):

* a header bar with the app title, a one-line ``pipeline.json`` chip, a
  ``Lock views`` switch, and a monospace subtitle showing the absolute
  output-root path;
* a two-column body: the filter sidebar (left) and a tabbed right
  column with two tabs — a scrollable cards ``Plate`` view (with an
  ``+ Add card`` button) and a per-colony ``Colony`` grid view; both
  tab bodies stay mounted so switching is a CSS-only operation;
* every shared ``dcc.Store`` instance the rest of the viewer reads
  (filter spec, image pairs, card list, lock-views, plus the colony-
  view curation, selection-delta, and grid-order stores).

Layout-level callbacks are minimal — only the ``Lock views`` switch
mirror — because the heavy callbacks live in the modules that own each
sub-tree (``_filter_panel`` for the sidebar, ``_viewer_card`` for the
cards, ``colony_view._callbacks`` for the colony grid). This keeps each
module self-contained and the top-level ``_callbacks.register_callbacks``
ends up as a thin orchestrator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import Input, Output, dcc, html
from dash.development.base_component import Component

from phenotypic.gui.results_viewer import _filter_panel, _ids as ids, colony_view
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.colony_view import _layout as _colony_layout  # noqa: F401

if TYPE_CHECKING:
    from phenotypic.gui.results_viewer._filtered_state import FilteredMeasurements

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
    """Build the right-side cards column (container + add-card button)."""
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


def _build_stores(filtered_state: "FilteredMeasurements") -> Component:
    """Mount every shared ``dcc.Store`` the viewer reads.

    In addition to the four session-storage stores backing the filter spec,
    image pair list, card list, and lock-views toggle, this also mounts two
    hidden trigger stores (:data:`ids.OSD_MOUNT_TRIGGER_ID` and
    :data:`ids.LOCK_VIEWS_EFFECT_ID`) used by the clientside callbacks
    to bridge Dash state changes into the OpenSeadragon JS lifecycle, plus
    four memory-storage stores backing the colony-view curation and
    multi-select state (removed keys, current selection, selection delta,
    and visual grid order).

    Args:
        filtered_state: The on-disk curation state loaded by
            :func:`phenotypic.gui.results_viewer._app.create_app`. Used to
            seed :data:`ids.STORE_REMOVED_KEYS` so the UI reflects existing
            curation at boot.

    Returns:
        A :class:`dash.html.Div` wrapping every ``dcc.Store``.
    """
    initial_removed_keys = filtered_state.removed_keys_payload()

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
            # Colony-view curation + selection stores. Memory-storage so
            # the curation/selection state survives tab switches but not
            # full page reloads (the on-disk parquet mirror is the
            # source of truth across sessions; the store is rehydrated
            # from it at every boot via ``filtered_state``).
            dcc.Store(
                id=ids.STORE_REMOVED_KEYS,
                data=initial_removed_keys,
                storage_type="memory",
            ),
            dcc.Store(
                id=ids.STORE_COLONY_SELECTION,
                data={"anchor": None, "selected": []},
                storage_type="memory",
            ),
            dcc.Store(
                id=ids.STORE_COLONY_SELECTION_DELTA,
                data=None,
                storage_type="memory",
            ),
            dcc.Store(
                id=ids.STORE_COLONY_GRID_ORDER,
                data=[],
                storage_type="memory",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_app_layout(
    output_root: OutputRoot,
    filtered_state: "FilteredMeasurements",
) -> Component:
    """Compose the top-level Dash component tree for the results viewer.

    Mounts every shared ``dcc.Store``, the header bar, the dismissable
    startup banner, and the two-column body. The right-hand column is
    a :class:`dbc.Tabs` switching between the existing per-image
    ``Plate`` view (cards) and the per-colony ``Colony`` grid view; both
    tab bodies stay mounted at all times so switching is a CSS-only
    operation (no callback re-render of either subtree). Sub-trees defer
    to their owning modules (``_filter_panel`` for the sidebar;
    ``_viewer_card`` for cards; ``colony_view._layout`` for the grid).

    Args:
        output_root: Validated handle on the CLI output directory.
        filtered_state: On-disk curation state, used to seed
            :data:`ids.STORE_REMOVED_KEYS` at boot so the colony view
            reflects existing manual curation.

    Returns:
        A :class:`dash.html.Div` ready to assign to ``app.layout``.
    """
    header = _build_header(output_root)
    banner = _build_startup_banner(output_root)
    sidebar = _filter_panel.layout(output_root)
    cards_column = _build_cards_column()
    colony_tab_body = colony_view._layout.layout(output_root)
    stores = _build_stores(filtered_state)

    tabs = dbc.Tabs(
        [
            dbc.Tab(
                cards_column,
                label="Plate",
                tab_id=ids.TAB_PLATE_ID,
            ),
            dbc.Tab(
                colony_tab_body,
                label="Colony",
                tab_id=ids.TAB_COLONY_ID,
            ),
        ],
        id=ids.TABS_ID,
        active_tab=ids.TAB_PLATE_ID,
    )

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
                tabs,
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

    Bridges the ``Lock views`` ``dbc.Switch`` to ``STORE_LOCK_VIEWS``.
    The clientside callback in :mod:`._callbacks` subscribes to a Store
    rather than the Switch so Dash's pattern-matching can include the
    boolean alongside other state without coupling to component types.

    Other layout-area callbacks (filter rows, card spawning, etc.)
    belong to the modules that own those component trees.

    Args:
        app: The Dash application to attach the callbacks to.
        output_root: Validated handle on the CLI output directory.
            Unused at this level; kept in the signature so the
            orchestrator can call every module's ``register_callbacks``
            with a uniform shape.
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
