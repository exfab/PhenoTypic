"""Top-level layout shell for the results viewer.

The shell composes (left → right):

* a header bar with the app title, a one-line ``pipeline.json`` chip, a
  ``Filters`` toggle (with an active-filter count badge), a ``Lock
  views`` switch, and a monospace subtitle showing the absolute
  output-root path;
* a full-width tabbed body — a scrollable cards ``Plate`` view (with an
  ``+ Add card`` button), a per-colony ``Colony`` grid, and the ``QC``
  and ``Heatmap`` tabs; all tab bodies stay mounted so switching is a
  CSS-only operation;
* a right-docked ``dbc.Offcanvas`` (boots closed) that hosts the filter
  sidebar, opened/closed by the header ``Filters`` toggle;
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

from phenotypic.gui._config import (
    CFG_QC_RECIPE,
    COLONY_TILE_SIZE_DEFAULT,
    MOUNT_HOME,
    SSH_TUNNEL_HINT,
    TILE_DIM_DEFAULT,
)
from phenotypic.gui._schema_cache import MeasurementSchema
from phenotypic.sdk_._qc_recipe import QcRecipe
from phenotypic.gui._shared import SHARED_LOGO_PATH
from phenotypic.gui._design import (
    COLOR_BG,
    COLOR_BLUE,
    COLOR_GOLD,
    COLOR_MUTED,
    COLOR_NAVY,
    COLOR_RULE,
    COLOR_SURFACE,
    FONT_SIZE_CAPTION,
    FONT_SIZE_LABEL,
    OI_ORANGE,
    OI_ORANGE_TEXT,
)
from phenotypic.gui.results_viewer import _filter_panel, _ids as ids, colony_view
from phenotypic.gui.results_viewer._error_tab import build_error_tab_body
from phenotypic.gui.results_viewer._heatmap_tab import build_heatmap_tab_body
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._qc_tab import build_qc_tab_body
from phenotypic.gui.results_viewer.colony_view import _layout as _colony_layout  # noqa: F401
from phenotypic.gui.results_viewer.timeline_view import _layout as _timeline_layout

if TYPE_CHECKING:
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style tokens -- aliases of phenotypic.gui._design constants kept for
# call-site readability; never override the underlying hex values.
# ---------------------------------------------------------------------------

_NAVY = COLOR_NAVY
_BLUE = COLOR_BLUE
_GOLD = COLOR_GOLD
_BG = COLOR_BG


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def _build_filters_toggle() -> Component:
    """The Filters offcanvas toggle (with active-filter count badge).

    Rendered into the sticky tab-bar actions strip by
    :func:`build_app_layout` (not the header), so it rides on the tab row
    and stays pinned while tab content scrolls.
    """
    return dbc.Button(
        [
            "Filters",
            dbc.Badge(
                "",
                id=ids.FILTER_TOGGLE_BADGE_ID,
                color="primary",
                className="ms-2",
                style={"display": "none"},
            ),
        ],
        id=ids.BTN_FILTERS_TOGGLE,
        color="secondary",
        outline=True,
        size="sm",
        n_clicks=0,
    )


def build_mode_badge(output_root: OutputRoot) -> Component:
    """Build the header mode badge: "Full run" vs "Standalone bundle".

    Reads :attr:`OutputRoot.has_results` (the only attribute touched, so a
    lightweight duck-typed stand-in works in unit tests). A full
    ``python -m phenotypic`` run carries per-image ``results/`` HDFs and shows
    **Full run**; a portable, deliverables-only bundle (master + mirror +
    overlays, no ``results/``) shows **Standalone bundle** so the user knows
    the per-image pixel-layer toggle is unavailable.

    Args:
        output_root: The active output handle (or any object exposing a
            boolean ``has_results`` attribute).

    Returns:
        A pill-shaped :class:`dbc.Badge`. Colours come from
        :mod:`phenotypic.gui._design` tokens (navy for a full run; a darkened
        Okabe-Ito orange text variant for a bundle, AA-legible on white per
        DESIGN.md "05 — Badges").
    """
    full_run = bool(getattr(output_root, "has_results", False))
    if full_run:
        text = "Full run"
        fg, bg, border = COLOR_NAVY, f"{COLOR_NAVY}12", f"{COLOR_NAVY}2e"
    else:
        text = "Standalone bundle"
        fg, bg, border = OI_ORANGE_TEXT, f"{OI_ORANGE}1f", f"{OI_ORANGE}45"
    return dbc.Badge(
        text,
        id=ids.HEADER_MODE_BADGE_ID,
        className="me-3 results-viewer-mode-badge",
        style={
            "color": fg,
            "background": bg,
            "border": f"1px solid {border}",
            "borderRadius": "9999px",
            "fontSize": FONT_SIZE_CAPTION,
            "fontWeight": 500,
            "letterSpacing": "0.04em",
            "padding": "0.2rem 0.6rem",
        },
    )


def _build_header(output_root: OutputRoot, *, url_prefix: str = MOUNT_HOME) -> Component:
    """Build the top header bar.

    Args:
        output_root: Validated handle on the CLI output directory; the
            pipeline summary and root path are surfaced as info chips.
        url_prefix: Mount-point prefix used to resolve the dashboard
            logo URL. Defaults to ``MOUNT_HOME`` ("/") for standalone
            launches; the hub passes ``MOUNT_VIEWER``.

    Returns:
        A header :class:`dash.html.Div` styled as a navy-on-white bar.
    """
    pipeline_label = output_root.pipeline_summary or "unknown"

    pipeline_chip = html.Span(
        [
            html.Span(
                "Pipeline:",
                className="me-1",
                style={"color": COLOR_MUTED},
            ),
            html.Span(
                pipeline_label,
                id=ids.HEADER_PIPELINE_CHIP_ID,
                style={"color": _NAVY, "fontWeight": 500},
            ),
        ],
        className="me-3 results-viewer-pipeline-chip",
        style={
            "fontSize": FONT_SIZE_LABEL,
            "padding": "0.25rem 0.6rem",
            "border": f"1px solid {_BLUE}33",
            "borderRadius": "var(--radius)",
            "background": COLOR_SURFACE,
        },
    )

    lock_switch = dbc.Switch(
        id=ids.BTN_LOCK_VIEWS_TOGGLE,
        label="Lock views",
        value=False,
        className="ms-2 mb-0",
    )

    logo = html.Img(
        src=f"{url_prefix}{SHARED_LOGO_PATH}",
        alt="PhenoTypic",
        className="results-viewer-header__logo",
    )

    title = html.H4(
        "Results Viewer",
        className="mb-0 me-3 results-viewer-header-title",
        style={"color": _NAVY},
    )

    subtitle = html.Div(
        [
            html.Span(str(output_root.root)),
            html.Span(" · "),
            html.Span(
                "Snapshot "
                f"{output_root.snapshot.captured_at.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}"
            ),
            html.Span(" · "),
            html.Code(
                output_root.snapshot.processing_fingerprint[:12],
                title=output_root.snapshot.processing_fingerprint,
            ),
            dbc.Badge(
                (
                    "Active run snapshot"
                    if output_root.snapshot.active_run
                    else "Current"
                ),
                id=ids.HEADER_SNAPSHOT_STATUS_ID,
                color=(
                    "warning"
                    if output_root.snapshot.active_run
                    else "success"
                ),
                className="ms-2",
            ),
            dbc.Button(
                "Refresh snapshot",
                id=ids.BTN_REFRESH_SNAPSHOT,
                color="secondary",
                outline=True,
                size="sm",
                n_clicks=0,
                disabled=output_root.snapshot.active_run,
                className="ms-2",
            ),
            html.Span(
                id=ids.HEADER_REFRESH_ERROR_ID,
                className="text-danger ms-2",
            ),
        ],
        className=(
            "text-muted small results-viewer-header-subtitle "
            "d-flex align-items-center flex-wrap gap-1"
        ),
        style={"marginTop": "0.1rem"},
    )

    top_row = html.Div(
        [
            logo,
            title,
            pipeline_chip,
            build_mode_badge(output_root),
            html.Div(style={"flex": "1 1 auto"}),  # spacer
            lock_switch,
        ],
        className="d-flex align-items-center",
    )

    return html.Div(
        [top_row, subtitle],
        className="results-viewer-header px-3 py-2",
        style={
            "background": COLOR_SURFACE,
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
                f"Forward this port over SSH with `{SSH_TUNNEL_HINT}` and "
                "open the URL in a local browser. Stale tiles? Nuke the "
                "DZI cache with: ",
            ),
            html.Code(
                f"rm -rf {cache_dir}",
                style={
                    "background": COLOR_RULE,
                    "color": _NAVY,
                    "padding": "1px 5px",
                    "borderRadius": "var(--radius-sm)",
                },
            ),
        ],
        id="results-viewer-startup-banner",
        color="info",
        dismissable=True,
        is_open=True,
        className="mx-3 mt-2 mb-0 small",
        style={
            "borderLeft": f"4px solid {COLOR_BLUE}",
            "background": "rgba(27,117,188,0.08)",
            "color": COLOR_BLUE,
        },
    )


def _build_stores(filtered_state: "CurationLabels") -> Component:
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
            dcc.Store(id=ids.COLONY_SELECTION_EFFECT_ID, data=0),
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
                id=ids.STORE_PLOT_REFRESH_REVISION,
                data=0,
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
            # Category-vocabulary revision ticker — bumped whenever a custom
            # category is registered (Task 7) so the bulk-mark dropdowns and
            # open radial wheels refresh their options/body. Memory storage:
            # the registry json is the cross-session source of truth.
            dcc.Store(
                id=ids.STORE_CATEGORY_VOCAB_REVISION,
                data=0,
                storage_type="memory",
            ),
            dcc.Store(
                id=ids.STORE_COLONY_TILE_SIZE,
                data=COLONY_TILE_SIZE_DEFAULT,
                storage_type="memory",
            ),
            # Tile-spotlight ``dim`` strength shared by the colony-view and
            # QC-Review galleries' ``−``/``+`` steppers. ``local`` storage
            # so the chosen strength survives a full page reload (the
            # effect is on by default at ``TILE_DIM_DEFAULT``).
            dcc.Store(
                id=ids.STORE_TILE_DIM_ALPHA,
                data=TILE_DIM_DEFAULT,
                storage_type="local",
            ),
            # QC revision tickers - mounted by Wave D so the Heatmap
            # tab's callbacks can subscribe before Wave E ships the QC
            # tab proper. Wave E mutates these stores from the QC card
            # callbacks; until then they stay at 0.
            dcc.Store(
                id=ids.STORE_QC_RECIPE_REVISION,
                data=0,
                storage_type="memory",
            ),
            dcc.Store(
                id=ids.STORE_QC_AUGMENTED_REVISION,
                data=0,
                storage_type="memory",
            ),
        ]
    )


def _resolve_measurement_schema(output_root: OutputRoot) -> MeasurementSchema:
    """Return the measurement-schema cache for the active output root.

    Created at layout build time and intentionally NOT stashed on
    ``app.server.config`` here - the schema is read by callbacks that
    pull it from the config directly (see
    :func:`._heatmap_tab._callbacks._refresh_heatmap_controls`); the
    Dash app factory in :mod:`._app` is responsible for the stash so
    construction stays a layout-time concern.
    """
    return MeasurementSchema.from_layout(output_root.layout)


def _resolve_qc_recipe(output_root: OutputRoot) -> QcRecipe:
    """Return the QC recipe for the active output root.

    Prefer the app-config-stashed instance (set by :func:`._app.create_app`)
    so layout and callbacks share the same in-memory object. Falls back
    to a fresh :meth:`QcRecipe.load` for tests or standalone callers that
    invoke :func:`build_app_layout` without the app factory.
    """
    try:
        from flask import current_app

        recipe = current_app.config.get(CFG_QC_RECIPE)
        if isinstance(recipe, QcRecipe):
            return recipe
    except RuntimeError:
        pass  # No application context (test harness, etc.).
    return QcRecipe.from_layout(output_root.layout)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_app_layout(
    output_root: OutputRoot,
    filtered_state: "CurationLabels",
    *,
    url_prefix: str = MOUNT_HOME,
    binding_generation: str | None = None,
) -> Component:
    """Compose the top-level Dash component tree for the results viewer.

    Mounts every shared ``dcc.Store``, the header bar, the dismissable
    startup banner, a full-width :class:`dbc.Tabs` body (``Plate`` cards,
    per-colony ``Colony`` grid, ``QC``, ``Heatmap`` — all kept mounted so
    switching is a CSS-only operation with no subtree re-render), and a
    right-docked :class:`dbc.Offcanvas` that hosts the filter sidebar and
    boots closed (opened from the header ``Filters`` toggle). Sub-trees
    defer to their owning modules (``_filter_panel`` for the sidebar;
    ``_viewer_card`` for cards; ``colony_view._layout`` for the grid).

    Args:
        output_root: Validated handle on the CLI output directory.
        filtered_state: On-disk curation state, used to seed
            :data:`ids.STORE_REMOVED_KEYS` at boot so the colony view
            reflects existing manual curation.
        url_prefix: Mount-point prefix passed through to
            :func:`_build_header` so the dashboard logo resolves
            correctly under both standalone and hub-mounted launches.
        binding_generation: Optional shell generation embedded in the page.

    Returns:
        A :class:`dash.html.Div` ready to assign to ``app.layout``.
    """
    header = _build_header(output_root, url_prefix=url_prefix)
    banner = _build_startup_banner(output_root)
    sidebar = _filter_panel.layout(output_root)
    cards_column = _build_cards_column()
    colony_tab_body = colony_view._layout.layout(output_root)

    # Heatmap tab uses the measurement-schema cache; lazily attach it to
    # ``app.server.config`` here if ``create_app`` did not. The
    # analysis sub-app already stashes one when mounted; reusing it
    # keeps the cache hits warm across tabs.
    schema = _resolve_measurement_schema(output_root)
    heatmap_tab_body = build_heatmap_tab_body(output_root, schema)
    error_tab_body = build_error_tab_body(output_root, schema)
    qc_tab_body = build_qc_tab_body(_resolve_qc_recipe(output_root))
    timeline_tab_body = _timeline_layout.layout(output_root)
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
            dbc.Tab(
                qc_tab_body,
                label="QC",
                tab_id=ids.TAB_QC_ID,
            ),
            dbc.Tab(
                heatmap_tab_body,
                label="Heatmap",
                tab_id=ids.TAB_HEATMAP_ID,
            ),
            dbc.Tab(
                error_tab_body,
                label="Error",
                tab_id=ids.TAB_ERROR_ID,
            ),
            dbc.Tab(
                timeline_tab_body,
                label="Timeline",
                tab_id=ids.TAB_TIMELINE_ID,
            ),
        ],
        id=ids.TABS_ID,
        active_tab=ids.TAB_PLATE_ID,
    )

    tabbar = html.Div(
        [
            html.Div(
                _build_filters_toggle(),
                className="results-viewer-tabbar__actions",
            ),
            tabs,
        ],
        className="results-viewer-tabbar",
    )

    body = html.Div(
        tabbar,
        className="results-viewer-body",
        style={
            "background": _BG,
            "minHeight": "calc(100vh - 7rem)",
        },
    )

    # The filter panel now lives in a right-docked offcanvas opened from
    # the top-bar Filters toggle, so every tab renders full-width by
    # default and filtering stays one click away.
    filter_offcanvas = dbc.Offcanvas(
        sidebar,
        id=ids.OFFCANVAS_FILTER_ID,
        title="Filter",
        placement="end",
        is_open=False,
        scrollable=True,
        backdrop=True,
    )

    children: list[Component] = [
        stores,
        dcc.Interval(
            id=ids.SNAPSHOT_STATUS_INTERVAL_ID,
            interval=10_000,
            n_intervals=0,
        ),
        header,
        banner,
        body,
        filter_offcanvas,
    ]
    if binding_generation is not None:
        children.insert(
            0,
            dcc.Store(
                id=ids.STORE_BINDING_GENERATION,
                data=binding_generation,
            ),
        )
    return html.Div(
        children,
        id="results-viewer-root",
        style={"background": _BG, "minHeight": "100vh"},
    )


def build_active_snapshot_layout(
    output_root: OutputRoot,
    *,
    url_prefix: str = MOUNT_HOME,
    binding_generation: str | None = None,
) -> Component:
    """Build a mutation-free placeholder for a nonterminal output."""
    children: list[Component] = [
        dcc.Interval(
            id=ids.SNAPSHOT_STATUS_INTERVAL_ID,
            interval=5_000,
            n_intervals=0,
        ),
        _build_header(output_root, url_prefix=url_prefix),
        dbc.Alert(
            [
                html.H5("Processing output is read-only"),
                html.P(
                    "This output still has a nonterminal run owner. Results, "
                    "curation, and QC callbacks are not loaded for this page."
                ),
                html.P(
                    "When processing finishes, use Refresh snapshot to load a "
                    "stable Results and Analysis revision."
                ),
            ],
            color="warning",
            className="m-4",
        ),
    ]
    if binding_generation is not None:
        children.insert(
            0,
            dcc.Store(
                id=ids.STORE_BINDING_GENERATION,
                data=binding_generation,
            ),
        )
    return html.Div(
        children,
        id="results-viewer-active-snapshot",
        style={"background": _BG, "minHeight": "100vh"},
    )


def build_empty_state_layout(
    *,
    binding_generation: str | None = None,
) -> Component:
    """Compose a placeholder layout when no ``OutputRoot`` is available.

    The hub mounts the viewer with ``output_root=None`` so the page is
    reachable before the user has chosen an output directory. This layout
    explains the situation and renders a hand-off banner that consumes
    :data:`SHELL_SIDEBAR_SELECTION_STORE` from the wrapping shell chrome.
    Clicking ``Open in viewer`` POSTs the selection to
    ``/sandbox/api/viewer/output-root``, which validates the layout via
    :meth:`OutputRoot.discover`, swaps the viewer ``ToolSession`` state,
    and triggers a hard navigation back to ``/results/`` so the
    :class:`_ViewerProxy` resolves a freshly-built loaded viewer.

    Returns:
        A :class:`dash.html.Div` that renders a friendly message, the
        hand-off banner, and an inline error slot. The successful-POST
        redirect is performed by the empty-state clientside callback in
        :mod:`._app` via ``window.location.assign(url_prefix)``.
    """
    handoff_banner = html.Div(
        [
            html.Span(
                "Selected: ",
                className="results-viewer-empty-handoff-prefix",
            ),
            html.Code(
                "(none)",
                id=ids.EMPTY_HANDOFF_LABEL,
                className="results-viewer-empty-handoff-label",
            ),
            dbc.Button(
                "↩ Open in viewer",
                id=ids.EMPTY_HANDOFF_OPEN_BUTTON,
                color="primary",
                size="sm",
                disabled=True,
                className="results-viewer-empty-handoff-open ms-2",
                n_clicks=0,
            ),
        ],
        id=ids.EMPTY_HANDOFF_BANNER,
        className="results-viewer-empty-handoff-banner",
        style={
            "display": "none",
            "alignItems": "center",
            "gap": "0.5rem",
            "marginTop": "1rem",
            "padding": "0.5rem 0.75rem",
            "background": COLOR_SURFACE,
            "border": f"1px solid {_BLUE}",
            "borderRadius": "var(--radius)",
        },
    )

    error_slot = html.Div(
        "",
        id=ids.EMPTY_HANDOFF_ERROR,
        className="results-viewer-empty-handoff-error text-danger small",
        style={"marginTop": "0.5rem", "minHeight": "1.25rem"},
    )

    children: list[Component] = [
            html.Div(
                [
                    html.H2(
                        "No output selected",
                        className="results-viewer-empty-title",
                    ),
                    html.P(
                        "Pick a CLI output directory in the sidebar, "
                        "then click ↩ Open in viewer to load it. The "
                        "viewer rebuilds in place once the chosen "
                        "directory passes layout validation.",
                        className="results-viewer-empty-body",
                    ),
                    handoff_banner,
                    error_slot,
                ],
                className="results-viewer-empty-card",
            ),
        ]
    if binding_generation is not None:
        children.insert(
            0,
            dcc.Store(
                id=ids.STORE_BINDING_GENERATION,
                data=binding_generation,
            ),
        )
    return html.Div(
        children,
        id="results-viewer-empty-state",
        className="results-viewer-empty",
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "minHeight": "calc(100vh - 7rem)",
            "background": _BG,
            "padding": "2rem",
        },
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


__all__ = ["build_app_layout", "build_mode_badge", "register_callbacks"]
