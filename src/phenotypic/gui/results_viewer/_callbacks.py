"""Callback orchestrator for the results viewer.

This module wires three pieces together:

1. **Per-module Python callbacks.** Each layout module
   (:mod:`._layout`, :mod:`._filter_panel`, :mod:`._viewer_card`) owns
   the callbacks for its own component subtree and exposes a
   ``register_callbacks(app, output_root)`` entry point. The
   orchestrator imports them and dispatches in a fixed order so the
   registration is idempotent and obvious to future maintainers.

2. **Clientside bridge: source spec -> Viv mount.** Each card's
   :data:`~phenotypic.gui.results_viewer._ids.card_source_store_id`
   ``dcc.Store`` holds ``build_source_spec``'s dict for the selected
   image's OME-Zarr store, and its
   :data:`~phenotypic.gui.results_viewer._ids.card_display_state_id`
   store holds what the Layers panel has been set to. A clientside
   callback subscribes to **every** such store via pattern-matching
   ``ALL`` (reading both ``data`` and ``id``) and forwards a
   homogeneous list of records to
   ``window.__phenotypicResultsViewer.applyPlateSources``, which drives
   ``window.phenotypicViv``. **The spec crosses unmodified** -- it is
   built at the facade's own key names server-side, so nothing here
   re-packs it and no second vocabulary exists to drift.

3. **Clientside bridge: lock-views toggle -> viewport broadcast.** A
   second clientside callback subscribes to
   :data:`~phenotypic.gui.results_viewer._ids.STORE_LOCK_VIEWS` and
   forwards changes to
   ``window.__phenotypicResultsViewer.setLockViews`` so the JS
   viewer registry can attach/detach its viewport-broadcast handlers.

Both clientside callbacks write into hidden trigger ``dcc.Store``
instances (:data:`~phenotypic.gui.results_viewer._ids
.VIV_MOUNT_TRIGGER_ID`,
:data:`~phenotypic.gui.results_viewer._ids.LOCK_VIEWS_EFFECT_ID`)
mounted by :mod:`._layout`. The trigger-store payload is a
millisecond timestamp solely used to satisfy Dash's "every callback
must have an Output" contract; nothing on the Python side ever reads
those stores.

Pattern-matching id serialization
---------------------------------
Dash renders pattern-matching component ids into the DOM ``id``
attribute as a JSON string with **alphabetically sorted keys**.
Concretely, a Python dict ``{"type": "card-viv-stage", "index":
"abc"}`` becomes the DOM id string
``'{"index":"abc","type":"card-viv-stage"}'``. The clientside
callback below mirrors that ordering so the JS-side
``document.getElementById`` lookups succeed.
"""

from __future__ import annotations

import logging
from typing import Any

import dash
from dash import ALL, Input, Output, State, no_update

from phenotypic.gui._config import (
    CFG_FILTERED_STATE,
    CFG_QC_PIPELINE,
)
from phenotypic.gui.results_viewer import (
    _filter_offcanvas,
    _filter_panel,
    _ids as ids,
    _layout,
    _viewer_card,
)
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._filtered_state import get_curated_frame
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._mutation_guard import (
    OutputMutationBlocked,
    require_output_mutation,
)
from phenotypic.gui.results_viewer._scatter_tab import (
    _callbacks as _scatter_callbacks,
)
from phenotypic.gui.results_viewer.colony_view import (
    _callbacks as _colony_callbacks,
)

logger = logging.getLogger(__name__)


def register_callbacks(app: dash.Dash, output_root: OutputRoot) -> None:
    """Register every callback the results viewer needs on *app*.

    Dispatches to each layout module's ``register_callbacks`` in a
    deterministic order, then attaches the two clientside callbacks
    that bridge Dash state to the OpenSeadragon JS layer. The colony
    view's curation callbacks receive the shared
    :class:`CurationLabels` instance pulled off
    ``app.server.config["filtered_state"]`` (seeded by
    :func:`._app.create_app`).

    Args:
        app: The Dash application owning the viewer's layout.
        output_root: Validated handle on the CLI output directory; passed
            by closure to every per-module callback that needs to slice
            ``master_df`` or resolve overlay paths.
    """
    filtered_state: CurationLabels = app.server.config[CFG_FILTERED_STATE]
    _layout.register_callbacks(app, output_root)
    _filter_panel.register_callbacks(app, output_root, filtered_state)
    _filter_offcanvas.register_filter_offcanvas_callbacks(app)
    _viewer_card.register_callbacks(app, output_root)
    _colony_callbacks.register_callbacks(app, output_root, filtered_state)
    _scatter_callbacks.register_callbacks(app, output_root)
    _register_plot_refresh_callback(app, output_root, filtered_state)
    _register_clientside_callbacks(app)


def _register_plot_refresh_callback(
    app: dash.Dash,
    output_root: OutputRoot,
    filtered_state: CurationLabels,
) -> None:
    """Refresh configured ``PlotMeas`` outputs after each GUI curation write."""

    @app.callback(
        Output(ids.STORE_PLOT_REFRESH_REVISION, "data"),
        Input(ids.STORE_REMOVED_KEYS, "data"),
        State(ids.STORE_PLOT_REFRESH_REVISION, "data"),
        prevent_initial_call=True,
    )
    def _refresh_measurement_plots(
        _removed_keys: list | None,
        revision: int | None,
    ) -> int | Any:
        pipeline = app.server.config.get(CFG_QC_PIPELINE)
        if pipeline is None:
            return no_update
        try:
            require_output_mutation("Measurement plot refresh")
            from phenotypic.gui._plot_refresh import refresh_measurement_plots

            measurements = get_curated_frame(
                filtered_state,
                output_root,
            ).to_pandas()
            refresh_measurement_plots(
                pipeline,
                output_root.layout,
                measurements,
                publication_guard=lambda: _output_publication_is_safe(
                    "Measurement plot refresh"
                ),
            )
        except OutputMutationBlocked as exc:
            logger.warning("%s", exc)
            return no_update
        except Exception:  # noqa: BLE001 - curation remains authoritative
            logger.warning(
                "GUI measurement plot refresh failed after curation",
                exc_info=True,
            )
            return no_update
        return (revision or 0) + 1


def _output_publication_is_safe(action: str) -> bool:
    """Reauthorize immediately before a Results artifact replacement."""
    try:
        require_output_mutation(action)
    except OutputMutationBlocked:
        return False
    return True


def _register_clientside_callbacks(app: dash.Dash) -> None:
    """Wire the two clientside callbacks bridging Dash to the Viv facade.

    Args:
        app: The Dash application that will own the clientside callbacks.
    """
    # ----------------------------------------------------------------------
    # Source spec + display state -> Viv mount / setSource / layer controls
    # ----------------------------------------------------------------------
    # Subscribes to every card's source-spec and display-state stores
    # (pattern-matching ALL) and reads the matching ``id`` objects so the JS
    # handler can address the card's stage, pyramid readout and zoom readout
    # elements.
    #
    # The DOM id strings are built using the same JSON serialization Dash
    # uses internally for pattern-matching ids -- keys sorted alphabetically.
    app.clientside_callback(
        """
        function(specList, displayList, idList, _cardList) {
            const ns = window.__phenotypicResultsViewer;
            if (!ns || !ns.applyPlateSources) {
                return window.dash_clientside.no_update;
            }
            const domId = function (type, index) {
                // Dash renders pattern-matching ids as JSON strings with
                // keys sorted alphabetically. Mirror that exactly.
                return JSON.stringify({index: index, type: type});
            };
            const states = (specList || []).map(function (spec, i) {
                const idObj = (idList && idList[i]) || {};
                return {
                    id: domId("card-viv-stage", idObj.index),
                    levelReadoutId: domId(
                        "card-pyramid-readout", idObj.index
                    ),
                    zoomReadoutId: domId("card-zoom-readout", idObj.index),
                    // Crosses UNMODIFIED: `build_source_spec` already
                    // returns the facade's own key names.
                    spec: spec || null,
                    display: (displayList && displayList[i]) || null
                };
            });
            // Defer one frame so the DOM has finished re-rendering after a
            // cards-container update; otherwise a fresh stage div can be
            // missed by getElementById.
            requestAnimationFrame(function () {
                ns.applyPlateSources(states);
            });
            return Date.now();
        }
        """,
        Output(ids.VIV_MOUNT_TRIGGER_ID, "data"),
        Input({"type": "card-source-spec", "index": ALL}, "data"),
        Input({"type": "card-display-state", "index": ALL}, "data"),
        Input({"type": "card-source-spec", "index": ALL}, "id"),
        Input(ids.STORE_CARD_LIST, "data"),
    )

    # ----------------------------------------------------------------------
    # Lock-views toggle -> viewport broadcast attach/detach
    # ----------------------------------------------------------------------
    # Reads the boolean STORE_LOCK_VIEWS and pokes
    # ``window.__phenotypicResultsViewer.setLockViews`` so the plate registry
    # starts or stops mirroring one stage's view state onto its peers.
    app.clientside_callback(
        """
        function(active) {
            const ns = window.__phenotypicResultsViewer;
            if (!ns || !ns.setLockViews) {
                return window.dash_clientside.no_update;
            }
            ns.setLockViews(!!active);
            return Date.now();
        }
        """,
        Output(ids.LOCK_VIEWS_EFFECT_ID, "data"),
        Input(ids.STORE_LOCK_VIEWS, "data"),
    )


__all__ = ["register_callbacks"]
