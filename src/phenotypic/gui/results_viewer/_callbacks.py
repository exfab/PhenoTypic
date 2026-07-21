"""Callback orchestrator for the results viewer.

This module wires three pieces together:

1. **Per-module Python callbacks.** Each layout module
   (:mod:`._layout`, :mod:`._filter_panel`, :mod:`._viewer_card`) owns
   the callbacks for its own component subtree and exposes a
   ``register_callbacks(app, output_root)`` entry point. The
   orchestrator imports them and dispatches in a fixed order so the
   registration is idempotent and obvious to future maintainers.

2. **Clientside bridge: image selection -> OSD mount.** Each card
   stores its ``(dataset, stem)`` selection in a per-card
   :data:`~phenotypic.gui.results_viewer._ids.card_state_store_id`
   ``dcc.Store``. A clientside callback subscribes to **every** such
   store via pattern-matching ``ALL`` (reading both ``data`` and
   ``id``) and forwards a homogeneous list of
   ``{id, dataset, stem}`` records to
   ``window.__phenotypicResultsViewer.applyImageSelection``. The JS
   layer then mounts/disposes OpenSeadragon viewers as needed.

3. **Clientside bridge: lock-views toggle -> OSD broadcast.** A
   second clientside callback subscribes to
   :data:`~phenotypic.gui.results_viewer._ids.STORE_LOCK_VIEWS` and
   forwards changes to
   ``window.__phenotypicResultsViewer.setLockViews`` so the JS
   viewer registry can attach/detach its viewport-broadcast handlers.

Both clientside callbacks write into hidden trigger ``dcc.Store``
instances (:data:`~phenotypic.gui.results_viewer._ids
.OSD_MOUNT_TRIGGER_ID`,
:data:`~phenotypic.gui.results_viewer._ids.LOCK_VIEWS_EFFECT_ID`)
mounted by :mod:`._layout`. The trigger-store payload is a
millisecond timestamp solely used to satisfy Dash's "every callback
must have an Output" contract; nothing on the Python side ever reads
those stores.

Pattern-matching id serialization
---------------------------------
Dash renders pattern-matching component ids into the DOM ``id``
attribute as a JSON string with **alphabetically sorted keys**.
Concretely, a Python dict ``{"type": "card-osd-div", "index":
"abc"}`` becomes the DOM id string
``'{"index":"abc","type":"card-osd-div"}'``. The clientside
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
from phenotypic.gui.results_viewer._error_tab import register_error_callbacks
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._heatmap_tab import register_heatmap_callbacks
from phenotypic.gui.results_viewer._qc_tab import register_qc_callbacks
from phenotypic.gui.results_viewer.colony_view import (
    _callbacks as _colony_callbacks,
)
from phenotypic.gui.results_viewer.timeline_view import (
    _callbacks as _timeline_callbacks,
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
    register_heatmap_callbacks(app)
    register_qc_callbacks(app)
    register_error_callbacks(app, output_root, filtered_state)
    _timeline_callbacks.register_callbacks(app, output_root)
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
            from phenotypic.gui._plot_refresh import refresh_measurement_plots

            measurements = get_curated_frame(
                filtered_state,
                output_root,
            ).to_pandas()
            refresh_measurement_plots(
                pipeline,
                output_root.layout,
                measurements,
            )
        except Exception:  # noqa: BLE001 - curation remains authoritative
            logger.warning(
                "GUI measurement plot refresh failed after curation",
                exc_info=True,
            )
            return no_update
        return (revision or 0) + 1


def _register_clientside_callbacks(app: dash.Dash) -> None:
    """Wire the two clientside callbacks bridging Dash to OpenSeadragon.

    Args:
        app: The Dash application that will own the clientside callbacks.
    """
    # ----------------------------------------------------------------------
    # Card-state -> OSD mount/dispose
    # ----------------------------------------------------------------------
    # Subscribes to every card's ``card-state`` store (pattern-matching
    # ALL) and reads both the ``data`` payload and the matching ``id``
    # objects so the JS handler can map state -> the corresponding
    # ``card-osd-div`` element id.
    #
    # The DOM id string is built using the same JSON serialization Dash
    # uses internally for pattern-matching ids, namely
    # ``JSON.stringify({"index": idx, "type": "card-osd-div"})`` (keys
    # sorted alphabetically).
    app.clientside_callback(
        """
        function(stateList, idList, _cardList) {
            const ns = window.__phenotypicResultsViewer;
            if (!ns || !ns.applyImageSelection) {
                return window.dash_clientside.no_update;
            }
            // Defer one frame so the DOM has finished re-rendering after
            // a cards-container update; otherwise a fresh osd-canvas div
            // can be missed by getElementById.
            const states = (stateList || []).map(function (s, i) {
                const idObj = (idList && idList[i]) || {};
                // Dash renders pattern-matching ids as JSON strings with
                // keys sorted alphabetically. Mirror that exactly.
                const divId = JSON.stringify({
                    index: idObj.index,
                    type: "card-osd-div"
                });
                if (!s) {
                    return {id: divId, dataset: null, stem: null};
                }
                return {
                    id: divId,
                    dataset: s.dataset || null,
                    stem: s.stem || null
                };
            });
            requestAnimationFrame(function () {
                ns.applyImageSelection(states);
            });
            return Date.now();
        }
        """,
        Output(ids.OSD_MOUNT_TRIGGER_ID, "data"),
        Input({"type": "card-state", "index": ALL}, "data"),
        Input({"type": "card-state", "index": ALL}, "id"),
        Input(ids.STORE_CARD_LIST, "data"),
    )

    # ----------------------------------------------------------------------
    # Lock-views toggle -> JS broadcast attach/detach
    # ----------------------------------------------------------------------
    # Reads the boolean STORE_LOCK_VIEWS and pokes
    # ``window.__phenotypicResultsViewer.setLockViews`` so the JS viewer
    # registry attaches/detaches its cross-viewer pan/zoom handlers.
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
