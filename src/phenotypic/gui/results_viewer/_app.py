"""Dash app factory for the results viewer.

Builds a configured :class:`dash.Dash` instance with its tile-serving
and colony-crop Flask blueprints mounted, the validated
:class:`~phenotypic.gui.results_viewer._output_root.OutputRoot` plus the
curation-state
:class:`~phenotypic.gui.results_viewer._curation_labels.CurationLabels`
stashed on ``app.server.config``, the layout assembled by
:func:`~phenotypic.gui.results_viewer._layout.build_app_layout`, and
all callbacks registered via
:func:`~phenotypic.gui.results_viewer._callbacks.register_callbacks`.

Phase 5 additions:
    * Optional ``output_root`` — when ``None`` the factory returns a
      Dash app whose layout is :func:`._layout.build_empty_state_layout`
      and which has NO blueprints, NO callbacks, and NO ``filtered_state``
      on ``app.server.config``. The hub uses this path to reach the
      ``/results/`` page before the user has selected a CLI output.
    * ``url_prefix`` — Mount-point prefix passed to ``dash.Dash`` as
      both ``requests_pathname_prefix`` and ``routes_pathname_prefix``,
      and stashed on ``app.server.config["pheno_url_prefix"]`` so
      callbacks (notably the colony-grid crop URLs) can construct
      hub-aware URLs at request time.
    * ``window.__phenotypicAppPrefix`` — injected via
      ``app.index_string`` so ``results_viewer.js`` can build DZI tile
      URLs that include the mount prefix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import Input, Output, State

from phenotypic.gui._config import (
    CFG_FILTERED_STATE,
    CFG_MEASUREMENT_SCHEMA,
    CFG_OPERATION_REGISTRY,
    CFG_OUTPUT_ROOT,
    CFG_QC_AUGMENTED_FRAME,
    CFG_QC_INSTANCES_CACHE,
    CFG_QC_PIPELINE,
    CFG_QC_RECIPE,
    CFG_URL_PREFIX,
    DEFAULT_URL_PREFIX,
    MOUNT_HOME,
    QC_CROPS_URL_SEGMENT,
    SANDBOX_API_VIEWER_OUTPUT_ROOT,
    TITLE_VIEWER,
    join_url_prefix,
)
from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui._schema_cache import MeasurementSchema
from phenotypic.gui._design import COLOR_BLUE, COLOR_SURFACE, inject_design_tokens
from phenotypic.gui._shared import register_shared_static
from phenotypic.gui._shared.tiles import register_crop_route
from phenotypic.gui._url_prefix import (
    configure_url_prefix_routing,
    dash_index_string_with_app_prefix,
)
from phenotypic.gui.results_viewer import _ids as ids, _tile_routes
from phenotypic.gui.results_viewer._callbacks import register_callbacks
from phenotypic.gui.results_viewer.timeline_view import (
    _thumb_routes as timeline_thumb_routes,
)
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._layout import (
    build_app_layout,
    build_empty_state_layout,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer.colony_view import _crop_routes as colony_crop_routes
from phenotypic.gui.shell._ids import SHELL_SIDEBAR_SELECTION_STORE
from phenotypic.sdk_._qc_recipe import QcRecipe

logger = logging.getLogger(__name__)


def create_app(
    output_root: Optional[OutputRoot] = None,
    *,
    url_prefix: str = MOUNT_HOME,
    api_url_prefix: str = DEFAULT_URL_PREFIX,
) -> dash.Dash:
    """Build a Dash application instance for the results viewer.

    Args:
        output_root: Validated, read-only handle on a CLI output
            directory (see
            :meth:`phenotypic.gui.results_viewer._output_root.OutputRoot.discover`).
            ``None`` triggers the empty-state pathway: the factory skips
            blueprint registration, ``FilteredMeasurements.load``, and
            callback registration; ``app.layout`` is the empty-state
            placeholder.
        url_prefix: Mount-point prefix. Defaults to ``"/"`` (standalone
            launcher); the hub composer passes ``"/results/"``. Set as
            ``requests_pathname_prefix``/``routes_pathname_prefix`` on
            the Dash constructor and stashed on
            ``app.server.config["pheno_url_prefix"]``.
        api_url_prefix: Browser-visible base prefix for shell-level
            Flask APIs. Defaults to ``"/"``; the hub passes the external
            proxy prefix when configured.

    Returns:
        A configured :class:`dash.Dash` instance whose ``app.run(...)``
        is the responsibility of the caller.
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title=TITLE_VIEWER,
        # Pin to the in-package directory so the assets ship correctly
        # regardless of the user's CWD at launch.
        assets_folder=str(Path(__file__).parent / "_assets"),
        # See builder/_app.py for the rationale: when mounted under
        # the hub's DispatcherMiddleware the dispatcher strips the
        # mount prefix from PATH_INFO, so Dash must route at "/".
        requests_pathname_prefix=url_prefix,
        routes_pathname_prefix=MOUNT_HOME,
    )

    # Inject window.__phenotypicAppPrefix so results_viewer.js can build
    # mount-aware URLs for DZI tiles and OSD assets.
    app.index_string = dash_index_string_with_app_prefix(url_prefix)

    inject_design_tokens(app)
    register_shared_static(app.server)

    app.server.config[CFG_URL_PREFIX] = url_prefix

    if output_root is None:
        app.layout = build_empty_state_layout()
        _register_empty_state_callbacks(
            app,
            url_prefix=url_prefix,
            api_url_prefix=api_url_prefix,
        )
        logger.debug(
            "Results viewer built in empty-state mode (url_prefix=%s)",
            url_prefix,
        )
        return configure_url_prefix_routing(app, url_prefix)

    output_root.require_refresh_state_current(
        context="Results session pre-read",
    )
    app.server.config[CFG_OUTPUT_ROOT] = output_root

    _tile_routes.register(app, output_root)
    timeline_thumb_routes.register(app, output_root)

    filtered_state = CurationLabels.load(output_root.layout, output_root.master_df)
    app.server.config[CFG_FILTERED_STATE] = filtered_state
    colony_crop_routes.register(app, output_root)
    # QC Review tab serves the same centered crops under its own segment
    # so the colony-view ``/crops`` and the Review gallery never collide.
    register_crop_route(app, output_root, QC_CROPS_URL_SEGMENT)

    # MeasurementSchema cache shared by the Heatmap tab (and a future
    # QC tab) - lazily built once per app instance. Idempotent: do not
    # clobber an existing instance e.g. when the analysis sub-app has
    # already populated the key.
    if app.server.config.get(CFG_MEASUREMENT_SCHEMA) is None:
        app.server.config[CFG_MEASUREMENT_SCHEMA] = MeasurementSchema.from_layout(
            output_root.layout
        )
    # QC tab's augmented-frame cache starts empty; Wave E's QC writer
    # fills it on its first card refresh. The heatmap render callback
    # gracefully falls back to the plain filtered frame until then.
    app.server.config.setdefault(CFG_QC_AUGMENTED_FRAME, None)

    # The QC tab's Add/Edit modal reads its class picker, param form, and
    # submit-time class resolution from an ``OperationRegistry`` on *this*
    # server's config. Under the hub's ``DispatcherMiddleware`` each sub-app
    # has its own Flask server, so the builder's registry is not visible
    # here — without this the picker renders empty and no QC check can be
    # added. Build one per viewer app, idempotently (a hub composer may
    # inject a shared registry ahead of ``create_app``).
    if app.server.config.get(CFG_OPERATION_REGISTRY) is None:
        operation_registry = OperationRegistry()
        operation_registry.discover()
        app.server.config[CFG_OPERATION_REGISTRY] = operation_registry

    # QC recipe is now the ``qc`` section of ``pipeline.json`` (pipeline-
    # backed adapter), not the legacy ``.viewer_cache/qc_recipe.json``
    # sidecar. Fold any legacy sidecar into the pipeline exactly once, then
    # load the recipe + the full pipeline (for the Review tab's in-session
    # recompute via ``run_qc``). The per-revision instance cache is keyed
    # on the recipe revision counter so a stale entry can never serve a
    # moved configuration.
    # Legacy ``.viewer_cache/qc_recipe.json`` sidecar migration only applies to
    # full runs (the sidecar lived at the *output root*). A standalone bundle
    # (``layout.output_root is None``) never has one, so skip it (review W6).
    if output_root.layout.output_root is not None:
        QcRecipe.migrate_from_sidecar(output_root.layout.output_root)
    app.server.config[CFG_QC_RECIPE] = QcRecipe.from_layout(output_root.layout)
    app.server.config[CFG_QC_PIPELINE] = _load_qc_pipeline(output_root)
    app.server.config.setdefault(CFG_QC_INSTANCES_CACHE, {})

    app.layout = build_app_layout(output_root, filtered_state, url_prefix=url_prefix)
    output_root.require_refresh_state_current(
        context="Results session post-read",
    )
    register_callbacks(app, output_root)

    return configure_url_prefix_routing(app, url_prefix)


def _load_qc_pipeline(output_root: OutputRoot):
    """Deserialize the output root's ``pipeline.json`` for QC recompute.

    The QC Review tab's per-group recompute hands this pipeline to
    :func:`phenotypic.sdk_._qc_recipe._runner.run_qc`, so it must carry the same ``qc``
    entries the CLI persisted. Loaded tolerantly (``skip_unknown_analyzers``)
    so a stale analyzer class never blocks viewer boot, and degrades to
    ``None`` when the file is absent or unreadable — recompute then no-ops
    rather than raising.

    Resolves the config through :attr:`OutputRoot.layout` via
    ``layout.resolved_pipeline_config_path``: the canonical typed config, else
    the legacy plain ``pipeline.json`` inside the bundle, anchored on the
    deliverables base so a standalone bundle (``output_root is None``) never
    double-joins ``deliverables/`` (mirroring ``QcRecipe.from_layout``).

    Args:
        output_root: The results-viewer output root handle.

    Returns:
        The deserialized ``ImagePipeline``, or ``None`` when no usable
        ``pipeline.json`` exists.
    """
    from phenotypic._core._image_pipeline import ImagePipeline

    pipeline_path = output_root.layout.resolved_pipeline_config_path
    if not pipeline_path.exists():
        return None
    try:
        return ImagePipeline.from_json(
            pipeline_path, skip_unknown_analyzers=True, load_warnings=[]
        )
    except Exception:  # noqa: BLE001 - boot-time tolerance; recompute no-ops
        logger.warning(
            "Could not load pipeline.json at %s for QC recompute; the Review "
            "tab's per-group recompute will be unavailable this session.",
            pipeline_path,
            exc_info=True,
        )
        return None


def _handoff_banner_state(
    selection: "dict | None",
) -> "tuple[dict, str, bool]":
    """Return the empty-state hand-off banner's ``(style, label, disabled)``.

    Pure helper behind the selection-store callback (extracted so the
    Open-button gate is unit-testable). The Open button is enabled when the
    sidebar selection is viewer-openable — either a full CLI output
    (``is_cli_output``) **or** a standalone deliverables bundle
    (``is_deliverables_bundle``); both boot the viewer.

    Args:
        selection: The :data:`SHELL_SIDEBAR_SELECTION_STORE` payload (or
            ``None``/non-dict before any selection).

    Returns:
        A ``(style, label, disabled)`` triple for the banner's style,
        path label, and Open-button ``disabled`` flag.
    """
    hidden = {"display": "none"}
    visible = {
        "display": "flex",
        "alignItems": "center",
        "gap": "0.5rem",
        "marginTop": "1rem",
        "padding": "0.5rem 0.75rem",
        "background": COLOR_SURFACE,
        "border": f"1px solid {COLOR_BLUE}",
        "borderRadius": "6px",
    }
    if not selection or not isinstance(selection, dict):
        return hidden, "(none)", True
    path = selection.get("path") or ""
    if not path:
        return hidden, "(none)", True
    caps = selection.get("capabilities") or {}
    openable = bool(caps.get("is_cli_output")) or bool(
        caps.get("is_deliverables_bundle")
    )
    return visible, path, not openable


def _register_empty_state_callbacks(
    app: dash.Dash,
    *,
    url_prefix: str,
    api_url_prefix: str,
) -> None:
    """Wire the empty-state hand-off banner.

    Two callbacks:

    1. **Mirror selection -> banner.** A serverside callback watches
       :data:`SHELL_SIDEBAR_SELECTION_STORE` (mounted on this app's
       chrome wrapper) and toggles the banner's visibility, label, and
       Open-button ``disabled`` flag based on whether the selection
       has the ``is_cli_output`` capability.

    2. **Open button -> POST + redirect.** A clientside callback fetches
       ``/sandbox/api/viewer/output-root`` with the selection's
       ``abs_path`` (or rel ``path`` as fallback). On success the
       browser navigates to ``url_prefix`` so :class:`_ViewerProxy`
       resolves a freshly-built loaded viewer; on 4xx the JSON
       ``error`` is rendered into the inline error slot.
    """
    @app.callback(
        Output(ids.EMPTY_HANDOFF_BANNER, "style"),
        Output(ids.EMPTY_HANDOFF_LABEL, "children"),
        Output(ids.EMPTY_HANDOFF_OPEN_BUTTON, "disabled"),
        Input(SHELL_SIDEBAR_SELECTION_STORE, "data"),
    )
    def _populate_handoff_banner(
        selection: "dict | None",
    ) -> "tuple":
        return _handoff_banner_state(selection)

    # Clientside POST + navigate. Uses ``window.fetch`` with the prefix
    # so it works under any DispatcherMiddleware mount. On success the
    # callback calls ``window.location.assign(prefix)`` directly (forces
    # a full reload even though the URL is unchanged), which is what
    # ``_ViewerProxy`` needs to resolve the freshly-built session. On
    # 4xx the JSON ``error`` is rendered into the inline error slot.
    api_output_root = join_url_prefix(api_url_prefix, SANDBOX_API_VIEWER_OUTPUT_ROOT)

    app.clientside_callback(
        """
        async function(n_clicks, selection) {
            if (!n_clicks || !selection) {
                return window.dash_clientside.no_update;
            }
            const path = selection.path;
            if (!path) { return "No sidebar selection."; }
            try {
                const resp = await fetch(
                    "__PHENO_API_OUTPUT_ROOT__",
                    {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({path: path}),
                    }
                );
                const data = await resp.json().catch(() => ({}));
                if (!resp.ok) {
                    return (data && data.error) || ("HTTP " + resp.status);
                }
                window.location.assign(__PHENO_VIEWER_PREFIX__);
                return "";
            } catch (err) {
                return String(err);
            }
        }
        """.replace("__PHENO_API_OUTPUT_ROOT__", api_output_root).replace("__PHENO_VIEWER_PREFIX__", repr(url_prefix)),
        Output(ids.EMPTY_HANDOFF_ERROR, "children"),
        Input(ids.EMPTY_HANDOFF_OPEN_BUTTON, "n_clicks"),
        State(SHELL_SIDEBAR_SELECTION_STORE, "data"),
        prevent_initial_call=True,
    )


__all__ = ["create_app"]
