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
from functools import lru_cache
from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import Input, Output, State

from phenotypic.gui._config import (
    CFG_FILTERED_STATE,
    CFG_MEASUREMENT_SCHEMA,
    CFG_OPERATION_REGISTRY,
    CFG_OUTPUT_MUTATION_GUARD,
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
from phenotypic.gui._async_binding_client import (
    async_binding_callback_source,
)
from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui._binding_generation import (
    BindingRequestFence,
    binding_generation_hooks,
    install_bound_output_callback_guard,
    install_binding_generation_guard,
)
from phenotypic.gui._schema_cache import MeasurementSchema
from phenotypic.gui._design import (
    COLOR_BLUE,
    COLOR_SURFACE,
    inject_design_tokens,
)
from phenotypic.gui._shared import register_shared_static
from phenotypic.gui._shared.tiles import register_crop_route
from phenotypic.gui._snapshot_status import snapshot_refresh_status
from phenotypic.gui._url_prefix import (
    configure_url_prefix_routing,
    dash_index_string_with_app_prefix,
)
from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._callbacks import register_callbacks
from phenotypic.gui.results_viewer._curation_labels import CurationLabels
from phenotypic.gui.results_viewer._layout import (
    build_active_snapshot_layout,
    build_app_layout,
    build_empty_state_layout,
)
from phenotypic.gui.results_viewer._mutation_guard import (
    OutputMutationBlocked,
    OutputMutationGuard,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.results_viewer._zarr_routes import (
    register_zarr_routes,
)
from phenotypic.gui.results_viewer.colony_view import (
    _crop_routes as colony_crop_routes,
)
from phenotypic.gui.shell._ids import SHELL_SIDEBAR_SELECTION_STORE
from phenotypic.gui.shell._binding_ui import binding_error_text
from phenotypic.gui.shell._ids import SHELL_RESULTS_BINDING_JOB_STORE
from phenotypic.sdk_._qc_recipe import QcRecipe

logger = logging.getLogger(__name__)

#: Banner prefix ``tools/viv-bundle/build.mjs`` stamps into the artifact.
_VIV_BANNER_MARK = "PhenoTypic vendored Viv bundle -- "

#: Only the banner is read; the artifact itself is ~2.5 MiB.
_VIV_BANNER_BYTES = 512


@lru_cache(maxsize=1)
def viv_bundle_version() -> str:
    """The version string stamped into the vendored Viv bundle.

    There is no npm in CI by design (spec section 3), so nothing rebuilds
    the committed artifact to prove it still matches
    ``tools/viv-bundle/package-lock.json``. Logging what shipped is the
    mitigation, not a fix: it tells a reader which bundle a browser is
    running when a render misbehaves, and it is the only such signal.

    ``tools/viv-bundle/VERSION`` is *not* read here -- it lives in the repo,
    not in the installed package, so an installed PhenoTypic has no copy.
    The banner travels with the artifact.
    ``tests/unit/gui/results_viewer/test_viv_asset_order.py`` pins the two
    to each other.

    Returns:
        The recorded version, or ``"unknown"`` when the banner is missing
        or unreadable -- a viewer must still start without it.
    """
    bundle = Path(__file__).parent / "_assets" / "viv" / "viv-bundle.min.js"
    try:
        with bundle.open("r", encoding="utf-8", errors="replace") as handle:
            banner = handle.read(_VIV_BANNER_BYTES)
    except OSError:
        return "unknown"
    head = banner.split("\n", 1)[0]
    _, mark, version = head.partition(_VIV_BANNER_MARK)
    return version.strip() if mark else "unknown"


def create_app(
    output_root: Optional[OutputRoot] = None,
    *,
    url_prefix: str = MOUNT_HOME,
    api_url_prefix: str = DEFAULT_URL_PREFIX,
    binding_generation: str | None = None,
    binding_fence: BindingRequestFence | None = None,
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
        binding_generation: Optional shell generation used to fence stale
            browser callback requests after a rebind.
        binding_fence: Shared Results/Analysis request fence for the binding.

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
        hooks=binding_generation_hooks(binding_generation),
    )

    # Inject window.__phenotypicAppPrefix so results_viewer.js can build
    # mount-aware URLs for its assets and byte routes.
    app.index_string = dash_index_string_with_app_prefix(
        url_prefix,
        binding_generation=binding_generation,
    )

    logger.info("viv bundle: %s", viv_bundle_version())

    inject_design_tokens(app)
    register_shared_static(app.server)

    app.server.config[CFG_URL_PREFIX] = url_prefix
    install_binding_generation_guard(
        app,
        binding_generation,
        binding_fence,
    )

    if output_root is None:
        app.layout = build_empty_state_layout(
            binding_generation=binding_generation,
        )
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

    output_root.require_session_snapshot_current(
        context="Results session pre-read",
    )
    app.server.config[CFG_OUTPUT_ROOT] = output_root
    mutation_guard = OutputMutationGuard(
        output_root=output_root,
        binding_generation=binding_generation,
    )
    app.server.config[CFG_OUTPUT_MUTATION_GUARD] = mutation_guard

    def _results_mutation_is_safe() -> bool:
        try:
            mutation_guard.authorize("Results mutation")
        except OutputMutationBlocked:
            return False
        return True

    install_bound_output_callback_guard(
        app,
        mutation_is_safe=_results_mutation_is_safe,
        protected_output_ids=(
            ids.STORE_REMOVED_KEYS,
            ids.STORE_QC_RECIPE_REVISION,
        ),
    )
    if output_root.snapshot.active_run:
        app.layout = build_active_snapshot_layout(
            output_root,
            url_prefix=url_prefix,
            binding_generation=binding_generation,
        )
        output_root.require_session_snapshot_current(
            context="Results active session post-read",
        )
        _register_snapshot_refresh_callbacks(
            app,
            output_root,
            url_prefix=url_prefix,
            api_url_prefix=api_url_prefix,
            refresh_supported=binding_generation is not None,
        )
        return configure_url_prefix_routing(app, url_prefix)

    # Raw store bytes with HTTP Range: the ONLY pixel source the Plate
    # surface has. The `/tiles/<ds>/<stem>.dzi` routes that used to sit
    # beside this are gone -- Plate reads store chunks in the browser, and
    # the Timeline surfaces that were the only other consumer were
    # unmounted. Browse and the builder keep their own libvips -> DZI ->
    # OpenSeadragon path in their own blueprints.
    register_zarr_routes(app, output_root)

    filtered_state = CurationLabels.load(
        output_root.layout, output_root.master_df
    )
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
        app.server.config[CFG_MEASUREMENT_SCHEMA] = (
            MeasurementSchema.from_layout(output_root.layout)
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

    # Viewer boot is source-preserving. Legacy sidecar migration is an
    # explicit compatibility action, never an implicit consequence of bind
    # or Refresh.
    app.server.config[CFG_QC_RECIPE] = QcRecipe.from_layout(output_root.layout)
    app.server.config[CFG_QC_PIPELINE] = _load_qc_pipeline(output_root)
    app.server.config.setdefault(CFG_QC_INSTANCES_CACHE, {})

    app.layout = build_app_layout(
        output_root,
        filtered_state,
        url_prefix=url_prefix,
        binding_generation=binding_generation,
        refresh_supported=binding_generation is not None,
    )
    output_root.require_session_snapshot_current(
        context="Results session post-read",
    )
    register_callbacks(app, output_root)
    _register_snapshot_refresh_callbacks(
        app,
        output_root,
        url_prefix=url_prefix,
        api_url_prefix=api_url_prefix,
        refresh_supported=binding_generation is not None,
    )

    return configure_url_prefix_routing(app, url_prefix)


def _register_snapshot_refresh_callbacks(
    app: dash.Dash,
    output_root: OutputRoot,
    *,
    url_prefix: str,
    api_url_prefix: str,
    refresh_supported: bool,
) -> None:
    """Wire status-only polling and explicit shared-session Refresh."""

    @app.callback(
        Output(ids.HEADER_SNAPSHOT_STATUS_ID, "children"),
        Output(ids.HEADER_SNAPSHOT_STATUS_ID, "color"),
        Output(ids.BTN_REFRESH_SNAPSHOT, "disabled"),
        Input(ids.SNAPSHOT_STATUS_INTERVAL_ID, "n_intervals"),
    )
    def _snapshot_status(_n_intervals: int) -> tuple[str, str, bool]:
        return snapshot_refresh_status(
            output_root,
            refresh_supported=refresh_supported,
        )

    if not refresh_supported:
        return

    api_output_root = join_url_prefix(
        api_url_prefix,
        SANDBOX_API_VIEWER_OUTPUT_ROOT,
    )
    app.clientside_callback(
        async_binding_callback_source(
            api_url=api_output_root,
            redirect_url=url_prefix,
            selection_required=False,
        ),
        Output(
            SHELL_RESULTS_BINDING_JOB_STORE,
            "data",
            allow_duplicate=True,
        ),
        Input(ids.BTN_REFRESH_SNAPSHOT, "n_clicks"),
        State(SHELL_RESULTS_BINDING_JOB_STORE, "data"),
        prevent_initial_call=True,
    )
    app.callback(
        Output(ids.HEADER_REFRESH_ERROR_ID, "children"),
        Input(SHELL_RESULTS_BINDING_JOB_STORE, "data"),
    )(binding_error_text)


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
       ``/sandbox/api/viewer/output-root`` with the selection's rel ``path``.
       On success the browser navigates to ``url_prefix`` so
       :class:`_ViewerProxy` serves the atomically published loaded viewer;
       on failure the JSON ``error`` is rendered into the inline error slot.
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
    # a full reload even though the URL is unchanged), which makes the
    # ``_ViewerProxy`` serve the newly published session. On failure the
    # JSON ``error`` is rendered into the inline error slot.
    api_output_root = join_url_prefix(
        api_url_prefix, SANDBOX_API_VIEWER_OUTPUT_ROOT
    )

    app.clientside_callback(
        async_binding_callback_source(
            api_url=api_output_root,
            redirect_url=url_prefix,
            selection_required=True,
        ),
        Output(
            SHELL_RESULTS_BINDING_JOB_STORE,
            "data",
            allow_duplicate=True,
        ),
        Input(ids.EMPTY_HANDOFF_OPEN_BUTTON, "n_clicks"),
        State(SHELL_SIDEBAR_SELECTION_STORE, "data"),
        State(SHELL_RESULTS_BINDING_JOB_STORE, "data"),
        prevent_initial_call=True,
    )
    app.callback(
        Output(ids.EMPTY_HANDOFF_ERROR, "children"),
        Input(SHELL_RESULTS_BINDING_JOB_STORE, "data"),
    )(binding_error_text)


__all__ = ["create_app"]
