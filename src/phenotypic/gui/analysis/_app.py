"""Dash app factory for the analysis sub-app.

Mirrors the pattern in
:mod:`phenotypic.gui.results_viewer._app`: builds a configured
:class:`dash.Dash` instance with the validated
:class:`~phenotypic.gui.results_viewer._output_root.OutputRoot` plus the
loaded :class:`~phenotypic.gui.analysis._recipe_state.RecipeState`
stashed on ``app.server.config``, the layout assembled by
:func:`._layout.build_app_layout` (or
:func:`._layout.build_empty_state_layout` when no output root is bound),
and all callbacks registered via :func:`._callbacks.register_callbacks`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

from phenotypic.gui._config import (
    CFG_MEASUREMENT_SCHEMA,
    CFG_OUTPUT_ROOT,
    CFG_RECIPE_STATE,
    CFG_URL_PREFIX,
    DEFAULT_URL_PREFIX,
    MOUNT_HOME,
    SANDBOX_API_VIEWER_OUTPUT_ROOT,
    TITLE_ANALYSIS,
    join_url_prefix,
)
from phenotypic.gui._binding_generation import (
    BindingRequestFence,
    binding_generation_hooks,
    install_bound_output_callback_guard,
    install_binding_generation_guard,
)
from dash import Input, Output, State

from phenotypic.gui._design import (
    COLOR_BLUE,
    COLOR_SURFACE,
    RADIUS,
    inject_design_tokens,
)
from phenotypic.gui._shared import register_shared_static
from phenotypic.gui._snapshot_status import snapshot_refresh_status
from phenotypic.gui._url_prefix import (
    configure_url_prefix_routing,
    dash_index_string_with_app_prefix,
)
from phenotypic.gui.analysis import _ids as analysis_ids
from phenotypic.gui.analysis._callbacks import register_callbacks
from phenotypic.gui.analysis._layout import (
    build_active_snapshot_layout,
    build_app_layout,
    build_empty_state_layout,
)
from phenotypic.gui.analysis._recipe_state import RecipeState
from phenotypic.gui._schema_cache import MeasurementSchema
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.shell._ids import SHELL_SIDEBAR_SELECTION_STORE

logger = logging.getLogger(__name__)


def create_app(
    *,
    output_root: Optional[OutputRoot] = None,
    url_prefix: str = MOUNT_HOME,
    api_url_prefix: str = DEFAULT_URL_PREFIX,
    binding_generation: str | None = None,
    binding_fence: BindingRequestFence | None = None,
) -> dash.Dash:
    """Build a configured Dash instance for the analysis sub-app.

    Args:
        output_root: Validated CLI output root; when ``None`` (e.g. the
            hub mounted ``/analysis/`` before the user picked an output
            directory), the factory returns a Dash whose layout is the
            empty-state placeholder and which has no callbacks
            registered.
        url_prefix: Mount-point prefix passed to ``dash.Dash`` as both
            ``requests_pathname_prefix`` and ``routes_pathname_prefix``.
            Standalone launches collapse to ``MOUNT_HOME`` ("/");
            :func:`compose_hub` passes ``MOUNT_ANALYSIS``.
        api_url_prefix: Browser-visible base prefix for shell-level
            Flask APIs. Defaults to ``"/"``; the hub passes the external
            proxy prefix when configured.
        binding_generation: Optional immutable shell bind UUID used to reject
            callbacks from a browser page rendered for an older output.
        binding_fence: Shared Results/Analysis request fence for this binding.

    Returns:
        Configured :class:`dash.Dash` instance.
    """
    assets_folder = str(Path(__file__).parent / "_assets")
    # ``DispatcherMiddleware`` strips the mount prefix before forwarding, so
    # the Dash internal routes must answer at ``/``. ``requests_pathname_prefix``
    # is what Dash uses to build client-facing URLs (so it gets the mount
    # prefix); ``routes_pathname_prefix`` is what Dash listens on (always
    # ``/`` so the dispatcher's stripped path matches). Standalone launches
    # collapse to identical prefixes when ``url_prefix == MOUNT_HOME``.
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=assets_folder,
        title=TITLE_ANALYSIS,
        requests_pathname_prefix=url_prefix,
        routes_pathname_prefix=MOUNT_HOME,
        hooks=binding_generation_hooks(binding_generation),
    )
    app.index_string = dash_index_string_with_app_prefix(
        url_prefix,
        binding_generation=binding_generation,
    )
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
        return configure_url_prefix_routing(app, url_prefix)

    # Route path resolution through the resolved BundleLayout, never
    # ``output_root.root``: for a standalone deliverables bundle ``root`` IS the
    # deliverables folder, so any helper that internally joins ``deliverables/``
    # would double-join. ``from_layout`` anchors on ``layout.deliverables_base``.
    output_root.require_session_snapshot_current(
        context="Analysis session pre-read",
    )
    app.server.config[CFG_OUTPUT_ROOT] = output_root
    install_bound_output_callback_guard(
        app,
        mutation_is_safe=output_root.mutation_snapshot_is_safe,
        status_output_id=analysis_ids.ANALYSIS_SNAPSHOT_STATUS,
    )
    if output_root.snapshot.active_run:
        app.layout = build_active_snapshot_layout(
            output_root,
            url_prefix=url_prefix,
            binding_generation=binding_generation,
        )
        output_root.require_session_snapshot_current(
            context="Analysis active session post-read",
        )
        _register_snapshot_refresh_callbacks(
            app,
            output_root,
            url_prefix=url_prefix,
            api_url_prefix=api_url_prefix,
            refresh_supported=binding_generation is not None,
        )
        return configure_url_prefix_routing(app, url_prefix)

    recipe = RecipeState.from_layout(output_root.layout)
    recipe.publication_guard = output_root.mutation_snapshot_is_safe
    schema = MeasurementSchema.from_layout(output_root.layout)
    app.server.config[CFG_RECIPE_STATE] = recipe
    app.server.config[CFG_MEASUREMENT_SCHEMA] = schema

    app.layout = build_app_layout(
        output_root,
        recipe,
        url_prefix=url_prefix,
        columns_provider=schema.columns_for,
        binding_generation=binding_generation,
        refresh_supported=binding_generation is not None,
    )
    output_root.require_session_snapshot_current(
        context="Analysis session post-read",
    )
    register_callbacks(app)
    _register_snapshot_refresh_callbacks(
        app,
        output_root,
        url_prefix=url_prefix,
        api_url_prefix=api_url_prefix,
        refresh_supported=binding_generation is not None,
    )

    logger.info(
        "Analysis sub-app ready: output_root=%s pipeline=%s",
        output_root.root,
        recipe.pipeline.name,
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
        Output(analysis_ids.ANALYSIS_SNAPSHOT_STATUS, "children"),
        Output(analysis_ids.ANALYSIS_SNAPSHOT_STATUS, "color"),
        Output(analysis_ids.ANALYSIS_REFRESH_SNAPSHOT, "disabled"),
        Input(analysis_ids.ANALYSIS_SNAPSHOT_INTERVAL, "n_intervals"),
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
        """
        async function(n_clicks) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            try {
                const resp = await fetch(
                    "__PHENO_API_OUTPUT_ROOT__",
                    {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({refresh: true}),
                    }
                );
                const data = await resp.json().catch(() => ({}));
                if (!resp.ok) {
                    return (data && data.error) || ("HTTP " + resp.status);
                }
                window.location.assign(__PHENO_ANALYSIS_PREFIX__);
                return "";
            } catch (err) {
                return String(err);
            }
        }
        """.replace(
            "__PHENO_API_OUTPUT_ROOT__",
            api_output_root,
        ).replace("__PHENO_ANALYSIS_PREFIX__", repr(url_prefix)),
        Output(analysis_ids.ANALYSIS_REFRESH_ERROR, "children"),
        Input(analysis_ids.ANALYSIS_REFRESH_SNAPSHOT, "n_clicks"),
        prevent_initial_call=True,
    )


def _handoff_banner_state(selection):
    """Return the empty-state hand-off banner's ``(style, label, disabled)``.

    Pure helper behind the selection-store callback (extracted so the
    Open-button gate is unit-testable), mirroring
    :func:`phenotypic.gui.results_viewer._app._handoff_banner_state`. The
    Open button is enabled when the sidebar selection is viewer-openable —
    either a full CLI output (``is_cli_output``) **or** a standalone
    deliverables bundle (``is_deliverables_bundle``).

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
        "borderRadius": RADIUS,
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
    """Wire the hand-off banner: selection store -> banner; click -> bind.

    Mirrors the results-viewer empty-state pattern. The clientside
    callback POSTs to the shared ``/sandbox/api/viewer/output-root``
    endpoint, which builds and atomically publishes both the Results and
    Analysis ToolSessions against one descriptor. On success the page
    navigates to ``url_prefix`` so the dispatcher proxy serves the newly
    published analysis app.
    """

    @app.callback(
        Output(analysis_ids.EMPTY_HANDOFF_BANNER, "style"),
        Output(analysis_ids.EMPTY_HANDOFF_LABEL, "children"),
        Output(analysis_ids.EMPTY_HANDOFF_OPEN_BUTTON, "disabled"),
        Input(SHELL_SIDEBAR_SELECTION_STORE, "data"),
    )
    def _populate_handoff_banner(selection):
        return _handoff_banner_state(selection)

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
                window.location.assign(__PHENO_ANALYSIS_PREFIX__);
                return "";
            } catch (err) {
                return String(err);
            }
        }
        """.replace("__PHENO_API_OUTPUT_ROOT__", api_output_root).replace("__PHENO_ANALYSIS_PREFIX__", repr(url_prefix)),
        Output(analysis_ids.EMPTY_HANDOFF_ERROR, "children"),
        Input(analysis_ids.EMPTY_HANDOFF_OPEN_BUTTON, "n_clicks"),
        State(SHELL_SIDEBAR_SELECTION_STORE, "data"),
        prevent_initial_call=True,
    )


__all__ = ["create_app"]
