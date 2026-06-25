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
from dash import Input, Output, State

from phenotypic.gui._design import (
    COLOR_BLUE,
    COLOR_SURFACE,
    RADIUS,
    inject_design_tokens,
)
from phenotypic.gui._shared import register_shared_static
from phenotypic.gui._url_prefix import configure_url_prefix_routing
from phenotypic.gui.analysis import _ids as analysis_ids
from phenotypic.gui.analysis._callbacks import register_callbacks
from phenotypic.gui.analysis._layout import (
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
    )
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
        return configure_url_prefix_routing(app, url_prefix)

    # Route path resolution through the resolved BundleLayout, never
    # ``output_root.root``: for a standalone deliverables bundle ``root`` IS the
    # deliverables folder, so any helper that internally joins ``deliverables/``
    # would double-join. ``from_layout`` anchors on ``layout.deliverables_base``.
    recipe = RecipeState.from_layout(output_root.layout)
    schema = MeasurementSchema.from_layout(output_root.layout)
    app.server.config[CFG_OUTPUT_ROOT] = output_root
    app.server.config[CFG_RECIPE_STATE] = recipe
    app.server.config[CFG_MEASUREMENT_SCHEMA] = schema

    app.layout = build_app_layout(
        output_root,
        recipe,
        url_prefix=url_prefix,
        columns_provider=schema.columns_for,
    )
    register_callbacks(app)

    logger.info(
        "Analysis sub-app ready: output_root=%s pipeline=%s",
        output_root.root,
        recipe.pipeline.name,
    )
    return configure_url_prefix_routing(app, url_prefix)


def _register_empty_state_callbacks(
    app: dash.Dash,
    *,
    url_prefix: str,
    api_url_prefix: str,
) -> None:
    """Wire the hand-off banner: selection store -> banner; click -> bind.

    Mirrors the results-viewer empty-state pattern. The clientside
    callback POSTs to the shared ``/sandbox/api/viewer/output-root``
    endpoint, which releases both viewer + analysis ToolSessions and
    rebuilds them against the new ``viewer_state["output_root"]``. On
    success the page navigates to ``url_prefix`` so the dispatcher
    proxy resolves a freshly-built loaded analysis app.
    """

    @app.callback(
        Output(analysis_ids.EMPTY_HANDOFF_BANNER, "style"),
        Output(analysis_ids.EMPTY_HANDOFF_LABEL, "children"),
        Output(analysis_ids.EMPTY_HANDOFF_OPEN_BUTTON, "disabled"),
        Input(SHELL_SIDEBAR_SELECTION_STORE, "data"),
    )
    def _populate_handoff_banner(selection):
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
        is_cli_output = bool(caps.get("is_cli_output"))
        return visible, path, not is_cli_output

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
