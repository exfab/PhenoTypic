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
    CFG_OUTPUT_ROOT,
    CFG_RECIPE_STATE,
    CFG_URL_PREFIX,
    MOUNT_HOME,
    TITLE_ANALYSIS,
)
from phenotypic.gui._design import inject_design_tokens
from phenotypic.gui.analysis._callbacks import register_callbacks
from phenotypic.gui.analysis._layout import (
    build_app_layout,
    build_empty_state_layout,
)
from phenotypic.gui.analysis._recipe_state import RecipeState
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


def create_app(
    *,
    output_root: Optional[OutputRoot] = None,
    url_prefix: str = MOUNT_HOME,
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
    app.server.config[CFG_URL_PREFIX] = url_prefix

    if output_root is None:
        app.layout = build_empty_state_layout()
        return app

    recipe = RecipeState.load(Path(output_root.root))
    app.server.config[CFG_OUTPUT_ROOT] = output_root
    app.server.config[CFG_RECIPE_STATE] = recipe

    app.layout = build_app_layout(output_root, recipe)
    register_callbacks(app)

    logger.info(
        "Analysis sub-app ready: output_root=%s pipeline=%s",
        output_root.root,
        recipe.pipeline.name,
    )
    return app


__all__ = ["create_app"]
