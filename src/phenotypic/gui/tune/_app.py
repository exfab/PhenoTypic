"""Dash app factory for the ``/tune/`` co-pilot.

Mirrors :mod:`phenotypic.gui.results_viewer._app` and
:mod:`phenotypic.gui.analysis._app`: builds a configured :class:`dash.Dash`
whose layout is the tune page for a bound :class:`~phenotypic.gui.tune.TuneRunRoot`,
or the pick-a-run empty state when no run is bound.

The factory is deliberately **optuna-free** at import / build time: the live
study is opened only inside the Monitor poll callback (gated on the ``tune``
extra being importable), never here. Importing this module — and therefore
:mod:`phenotypic.gui.tune` — must not pull ``optuna`` into ``sys.modules``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

from phenotypic.gui._config import MOUNT_HOME, TITLE_TUNE
from phenotypic.gui._design import inject_design_tokens
from phenotypic.gui._shared import register_shared_static
from phenotypic.gui.tune._layout import build_empty_state_layout, build_layout
from phenotypic.gui.tune._run_root import TuneRunRoot

logger = logging.getLogger(__name__)


def create_app(
    root: Optional[TuneRunRoot] = None,
    *,
    url_prefix: str = MOUNT_HOME,
) -> dash.Dash:
    """Build a configured Dash instance for the tune co-pilot.

    Args:
        root: Validated tune output handle (see
            :meth:`phenotypic.gui.tune.TuneRunRoot.discover`). ``None``
            triggers the empty-state pathway: the factory renders the
            pick-a-run prompt and registers no poll/figure callbacks.
        url_prefix: Mount-point prefix passed to ``dash.Dash`` as
            ``requests_pathname_prefix``; ``routes_pathname_prefix`` stays
            ``MOUNT_HOME`` so the DispatcherMiddleware-stripped path matches.
            Standalone launches collapse to ``MOUNT_HOME`` ("/");
            :func:`phenotypic.gui.shell._app.compose_hub` passes ``MOUNT_TUNE``.

    Returns:
        A configured :class:`dash.Dash`; ``app.run(...)`` is the caller's
        responsibility.
    """
    assets_folder = str(Path(__file__).parent / "_assets")
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=assets_folder,
        title=TITLE_TUNE,
        # See results_viewer/analysis _app.py: the dispatcher strips the mount
        # prefix before Dash routes, so requests build URLs with the prefix but
        # routing answers at "/".
        requests_pathname_prefix=url_prefix,
        routes_pathname_prefix=MOUNT_HOME,
    )
    inject_design_tokens(app)
    register_shared_static(app.server)

    if root is None:
        app.layout = build_empty_state_layout()
        logger.debug("Tune co-pilot built in empty-state mode (url_prefix=%s)", url_prefix)
        return app

    app.layout = build_layout(root)
    logger.info("Tune co-pilot ready: run=%s", root.path)
    return app


__all__ = ["create_app"]
