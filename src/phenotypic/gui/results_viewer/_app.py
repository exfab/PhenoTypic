"""Dash app factory for the results viewer.

Builds a configured :class:`dash.Dash` instance with its tile-serving
Flask blueprint mounted, the validated
:class:`~phenotypic.gui.results_viewer._output_root.OutputRoot` stashed
on ``app.server.config``, the layout assembled by
:func:`~phenotypic.gui.results_viewer._layout.build_app_layout`, and
all callbacks registered via
:func:`~phenotypic.gui.results_viewer._callbacks.register_callbacks`.

The package's ``_assets`` directory (which ships the vendored
OpenSeadragon JS plus viewer CSS/JS) is registered explicitly via
``assets_folder="_assets"`` so Dash picks it up regardless of the user's
current working directory at launch time.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

from phenotypic.gui.results_viewer import _tile_routes
from phenotypic.gui.results_viewer._callbacks import register_callbacks
from phenotypic.gui.results_viewer._layout import build_app_layout
from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


def create_app(output_root: OutputRoot) -> dash.Dash:
    """Build a Dash application instance for the results viewer.

    Constructs a :class:`dash.Dash` with Bootstrap styling, points the
    asset loader at the in-package ``_assets`` directory (so vendored
    OpenSeadragon and viewer CSS are auto-served), stashes
    *output_root* on ``app.server.config["output_root"]`` for later
    callback access, mounts the DZI tile-serving blueprint via
    :func:`phenotypic.gui.results_viewer._tile_routes.register`,
    assembles the layout via
    :func:`phenotypic.gui.results_viewer._layout.build_app_layout`,
    and finally hooks every per-module + clientside callback through
    :func:`phenotypic.gui.results_viewer._callbacks.register_callbacks`.

    Args:
        output_root: Validated, read-only handle on a CLI output
            directory (see
            :meth:`phenotypic.gui.results_viewer._output_root.OutputRoot.discover`).

    Returns:
        A configured :class:`dash.Dash` instance whose ``app.run(...)``
        is the responsibility of the caller (typically
        ``__main__.py``).
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="PhenoTypic Results Viewer",
        # Pin to the in-package directory so the assets ship correctly
        # regardless of the user's CWD at launch.
        assets_folder=str(Path(__file__).parent / "_assets"),
    )

    # Stash the output root on the underlying Flask server so future
    # callbacks (and the tile blueprint) can fetch it via
    # ``flask.current_app.config["output_root"]`` without re-discovering
    # the directory layout per request.
    app.server.config["output_root"] = output_root

    _tile_routes.register(app, output_root)

    app.layout = build_app_layout(output_root)
    register_callbacks(app, output_root)

    return app


__all__ = ["create_app"]
