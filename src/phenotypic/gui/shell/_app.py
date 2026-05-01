"""Shell Dash factory.

Phase 3 ships the standalone shell variant: ``create_app(sandbox)`` returns
a single :class:`dash.Dash` app whose body is the home page wrapped in
chrome, with the Phase 2 Flask blueprints (``/sandbox/api/*`` and
``/runs/*``) registered on its server. No sub-app mounting yet — that's
Phase 5, which will add the ``DispatcherMiddleware`` composer + viewer
``ToolSession`` + per-app ``wrap_in_chrome`` passes.

The factory pins ``assets_folder`` to the package's ``_assets/`` directory
so the shell's CSS works regardless of CWD.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import html

from phenotypic.gui.shell._home import build_home_layout
from phenotypic.gui.shell._ids import SHELL_TAB_HOME
from phenotypic.gui.shell._layout import wrap_in_chrome
from phenotypic.gui.shell._routes import register_sandbox_api
from phenotypic.gui.shell._runs_blueprint import register as register_runs
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._session import ToolSession

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["create_app"]


def create_app(
    sandbox: SandboxRoot,
    *,
    url_prefix: str = "/",
    viewer_session: "ToolSession[object] | None" = None,
) -> dash.Dash:
    """Build the shell Dash app.

    Args:
        sandbox: Frozen-at-launch sandbox root. Echoed in the top-bar label;
            used by the JSON API + ``/runs/`` blueprints.
        url_prefix: Phase 5 mounts the shell at ``/`` and the sub-apps at
            ``/builder/``, ``/results/``, ``/run/``. The Phase 3 standalone
            launch keeps the default ``"/"``.
        viewer_session: Optional :class:`ToolSession` whose ``touch()`` is
            called by the JSON API + runs blueprints. Phase 5 wires the
            real viewer session; Phase 3 standalone tests pass ``None``.

    Returns:
        Configured :class:`dash.Dash` app. ``app.run()`` starts the server.
    """
    assets_folder = str(Path(__file__).parent / "_assets")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=assets_folder,
        title="PhenoTypic GUI",
        requests_pathname_prefix=url_prefix,
        routes_pathname_prefix=url_prefix,
    )

    # Body BEFORE chrome: the home pane wrapped in a top-level html.Div so
    # ``wrap_in_chrome`` has a single root to splice.
    app.layout = html.Div(build_home_layout(sandbox), className="shell-page")

    # Chrome wrap: top bar + sidebar + RSS interval + help modal +
    # registers chrome callbacks on this Dash app.
    wrap_in_chrome(app, active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    # Flask blueprints: register on the SAME Flask server so they answer
    # regardless of which Dash sub-app is active in Phase 5.
    register_sandbox_api(app.server, sandbox, viewer_session=viewer_session)
    register_runs(app.server, sandbox, viewer_session=viewer_session)

    logger.debug(
        "Shell Dash app built: sandbox=%s url_prefix=%s",
        sandbox.root,
        url_prefix,
    )
    return app
