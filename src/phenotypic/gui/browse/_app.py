"""Dash app factory for the Browse tab.

Eager, lightweight (no heavy parquet load → no ToolSession). Mounts the
token-keyed tile blueprint, wipes + initialises the ephemeral cache, injects
``window.__phenotypicAppPrefix`` (so ``browse.js`` builds hub-aware tile +
OSD-asset URLs), builds the layout, and registers callbacks.
"""
from __future__ import annotations

import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

from phenotypic.gui._config import CFG_URL_PREFIX, MOUNT_HOME, TITLE_BROWSE
from phenotypic.gui._design import inject_design_tokens
from phenotypic.gui._shared import register_shared_static
from phenotypic.gui.browse import _source_render, _tile_routes
from phenotypic.gui.browse._callbacks import register_callbacks
from phenotypic.gui.browse._layout import build_browse_layout
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["create_app"]


def _index_string_with_prefix(url_prefix: str) -> str:
    """Dash ``index_string`` that exposes ``window.__phenotypicAppPrefix``."""
    safe_prefix = (
        url_prefix.replace("\\", "\\\\").replace('"', '\\"').replace("</", "<\\/")
    )
    return (
        "<!DOCTYPE html>\n<html>\n    <head>\n"
        "        {%metas%}\n        <title>{%title%}</title>\n"
        "        {%favicon%}\n        {%css%}\n"
        f'        <script>window.__phenotypicAppPrefix = "{safe_prefix}";</script>\n'
        "    </head>\n    <body>\n        {%app_entry%}\n        <footer>\n"
        "            {%config%}\n            {%scripts%}\n            {%renderer%}\n"
        "        </footer>\n    </body>\n</html>"
    )


def create_app(sandbox: SandboxRoot, *, url_prefix: str = MOUNT_HOME) -> dash.Dash:
    """Build the Browse Dash app.

    Args:
        sandbox: Frozen-at-launch sandbox root (security boundary + path base).
        url_prefix: Mount prefix. ``"/"`` standalone; hub passes ``"/browse/"``.
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=str(Path(__file__).parent / "_assets"),
        title=TITLE_BROWSE,
        requests_pathname_prefix=url_prefix,
        routes_pathname_prefix=MOUNT_HOME,
    )
    app.index_string = _index_string_with_prefix(url_prefix)
    inject_design_tokens(app)
    register_shared_static(app.server)
    app.server.config[CFG_URL_PREFIX] = url_prefix

    _source_render.init_cache()  # wipe stale tiles + register atexit cleanup
    _tile_routes.register(app, sandbox)
    app.layout = build_browse_layout()
    register_callbacks(app, sandbox)

    logger.debug(
        "Browse app built: sandbox=%s url_prefix=%s", sandbox.root, url_prefix
    )
    return app
