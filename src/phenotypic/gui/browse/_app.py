"""Dash app factory for the Browse tab.

Eager, lightweight (no heavy parquet load → no ToolSession). Mounts the
revision-addressed asset and preparation blueprints, injects
``window.__phenotypicAppPrefix`` (so ``browse.js`` builds hub-aware tile +
OSD-asset URLs), builds the layout, and registers callbacks.
"""

from __future__ import annotations

import atexit
import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

from phenotypic.gui._config import (
    CFG_BROWSE_CACHE,
    CFG_BROWSE_PREPARATION_MANAGER,
    CFG_URL_PREFIX,
    MOUNT_HOME,
    TITLE_BROWSE,
)
from phenotypic.gui._design import inject_design_tokens
from phenotypic.gui._shared import register_shared_static
from phenotypic.gui._url_prefix import (
    configure_url_prefix_routing,
    dash_index_string_with_app_prefix,
)
from phenotypic.gui.browse import (
    _preparation_routes,
    _thumb_routes,
    _tile_routes,
)
from phenotypic.gui.browse._cache import BrowseCache
from phenotypic.gui.browse._callbacks import register_callbacks
from phenotypic.gui.browse._layout import build_browse_layout
from phenotypic.gui.browse._preparation import BrowsePreparationManager
from phenotypic.gui.results_viewer._dzi_tiler import DZI_BACKEND_INFO
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["create_app"]


def create_app(
    sandbox: SandboxRoot, *, url_prefix: str = MOUNT_HOME
) -> dash.Dash:
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
    app.index_string = dash_index_string_with_app_prefix(url_prefix)
    inject_design_tokens(app)
    register_shared_static(app.server)
    app.server.config[CFG_URL_PREFIX] = url_prefix

    cache = BrowseCache.for_sandbox(sandbox.root)
    manager = BrowsePreparationManager(cache)
    preparation_api = _preparation_routes.BrowsePreparationApi(
        sandbox=sandbox,
        cache=cache,
        manager=manager,
    )
    app.server.config[CFG_BROWSE_CACHE] = cache
    app.server.config[CFG_BROWSE_PREPARATION_MANAGER] = manager
    app.server.config["PHENOTYPIC_BROWSE_PREPARATION_API"] = preparation_api
    atexit.register(manager.close)

    _tile_routes.register(app, preparation_api)
    _thumb_routes.register(app, sandbox, preparation_api)
    _preparation_routes.register(app, preparation_api)
    app.layout = build_browse_layout()
    register_callbacks(app, sandbox, preparation_api)

    logger.info(
        "Browse DZI backend selected: backend=%s version=%s fallback=%s",
        DZI_BACKEND_INFO.name,
        DZI_BACKEND_INFO.version or "unknown",
        DZI_BACKEND_INFO.fallback_reason or "none",
    )

    logger.debug(
        "Browse app built: sandbox=%s url_prefix=%s", sandbox.root, url_prefix
    )
    return configure_url_prefix_routing(app, url_prefix)
