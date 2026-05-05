"""Dash app factory for the pipeline builder.

Phase 2 deliverable: build a fully-laid-out Dash application without any
``@callback`` registration. Phase 3 imports :func:`create_app`, mutates the
returned instance to add callbacks via the IDs exported from
:mod:`phenotypic.gui.builder._ids`, and wires preview/save/load logic.

The app is ready to ``run(host=..., port=...)`` for visual inspection — the
canvas, palette, and footer all render — but no buttons are functional until
Phase 3 lands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

from phenotypic.gui._design import inject_design_tokens
from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui.builder._callbacks import register_callbacks
from phenotypic.gui.builder._layout import build_app_layout
from phenotypic.gui.builder._point_picker import (
    register_point_picker_callbacks,
    register_point_picker_routes,
)
from phenotypic.gui.builder._state import BuilderState


def create_app(
    image_root: Optional[Path] = None,
    *,
    registry: Optional[OperationRegistry] = None,
    url_prefix: str = "/",
) -> dash.Dash:
    """Build a Dash application instance for the pipeline builder.

    Args:
        image_root: Optional server-side directory used as the root of the
            directory-tree picker.
        registry: Pre-populated :class:`OperationRegistry` to share across
            requests. When ``None``, a fresh registry is constructed and
            :meth:`OperationRegistry.discover` is called immediately so the
            palette renders on the first paint.
        url_prefix: Mount-point prefix passed to :class:`dash.Dash` as
            both ``requests_pathname_prefix`` and ``routes_pathname_prefix``.
            Defaults to ``"/"`` so the standalone launcher works unchanged.
            The hub composer passes ``"/builder/"``.

    Returns:
        A configured :class:`dash.Dash` instance whose ``app.run(...)`` is
        the responsibility of the caller.

    Examples:
        >>> from phenotypic.gui.builder._app import create_app
        >>> app = create_app(image_root=None)
        >>> app.title
        'PhenoTypic Pipeline Builder'
    """

    if registry is None:
        registry = OperationRegistry()
        registry.discover()

    state = BuilderState()

    # Dash split: ``requests_pathname_prefix`` is the prefix browsers see
    # (URL construction); ``routes_pathname_prefix`` is what Dash matches
    # against incoming PATH_INFO. When the hub mounts this app via
    # ``DispatcherMiddleware`` the dispatcher strips the mount prefix
    # before forwarding, so Dash must route at ``/``. Standalone (default
    # ``url_prefix == "/"``) collapses to identical prefixes.
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="PhenoTypic Pipeline Builder",
        requests_pathname_prefix=url_prefix,
        routes_pathname_prefix="/",
    )

    inject_design_tokens(app)

    app.layout = build_app_layout(state, registry, image_root, url_prefix=url_prefix)

    app.server.config["pheno_image_root"] = image_root
    app.server.config["pheno_registry"] = registry
    app.server.config["pheno_url_prefix"] = url_prefix

    register_callbacks(app)
    register_point_picker_routes(app, image_root)
    register_point_picker_callbacks(app)

    return app


__all__ = ["create_app"]
