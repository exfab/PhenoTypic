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

from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui.builder._callbacks import register_callbacks
from phenotypic.gui.builder._layout import build_app_layout
from phenotypic.gui.builder._state import BuilderState


def create_app(
    image_root: Optional[Path] = None,
    *,
    registry: Optional[OperationRegistry] = None,
) -> dash.Dash:
    """Build a Dash application instance for the pipeline builder.

    Constructs (or reuses) an :class:`OperationRegistry`, builds an empty
    :class:`BuilderState`, instantiates a :class:`dash.Dash` with Bootstrap
    styling, sets ``app.layout`` from :func:`build_app_layout`, and stashes
    *image_root* and *registry* on ``app.server.config`` so Phase 3 callbacks
    can retrieve them via ``flask.current_app``.

    Args:
        image_root: Optional server-side directory used as the root of the
            directory-tree picker. ``None`` disables the tree (the user can
            still type a path or fall back to the synthetic plate).
        registry: Pre-populated :class:`OperationRegistry` to share across
            requests. When ``None``, a fresh registry is constructed and
            :meth:`OperationRegistry.discover` is called immediately so the
            palette renders on the first paint.

    Returns:
        A configured :class:`dash.Dash` instance whose ``app.run(...)`` is
        the responsibility of the caller (typically the Phase-4 CLI).

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

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        title="PhenoTypic Pipeline Builder",
    )

    app.layout = build_app_layout(state, registry, image_root)

    # Stash dependencies on the underlying Flask server so Phase 3 callbacks
    # can fetch them via ``flask.current_app.config[...]`` without rebuilding
    # the registry per request.
    app.server.config["pheno_image_root"] = image_root
    app.server.config["pheno_registry"] = registry

    register_callbacks(app)

    return app


__all__ = ["create_app"]
