"""Run console Dash factory.

Phase 5 ships a minimal placeholder factory whose job is to boot a
mountable Dash app under ``/run/`` so the unified hub can route to it.
The full Run console UI (form pickers, log tail, Recent Runs panel,
local + SLURM submit, dashboard iframe) lands in Phase 6.

Signature mirrors builder + results_viewer: ``create_app(sandbox, *,
url_prefix="/")``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import html

from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["create_app"]


def _build_placeholder_layout(sandbox: SandboxRoot) -> html.Div:
    """Render a holding-page that signals the Run console is reachable.

    Phase 6 replaces the body with the real form + iframe + Recent Runs
    panel. Until then, the user lands here when they click the Run tab.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.H2(
                        "Run console",
                        className="run-console-empty-title",
                    ),
                    html.P(
                        "Phase 6 will fill this view with the pipeline form, "
                        "the live log tail, and the Recent Runs panel. The "
                        "route is mountable now so the hub can wire it in.",
                        className="run-console-empty-body",
                    ),
                    html.P(
                        f"Sandbox: {sandbox.root}",
                        className="run-console-empty-meta",
                        style={"fontFamily": "monospace", "fontSize": "0.85rem"},
                    ),
                ],
                className="run-console-empty-card",
            ),
        ],
        id="run-console-root",
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "minHeight": "calc(100vh - 7rem)",
            "padding": "2rem",
        },
    )


def create_app(
    sandbox: SandboxRoot,
    *,
    url_prefix: str = "/",
) -> dash.Dash:
    """Build the Run console Dash app.

    Args:
        sandbox: Frozen-at-launch sandbox root. Phase 6 callbacks will
            use this to constrain the file pickers; the Phase 5
            placeholder only echoes the path.
        url_prefix: Mount-point prefix. Defaults to ``"/"`` (standalone
            launcher); the hub composer passes ``"/run/"``.

    Returns:
        A configured :class:`dash.Dash` instance ready to mount under
        ``url_prefix`` or run standalone via ``app.run(...)``.
    """
    assets_folder = str(Path(__file__).parent / "_assets")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=assets_folder,
        title="PhenoTypic Run Console",
        # Dispatcher strips mount prefix; Dash must route at "/".
        requests_pathname_prefix=url_prefix,
        routes_pathname_prefix="/",
    )
    app.layout = _build_placeholder_layout(sandbox)
    app.server.config["pheno_url_prefix"] = url_prefix
    app.server.config["pheno_sandbox_root"] = str(sandbox.root)

    logger.debug(
        "Run console placeholder built: sandbox=%s url_prefix=%s",
        sandbox.root,
        url_prefix,
    )
    return app
