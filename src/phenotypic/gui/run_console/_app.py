"""Run console Dash factory (Phase 6).

The Run console mounts under ``/run/`` in the unified hub. This factory
builds the page (form + iframe panel + log tail + Recent Runs) and
registers every callback the form needs.

The factory expects a process-wide :class:`LocalRunner` and
:class:`RunRegistry` to be passed in — both must be shared with the
shell's ``/runs/`` blueprint (so the Recent Runs panel sees the same
records the boot-time rehydrate populated). The shell composer creates
these singletons in :func:`phenotypic.gui.shell._app.compose_hub` and
forwards them.

Standalone debugging via ``python -m phenotypic.gui.run_console`` still
works: when ``registry`` / ``runner`` are ``None``, the factory builds
fresh local instances. They will not be visible to the shell, but a
single-tool invocation does not need cross-tool state sharing.
"""
from __future__ import annotations

import logging
from pathlib import Path

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

from phenotypic.gui._design import inject_design_tokens
from phenotypic.gui.run_console._callbacks import register_callbacks
from phenotypic.gui.run_console._layout import build_run_console_layout
from phenotypic.gui.run_console._runner import LocalRunner
from phenotypic.gui.shell._runs_registry import RunRegistry
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["create_app"]


def create_app(
    sandbox: SandboxRoot,
    *,
    url_prefix: str = "/",
    registry: RunRegistry | None = None,
    runner: LocalRunner | None = None,
) -> dash.Dash:
    """Build the Run console Dash app.

    Args:
        sandbox: Frozen-at-launch sandbox root. Constrains the file
            pickers + drives Recent Runs scan.
        url_prefix: Mount-point prefix. Defaults to ``"/"`` (standalone
            launcher); the hub composer passes ``"/run/"``.
        registry: Process-wide :class:`RunRegistry`. When ``None`` (the
            standalone case) a fresh local registry is built and the
            sandbox is scanned to seed Recent Runs.
        runner: Process-wide :class:`LocalRunner`. When ``None`` a
            fresh local runner is built. Production callers should
            ALWAYS pass a shared runner so a navigation-away does not
            kill in-flight subprocesses (the hub keeps the runner alive
            even when the Run console UI is released).

    Returns:
        A configured :class:`dash.Dash` instance ready to mount under
        ``url_prefix`` or run standalone via ``app.run(...)``.
    """
    if registry is None:
        registry = RunRegistry()
        registry.rehydrate_from_sandbox(sandbox)

    if runner is None:
        runner = LocalRunner()

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
    inject_design_tokens(app)
    app.layout = build_run_console_layout(sandbox, registry=registry, runner=runner)
    app.server.config["pheno_url_prefix"] = url_prefix
    app.server.config["pheno_sandbox_root"] = str(sandbox.root)
    # Stream A reads the runner here (per its integration note) rather
    # than threading it through every callback closure.
    app.server.config["pheno_runner"] = runner
    app.server.config["pheno_registry"] = registry

    register_callbacks(app, sandbox, registry=registry, runner=runner)

    logger.debug(
        "Run console built: sandbox=%s url_prefix=%s",
        sandbox.root,
        url_prefix,
    )
    return app
