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

import atexit
import logging
import os
from pathlib import Path
from uuid import UUID

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

from phenotypic.gui._config import (
    CFG_RUN_REGISTRY,
    CFG_RUNNER,
    CFG_SANDBOX_ROOT,
    CFG_URL_PREFIX,
    DEFAULT_URL_PREFIX,
    MOUNT_HOME,
    TITLE_RUN,
)
from phenotypic.gui._design import inject_design_tokens
from phenotypic.gui._shared import register_shared_static
from phenotypic.gui._url_prefix import configure_url_prefix_routing
from phenotypic.gui.run_console._callbacks import register_callbacks
from phenotypic.gui.run_console._layout import build_run_console_layout
from phenotypic.gui.run_console._runner import LocalRunner
from phenotypic.gui.run_console._slurm_observer import SlurmLifecycleObserver
from phenotypic.gui.shell._runs_registry import RunRegistry
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle

logger = logging.getLogger(__name__)

__all__ = ["SLURM_OBSERVER_EXTENSION", "create_app"]

SLURM_OBSERVER_EXTENSION = "phenotypic_slurm_observer"


def _bind_rehydrated_slurm_records(
    observer: SlurmLifecycleObserver,
    registry: RunRegistry,
) -> None:
    """Restore exact lifecycle bindings for nonterminal durable records."""
    for record in registry.list():
        if (
            record.mode != "slurm"
            or record.generation is None
            or record.status in {"complete", "failed", "cancelled"}
        ):
            continue
        lifecycle = load_slurm_lifecycle(record.output_dir)
        if lifecycle is None:
            continue
        try:
            scheduler_generation = UUID(str(lifecycle["generation"]))
            observer.bind_generation(
                run_id=record.run_id,
                record_generation=record.generation,
                scheduler_generation=scheduler_generation,
            )
        except (KeyError, ValueError):
            logger.warning(
                "Could not restore SLURM lifecycle binding for %s",
                record.run_id,
                exc_info=True,
            )


def create_app(
    sandbox: SandboxRoot,
    *,
    url_prefix: str = MOUNT_HOME,
    server_url_prefix: str = DEFAULT_URL_PREFIX,
    registry: RunRegistry | None = None,
    runner: LocalRunner | None = None,
    slurm_observer: SlurmLifecycleObserver | None = None,
    start_slurm_observer: bool | None = None,
) -> dash.Dash:
    """Build the Run console Dash app.

    Args:
        sandbox: Frozen-at-launch sandbox root. Constrains the file
            pickers + drives Recent Runs scan.
        url_prefix: Mount-point prefix. Defaults to ``"/"`` (standalone
            launcher); the hub composer passes ``"/run/"``.
        server_url_prefix: Browser-visible base prefix for shell-level
            Flask routes such as ``/runs``. Defaults to ``"/"``.
        registry: Process-wide :class:`RunRegistry`. When ``None`` (the
            standalone case) a fresh local registry is built and the
            sandbox is scanned to seed Recent Runs.
        runner: Process-wide :class:`LocalRunner`. When ``None`` a
            fresh local runner is built. Production callers should
            ALWAYS pass a shared runner so a navigation-away does not
            kill in-flight subprocesses (the hub keeps the runner alive
            even when the Run console UI is released).
        slurm_observer: Process-wide scheduler lifecycle observer. A fresh
            observer is created for standalone use when omitted.
        start_slurm_observer: Whether to start the observer daemon. Defaults
            to production-on and pytest-off.

    Returns:
        A configured :class:`dash.Dash` instance ready to mount under
        ``url_prefix`` or run standalone via ``app.run(...)``.
    """
    if registry is None:
        registry = RunRegistry()
        registry.rehydrate_from_sandbox(sandbox)

    if runner is None:
        runner = LocalRunner()
    if slurm_observer is None:
        slurm_observer = SlurmLifecycleObserver(registry)
    _bind_rehydrated_slurm_records(slurm_observer, registry)
    if start_slurm_observer is None:
        start_slurm_observer = "PYTEST_CURRENT_TEST" not in os.environ
    if start_slurm_observer:
        slurm_observer.start()
        atexit.register(slurm_observer.stop)

    assets_folder = str(Path(__file__).parent / "_assets")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=assets_folder,
        title=TITLE_RUN,
        # Dispatcher strips mount prefix; Dash must route at "/".
        requests_pathname_prefix=url_prefix,
        routes_pathname_prefix=MOUNT_HOME,
    )
    inject_design_tokens(app)
    register_shared_static(app.server)
    app.layout = build_run_console_layout(
        sandbox, registry=registry, runner=runner, url_prefix=url_prefix
    )
    app.server.config[CFG_URL_PREFIX] = url_prefix
    app.server.config[CFG_SANDBOX_ROOT] = str(sandbox.root)
    # Stream A reads the runner here (per its integration note) rather
    # than threading it through every callback closure.
    app.server.config[CFG_RUNNER] = runner
    app.server.config[CFG_RUN_REGISTRY] = registry
    app.server.extensions[SLURM_OBSERVER_EXTENSION] = slurm_observer

    register_callbacks(
        app,
        sandbox,
        registry=registry,
        runner=runner,
        slurm_observer=slurm_observer,
        server_url_prefix=server_url_prefix,
    )

    logger.debug(
        "Run console built: sandbox=%s url_prefix=%s",
        sandbox.root,
        url_prefix,
    )
    return configure_url_prefix_routing(app, url_prefix)
