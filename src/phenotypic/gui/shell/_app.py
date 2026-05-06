"""Shell Dash factory + DispatcherMiddleware composer.

Phase 5 ships the unified hub: ``create_app(sandbox)`` returns a
:class:`dash.Dash` whose ``server.wsgi_app`` is wrapped in a
:class:`werkzeug.middleware.dispatcher.DispatcherMiddleware`. The
dispatcher routes:

    * ``/builder/...``  → builder Dash factory (eager — small).
    * ``/results/...``  → :class:`_ViewerProxy` over a viewer
      :class:`ToolSession` (lazy — viewer is heavy).
    * ``/run/...``      → run-console Dash factory (eager — small,
      Phase 5 placeholder; Phase 6 fills in the form/iframe/log/recents).
    * everything else   → the shell's Flask server. Flask blueprints
      (``/sandbox/api/*``, ``/runs/*``) are registered on that server
      so they answer regardless of which sub-app is active.

The viewer is wrapped in a :class:`ToolSession` because it loads heavy
parquet tables on every build; the session lets the user release+rebuild
to reclaim object-graph memory between explorations. Each Dash app gets
its chrome wrapped via :func:`wrap_in_chrome` so the top-bar tabs and
sidebar appear identically across mounts.

Standalone shell mode (Phase 3 backwards-compat): the same
``create_app(sandbox)`` is used by Phase 3 tests; the dispatcher's
fallback is the shell Flask, so every Phase 3 endpoint (``/``,
``/_dash-layout``, ``/sandbox/api/*``, ``/runs/*``) keeps working.
The sub-app mounts simply add new routable prefixes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import dash
import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import html
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from phenotypic.gui._config import (
    CFG_RUN_REGISTRY,
    CFG_RUNNER,
    DEFAULT_IDLE_RELEASE_SECONDS,
    MOUNT_BUILDER,
    MOUNT_HOME,
    MOUNT_RUN,
    MOUNT_VIEWER,
    TITLE_HUB,
)
from phenotypic.gui.shell._home import build_home_layout
from phenotypic.gui.shell._ids import (
    SHELL_TAB_BUILDER,
    SHELL_TAB_HOME,
    SHELL_TAB_RUN,
    SHELL_TAB_VIEWER,
)
from phenotypic.gui.shell._layout import wrap_in_chrome
from phenotypic.gui.shell._routes import register_sandbox_api
from phenotypic.gui.shell._runs_blueprint import register as register_runs
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._session import ToolSession, start_idle_release_thread

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["create_app", "compose_hub"]


# ---------------------------------------------------------------------------
# _ViewerProxy — per-request resolution of the viewer's WSGI app
# ---------------------------------------------------------------------------

class _ViewerProxy:
    """WSGI callable that resolves the viewer's app per request.

    The dispatcher's mount point is fixed at composition time, but the
    underlying Dash instance changes whenever the :class:`ToolSession`
    rebuilds (release + first ``get`` after release). Going through a
    proxy keeps the dispatcher's mount stable while the wrapped state
    floats: each request asks the session for the current Dash app and
    forwards the WSGI tuple to its Flask ``wsgi_app``.

    ``ToolSession.get()`` is itself thread-safe and updates
    ``_last_access`` so a steady stream of requests prevents the idle
    daemon from releasing the viewer mid-session.
    """

    def __init__(self, session: "ToolSession[dash.Dash]") -> None:
        self._session = session

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: Callable[..., Any],
    ) -> Any:
        app = self._session.get()
        return app.server.wsgi_app(environ, start_response)


# ---------------------------------------------------------------------------
# Shell-only Dash builder (used as the dispatcher's default fallback app)
# ---------------------------------------------------------------------------

def _build_shell_dash_app(
    sandbox: SandboxRoot,
    *,
    url_prefix: str = MOUNT_HOME,
    viewer_session: "ToolSession[Any] | None" = None,
    viewer_state: "dict[str, Any] | None" = None,
) -> dash.Dash:
    """Build the shell's home Dash (chrome + home pane + Flask blueprints).

    The returned Dash app is the dispatcher's default fallback in the
    composed hub: any path that doesn't start with ``/builder``,
    ``/results``, or ``/run`` reaches its Flask server, which carries
    the ``/sandbox/api/*`` and ``/runs/*`` blueprints alongside the
    home Dash routes.

    Args:
        sandbox: Frozen-at-launch sandbox root.
        url_prefix: Mount prefix. Hub uses ``"/"`` (the shell IS the
            fallback, so it serves the root); standalone same.
        viewer_session: Optional :class:`ToolSession` for the viewer.
            The Phase 2 blueprints call ``viewer_session.touch()``
            on each request so iframe-driven dashboard polls keep
            the viewer alive even when no Dash callback fires.
    """
    assets_folder = str(Path(__file__).parent / "_assets")

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=assets_folder,
        title=TITLE_HUB,
        requests_pathname_prefix=url_prefix,
        routes_pathname_prefix=url_prefix,
    )
    app.layout = html.Div(build_home_layout(sandbox), className="shell-page")
    wrap_in_chrome(app, active_tab=SHELL_TAB_HOME, sandbox=sandbox)

    register_sandbox_api(
        app.server,
        sandbox,
        viewer_session=viewer_session,
        viewer_state=viewer_state,
    )
    register_runs(app.server, sandbox, viewer_session=viewer_session)
    return app


# ---------------------------------------------------------------------------
# Composer — the full hub
# ---------------------------------------------------------------------------

def compose_hub(
    sandbox: SandboxRoot,
    *,
    idle_release_seconds: float = DEFAULT_IDLE_RELEASE_SECONDS,
    start_idle_thread: bool = True,
) -> tuple[dash.Dash, ToolSession[dash.Dash]]:
    """Build the shell + builder + viewer-session + run console; mount via DispatcherMiddleware.

    Returns:
        Tuple of ``(shell_app, viewer_session)``. The shell Dash's
        ``server.wsgi_app`` has been replaced with a
        :class:`DispatcherMiddleware` so HTTP requests through any of
        ``/``, ``/builder/...``, ``/results/...``, ``/run/...``,
        ``/sandbox/api/...``, ``/runs/...`` resolve correctly.

    Args:
        sandbox: Frozen-at-launch sandbox root.
        idle_release_seconds: How long the viewer session may go without
            a ``get`` or ``touch`` before the daemon releases it. The
            default (15 minutes) matches the spec.
        start_idle_thread: When ``True`` (default), spawn the daemon
            thread that releases idle sessions. Tests pass ``False``
            so they don't leak background threads.
    """
    # Local imports to keep boot-time cycles minimal.
    from phenotypic.gui import builder, results_viewer, run_console
    from phenotypic.gui.results_viewer._output_root import OutputRoot

    # Mutable handoff slot so the sidebar can hand a CLI output path to the
    # viewer without changing the ToolSession's build identity. The
    # ``/sandbox/api/viewer/output-root`` endpoint validates an incoming
    # path with ``OutputRoot.discover``, stamps the result here, and calls
    # ``viewer_session.release()``; the next GET to ``/results/`` rebuilds
    # the viewer with this OutputRoot.
    viewer_state: dict[str, "OutputRoot | None"] = {"output_root": None}

    # 1. Viewer session (lazy — heavy parquet load deferred to first GET).
    def _build_viewer() -> dash.Dash:
        viewer_app = results_viewer.create_app(
            output_root=viewer_state["output_root"], url_prefix=MOUNT_VIEWER
        )
        wrap_in_chrome(viewer_app, active_tab=SHELL_TAB_VIEWER, sandbox=sandbox)
        return viewer_app

    def _teardown_viewer(viewer_app: dash.Dash) -> None:
        # Intentionally a no-op: the released ``viewer_app`` is dropped
        # by ``ToolSession.release()`` and reaches GC once all in-flight
        # requests through ``_ViewerProxy`` finish. Eagerly popping
        # ``filtered_state`` / ``output_root`` from ``app.server.config``
        # would race in-flight callbacks reading those keys (a Phase-6
        # callback issuing ``current_app.config["filtered_state"]`` could
        # see ``KeyError``). Letting GC reclaim the heavy state via the
        # config-dict-of-the-released-app is one cycle slower but
        # race-free; Phase 6 can add an in-flight ref counter if eager
        # reclamation becomes a hard requirement.
        del viewer_app  # pragma: no cover - intentional no-op

    viewer_session: ToolSession[dash.Dash] = ToolSession(
        "viewer",
        build=_build_viewer,
        teardown=_teardown_viewer,
    )

    # 2. Shell Dash (registers the API + runs blueprints with the
    #    viewer-session touch hook wired in).
    shell_app = _build_shell_dash_app(
        sandbox,
        viewer_session=viewer_session,
        viewer_state=viewer_state,
    )

    # 3. Builder Dash (eager — single-process registry build).
    builder_app = builder.create_app(
        image_root=sandbox.root, url_prefix=MOUNT_BUILDER
    )
    wrap_in_chrome(builder_app, active_tab=SHELL_TAB_BUILDER, sandbox=sandbox)

    # 4. Run console Dash (eager). Build the process-wide runner +
    #    registry HERE so the shell's ``/runs/`` blueprint, the Recent
    #    Runs panel, and the run-console callbacks all share the same
    #    state. Rehydrate the registry from disk so historical runs are
    #    visible immediately without waiting for a refresh.
    from phenotypic.gui.run_console._runner import LocalRunner
    from phenotypic.gui.shell._runs_registry import RunRegistry

    registry = RunRegistry()
    registry.rehydrate_from_sandbox(sandbox)
    runner = LocalRunner()

    run_app = run_console.create_app(
        sandbox,
        url_prefix=MOUNT_RUN,
        registry=registry,
        runner=runner,
    )
    wrap_in_chrome(run_app, active_tab=SHELL_TAB_RUN, sandbox=sandbox)
    # Stash on the shell server too so any future cross-tool callback
    # (e.g. the sidebar's "open in run console" hand-off) can reach the
    # same singletons.
    shell_app.server.config[CFG_RUNNER] = runner
    shell_app.server.config[CFG_RUN_REGISTRY] = registry

    # 5. Compose at the WSGI layer. The dispatcher receives the shell's
    #    Flask app as its default; any path not matching a mount prefix
    #    falls through to it (which carries the API + runs blueprints).
    viewer_proxy = _ViewerProxy(viewer_session)
    # ``wsgi_app`` is the standard Flask seam for WSGI middleware
    # injection (this is the same recipe Werkzeug docs recommend).
    # DispatcherMiddleware mount keys are prefixes WITHOUT the trailing "/"
    # (e.g. "/builder", not "/builder/"). Strip the trailing slash from
    # the MOUNT_* constants by index.
    shell_app.server.wsgi_app = DispatcherMiddleware(  # type: ignore[method-assign]
        shell_app.server.wsgi_app,
        {
            MOUNT_BUILDER.rstrip("/"): builder_app.server,
            MOUNT_VIEWER.rstrip("/"): viewer_proxy,
            MOUNT_RUN.rstrip("/"): run_app.server,
        },
    )

    logger.info(
        "GUI hub composed: sandbox=%s mounts=%s, %s, %s",
        sandbox.root,
        MOUNT_BUILDER,
        MOUNT_VIEWER,
        MOUNT_RUN,
    )

    if start_idle_thread:
        start_idle_release_thread(
            [viewer_session],  # type: ignore[list-item]
            idle_release_seconds=idle_release_seconds,
        )

    return shell_app, viewer_session


def create_app(
    sandbox: SandboxRoot,
    *,
    url_prefix: str = MOUNT_HOME,
    viewer_session: "ToolSession[Any] | None" = None,
    idle_release_seconds: float = DEFAULT_IDLE_RELEASE_SECONDS,
    start_idle_thread: bool | None = None,
) -> dash.Dash:
    """Build the unified GUI hub Dash app.

    Phase 5 default: composes the full hub (shell + builder + viewer
    via :class:`ToolSession` + run console) and returns the shell's
    Dash app with its ``server.wsgi_app`` replaced by a
    :class:`DispatcherMiddleware`. Tests can hit any HTTP route through
    ``app.server.test_client()`` (which goes through ``wsgi_app``).

    Backwards-compat for Phase 3 tests: passing ``viewer_session``
    explicitly opts out of full composition — the call returns just
    the shell Dash app with the API + runs blueprints registered
    against the supplied session. Phase 3 lifecycle tests rely on this
    to inject a stub session and assert ``touch()`` is called.

    Args:
        sandbox: Frozen-at-launch sandbox root.
        url_prefix: Reserved for future composer nesting. Hub keeps
            it at ``"/"`` since the shell Dash is itself the dispatcher's
            fallback. Standalone shell launches accept any value.
        viewer_session: Phase 3 escape hatch — pass to test the shell
            in isolation against a specific session. When set,
            ``create_app`` returns just the shell Dash without composing
            the sub-apps. ``None`` (default) triggers the full hub
            composition.
        idle_release_seconds: Forwarded to :func:`compose_hub`.
        start_idle_thread: Forwarded to :func:`compose_hub`. When
            ``None`` (default) the daemon is started under production
            launch but skipped under pytest (``PYTEST_CURRENT_TEST``
            in env). Tests that want the daemon explicitly should
            pass ``True``.

    Returns:
        Configured :class:`dash.Dash` instance. ``app.run()`` (or
        Werkzeug ``run_simple``) starts the unified server.
    """
    if viewer_session is not None:
        # Phase 3 backwards-compat: test path injects a stub session
        # and stops at the shell Dash (no sub-app composition).
        return _build_shell_dash_app(
            sandbox, url_prefix=url_prefix, viewer_session=viewer_session
        )

    if start_idle_thread is None:
        # Don't leak daemon threads in pytest unless the test asks.
        import os

        start_idle_thread = "PYTEST_CURRENT_TEST" not in os.environ

    shell_app, _viewer_session = compose_hub(
        sandbox,
        idle_release_seconds=idle_release_seconds,
        start_idle_thread=start_idle_thread,
    )
    return shell_app
