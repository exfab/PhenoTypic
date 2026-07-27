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
from uuid import uuid4

from phenotypic.gui._config import (
    CFG_ANALYSIS_SESSION,
    CFG_RESULTS_BINDING_STATE,
    CFG_RESULTS_BINDING_COORDINATOR,
    CFG_RESULTS_BINDING_JOBS,
    CFG_RUN_REGISTRY,
    CFG_RUNNER,
    CFG_VIEWER_SESSION,
    DEFAULT_URL_PREFIX,
    DEFAULT_IDLE_RELEASE_SECONDS,
    MOUNT_ANALYSIS,
    MOUNT_BROWSE,
    MOUNT_BUILDER,
    MOUNT_HOME,
    MOUNT_RUN,
    MOUNT_TUNE,
    MOUNT_VIEWER,
    TITLE_HUB,
    join_url_prefix,
    normalize_url_prefix,
)
from phenotypic.gui._binding_generation import BindingRequestFence
from phenotypic.gui.shell._home import build_home_layout
from phenotypic.gui.shell._ids import (
    SHELL_TAB_ANALYSIS,
    SHELL_TAB_BROWSE,
    SHELL_TAB_BUILDER,
    SHELL_TAB_HOME,
    SHELL_TAB_RUN,
    SHELL_TAB_TUNE,
    SHELL_TAB_VIEWER,
)
from phenotypic.gui.shell._layout import wrap_in_chrome
from phenotypic.gui._shared import register_shared_static
from phenotypic.gui.shell._routes import register_sandbox_api
from phenotypic.gui.shell._binding import BindingCoordinator
from phenotypic.gui.shell._binding_jobs import (
    ResultsBindJobContext,
    ResultsBindJobFailure,
    ResultsBindJobManager,
)
from phenotypic.gui.shell._runs_blueprint import register as register_runs
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._session import (
    ToolSession,
    start_idle_release_thread,
    swap_tool_session_states,
)
from phenotypic.gui._url_prefix import configure_url_prefix_routing

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["create_app", "compose_hub"]


# ---------------------------------------------------------------------------
# _ViewerProxy — per-request resolution of the viewer's WSGI app
# ---------------------------------------------------------------------------

class _SessionProxy:
    """WSGI callable that resolves a tool's Dash app per request.

    The dispatcher's mount point is fixed at composition time, but the
    underlying Dash instance changes whenever the :class:`ToolSession`
    rebuilds (release + first ``get`` after release). Going through a
    proxy keeps the dispatcher's mount stable while the wrapped state
    floats: each request asks the session for the current Dash app and
    forwards the WSGI tuple to its Flask ``wsgi_app``.

    ``ToolSession.get()`` is itself thread-safe and updates
    ``_last_access`` so a steady stream of requests prevents the idle
    daemon from releasing the tool mid-session.
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


# Back-compat alias — the analysis sub-app reuses the same proxy machinery,
# so the historical ``_ViewerProxy`` name now points at the generic class.
_ViewerProxy = _SessionProxy
_AnalysisProxy = _SessionProxy


# ---------------------------------------------------------------------------
# Shell-only Dash builder (used as the dispatcher's default fallback app)
# ---------------------------------------------------------------------------

def _build_shell_dash_app(
    sandbox: SandboxRoot,
    *,
    url_prefix: str = MOUNT_HOME,
    viewer_session: "ToolSession[Any] | None" = None,
    viewer_state: "dict[str, Any] | None" = None,
    extra_release_sessions: "tuple[ToolSession[Any], ...] | None" = None,
    bind_output: "Callable[[Path | None], Any] | None" = None,
    binding_jobs: "ResultsBindJobManager | None" = None,
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
        routes_pathname_prefix=MOUNT_HOME,
    )
    app.layout = html.Div(build_home_layout(sandbox), className="shell-page")
    wrap_in_chrome(
        app,
        active_tab=SHELL_TAB_HOME,
        sandbox=sandbox,
        url_prefix=url_prefix,
    )

    register_sandbox_api(
        app.server,
        sandbox,
        viewer_session=viewer_session,
        viewer_state=viewer_state,
        extra_release_sessions=extra_release_sessions,
        bind_output=bind_output,
        binding_jobs=binding_jobs,
        browser_url_prefix=url_prefix,
    )
    register_runs(app.server, sandbox, viewer_session=viewer_session)
    register_shared_static(app.server)
    return app


# ---------------------------------------------------------------------------
# Composer — the full hub
# ---------------------------------------------------------------------------

def compose_hub(
    sandbox: SandboxRoot,
    *,
    url_prefix: str = DEFAULT_URL_PREFIX,
    idle_release_seconds: float = DEFAULT_IDLE_RELEASE_SECONDS,
    start_idle_thread: bool = True,
    binding_drain_timeout_seconds: float = 5.0,
    progress: Callable[[str], None] | None = None,
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
        binding_drain_timeout_seconds: Maximum time an output Refresh waits
            for callbacks admitted by the previous binding to finish.
        progress: Optional callback invoked with a short label before each
            eager sub-app is built (``"sub-app modules"``, ``"shell"``,
            ``"builder"``, …). The launcher passes
            :meth:`StartupReporter.detail` so the startup bar reflects which
            sub-app is currently being composed. ``None`` (default) is a
            no-op for non-interactive / test callers.
    """
    base_url_prefix = normalize_url_prefix(url_prefix)

    def _tick(label: str) -> None:
        if progress is not None:
            progress(label)

    # Local imports to keep boot-time cycles minimal.
    _tick("sub-app modules")
    from phenotypic.gui import (
        analysis,
        browse,
        builder,
        results_viewer,
        run_console,
        tune,
    )
    from phenotypic.gui.results_viewer._output_root import (
        OutputRoot,
        OutputSnapshotChangedError,
        sandbox_viewer_cache_root,
    )

    # Shared binding record. A successful bind or explicit Refresh publishes
    # all four fields together with the paired ToolSession states.
    initial_binding_fence = BindingRequestFence()
    viewer_state: dict[str, Any] = {
        "bound_path": None,
        "output_root": None,
        "snapshot": None,
        "status": "unavailable",
        "error": None,
        "binding_generation": str(uuid4()),
        "binding_fence": initial_binding_fence,
    }
    binding_coordinator = BindingCoordinator()

    def _make_viewer(
        output_root: OutputRoot | None,
        binding_generation: str,
        binding_fence: BindingRequestFence,
    ) -> dash.Dash:
        viewer_app = results_viewer.create_app(
            output_root=output_root,
            url_prefix=join_url_prefix(base_url_prefix, MOUNT_VIEWER),
            api_url_prefix=base_url_prefix,
            binding_generation=binding_generation,
            binding_fence=binding_fence,
        )
        wrap_in_chrome(
            viewer_app,
            active_tab=SHELL_TAB_VIEWER,
            sandbox=sandbox,
            url_prefix=base_url_prefix,
        )
        return viewer_app

    # 1. Viewer session (lazy — heavy parquet load deferred to first GET).
    def _build_viewer() -> dash.Dash:
        return _make_viewer(
            viewer_state["output_root"],
            viewer_state["binding_generation"],
            viewer_state["binding_fence"],
        )

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

    def _make_analysis(
        output_root: OutputRoot | None,
        binding_generation: str,
        binding_fence: BindingRequestFence,
    ) -> dash.Dash:
        analysis_app = analysis.create_app(
            output_root=output_root,
            url_prefix=join_url_prefix(base_url_prefix, MOUNT_ANALYSIS),
            api_url_prefix=base_url_prefix,
            binding_generation=binding_generation,
            binding_fence=binding_fence,
        )
        wrap_in_chrome(
            analysis_app,
            active_tab=SHELL_TAB_ANALYSIS,
            sandbox=sandbox,
            url_prefix=base_url_prefix,
        )
        return analysis_app

    # 1b. Analysis shares the same binding record as Results.
    def _build_analysis() -> dash.Dash:
        return _make_analysis(
            viewer_state["output_root"],
            viewer_state["binding_generation"],
            viewer_state["binding_fence"],
        )

    def _teardown_analysis(_app: dash.Dash) -> None:
        del _app  # pragma: no cover

    analysis_session: ToolSession[dash.Dash] = ToolSession(
        "analysis",
        build=_build_analysis,
        teardown=_teardown_analysis,
    )

    def _bind_output(context: ResultsBindJobContext) -> dict[str, Any]:
        """Build candidates off-lock and publish one fenced revision."""
        selected = context.target
        try:
            candidate_root = OutputRoot.discover(
                selected,
                cache_root=sandbox_viewer_cache_root(sandbox.root),
                cancellation=context.cancellation,
                progress_callback=context.report_discovery,
            )
            context.set_phase(
                "building_results",
                "Building the Results session candidate.",
            )
            binding_generation = str(uuid4())
            candidate_fence = BindingRequestFence()
            candidate_viewer = _make_viewer(
                candidate_root,
                binding_generation,
                candidate_fence,
            )
            context.set_phase(
                "building_analysis",
                "Building the Analysis session candidate.",
            )
            candidate_analysis = _make_analysis(
                candidate_root,
                binding_generation,
                candidate_fence,
            )
            context.set_phase(
                "publishing",
                "Publishing the shared Results and Analysis revision.",
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ResultsBindJobFailure("invalid", str(exc)) from exc
        except OutputSnapshotChangedError as exc:
            raise ResultsBindJobFailure("stale", str(exc)) from exc

        def _commit_binding() -> None:
            # Runs with both ToolSession locks held. This closes the final
            # candidate-build-to-publish gap and acts as the request CAS.
            def _publish_latest() -> None:
                viewer_state.update(
                    {
                        "bound_path": selected,
                        "output_root": candidate_root,
                        "snapshot": candidate_root.snapshot,
                        "status": "current",
                        "error": None,
                        "binding_generation": binding_generation,
                        "binding_fence": candidate_fence,
                    }
                )

            binding_coordinator.commit_if_latest(
                context.ticket,
                _publish_latest,
            )

        snapshot = candidate_root.snapshot
        publication_result = {
            "abs_path": str(candidate_root.root),
            "binding_generation": binding_generation,
            "snapshot": {
                "processing_fingerprint": snapshot.processing_fingerprint,
                "consumed_state_fingerprint": (
                    snapshot.consumed_state_fingerprint
                ),
                "captured_at": snapshot.captured_at.isoformat(),
                "active_run": snapshot.active_run,
            },
        }

        # Discovery and both candidate constructors above intentionally run
        # outside this short publication lock. Only callback draining and the
        # paired session/state CAS are serialized.
        try:
            context.require_active()
            with binding_coordinator.serialized():
                context.require_active()
                binding_coordinator.require_latest(context.ticket)
                old_fence = viewer_state["binding_fence"]
                if not isinstance(old_fence, BindingRequestFence):
                    raise RuntimeError(
                        "bound output request fence is unavailable"
                    )
                try:
                    old_fence.close_and_wait(
                        timeout_seconds=binding_drain_timeout_seconds,
                    )
                    candidate_root.require_session_snapshot_current(
                        context="Shared Results/Analysis publish",
                    )
                    context.commit_publication(
                        lambda: swap_tool_session_states(
                            (
                                (viewer_session, candidate_viewer),
                                (analysis_session, candidate_analysis),
                            ),
                            commit=_commit_binding,
                        ),
                        result=publication_result,
                    )
                except Exception:
                    old_fence.reopen()
                    raise
        except OutputSnapshotChangedError as exc:
            raise ResultsBindJobFailure("stale", str(exc)) from exc
        return publication_result

    binding_jobs = ResultsBindJobManager(
        _bind_output,
        issue_ticket=binding_coordinator.issue_request,
    )

    # 2. Shell Dash (registers the API + runs blueprints with the
    #    viewer-session touch hook + atomic Results/Analysis binder).
    _tick("shell")
    shell_app = _build_shell_dash_app(
        sandbox,
        url_prefix=base_url_prefix,
        viewer_session=viewer_session,
        viewer_state=viewer_state,
        extra_release_sessions=(analysis_session,),
        binding_jobs=binding_jobs,
    )
    shell_app.server.config[CFG_VIEWER_SESSION] = viewer_session
    shell_app.server.config[CFG_ANALYSIS_SESSION] = analysis_session
    shell_app.server.config[CFG_RESULTS_BINDING_STATE] = viewer_state
    shell_app.server.config[CFG_RESULTS_BINDING_COORDINATOR] = (
        binding_coordinator
    )
    shell_app.server.config[CFG_RESULTS_BINDING_JOBS] = binding_jobs

    # 3. Builder Dash (eager — single-process registry build).
    _tick("builder")
    builder_app = builder.create_app(
        image_root=sandbox.root,
        url_prefix=join_url_prefix(base_url_prefix, MOUNT_BUILDER),
    )
    wrap_in_chrome(
        builder_app,
        active_tab=SHELL_TAB_BUILDER,
        sandbox=sandbox,
        url_prefix=base_url_prefix,
    )

    # 4. Run console Dash (eager). Build the process-wide runner +
    #    registry HERE so the shell's ``/runs/`` blueprint, the Recent
    #    Runs panel, and the run-console callbacks all share the same
    #    state. Rehydrate the registry from disk so historical runs are
    #    visible immediately without waiting for a refresh.
    _tick("run console")
    from phenotypic.gui.run_console._runner import LocalRunner
    from phenotypic.gui.run_console._slurm_observer import (
        SlurmLifecycleObserver,
    )
    from phenotypic.gui.run_console._app import SLURM_OBSERVER_EXTENSION
    from phenotypic.gui.shell._runs_registry import RunRegistry

    registry = RunRegistry()
    registry.rehydrate_from_sandbox(sandbox)
    runner = LocalRunner()
    slurm_observer = SlurmLifecycleObserver(registry)

    run_app = run_console.create_app(
        sandbox,
        url_prefix=join_url_prefix(base_url_prefix, MOUNT_RUN),
        server_url_prefix=base_url_prefix,
        registry=registry,
        runner=runner,
        slurm_observer=slurm_observer,
        start_slurm_observer=start_idle_thread,
    )
    wrap_in_chrome(
        run_app,
        active_tab=SHELL_TAB_RUN,
        sandbox=sandbox,
        url_prefix=base_url_prefix,
    )

    # 4b. Tune co-pilot Dash (eager). Read-only and lightweight — no heavy
    #     parquet load, so no ToolSession is needed. Mounted empty-state: the
    #     user binds a tune run from the sidebar (Chunk C), at which point the
    #     page re-reads the bound run. The factory stays optuna-free; the live
    #     study is opened lazily inside the Monitor poll callback only.
    _tick("tune")
    tune_app = tune.create_app(
        root=None,
        url_prefix=join_url_prefix(base_url_prefix, MOUNT_TUNE),
        sandbox=sandbox,
        registry=registry,
        runner=runner,
    )
    wrap_in_chrome(
        tune_app,
        active_tab=SHELL_TAB_TUNE,
        sandbox=sandbox,
        url_prefix=base_url_prefix,
    )

    # 4c. Browse Dash (eager — lightweight source-image viewer). No
    #     ToolSession: it loads no heavy parquet, just lists files + serves
    #     ephemeral tiles.
    _tick("browse")
    browse_app = browse.create_app(
        sandbox, url_prefix=join_url_prefix(base_url_prefix, MOUNT_BROWSE)
    )
    wrap_in_chrome(
        browse_app,
        active_tab=SHELL_TAB_BROWSE,
        sandbox=sandbox,
        url_prefix=base_url_prefix,
    )

    # Stash on the shell server too so any future cross-tool callback
    # (e.g. the sidebar's "open in run console" hand-off) can reach the
    # same singletons.
    shell_app.server.config[CFG_RUNNER] = runner
    shell_app.server.config[CFG_RUN_REGISTRY] = registry
    shell_app.server.extensions[SLURM_OBSERVER_EXTENSION] = slurm_observer

    # 5. Compose at the WSGI layer. The dispatcher receives the shell's
    #    Flask app as its default; any path not matching a mount prefix
    #    falls through to it (which carries the API + runs blueprints).
    viewer_proxy = _ViewerProxy(viewer_session)
    analysis_proxy = _AnalysisProxy(analysis_session)
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
            MOUNT_TUNE.rstrip("/"): tune_app.server,
            MOUNT_ANALYSIS.rstrip("/"): analysis_proxy,
            MOUNT_BROWSE.rstrip("/"): browse_app.server,
        },
    )

    logger.info(
        "GUI hub composed: sandbox=%s mounts=%s, %s, %s, %s, %s, %s",
        sandbox.root,
        MOUNT_BUILDER,
        MOUNT_VIEWER,
        MOUNT_RUN,
        MOUNT_TUNE,
        MOUNT_ANALYSIS,
        MOUNT_BROWSE,
    )

    if start_idle_thread:
        start_idle_release_thread(
            [viewer_session, analysis_session],  # type: ignore[list-item]
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
    progress: Callable[[str], None] | None = None,
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
        progress: Optional per-sub-app progress callback forwarded to
            :func:`compose_hub` (the launcher passes
            :meth:`StartupReporter.detail`). Ignored on the Phase 3
            ``viewer_session`` escape-hatch path.

    Returns:
        Configured :class:`dash.Dash` instance. ``app.run()`` (or
        Werkzeug ``run_simple``) starts the unified server.
    """
    if viewer_session is not None:
        # Phase 3 backwards-compat: test path injects a stub session
        # and stops at the shell Dash (no sub-app composition).
        app = _build_shell_dash_app(
            sandbox, url_prefix=url_prefix, viewer_session=viewer_session
        )
        return configure_url_prefix_routing(app, url_prefix)

    if start_idle_thread is None:
        # Don't leak daemon threads in pytest unless the test asks.
        import os

        start_idle_thread = "PYTEST_CURRENT_TEST" not in os.environ

    shell_app, _viewer_session = compose_hub(
        sandbox,
        url_prefix=url_prefix,
        idle_release_seconds=idle_release_seconds,
        start_idle_thread=start_idle_thread,
        progress=progress,
    )
    return configure_url_prefix_routing(shell_app, url_prefix)
