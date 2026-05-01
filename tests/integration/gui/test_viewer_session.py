"""Phase 5 viewer-session lifecycle through the dispatcher.

The viewer is hidden behind a :class:`ToolSession`; the dispatcher
forwards ``/results/*`` to a :class:`_ViewerProxy` that resolves the
session per request. After ``release()`` the next request rebuilds
the underlying Dash and the URL keeps working.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from phenotypic.gui.shell import SandboxRoot
from phenotypic.gui.shell._app import compose_hub
from phenotypic.gui.shell._session import start_idle_release_thread


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxRoot:
    return SandboxRoot.from_path(tmp_path)


def test_compose_hub_returns_session(sandbox: SandboxRoot) -> None:
    shell_app, viewer_session = compose_hub(sandbox, start_idle_thread=False)
    assert viewer_session.name == "viewer"
    # Lazy: not built until first request.
    assert viewer_session.is_built() is False


def test_first_results_request_builds_session(sandbox: SandboxRoot) -> None:
    shell_app, viewer_session = compose_hub(sandbox, start_idle_thread=False)
    assert viewer_session.is_built() is False
    client = shell_app.server.test_client()
    resp = client.get("/results/")
    assert resp.status_code == 200
    assert viewer_session.is_built() is True


def test_release_rebuilds_on_next_request(sandbox: SandboxRoot) -> None:
    """After ``release()`` the next ``/results/`` hit rebuilds the app.

    Confirms the ``_ViewerProxy`` resolves the session per request
    rather than caching the underlying Dash instance at composition.
    """
    shell_app, viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = shell_app.server.test_client()
    client.get("/results/")
    first_app = viewer_session.get()

    viewer_session.release()
    assert viewer_session.is_built() is False

    resp = client.get("/results/")
    assert resp.status_code == 200
    second_app = viewer_session.get()
    # Different Dash instance after rebuild.
    assert first_app is not second_app


def test_idle_release_thread_releases_built_session(
    sandbox: SandboxRoot,
) -> None:
    """``start_idle_release_thread`` releases sessions past the idle bound.

    Spawns the daemon ourselves with a tiny idle threshold + poll interval
    so the test runs in well under a second; uses a ``stop_event`` to
    terminate the thread cleanly so we don't leak a daemon.
    """
    shell_app, viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = shell_app.server.test_client()
    client.get("/results/")  # build the session
    assert viewer_session.is_built() is True

    stop_event = threading.Event()
    thread = start_idle_release_thread(
        [viewer_session],  # type: ignore[list-item]
        idle_release_seconds=0.05,
        poll_interval_seconds=0.05,
        stop_event=stop_event,
    )
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and viewer_session.is_built():
        time.sleep(0.05)
    stop_event.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert viewer_session.is_built() is False


def test_release_does_not_mutate_released_app_config(
    sandbox: SandboxRoot,
) -> None:
    """``_teardown_viewer`` is a no-op — the released app is GC'd as is.

    Eagerly popping ``filtered_state`` / ``output_root`` from the
    released app's ``server.config`` would race Phase-6 callbacks that
    read those keys mid-request (the proxy may have handed the released
    app to a request just before ``release()`` ran). The ToolSession
    teardown therefore avoids touching the released app at all, and
    relies on GC after all in-flight requests return.
    """
    shell_app, viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = shell_app.server.test_client()
    client.get("/results/")
    viewer_app = viewer_session.get()
    # Pre-release sentinel — the empty-state app does not set these
    # keys, so we plant fakes to confirm teardown leaves them alone.
    viewer_app.server.config["pheno_test_keep_me"] = "still here"
    viewer_session.release()
    # The released ``viewer_app`` is no longer reachable through the
    # session, but our local reference is still valid; teardown must
    # not have mutated it.
    assert viewer_app.server.config["pheno_test_keep_me"] == "still here"
