"""Smoke tests for the Phase 5 composed hub.

Boots the unified hub via :func:`phenotypic.gui.shell.create_app` and
exercises each mount point through the test client. Confirms:

* ``/`` (shell home) returns 200.
* ``/builder/``, ``/results/``, ``/run/`` each route to a Dash app
  that returns 200 (the index page is a React loader; what matters is
  the dispatcher correctly forwarded the request).
* ``/sandbox/api/*`` and ``/runs/*`` (registered on the shell's Flask
  fallback) still work — they are NOT shadowed by the dispatcher.
* ``/runs/<missing>`` returns 404.
* ``/results/_dash-layout`` returns the empty-state placeholder when
  no ``OutputRoot`` is selected.
* The dispatcher correctly strips the mount prefix so that each Dash
  app can route at ``/`` internally.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.gui.shell import SandboxRoot, create_app


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxRoot:
    return SandboxRoot.from_path(tmp_path)


def _find_string_in_json(node: object, needle: str) -> str | None:
    """Walk a JSON-like dict/list tree, return the first string containing needle."""
    if isinstance(node, str):
        if needle in node:
            return node
        return None
    if isinstance(node, dict):
        for v in node.values():
            found = _find_string_in_json(v, needle)
            if found is not None:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _find_string_in_json(v, needle)
            if found is not None:
                return found
    return None


def test_shell_home_returns_200(sandbox: SandboxRoot) -> None:
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/")
    assert resp.status_code == 200


def test_builder_mount_routes(sandbox: SandboxRoot) -> None:
    """``/builder/`` reaches the builder Dash app's index."""
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/builder/")
    assert resp.status_code == 200
    # Dash index page is a React loader; assert it carries the
    # ``react-entry-point`` div Dash injects.
    assert b"react-entry-point" in resp.data


def test_run_console_mount_routes(sandbox: SandboxRoot) -> None:
    """``/run/`` reaches the run-console Dash app's index."""
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/run/")
    assert resp.status_code == 200
    assert b"react-entry-point" in resp.data


def test_results_mount_routes_to_empty_state(sandbox: SandboxRoot) -> None:
    """``/results/`` lazily builds the empty-state viewer."""
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/results/")
    assert resp.status_code == 200
    assert b"react-entry-point" in resp.data


def test_results_layout_is_empty_state(sandbox: SandboxRoot) -> None:
    """``/results/_dash-layout`` carries the empty-state placeholder."""
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/results/_dash-layout")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    # Empty-state layout root id (build_empty_state_layout in viewer/_layout.py).
    assert "results-viewer-empty-state" in text


def test_sandbox_api_falls_through_dispatcher(sandbox: SandboxRoot) -> None:
    """``/sandbox/api/root`` is on the shell Flask, not under any mount."""
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/sandbox/api/root")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["root"] == str(sandbox.root)


def test_runs_blueprint_falls_through_dispatcher(tmp_path: Path) -> None:
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    out = sandbox_dir / "out"
    out.mkdir()
    (out / "dashboard.html").write_text("<html/>")
    sandbox = SandboxRoot.from_path(sandbox_dir)

    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/runs/out/dashboard.html")
    assert resp.status_code == 200
    assert b"<html/>" in resp.data


def test_runs_missing_returns_404(sandbox: SandboxRoot) -> None:
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/runs/no/such/file.html")
    assert resp.status_code == 404


def test_builder_dash_layout_endpoint(sandbox: SandboxRoot) -> None:
    """The builder's ``/_dash-layout`` resolves at ``/builder/_dash-layout``.

    Dispatcher strips ``/builder``; the builder Dash routes at ``/`` and
    therefore answers ``/_dash-layout`` (which the dispatcher reaches as
    ``/builder/_dash-layout``).
    """
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/builder/_dash-layout")
    assert resp.status_code == 200


def test_run_console_dash_layout_endpoint(sandbox: SandboxRoot) -> None:
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/run/_dash-layout")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    # Phase 5 placeholder root id.
    assert "run-console-root" in text


def test_dispatcher_threads_script_root(sandbox: SandboxRoot) -> None:
    """``DispatcherMiddleware`` sets ``SCRIPT_NAME`` so ``flask.request.script_root``
    reflects the mount point; Phase 6 callbacks rely on this for URL building.

    We attach a probe blueprint to the builder's Flask server, then issue
    a request through the shell composer's dispatcher to ``/builder/_probe``.
    The blueprint reports back what Flask sees for ``script_root`` /
    ``path`` / ``url`` — confirming the dispatcher correctly mutates the
    WSGI environ.
    """
    from flask import Blueprint, jsonify, request

    app = create_app(sandbox)
    # ``app.server.wsgi_app`` is the DispatcherMiddleware. To reach the
    # builder's Flask server we have to look it up through the dispatcher's
    # mounts dict.
    dispatcher = app.server.wsgi_app
    builder_flask = dispatcher.mounts["/builder"]

    bp = Blueprint("probe", __name__)

    @bp.route("/_probe")
    def _probe() -> object:
        return jsonify(
            {
                "script_root": request.script_root,
                "path": request.path,
                "url": request.url,
            }
        )

    builder_flask.register_blueprint(bp)

    client = app.server.test_client()
    resp = client.get("/builder/_probe")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["script_root"] == "/builder"
    assert data["path"] == "/_probe"
    assert "/builder/_probe" in data["url"]


def test_results_assets_are_prefix_aware(sandbox: SandboxRoot) -> None:
    """The viewer index injects ``window.__phenotypicAppPrefix``.

    ``results_viewer.js`` reads this to build hub-aware DZI tile URLs.
    """
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/results/")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    assert 'window.__phenotypicAppPrefix = "/results/"' in text


def test_builder_logo_uses_prefix(sandbox: SandboxRoot) -> None:
    """Builder logo ``<img src=...>`` carries the ``/builder/`` prefix.

    ``/_dash-layout`` is JSON; Dash escapes ``/`` as ``\\u002f`` in the
    payload, so the assertion checks for the encoded form.
    """
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/builder/_dash-layout")
    assert resp.status_code == 200
    payload = resp.get_json()
    # Walk the JSON tree looking for the logo Img's src.
    found = _find_string_in_json(payload, "pheno_logo.png")
    assert found is not None
    assert found.startswith("/builder/")
