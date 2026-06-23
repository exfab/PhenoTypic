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

import html
import re
from pathlib import Path
from typing import Any

import pytest

from phenotypic.gui.run_console._callbacks import _dashboard_url
from phenotypic.gui.shell import SandboxRoot, create_app


@pytest.fixture()
def sandbox(tmp_path: Path) -> SandboxRoot:
    return SandboxRoot.from_path(tmp_path)


OOD_NODE_PREFIX = "/node/hz01/30099/"


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


def _assert_content_type_prefix(resp: Any, expected: str) -> None:
    """Assert a Flask test response has a content type prefix."""
    content_type = resp.headers.get("content-type", "")
    assert content_type.startswith(expected), content_type


def _assert_not_dash_index(resp: Any) -> None:
    """Assert a route did not silently fall through to Dash index HTML."""
    body = resp.get_data(as_text=True)
    assert "<!DOCTYPE html>" not in body[:200]
    assert "react-entry-point" not in body[:500]


def _first_component_suite_script(index_html: str) -> str:
    """Return the first generated Dash component-suite JavaScript path."""
    match = re.search(r'src="([^"]+/_dash-component-suites/[^"]+\.js)"', index_html)
    assert match is not None, index_html[:1000]
    return html.unescape(match.group(1))


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


def test_results_layout_carries_shared_source_chrome(sandbox: SandboxRoot) -> None:
    """Viewer keeps its output-root flow while showing source chrome."""
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/results/_dash-layout")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    assert "shell-source-image-root-store" in text
    assert "shell-source-image-root-label" in text
    assert "source: unset" in text


def test_analysis_layout_carries_shared_source_chrome(sandbox: SandboxRoot) -> None:
    """Analysis gets shared source visibility without binding an output root."""
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/analysis/_dash-layout")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    assert "analysis-page" in text
    assert "shell-source-image-root-store" in text
    assert "shell-source-image-root-label" in text
    assert "source: unset" in text


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


def test_default_hub_nav_keeps_root_relative_prefixes(sandbox: SandboxRoot) -> None:
    """Default launch keeps the historical browser-visible mount prefixes."""
    app = create_app(sandbox)
    client = app.server.test_client()
    resp = client.get("/_dash-layout")
    assert resp.status_code == 200
    payload = resp.get_json()

    assert _find_string_in_json(payload, "/builder/") == "/builder/"
    assert _find_string_in_json(payload, "/node/hz01/30099/builder/") is None


def test_explicit_url_prefix_rewrites_chrome_navigation(
    sandbox: SandboxRoot,
) -> None:
    """Top-bar links use the browser-visible OOD prefix when supplied."""
    app = create_app(sandbox, url_prefix="/node/hz01/30099/")
    client = app.server.test_client()
    resp = client.get("/_dash-layout")
    assert resp.status_code == 200
    payload = resp.get_json()

    assert _find_string_in_json(payload, "/node/hz01/30099/") == (
        "/node/hz01/30099/"
    )
    assert _find_string_in_json(payload, "/node/hz01/30099/builder/") == (
        "/node/hz01/30099/builder/"
    )


def test_explicit_url_prefix_rewrites_results_browser_prefix(
    sandbox: SandboxRoot,
) -> None:
    """Dash assets for mounted sub-apps use the external OOD base path."""
    app = create_app(sandbox, url_prefix="/node/hz01/30099/")
    client = app.server.test_client()
    resp = client.get("/results/")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    assert 'window.__phenotypicAppPrefix = "/node/hz01/30099/results/"' in text


def test_explicit_url_prefix_rewrites_empty_state_api_fetches(
    sandbox: SandboxRoot,
) -> None:
    """Empty-state handoff callbacks POST to the prefixed sandbox API."""
    app = create_app(sandbox, url_prefix="/node/hz01/30099/")
    client = app.server.test_client()

    results_html = client.get("/results/").get_data(as_text=True)
    analysis_html = client.get("/analysis/").get_data(as_text=True)

    assert "/node/hz01/30099/sandbox/api/viewer/output-root" in results_html
    assert "/node/hz01/30099/sandbox/api/viewer/output-root" in analysis_html


def test_explicit_url_prefix_routes_backend_dash_endpoints(
    sandbox: SandboxRoot,
) -> None:
    """OOD /node forwards the full prefix, so Dash endpoints must strip it."""
    app = create_app(sandbox, url_prefix=OOD_NODE_PREFIX)
    client = app.server.test_client()

    layout = client.get(f"{OOD_NODE_PREFIX}_dash-layout")
    assert layout.status_code == 200
    _assert_content_type_prefix(layout, "application/json")
    assert "shell-source-image-root-store" in layout.get_data(as_text=True)

    deps = client.get(f"{OOD_NODE_PREFIX}_dash-dependencies")
    assert deps.status_code == 200
    _assert_content_type_prefix(deps, "application/json")
    _assert_not_dash_index(deps)


def test_explicit_url_prefix_routes_backend_assets_and_component_suites(
    sandbox: SandboxRoot,
) -> None:
    """CSS and generated Dash JS must not return the HTML app shell under /node."""
    app = create_app(sandbox, url_prefix=OOD_NODE_PREFIX)
    client = app.server.test_client()

    index = client.get(OOD_NODE_PREFIX)
    assert index.status_code == 200
    index_html = index.get_data(as_text=True)

    shell_css = client.get(f"{OOD_NODE_PREFIX}assets/shell.css")
    assert shell_css.status_code == 200
    _assert_content_type_prefix(shell_css, "text/css")
    _assert_not_dash_index(shell_css)
    assert "shell-" in shell_css.get_data(as_text=True)

    script_path = _first_component_suite_script(index_html)
    script = client.get(script_path)
    assert script.status_code == 200
    assert "javascript" in script.headers.get("content-type", "")
    _assert_not_dash_index(script)


def test_explicit_url_prefix_routes_backend_subapps_and_shell_blueprints(
    tmp_path: Path,
) -> None:
    """The prefix strip must run before DispatcherMiddleware and shell blueprints."""
    sandbox_dir = tmp_path / "sandbox"
    sandbox_dir.mkdir()
    out = sandbox_dir / "out"
    out.mkdir()
    (out / "dashboard.html").write_text("<html>dashboard</html>")
    sandbox = SandboxRoot.from_path(sandbox_dir)

    app = create_app(sandbox, url_prefix=OOD_NODE_PREFIX)
    client = app.server.test_client()

    for path, marker in (
        (f"{OOD_NODE_PREFIX}builder/_dash-layout", "builder-page-root"),
        (f"{OOD_NODE_PREFIX}results/_dash-layout", "results-viewer-empty-state"),
        (f"{OOD_NODE_PREFIX}run/_dash-layout", "run-console-root"),
        (f"{OOD_NODE_PREFIX}analysis/_dash-layout", "analysis-page"),
        (f"{OOD_NODE_PREFIX}tune/_dash-layout", "tune-"),
        (f"{OOD_NODE_PREFIX}browse/_dash-layout", "browse-page"),
    ):
        resp = client.get(path)
        assert resp.status_code == 200, path
        _assert_content_type_prefix(resp, "application/json")
        assert marker in resp.get_data(as_text=True)

    root_resp = client.get(f"{OOD_NODE_PREFIX}sandbox/api/root")
    assert root_resp.status_code == 200
    _assert_content_type_prefix(root_resp, "application/json")
    assert root_resp.get_json()["root"] == str(sandbox.root)

    run_resp = client.get(f"{OOD_NODE_PREFIX}runs/out/dashboard.html")
    assert run_resp.status_code == 200
    _assert_content_type_prefix(run_resp, "text/html")
    assert b"dashboard" in run_resp.data


def test_explicit_url_prefix_preserves_script_root_through_dispatcher(
    sandbox: SandboxRoot,
) -> None:
    """Flask should see both the external OOD prefix and internal mount."""
    from flask import Blueprint, jsonify, request

    app = create_app(sandbox, url_prefix=OOD_NODE_PREFIX)
    dispatcher = app.server.wsgi_app
    builder_flask = dispatcher.mounts["/builder"]

    bp = Blueprint("prefixed_probe", __name__)

    @bp.route("/_prefixed_probe")
    def _prefixed_probe() -> object:
        return jsonify(
            {
                "script_root": request.script_root,
                "path": request.path,
                "url": request.url,
            }
        )

    builder_flask.register_blueprint(bp)

    client = app.server.test_client()
    resp = client.get(f"{OOD_NODE_PREFIX}builder/_prefixed_probe")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["script_root"] == "/node/hz01/30099/builder"
    assert data["path"] == "/_prefixed_probe"
    assert "/node/hz01/30099/builder/_prefixed_probe" in data["url"]


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
    found = _find_string_in_json(payload, "dashboard_logo.svg")
    assert found is not None
    assert found.startswith("/builder/")


def test_explicit_url_prefix_rewrites_builder_logo(sandbox: SandboxRoot) -> None:
    """Builder layout assets carry both the OOD base prefix and sub-app prefix."""
    app = create_app(sandbox, url_prefix="/node/hz01/30099/")
    client = app.server.test_client()
    resp = client.get("/builder/_dash-layout")
    assert resp.status_code == 200
    payload = resp.get_json()

    found = _find_string_in_json(payload, "dashboard_logo.svg")
    assert found is not None
    assert found.startswith("/node/hz01/30099/builder/")


def test_builder_point_picker_assets_are_not_root_hardcoded(
    sandbox: SandboxRoot,
) -> None:
    """Point-picker OSD fallback assets derive from the injected app prefix."""
    app = create_app(sandbox, url_prefix="/node/hz01/30099/")
    client = app.server.test_client()
    resp = client.get("/builder/assets/point_picker.js")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)

    assert '"/results/assets/openseadragon' not in text
    assert "resultsPrefix + \"assets/openseadragon" in text


def test_dashboard_url_uses_explicit_url_prefix() -> None:
    """Run dashboard iframes use the browser-visible OOD base prefix."""
    assert _dashboard_url("out", url_prefix="/node/hz01/30099/") == (
        "/node/hz01/30099/runs/out/deliverables/dashboard.html"
    )


def test_shared_logo_served_under_each_mount(sandbox: SandboxRoot) -> None:
    """The single canonical ``dashboard_logo.svg`` is reachable under
    every sub-app's ``/_shared/`` URL prefix.

    Each sub-app's Flask server registers a shared-static blueprint that
    serves the same file from ``phenotypic/_assets/logos/``. This guards against
    accidental reintroduction of per-app duplicate copies.
    """
    app = create_app(sandbox)
    client = app.server.test_client()
    for mount in ("/builder", "/results", "/run", "/analysis"):
        resp = client.get(f"{mount}/_shared/dashboard_logo.svg")
        assert resp.status_code == 200, f"{mount} did not serve the shared logo"
        body = resp.get_data(as_text=True)
        assert body.startswith("<svg") or body.lstrip().startswith("<?xml"), (
            f"{mount} returned non-SVG content: {body[:120]!r}"
        )
