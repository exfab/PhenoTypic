# OOD Node Prefix Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `phenotypic-gui --url-prefix=/node/<host>/<port>/` work behind Open OnDemand `/node` while preserving existing `/rnode`, plain SSH tunnel, and mounted sub-app behavior.

**Architecture:** Keep Dash `requests_pathname_prefix` as the browser-visible prefix so generated URLs still include `/node/<host>/<port>/`. Add a small shared WSGI middleware that strips that same prefix from incoming backend `PATH_INFO` before Flask, Dash, or the hub `DispatcherMiddleware` route the request. Apply the middleware after hub composition for `phenotypic-gui`, and inside standalone GUI app factories so every launcher that exposes `--url-prefix` has the same behavior.

**Tech Stack:** Python 3.12, Dash, Flask/Werkzeug WSGI, `uv`, pytest, ruff, mypy.

---

## Background And Evidence

Observed user command:

```bash
phenotypic-gui --root /rhome/ejaco020 --host 0.0.0.0 --port 42793 --url-prefix=/node/hz01/42793/
```

Observed failure:

- The browser remains on Dash's static `Loading...` placeholder.
- Devtools shows JavaScript and CSS requests returning `200` but classified as `html`.
- The server log shows backend requests under
  `/node/hz01/42793/_dash-component-suites/`.
- A local in-process probe reproduced that `/node/hz01/42793/_dash-layout`, `/node/hz01/42793/assets/shell.css`, `/node/hz01/42793/builder/_dash-layout`, and `/node/hz01/42793/sandbox/api/root` all return Dash HTML, while unprefixed `/_dash-layout`, `/builder/_dash-layout`, and `/sandbox/api/root` route correctly.

Official Open OnDemand behavior:

- `/node/<host>/<port>/<app-path>` forwards the full request path to the backend.
- `/rnode/<host>/<port>/<app-path>` strips the reverse-proxy routing prefix and forwards only the relative app path.
- Source: Open OnDemand `ood_portal.yml` reverse proxy documentation, section "Configure Reverse Proxy": https://osc.github.io/ood-documentation/latest/reference/files/ood-portal-yml.html#configure-reverse-proxy

Current PhenoTypic behavior:

- `normalize_url_prefix` and `join_url_prefix` build browser-visible paths in `src/phenotypic/gui/_config.py`.
- The shell and sub-app factories set `requests_pathname_prefix=url_prefix` and `routes_pathname_prefix="/"`.
- The hub dispatcher mounts internal paths only: `/builder`, `/results`, `/run`, `/tune`, `/analysis`, and `/browse`.
- Existing tests verify generated prefixed URLs, but they still request unprefixed backend paths. They do not test what OOD `/node` actually sends to the backend.

## Non-Goals

- Do not add authentication or replace the development server.
- Do not change the public meaning of `--url-prefix`; it remains a path-only browser-visible prefix.
- Do not change internal mount constants such as `MOUNT_BUILDER` or `MOUNT_VIEWER`.
- Do not hand-code OOD-specific `/node` parsing. The fix must work for any explicit path prefix.

## File Structure

- Create `src/phenotypic/gui/_url_prefix.py`
  - Owns the shared WSGI prefix-strip middleware.
  - Owns a small install helper that mutates a Dash app's Flask `wsgi_app`.
  - Does not import shell, builder, viewer, or other sub-app modules.
- Modify `src/phenotypic/gui/shell/_app.py`
  - Installs the middleware around the fully composed hub dispatcher.
  - Installs the middleware for the shell-only test escape hatch.
- Modify these standalone-capable sub-app factories:
  - `src/phenotypic/gui/builder/_app.py`
  - `src/phenotypic/gui/results_viewer/_app.py`
  - `src/phenotypic/gui/run_console/_app.py`
  - `src/phenotypic/gui/analysis/_app.py`
  - `src/phenotypic/gui/tune/_app.py`
  - `src/phenotypic/gui/browse/_app.py`
- Modify `src/phenotypic/gui/_config.py`
  - Update launcher help text and banner guidance.
  - Export the new helper only if the final implementation places it there. Prefer `src/phenotypic/gui/_url_prefix.py`.
- Modify `src/phenotypic/gui/CLAUDE.md`
  - Correct the `--url-prefix` explanation for both OOD `/node` and `/rnode`.
- Modify `src/phenotypic/gui/FEATURES.md`
  - Required because CI gates edits under `src/phenotypic/gui/`.
- Modify tests:
  - `tests/unit/gui/test_url_prefix.py` or `tests/unit/gui/test_url_prefix_middleware.py`
  - `tests/unit/gui/test_config_and_design.py`
  - `tests/integration/gui/test_smoke_shell.py`

---

### Task 1: Add Failing Hub Tests For Backend `/node` Paths

**Files:**
- Modify: `tests/integration/gui/test_smoke_shell.py`

- [ ] **Step 1: Add imports and constants**

Add these imports near the top of `tests/integration/gui/test_smoke_shell.py`:

```python
import html
import re
```

Add this constant after the `sandbox` fixture:

```python
OOD_NODE_PREFIX = "/node/hz01/30099/"
```

- [ ] **Step 2: Add response helpers**

Add these helpers after `_find_string_in_json`:

```python
def _assert_content_type_prefix(resp: object, expected: str) -> None:
    """Assert a Flask test response has a content type prefix."""
    content_type = resp.headers.get("content-type", "")
    assert content_type.startswith(expected), content_type


def _assert_not_dash_index(resp: object) -> None:
    """Assert a route did not silently fall through to Dash index HTML."""
    body = resp.get_data(as_text=True)
    assert "<!DOCTYPE html>" not in body[:200]
    assert "react-entry-point" not in body[:500]


def _first_component_suite_script(index_html: str) -> str:
    """Return the first generated Dash component-suite JavaScript path."""
    match = re.search(r'src="([^"]+/_dash-component-suites/[^"]+\.js)"', index_html)
    assert match is not None, index_html[:1000]
    return html.unescape(match.group(1))
```

- [ ] **Step 3: Add the failing shell endpoint test**

Add this test near the existing explicit prefix tests:

```python
def test_explicit_url_prefix_routes_backend_dash_endpoints(
    sandbox: SandboxRoot,
) -> None:
    """OOD /node forwards the full prefix, so backend Dash endpoints must strip it."""
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
```

- [ ] **Step 4: Add the failing asset and component-suite test**

Add this test after the shell endpoint test:

```python
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
```

- [ ] **Step 5: Add the failing sub-app and API route test**

Add this test after the asset test:

```python
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
        (f"{OOD_NODE_PREFIX}builder/_dash-layout", "pipeline-builder"),
        (f"{OOD_NODE_PREFIX}results/_dash-layout", "results-viewer-empty-state"),
        (f"{OOD_NODE_PREFIX}run/_dash-layout", "run-console-root"),
        (f"{OOD_NODE_PREFIX}analysis/_dash-layout", "analysis-page"),
        (f"{OOD_NODE_PREFIX}tune/_dash-layout", "tune-"),
        (f"{OOD_NODE_PREFIX}browse/_dash-layout", "browse-"),
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
```

- [ ] **Step 6: Add a `SCRIPT_NAME` probe test**

Add this test near `test_dispatcher_threads_script_root`:

```python
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
```

- [ ] **Step 7: Run tests and verify they fail before implementation**

Run:

```bash
uv run pytest tests/integration/gui/test_smoke_shell.py -q
```

Expected before implementation:

```text
FAILED tests/integration/gui/test_smoke_shell.py::test_explicit_url_prefix_routes_backend_dash_endpoints
FAILED tests/integration/gui/test_smoke_shell.py::test_explicit_url_prefix_routes_backend_assets_and_component_suites
FAILED tests/integration/gui/test_smoke_shell.py::test_explicit_url_prefix_routes_backend_subapps_and_shell_blueprints
FAILED tests/integration/gui/test_smoke_shell.py::test_explicit_url_prefix_preserves_script_root_through_dispatcher
```

The expected failure should show `text/html` where the tests expect JSON, CSS, or JavaScript.

- [ ] **Step 8: Commit the failing tests**

```bash
git add tests/integration/gui/test_smoke_shell.py
git commit -m "test: cover OOD node-prefixed GUI routes"
```

---

### Task 2: Add Shared URL Prefix Strip Middleware

**Files:**
- Create: `src/phenotypic/gui/_url_prefix.py`
- Create: `tests/unit/gui/test_url_prefix_middleware.py`

- [ ] **Step 1: Write unit tests for the middleware**

Create `tests/unit/gui/test_url_prefix_middleware.py`:

```python
from __future__ import annotations

from typing import Any

import dash
from werkzeug.test import Client
from werkzeug.wrappers import Response

from phenotypic.gui._url_prefix import (
    URLPrefixStripMiddleware,
    install_url_prefix_strip_middleware,
)


def _capture_app(environ: dict[str, Any], start_response: object) -> list[bytes]:
    """Return PATH_INFO and SCRIPT_NAME for middleware assertions."""
    body = (
        f"path={environ.get('PATH_INFO', '')};"
        f"script={environ.get('SCRIPT_NAME', '')}"
    )
    response = Response(body, mimetype="text/plain")
    return response(environ, start_response)


def test_prefix_middleware_strips_exact_prefix_to_root() -> None:
    app = URLPrefixStripMiddleware(_capture_app, "/node/hz01/30099/")
    client = Client(app, Response)

    resp = client.get("/node/hz01/30099/")

    assert resp.status_code == 200
    assert resp.get_data(as_text=True) == "path=/;script=/node/hz01/30099"


def test_prefix_middleware_strips_nested_path_and_preserves_script_name() -> None:
    app = URLPrefixStripMiddleware(_capture_app, "/node/hz01/30099/")
    client = Client(app, Response)

    resp = client.get("/node/hz01/30099/builder/_dash-layout")

    assert resp.status_code == 200
    assert resp.get_data(as_text=True) == (
        "path=/builder/_dash-layout;script=/node/hz01/30099"
    )


def test_prefix_middleware_does_not_strip_near_miss_prefix() -> None:
    app = URLPrefixStripMiddleware(_capture_app, "/node/hz01/30099/")
    client = Client(app, Response)

    resp = client.get("/node/hz01/30099x/_dash-layout")

    assert resp.status_code == 200
    assert resp.get_data(as_text=True) == "path=/node/hz01/30099x/_dash-layout;script="


def test_install_url_prefix_strip_middleware_is_noop_for_default_prefix() -> None:
    app = dash.Dash(__name__)
    original = app.server.wsgi_app

    installed = install_url_prefix_strip_middleware(app, "/")

    assert installed is False
    assert app.server.wsgi_app is original


def test_install_url_prefix_strip_middleware_delegates_attributes() -> None:
    app = dash.Dash(__name__)
    installed = install_url_prefix_strip_middleware(app, "/node/hz01/30099/")

    assert installed is True
    assert app.server.wsgi_app.name == app.server.name
```

- [ ] **Step 2: Run middleware tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/gui/test_url_prefix_middleware.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'phenotypic.gui._url_prefix'
```

- [ ] **Step 3: Implement the middleware**

Create `src/phenotypic/gui/_url_prefix.py`:

```python
"""WSGI helpers for browser-visible GUI URL prefixes.

Dash needs ``requests_pathname_prefix`` to include the browser-visible
reverse-proxy prefix so generated asset, API, and navigation URLs are correct.
Some proxies, including Open OnDemand ``/node``, forward that same prefix to the
backend. ``URLPrefixStripMiddleware`` removes the prefix from incoming WSGI
``PATH_INFO`` before Flask, Dash, or ``DispatcherMiddleware`` route the request.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import dash

from phenotypic.gui._config import MOUNT_HOME, normalize_url_prefix

StartResponse = Callable[[str, list[tuple[str, str]], Any], Any]
WsgiApp = Callable[[dict[str, Any], StartResponse], Iterable[bytes]]


class URLPrefixStripMiddleware:
    """Strip a configured URL prefix from WSGI ``PATH_INFO``.

    Args:
        app: Wrapped WSGI application.
        url_prefix: Browser-visible path prefix, such as
            ``"/node/hz01/30099/"``. ``"/"`` is valid but should normally be
            skipped by :func:`install_url_prefix_strip_middleware`.
    """

    def __init__(self, app: WsgiApp, url_prefix: str) -> None:
        self.app = app
        self.url_prefix = normalize_url_prefix(url_prefix)
        self._strip_prefix = self.url_prefix.rstrip("/")

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        """Route the request after stripping the configured prefix when present."""
        if self.url_prefix == MOUNT_HOME:
            return self.app(environ, start_response)

        path_info = str(environ.get("PATH_INFO") or MOUNT_HOME)
        if path_info == self._strip_prefix:
            stripped_path = MOUNT_HOME
        elif path_info.startswith(f"{self._strip_prefix}/"):
            stripped_path = path_info[len(self._strip_prefix) :] or MOUNT_HOME
        else:
            return self.app(environ, start_response)

        script_name = str(environ.get("SCRIPT_NAME") or "")
        rewritten = environ.copy()
        rewritten["PATH_INFO"] = stripped_path
        rewritten["SCRIPT_NAME"] = f"{script_name.rstrip('/')}{self._strip_prefix}"
        return self.app(rewritten, start_response)

    def __getattr__(self, name: str) -> Any:
        """Delegate attributes so tests can still inspect wrapped middleware."""
        return getattr(self.app, name)


def install_url_prefix_strip_middleware(app: dash.Dash, url_prefix: str) -> bool:
    """Install prefix-stripping middleware on a Dash app's Flask server.

    Args:
        app: Dash app whose ``server.wsgi_app`` should be wrapped.
        url_prefix: Browser-visible path prefix.

    Returns:
        ``True`` when middleware was installed, ``False`` for the default
        ``"/"`` prefix.
    """
    prefix = normalize_url_prefix(url_prefix)
    if prefix == MOUNT_HOME:
        return False

    current = app.server.wsgi_app
    if isinstance(current, URLPrefixStripMiddleware):
        return False

    app.server.wsgi_app = URLPrefixStripMiddleware(current, prefix)
    return True


__all__ = ["URLPrefixStripMiddleware", "install_url_prefix_strip_middleware"]
```

- [ ] **Step 4: Run middleware tests and verify they pass**

Run:

```bash
uv run pytest tests/unit/gui/test_url_prefix_middleware.py -q
```

Expected:

```text
5 passed
```

- [ ] **Step 5: Run type and lint checks for the new module**

Run:

```bash
uv run ruff check src/phenotypic/gui/_url_prefix.py tests/unit/gui/test_url_prefix_middleware.py
uv run mypy src/phenotypic/gui/_url_prefix.py
```

Expected:

```text
All checks passed!
Success: no issues found in 1 source file
```

- [ ] **Step 6: Commit middleware**

```bash
git add src/phenotypic/gui/_url_prefix.py tests/unit/gui/test_url_prefix_middleware.py
git commit -m "fix: add GUI URL prefix strip middleware"
```

---

### Task 3: Install Middleware In Hub And Standalone App Factories

**Files:**
- Modify: `src/phenotypic/gui/shell/_app.py`
- Modify: `src/phenotypic/gui/builder/_app.py`
- Modify: `src/phenotypic/gui/results_viewer/_app.py`
- Modify: `src/phenotypic/gui/run_console/_app.py`
- Modify: `src/phenotypic/gui/analysis/_app.py`
- Modify: `src/phenotypic/gui/tune/_app.py`
- Modify: `src/phenotypic/gui/browse/_app.py`

- [ ] **Step 1: Install middleware around the composed hub**

In `src/phenotypic/gui/shell/_app.py`, add the import:

```python
from phenotypic.gui._url_prefix import install_url_prefix_strip_middleware
```

Then change the `viewer_session is not None` branch of `create_app` from returning directly to:

```python
    if viewer_session is not None:
        # Phase 3 backwards-compat: test path injects a stub session
        # and stops at the shell Dash (no sub-app composition).
        app = _build_shell_dash_app(
            sandbox, url_prefix=url_prefix, viewer_session=viewer_session
        )
        install_url_prefix_strip_middleware(app, url_prefix)
        return app
```

At the end of `create_app`, after the `compose_hub` call, install the middleware around the final dispatcher:

```python
    shell_app, _viewer_session = compose_hub(
        sandbox,
        url_prefix=url_prefix,
        idle_release_seconds=idle_release_seconds,
        start_idle_thread=start_idle_thread,
        progress=progress,
    )
    install_url_prefix_strip_middleware(shell_app, url_prefix)
    return shell_app
```

Do not install this middleware inside `_build_shell_dash_app()` or inside `compose_hub()` before the `DispatcherMiddleware` assignment. For the hub, it must be the outermost WSGI wrapper.

- [ ] **Step 2: Install middleware in builder**

In `src/phenotypic/gui/builder/_app.py`, add:

```python
from phenotypic.gui._url_prefix import install_url_prefix_strip_middleware
```

Before the final `return app`, add:

```python
    install_url_prefix_strip_middleware(app, url_prefix)
    return app
```

- [ ] **Step 3: Install middleware in results viewer**

In `src/phenotypic/gui/results_viewer/_app.py`, add:

```python
from phenotypic.gui._url_prefix import install_url_prefix_strip_middleware
```

In the `output_root is None` branch, change the return to:

```python
        install_url_prefix_strip_middleware(app, url_prefix)
        return app
```

Before the final `return app`, add:

```python
    install_url_prefix_strip_middleware(app, url_prefix)
    return app
```

- [ ] **Step 4: Install middleware in run console**

In `src/phenotypic/gui/run_console/_app.py`, add:

```python
from phenotypic.gui._url_prefix import install_url_prefix_strip_middleware
```

Before the final `return app`, add:

```python
    install_url_prefix_strip_middleware(app, url_prefix)
    return app
```

- [ ] **Step 5: Install middleware in analysis**

In `src/phenotypic/gui/analysis/_app.py`, add:

```python
from phenotypic.gui._url_prefix import install_url_prefix_strip_middleware
```

In the `output_root is None` branch, change the return to:

```python
        install_url_prefix_strip_middleware(app, url_prefix)
        return app
```

Before the final `return app`, add:

```python
    install_url_prefix_strip_middleware(app, url_prefix)
    return app
```

- [ ] **Step 6: Install middleware in tune**

In `src/phenotypic/gui/tune/_app.py`, add:

```python
from phenotypic.gui._url_prefix import install_url_prefix_strip_middleware
```

Before the final `return app`, add:

```python
    install_url_prefix_strip_middleware(app, url_prefix)
    return app
```

- [ ] **Step 7: Install middleware in browse**

In `src/phenotypic/gui/browse/_app.py`, add:

```python
from phenotypic.gui._url_prefix import install_url_prefix_strip_middleware
```

Before the final `return app`, add:

```python
    install_url_prefix_strip_middleware(app, url_prefix)
    return app
```

- [ ] **Step 8: Run the failing hub tests again**

Run:

```bash
uv run pytest tests/integration/gui/test_smoke_shell.py -q
```

Expected after implementation:

```text
all tests in the selected files passed
```

- [ ] **Step 9: Run focused standalone prefix probes**

Run:

```bash
uv run pytest tests/gui/browse/test_app.py tests/gui/builder/test_tile_blueprint.py tests/integration/gui/test_smoke_shell.py -q
```

Expected:

```text
all tests in the selected files passed
```

- [ ] **Step 10: Commit app factory installation**

```bash
git add src/phenotypic/gui/shell/_app.py src/phenotypic/gui/builder/_app.py src/phenotypic/gui/results_viewer/_app.py src/phenotypic/gui/run_console/_app.py src/phenotypic/gui/analysis/_app.py src/phenotypic/gui/tune/_app.py src/phenotypic/gui/browse/_app.py
git commit -m "fix: route GUI requests with explicit URL prefixes"
```

---

### Task 4: Update Launcher Guidance And GUI Documentation

**Files:**
- Modify: `src/phenotypic/gui/_config.py`
- Modify: `tests/unit/gui/test_config_and_design.py`
- Modify: `src/phenotypic/gui/CLAUDE.md`
- Modify: `src/phenotypic/gui/FEATURES.md`

- [ ] **Step 1: Update `--url-prefix` help text**

In `src/phenotypic/gui/_config.py`, update the `--url-prefix` help string in `add_launcher_args` to:

```python
        help=(
            "Browser-visible path prefix for reverse proxies such as "
            "Open OnDemand /node or /rnode. Pass only the path portion, "
            "e.g. /node/hz01/30099/. Default /."
        ),
```

- [ ] **Step 2: Update banner tunnel guidance**

Replace the single SSH hint line in `print_launcher_banner` with:

```python
    print(
        "  SSH tunnel from local: "
        f"ssh -N -L {port}:<server-host>:{port} <cluster>"
    )
    print("  If running on a compute node, use that node as <server-host>.")
    print("  For a plain SSH tunnel, omit --url-prefix and open localhost.")
```

Keep `extra_lines` printing after these lines.

- [ ] **Step 3: Update banner unit tests**

In `tests/unit/gui/test_config_and_design.py`, update `test_banner_contains_title_url_and_tunnel_hint` assertions:

```python
        assert "ssh -N -L 8050:<server-host>:8050" in out
        assert "If running on a compute node" in out
        assert "omit --url-prefix" in out
```

Keep the title, URL, and root assertions.

- [ ] **Step 4: Add argparse help test for OOD wording**

Add this test to `TestAddLauncherArgs`:

```python
    def test_url_prefix_help_mentions_node_and_rnode(self) -> None:
        parser = argparse.ArgumentParser()
        _config.add_launcher_args(parser)
        help_text = parser.format_help()
        assert "Open OnDemand /node or /rnode" in help_text
```

- [ ] **Step 5: Update GUI CLAUDE guide**

In `src/phenotypic/gui/CLAUDE.md`, replace the current `--url-prefix` paragraph with:

```markdown
`--url-prefix` is shared by all GUI launchers and defaults to `/`. It is a
path-only browser-visible prefix for reverse proxies, not a full URL. Open
OnDemand supports two relevant forms: `/node/<host>/<port>/`, which forwards
the full prefixed path to the backend, and `/rnode/<host>/<port>/`, which
forwards only the relative app path. PhenoTypic supports both: Dash still
generates browser URLs under `url_prefix`, and the GUI WSGI layer strips that
same prefix from incoming backend requests when the proxy preserves it. Use
`join_url_prefix` for browser-facing links, API fetches, and iframe URLs that
need to survive a proxy prefix.
```

- [ ] **Step 6: Update GUI feature ledger**

In `src/phenotypic/gui/FEATURES.md`, update the existing GUI `--url-prefix` row to mention `/node` backend routing. If the current row is:

```markdown
| GUI `--url-prefix` option              | Shared launcher args                        | Optional path-only browser prefix for OOD/path-stripping proxies; default `/` preserves existing URLs | ✅ shipping | integration | tests/integration/gui/test_smoke_shell.py::test_explicit_url_prefix_rewrites_results_browser_prefix |
```

Replace it with:

```markdown
| GUI `--url-prefix` option              | Shared launcher args + URL prefix WSGI middleware | Optional path-only browser prefix for OOD `/node` and `/rnode`; default `/` preserves existing URLs, while explicit prefixes route Dash assets, component suites, sub-apps, shell APIs, and run files through the backend prefix-strip layer | ✅ shipping | integration | tests/integration/gui/test_smoke_shell.py::test_explicit_url_prefix_routes_backend_subapps_and_shell_blueprints |
```

If line wrapping in the table differs, keep the same columns and update only the description and test reference.

- [ ] **Step 7: Run focused doc/config tests**

Run:

```bash
uv run pytest tests/unit/gui/test_config_and_design.py::TestAddLauncherArgs tests/unit/gui/test_config_and_design.py::TestPrintLauncherBanner -q
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 8: Commit docs and guidance**

```bash
git add src/phenotypic/gui/_config.py tests/unit/gui/test_config_and_design.py src/phenotypic/gui/CLAUDE.md src/phenotypic/gui/FEATURES.md
git commit -m "docs: clarify GUI URL prefix routing"
```

---

### Task 5: Final Verification And Review

**Files:**
- Verify: all files changed in Tasks 1 through 4

- [ ] **Step 1: Run focused GUI tests**

Run:

```bash
uv run pytest tests/unit/gui/test_url_prefix_middleware.py tests/unit/gui/test_config_and_design.py tests/integration/gui/test_smoke_shell.py tests/gui/browse/test_app.py tests/gui/builder/test_tile_blueprint.py -q
```

Expected:

```text
all selected tests passed
```

- [ ] **Step 2: Run lint**

Run:

```bash
uv run ruff check src/phenotypic/gui tests/unit/gui/test_url_prefix_middleware.py tests/integration/gui/test_smoke_shell.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 3: Run type checking**

Run:

```bash
uv run mypy src/phenotypic/gui
```

Expected:

```text
Success: no issues found
```

If existing unrelated GUI typing errors appear, record the exact error count and rerun a narrower command:

```bash
uv run mypy src/phenotypic/gui/_url_prefix.py src/phenotypic/gui/shell/_app.py
```

Expected for the narrower command:

```text
Success: no issues found
```

- [ ] **Step 4: Run manual in-process route probe**

Run:

```bash
uv run python - <<'PY'
import json
import tempfile
from pathlib import Path

from phenotypic.gui.shell import SandboxRoot, create_app

prefix = "/node/hz01/42793/"
sandbox = SandboxRoot.from_path(Path(tempfile.mkdtemp()))
app = create_app(sandbox, url_prefix=prefix, start_idle_thread=False)
client = app.server.test_client()

paths = [
    f"{prefix}_dash-layout",
    f"{prefix}_dash-dependencies",
    f"{prefix}assets/shell.css",
    f"{prefix}builder/_dash-layout",
    f"{prefix}results/_dash-layout",
    f"{prefix}sandbox/api/root",
]
print(
    json.dumps(
        [
            {
                "path": path,
                "status": client.get(path).status_code,
                "content_type": client.get(path).headers.get("content-type"),
            }
            for path in paths
        ],
        indent=2,
    )
)
PY
```

Expected content types:

```text
application/json for _dash-layout, _dash-dependencies, sub-app layouts, and /sandbox/api/root
text/css for assets/shell.css
```

- [ ] **Step 5: Request independent code review**

Dispatch a reviewer with this brief:

```text
Review the GUI URL prefix routing fix. Focus on whether the WSGI middleware is boundary-safe, preserves SCRIPT_NAME correctly, is installed at the right WSGI layer for the hub, does not break standalone sub-apps, and has tests that would fail without the fix. Do not edit files.
```

Required result:

```text
No Critical or Important findings remain open.
```

- [ ] **Step 6: Commit final adjustments**

If review produces fixes, apply them and commit:

```bash
git add <reviewed-files>
git commit -m "fix: address URL prefix routing review"
```

If review produces no changes, do not create an empty commit.

---

## Expected User-Facing Behavior After The Fix

Open OnDemand `/node`:

```bash
phenotypic-gui --root /rhome/ejaco020 --host 0.0.0.0 --port 42793 --url-prefix=/node/hz01/42793/
```

Browser URL:

```text
https://ondemand.hpcc.ucr.edu/node/hz01/42793/
```

Backend receives:

```text
GET /node/hz01/42793/_dash-layout
GET /node/hz01/42793/assets/shell.css
GET /node/hz01/42793/_dash-component-suites/<generated-js-path>
```

Middleware rewrites those to:

```text
PATH_INFO=/_dash-layout, SCRIPT_NAME=/node/hz01/42793
PATH_INFO=/assets/shell.css, SCRIPT_NAME=/node/hz01/42793
PATH_INFO=/_dash-component-suites/<generated-js-path>, SCRIPT_NAME=/node/hz01/42793
```

Open OnDemand `/rnode`:

```bash
phenotypic-gui --root /rhome/ejaco020 --host 0.0.0.0 --port 42793 --url-prefix=/rnode/hz01/42793/
```

OOD strips the prefix before the backend sees it. Middleware sees unprefixed paths and is a no-op. Dash still generates browser-facing `/rnode/hz01/42793/<app-path>` URLs.

Plain SSH tunnel:

```bash
phenotypic-gui --root /rhome/ejaco020 --host 127.0.0.1 --port 42793
ssh -N -L 42793:<server-host>:42793 ejaco020@cluster.hpcc.ucr.edu
```

Browser URL:

```text
http://localhost:42793/
```

Do not pass `--url-prefix` for this mode.

## Self-Review

- Spec coverage: The plan covers the confirmed `/node` root cause, the shared WSGI middleware, hub ordering before `DispatcherMiddleware`, standalone launchers, tests for JSON/CSS/JS/API/run-file routing, docs, feature ledger, and tunnel guidance.
- Placeholder scan: No `TBD`, `TODO`, "implement later", or vague "add tests" steps remain. Each code task includes concrete snippets and commands.
- Type consistency: The middleware class is consistently named `URLPrefixStripMiddleware`; the install helper is consistently named `install_url_prefix_strip_middleware`; tests import those exact names.
