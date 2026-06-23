"""Unit tests for GUI URL-prefix WSGI middleware."""
from __future__ import annotations

from typing import Any

import dash
from werkzeug.test import Client
from werkzeug.wrappers import Response

from phenotypic.gui._url_prefix import (
    URLPrefixStripMiddleware,
    configure_url_prefix_routing,
    install_url_prefix_strip_middleware,
)


def _capture_app(environ: dict[str, Any], start_response: object) -> list[bytes]:
    """Return PATH_INFO and SCRIPT_NAME for middleware assertions."""
    body = f"path={environ.get('PATH_INFO', '')};script={environ.get('SCRIPT_NAME', '')}"
    response = Response(body, mimetype="text/plain")
    return response(environ, start_response)


def test_prefix_middleware_strips_exact_prefix_to_root() -> None:
    """An exact prefix request reaches downstream routing as ``/``."""
    client = Client(
        URLPrefixStripMiddleware(_capture_app, "/node/hz01/30099/"),
        Response,
    )

    resp = client.get("/node/hz01/30099")

    assert resp.text == "path=/;script=/node/hz01/30099"


def test_prefix_middleware_strips_nested_path_and_preserves_script_name() -> None:
    """Nested requests strip only the configured prefix and extend SCRIPT_NAME."""
    client = Client(
        URLPrefixStripMiddleware(_capture_app, "/node/hz01/30099/"),
        Response,
    )

    resp = client.get(
        "/node/hz01/30099/builder/_dash-layout",
        environ_overrides={"SCRIPT_NAME": "/base"},
    )

    assert resp.text == "path=/builder/_dash-layout;script=/base/node/hz01/30099"


def test_prefix_middleware_leaves_near_miss_prefixes_untouched() -> None:
    """Boundary checks prevent stripping similar but different path segments."""
    client = Client(
        URLPrefixStripMiddleware(_capture_app, "/node/hz01/30099/"),
        Response,
    )

    resp = client.get("/node/hz01/30099x/builder/_dash-layout")

    assert resp.text == "path=/node/hz01/30099x/builder/_dash-layout;script="


def test_install_with_root_prefix_is_noop() -> None:
    """The default ``/`` prefix should not wrap the Flask WSGI app."""
    app = dash.Dash(__name__)
    original = _capture_app
    app.server.wsgi_app = original  # type: ignore[method-assign]

    installed = install_url_prefix_strip_middleware(app, "/")

    assert installed is False
    assert app.server.wsgi_app is original


def test_installed_wrapper_delegates_attributes_needed_by_tests() -> None:
    """The wrapper preserves access to DispatcherMiddleware test hooks."""

    class MountedApp:
        mounts = {"/builder": object()}

        def __call__(
            self,
            environ: dict[str, Any],
            start_response: object,
        ) -> list[bytes]:
            return _capture_app(environ, start_response)

    app = dash.Dash(__name__)
    app.server.wsgi_app = MountedApp()  # type: ignore[method-assign]

    installed = install_url_prefix_strip_middleware(app, "/node/hz01/30099/")

    assert installed is True
    assert "/builder" in app.server.wsgi_app.mounts


def test_configure_url_prefix_routing_returns_same_app() -> None:
    """Factory-tail helper wraps when needed and preserves app identity."""
    app = dash.Dash(__name__)

    returned = configure_url_prefix_routing(app, "/node/hz01/30099/")

    assert returned is app
    assert isinstance(app.server.wsgi_app, URLPrefixStripMiddleware)
