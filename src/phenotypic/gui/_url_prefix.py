"""WSGI helpers for browser-visible GUI URL prefixes.

Dash needs ``requests_pathname_prefix`` to include the browser-visible
reverse-proxy prefix so generated asset, API, and navigation URLs are correct.
Some proxies, including Open OnDemand ``/node``, forward that same prefix to
the backend. ``URLPrefixStripMiddleware`` removes the prefix from incoming
WSGI ``PATH_INFO`` before Flask, Dash, or ``DispatcherMiddleware`` route the
request.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, cast

import dash

from phenotypic.gui._config import MOUNT_HOME, normalize_url_prefix

StartResponse = Callable[[str, list[tuple[str, str]], Any], Any]
WsgiApp = Callable[[dict[str, Any], StartResponse], Iterable[bytes]]


class URLPrefixStripMiddleware:
    """Strip a configured URL prefix from WSGI ``PATH_INFO``."""

    def __init__(self, app: WsgiApp, url_prefix: str) -> None:
        """Store the wrapped app and normalized prefix.

        Args:
            app: Downstream WSGI application.
            url_prefix: Browser-visible path prefix to strip from incoming
                backend requests when the prefix is present as a complete path
                segment sequence.
        """
        self.app = app
        self.url_prefix = normalize_url_prefix(url_prefix)
        self._strip_prefix = self.url_prefix.rstrip("/")

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: StartResponse,
    ) -> Iterable[bytes]:
        """Delegate after stripping the configured prefix when it matches."""
        if self.url_prefix == MOUNT_HOME:
            return self.app(environ, start_response)

        path_info = str(environ.get("PATH_INFO") or MOUNT_HOME)
        if path_info == self._strip_prefix:
            stripped_path = MOUNT_HOME
        elif path_info.startswith(f"{self._strip_prefix}/"):
            stripped_path = path_info[len(self._strip_prefix):] or MOUNT_HOME
        else:
            return self.app(environ, start_response)

        script_name = str(environ.get("SCRIPT_NAME") or "")
        rewritten = environ.copy()
        rewritten["PATH_INFO"] = stripped_path
        rewritten["SCRIPT_NAME"] = f"{script_name.rstrip('/')}{self._strip_prefix}"
        return self.app(rewritten, start_response)

    def __getattr__(self, name: str) -> Any:
        """Delegate attributes exposed by wrapped WSGI middleware."""
        return getattr(self.app, name)


def install_url_prefix_strip_middleware(app: dash.Dash, url_prefix: str) -> bool:
    """Install prefix-stripping middleware on a Dash app's Flask server.

    Args:
        app: Dash application whose ``server.wsgi_app`` should be wrapped.
        url_prefix: Browser-visible path prefix. ``"/"`` is a no-op because
            there is no explicit proxy prefix to strip from backend requests.

    Returns:
        ``True`` when a wrapper was installed; ``False`` when no wrapper was
        needed or one was already installed.
    """
    prefix = normalize_url_prefix(url_prefix)
    if prefix == MOUNT_HOME:
        return False
    current = app.server.wsgi_app
    if isinstance(current, URLPrefixStripMiddleware):
        return False
    app.server.wsgi_app = URLPrefixStripMiddleware(  # type: ignore[method-assign]
        cast(WsgiApp, current),
        prefix,
    )
    return True


def configure_url_prefix_routing(app: dash.Dash, url_prefix: str) -> dash.Dash:
    """Install prefix routing middleware and return the app.

    App factories call this immediately before returning, after registering
    app-specific routes and callbacks. Keeping the call at the return boundary
    makes the hub's required post-``DispatcherMiddleware`` ordering visible
    while avoiding duplicated install-and-return boilerplate.

    Args:
        app: Dash application to finalize.
        url_prefix: Browser-visible path prefix.

    Returns:
        The same Dash application passed in.
    """
    install_url_prefix_strip_middleware(app, url_prefix)
    return app


__all__ = [
    "URLPrefixStripMiddleware",
    "configure_url_prefix_routing",
    "install_url_prefix_strip_middleware",
]
