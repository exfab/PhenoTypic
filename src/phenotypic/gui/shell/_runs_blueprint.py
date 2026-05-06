"""``/runs/<rel>/<path:file>`` static blueprint.

Iframed dashboards (``dashboard.html`` artefacts produced by
``python -m phenotypic``) poll their progress files using **relative URLs**
(``progress/manifest.json``, ``progress/failures.jsonl``). When a dashboard is
iframed at ``/runs/plate_2026-04/output/dashboard.html``, the browser resolves
those polls to ``/runs/plate_2026-04/output/progress/manifest.json``. Reaching
that file requires a multi-segment route parameter — Flask's ``<path:file>``
catch-all. A single-segment ``<file>`` would silently 404 the polls and the
iframed dashboard would render once but never update, making it look frozen.

Security
    Every request runs ``SandboxRoot.resolve(...)`` first. ``..``-style URLs,
    absolute paths, and symlinks pointing outside the sandbox all return
    ``404`` (the blueprint deliberately does NOT distinguish "outside
    sandbox" from "missing" — the sandbox API blueprint, by contrast,
    returns ``400`` for the same condition because its ``path=`` query
    parameter is a strict typed input). ``PermissionError`` returns ``403``
    (so the user sees something distinct from "missing").

Lifecycle
    Each handled request also calls ``viewer_session.touch()``. Iframed
    dashboards bypass Dash callbacks entirely; without ``touch`` here the
    idle daemon would release the viewer session while the user was actively
    reading an iframed dashboard. The blueprint is the canonical liveness
    signal for "user is viewing iframed content."

Registration
    Registered directly on ``shell_app.server`` (NOT under
    ``DispatcherMiddleware``) so it answers regardless of which Dash sub-app
    is currently mounted. Cloud-deploy auth (deferred) attaches a single
    ``@before_request`` hook here rather than per-Dash-callback.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from flask import Blueprint, Response, abort, send_from_directory

from phenotypic.gui._config import RUNS_BLUEPRINT_PREFIX

if TYPE_CHECKING:
    from flask import Flask

    from phenotypic.gui.shell._sandbox import SandboxRoot
    from phenotypic.gui.shell._session import ToolSession

logger = logging.getLogger(__name__)

__all__ = ["build_runs_blueprint", "register"]


def build_runs_blueprint(
    sandbox: "SandboxRoot",
    *,
    viewer_session: "ToolSession[object] | None" = None,
    name: str = "phenotypic_runs",
    url_prefix: str = RUNS_BLUEPRINT_PREFIX,
) -> Blueprint:
    """Build a ``/runs/<rel>/<path:file>`` blueprint.

    Args:
        sandbox: Containment primitive. Every request resolves its path
            argument through ``sandbox.resolve`` and returns 404 on escape.
        viewer_session: Optional :class:`ToolSession` whose ``touch()`` is
            called on every successfully-served request. Used by the shell
            so iframed dashboard polls keep the viewer alive. ``None`` for
            standalone tests / pre-Phase 5 callers.
        name: Blueprint name. Defaults to ``"phenotypic_runs"``. Override
            only if multiple sandboxes coexist in one Flask process (not a
            v1 case but cheap to allow).
        url_prefix: Route prefix. Defaults to ``"/runs"``.

    Returns:
        A configured :class:`flask.Blueprint`. Caller registers it on the
        Flask app with ``server.register_blueprint(bp)``.
    """
    bp = Blueprint(name, __name__, url_prefix=url_prefix)

    @bp.route("/<path:rel_file>")
    def serve_run_file(rel_file: str) -> Response:
        """Serve a single file under ``sandbox.root / rel_file``.

        Returns:
            ``send_from_directory`` response on success.

        Aborts:
            404 — path escapes the sandbox, or target file is missing.
            403 — target file is unreadable (PermissionError).
        """
        # ``rel_file`` may be e.g. "plate/output/progress/manifest.json".
        # We split off the directory and serve the file from it; this lets
        # ``send_from_directory`` apply its own safe-join + 404 semantics
        # for the leaf, while we apply sandbox containment to the whole
        # tree.
        try:
            resolved = sandbox.resolve(rel_file)
        except ValueError:
            logger.warning(
                "rejected /runs traversal attempt: %r", rel_file
            )
            return abort(404)  # type: ignore[return-value]

        if not resolved.exists() or resolved.is_dir():
            # 404 on missing OR on a directory — we do not enumerate.
            return abort(404)  # type: ignore[return-value]

        directory = resolved.parent
        filename = resolved.name
        try:
            response = send_from_directory(directory, filename)
        except PermissionError:
            # IMPORTANT: do NOT touch() before this point. A 403 on an
            # in-sandbox-but-unreadable file must NOT bump the viewer's
            # idle timer; otherwise an attacker hammering known-unreadable
            # paths could keep the viewer session alive indefinitely.
            logger.warning("/runs permission denied: %s", resolved)
            return abort(403)  # type: ignore[return-value]

        # Touch only on a fully successful serve.
        if viewer_session is not None:
            viewer_session.touch()
        return response

    return bp


def register(
    server: "Flask",
    sandbox: "SandboxRoot",
    *,
    viewer_session: "ToolSession[object] | None" = None,
) -> Blueprint:
    """Build and register a runs blueprint on ``server``.

    Convenience wrapper for the common case where the caller has a Flask
    app + sandbox in hand. Returns the registered blueprint so tests can
    introspect its rules.
    """
    bp = build_runs_blueprint(sandbox, viewer_session=viewer_session)
    server.register_blueprint(bp)
    logger.debug(
        "registered /runs blueprint on Flask app=%s sandbox=%s",
        server.name,
        sandbox.root,
    )
    return bp
