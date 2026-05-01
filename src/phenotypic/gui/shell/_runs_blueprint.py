"""``/runs/<rel>/<path:file>`` static blueprint (Phase 2).

Phase 0 placeholder — implementation lands in Phase 2. See ``GUI_SPEC_V1.md``
section 3 (Sandbox & file browser).

Per plan: catch-all ``<path:file>`` parameter is required because the
generated dashboard polls relative URLs like ``progress/manifest.json`` that
resolve to nested segments under ``/runs/<rel>/``. Every request also calls
``viewer_session.touch()`` so the idle timer resets while the user actively
views iframed dashboards. Registered on ``shell_app.server`` directly — NOT
under ``DispatcherMiddleware``.
"""
from __future__ import annotations

# TODO(Phase 2): ``register(server, sandbox, viewer_session)`` builder; rejects
# path traversal, returns 404 outside sandbox, 403 on permission errors.

__all__: list[str] = []
