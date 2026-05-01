"""Shell Dash factory + ``DispatcherMiddleware`` composer (Phase 5).

Phase 0 placeholder — implementation lands in Phase 5. See ``GUI_SPEC_V1.md``
section 6 (Builder + Viewer integration) for the factory signature contract
and ``_ViewerProxy`` requirement.
"""
from __future__ import annotations

# TODO(Phase 5): implement ``create_app(sandbox)`` returning the composed
# Werkzeug ``DispatcherMiddleware`` callable. Per plan:
#   * call ``builder.create_app(image_root=sandbox.root, url_prefix="/builder/")``
#   * call ``run_console.create_app(sandbox=sandbox, url_prefix="/run/")``
#   * wrap viewer in ``ToolSession`` with ``_ViewerProxy``
#   * register ``_routes`` + ``_runs_blueprint`` on ``shell_app.server``
#   * compose mounts: ``/builder/``, ``/results/``, ``/run/``
#   * start idle-release daemon thread

__all__: list[str] = []
