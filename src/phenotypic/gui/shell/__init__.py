"""Unified GUI shell — Dash hub composer.

The shell mounts ``builder``, ``results_viewer``, and ``run_console`` under one
URL via ``werkzeug.middleware.dispatcher.DispatcherMiddleware`` and registers
the sandbox JSON API + ``/runs/`` static blueprint on its own Flask server.
See ``GUI_SPEC_V1.md`` and ``docs/source/user_guide/gui.rst`` (Phase 8).

Public API (filled in across phases):
    * :class:`SandboxRoot` — sandbox primitive (Phase 1).
    * :func:`create_app` — composed shell app factory (Phase 5).
    * :func:`launch_gui` — convenience launcher (Phase 3).
"""
from __future__ import annotations

# Phase 0 scaffolding: __all__ stays empty until names are wired in by later
# phases. Lazy ``__getattr__`` re-exports land in Phase 3 (launch_gui /
# SandboxRoot) and Phase 5 (create_app).
__all__: list[str] = []
