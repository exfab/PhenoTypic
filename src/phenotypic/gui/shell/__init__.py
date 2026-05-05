"""Unified GUI shell — Dash hub composer.

The shell mounts ``builder``, ``results_viewer``, and ``run_console`` under one
URL via ``werkzeug.middleware.dispatcher.DispatcherMiddleware`` and registers
the sandbox JSON API + ``/runs/`` static blueprint on its own Flask server.
See ``GUI_SPEC_V1.md`` and ``docs/source/user_guide/gui.rst`` (Phase 8).

Public API:
    * :class:`SandboxRoot` — sandbox primitive (Phase 1).
    * :class:`ToolSession` — lifecycle wrapper (Phase 1).
    * :func:`create_app` — composed shell app factory (Phase 3 standalone;
      Phase 5 will compose sub-apps via ``DispatcherMiddleware``).
    * :func:`launch_gui` — convenience launcher (Phase 3).
"""
from __future__ import annotations

from phenotypic.gui.shell._app import create_app
from phenotypic.gui.shell._launcher import launch_gui, main
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._session import ToolSession

__all__ = [
    "SandboxRoot",
    "ToolSession",
    "create_app",
    "launch_gui",
    "main",
]
