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

Every name above resolves lazily (PEP 562). Importing a submodule of this
package must not execute the Dash app factory: ``_services`` modules reach
``_classifier`` and ``_sandbox`` through this package, and an eager ``__init__``
would drag ``dash``/``flask``/``werkzeug`` in behind them and fail the gate in
``tests/unit/services/test_import_purity.py``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # type-checker only; never executed at runtime
    from phenotypic.gui.shell._app import create_app  # noqa: F401
    from phenotypic.gui.shell._launcher import launch_gui, main  # noqa: F401
    from phenotypic.gui.shell._sandbox import SandboxRoot  # noqa: F401
    from phenotypic.gui.shell._session import ToolSession  # noqa: F401

__all__ = [
    "SandboxRoot",
    "ToolSession",
    "create_app",
    "launch_gui",
    "main",
]

_LAZY: dict[str, tuple[str, str]] = {
    "create_app": ("phenotypic.gui.shell._app", "create_app"),
    "launch_gui": ("phenotypic.gui.shell._launcher", "launch_gui"),
    "main": ("phenotypic.gui.shell._launcher", "main"),
    "SandboxRoot": ("phenotypic.gui.shell._sandbox", "SandboxRoot"),
    "ToolSession": ("phenotypic.gui.shell._session", "ToolSession"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(name) from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)
