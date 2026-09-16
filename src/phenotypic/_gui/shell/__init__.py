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

import importlib as _importlib
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

#: Public shell names by defining module. The console script imports this package before
#: ``_launcher``, so an eager ``_app`` import here would load Dash -- and through it the
#: sub-apps -- just to print ``phenotypic-gui --help``.
_LAZY_ATTRS: dict[str, str] = {
    "SandboxRoot": "phenotypic._gui.shell._sandbox",
    "ToolSession": "phenotypic._gui.shell._session",
    "create_app": "phenotypic._gui.shell._app",
    "launch_gui": "phenotypic._gui.shell._launcher",
    "main": "phenotypic._gui.shell._launcher",
}


def __getattr__(name: str) -> _Any:
    """Resolve a public shell name on first access and cache it on the package."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    # ``globals().get`` rather than a bare ``__all__``: this function sits above the eager
    # imports for the same reason ``__getattr__`` does, so it is callable in a window where
    # ``__all__`` has not been assigned yet. A bare reference raises ``NameError`` there.
    return sorted(set(globals()) | set(globals().get("__all__", ())))


if _TYPE_CHECKING:
    from phenotypic._gui.shell._app import create_app
    from phenotypic._gui.shell._launcher import launch_gui, main
    from phenotypic._gui.shell._sandbox import SandboxRoot
    from phenotypic._gui.shell._session import ToolSession

__all__ = [
    "SandboxRoot",
    "ToolSession",
    "create_app",
    "launch_gui",
    "main",
]
