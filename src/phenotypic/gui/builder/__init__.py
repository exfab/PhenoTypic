"""Dash + dash-cytoscape interactive ImagePipeline builder.

This subpackage provides a web-based node-graph editor for constructing
:class:`phenotypic.ImagePipeline` objects interactively. It is intended to
run on an HPCC head node and be reached from a workstation via SSH port
forwarding (``ssh -L 8050:localhost:8050 user@hpcc``).

Public API will be filled in by Phase 4 once :func:`create_app` exists.
"""

from __future__ import annotations

__all__ = ["create_app", "BuilderState"]


def __getattr__(name: str):  # noqa: D401 - lazy re-export
    """Lazily import builder objects to keep Dash off the import-time hot path."""
    if name == "create_app":
        from phenotypic.gui.builder._app import create_app

        return create_app
    if name == "BuilderState":
        from phenotypic.gui.builder._state import BuilderState

        return BuilderState
    raise AttributeError(f"module 'phenotypic.gui.builder' has no attribute {name!r}")
