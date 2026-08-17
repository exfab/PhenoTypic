"""Dash-free service tier shared by the GUI and the MCP server.

Modules here import only the standard library and other ``phenotypic``
internals. Nothing in this package may import ``dash``,
``dash_bootstrap_components``, ``flask``, or ``werkzeug`` — the boundary is
enforced by ``tests/unit/services/test_import_purity.py``.

This module is deliberately empty of submodule imports: eagerly importing them
here would make one heavy dependency contaminate every consumer, which is the
failure this tier exists to prevent.
"""

from __future__ import annotations

__all__: list[str] = []
