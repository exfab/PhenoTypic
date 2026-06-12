"""PhenoTypic GUI — Source Image Browse tab.

A deep-zoom viewer for the raw input images under the selected source
root. Mounted at ``/browse/`` in the unified hub. See
``docs/superpowers/specs/2026-06-11-gui-source-image-browse-tab-design.md``.
"""
from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str):  # lazy to avoid importing dash at package import
    if name == "create_app":
        from phenotypic.gui.browse._app import create_app

        return create_app
    raise AttributeError(name)
