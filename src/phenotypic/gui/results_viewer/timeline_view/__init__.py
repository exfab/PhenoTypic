"""Results-viewer Timeline tab — overlay matrix over OutputRoot.master_df axes.

Renders the shared timeline engine (``gui/_shared/timeline``) + the
focus-and-navigate ``timeline.js`` controller over the run's overlay PNGs,
with a (row × time) axis pair drawn from ``OutputRoot.master_df`` (spec §6/§16).
"""
from __future__ import annotations

from phenotypic.gui.results_viewer.timeline_view._callbacks import register_callbacks
from phenotypic.gui.results_viewer.timeline_view._grid import (
    build_timeline_records,
    has_eligible_time_axis,
    is_large_time_axis,
    selectable_time_columns,
)
from phenotypic.gui.results_viewer.timeline_view._layout import layout

__all__ = [
    "layout",
    "register_callbacks",
    "selectable_time_columns",
    "is_large_time_axis",
    "has_eligible_time_axis",
    "build_timeline_records",
]
