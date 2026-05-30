"""Heatmap tab for the results viewer.

Renders a Plotly heatmap of any measurement (or QC metric) column
laid out by ``Grid_RowNum`` x ``Grid_ColNum``. Curated colonies render
as an overlay of `x`-markers in the muted brand color so the user can
distinguish "excluded" from "low value".

Public surface:

* :func:`build_heatmap_figure` - pure figure builder (no Dash imports).
* :func:`build_heatmap_tab_body` - layout factory used by
  :mod:`phenotypic.gui.results_viewer._layout`.
* :func:`register_heatmap_callbacks` - registers the two callbacks
  owned by this tab (figure render + control population).
"""
from __future__ import annotations

from phenotypic.gui.results_viewer._heatmap_tab._callbacks import (
    register_heatmap_callbacks,
)
from phenotypic.gui.results_viewer._heatmap_tab._figure import build_heatmap_figure
from phenotypic.gui.results_viewer._heatmap_tab._layout import build_heatmap_tab_body

__all__ = [
    "build_heatmap_figure",
    "build_heatmap_tab_body",
    "register_heatmap_callbacks",
]
