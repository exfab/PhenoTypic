"""Viewer module for sweep results comparison and export.

Provides interactive and static viewers for comparing pipeline variant results.

Components:
- SweepComparisonWidget: Interactive Panel widget for side-by-side comparison
- SweepHTMLExporter: Static HTML export with keyboard navigation
"""

from __future__ import annotations

from ._comparison_widget import SweepComparisonWidget
from ._html_exporter import SweepHTMLExporter

__all__ = [
    "SweepComparisonWidget",
    "SweepHTMLExporter",
]
