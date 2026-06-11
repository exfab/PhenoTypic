"""Error-analysis tab for the results viewer.

For the focused error category, this tab runs
:class:`phenotypic.analysis.ErrorCutoffFinder` against a good baseline
(all-unlabeled or verified-only) and surfaces a ranked cutoff table, a
good-vs-error distribution figure with a draggable cutoff line, a live
recall/specificity readout, and a copy-able filter spec — recomputed as
the user marks objects on the other tabs.

Public surface:

* :func:`build_error_tab_body` — layout factory used by
  :mod:`phenotypic.gui.results_viewer._layout`.
* :func:`register_error_callbacks` — registers the tab's recompute /
  category-select / drag-readout / save-report callbacks.
"""
from __future__ import annotations

from phenotypic.gui.results_viewer._error_tab._callbacks import (
    register_error_callbacks,
)
from phenotypic.gui.results_viewer._error_tab._layout import build_error_tab_body

__all__ = [
    "build_error_tab_body",
    "register_error_callbacks",
]
