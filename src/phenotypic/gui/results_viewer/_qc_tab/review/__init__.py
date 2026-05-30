"""QC Review sub-view — the master–detail curation walkthrough.

The QC tab toggles between **Configure** (the per-check card editor) and
**Review** (this subtree). Review walks the worst-agreeing groups for a
selected QC module, reuses the colony-view tile gallery for per-colony
curation, recomputes the module after each curated group, and tracks
per-module review progress.

Public surface:

* :func:`build_review_view` — the Review sub-view layout factory, mounted
  by the QC tab body's :data:`._ids.QC_REVIEW_VIEW_ID` container.
* :func:`register_review_callbacks` — registers every Review callback;
  called from
  :func:`phenotypic.gui.results_viewer._qc_tab._callbacks.register_qc_callbacks`.
"""

from __future__ import annotations

from phenotypic.gui.results_viewer._qc_tab.review._callbacks import (
    register_review_callbacks,
)
from phenotypic.gui.results_viewer._qc_tab.review._layout import build_review_view

__all__ = [
    "build_review_view",
    "register_review_callbacks",
]
