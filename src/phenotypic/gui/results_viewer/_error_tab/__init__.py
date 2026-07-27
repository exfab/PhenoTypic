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

State and persistence:

* The good baseline is chosen by ``ERROR_GOOD_MODE_TOGGLE_ID``: ``all_unlabeled``
  (every unlabeled object is good) vs ``verified`` (good = ``verified_good_keys``,
  the unlabeled members of QC-reviewed groups, derived from ``qc/qc.duckdb`` via
  ``review/_db.py`` + ``qc/review_state.json``; diagnostic-only modules are skipped).
* State is server-side: the recompute callback reads ``filtered_state.labels`` under
  its lock (the shared ``CurationLabels`` — there is **no** ``STORE_LABELS`` Dash
  store) and gates on ``active_tab == TAB_ERROR_ID``.
* Recompute, activation, category focus, and verified-baseline preview are
  compute-only. They never write canonical deliverables.
* ``Publish all categories`` computes the complete configured vocabulary from
  one fingerprinted snapshot and transactionally publishes checksummed
  ``error_analysis.{parquet,csv,html}`` plus ``verified.parquet`` in verified
  mode. Empty categories remain explicit manifest entries.
* ``reemit_error_deliverables`` (``_cli/_cli_error_outputs.py``, called from
  ``finalize_post_master_outputs``) authoritatively rewrites ``errors/*`` +
  ``error_analysis.*`` from the durable labels on headless finalize, so CLI output
  matches the live GUI. ``verified.parquet`` is **GUI-only** — finalize never writes
  it. Resolve all paths via ``phenotypic.sdk_`` helpers.
"""
from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazy-load Dash UI while keeping publication services headless."""
    if name == "register_error_callbacks":
        from phenotypic.gui.results_viewer._error_tab._callbacks import (
            register_error_callbacks,
        )

        return register_error_callbacks
    if name == "build_error_tab_body":
        from phenotypic.gui.results_viewer._error_tab._layout import (
            build_error_tab_body,
        )

        return build_error_tab_body
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "build_error_tab_body",
    "register_error_callbacks",
]
