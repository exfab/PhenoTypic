"""Component IDs owned by the Error-analysis tab.

Every control is a single static component (no pattern-matching ids), so
plain string ids are sufficient and grep-able. The two per-tab
``dcc.Store``s carry the focused-measurement value arrays (so the drag
readout recomputes without re-reading parquet) and the good-baseline
mode.
"""
from __future__ import annotations

# -- category switcher -------------------------------------------------------

#: Container the recompute callback fills with the per-category chip row
#: (each chip carries a live count; the selected chip is the focused
#: category).
ERROR_CATEGORY_CHIPS_ID = "error-category-chips"

# -- good-baseline toggle + verified badge -----------------------------------

#: Segmented control (``dbc.RadioItems``) choosing the good baseline:
#: ``all_unlabeled`` (default) or ``verified``.
ERROR_GOOD_MODE_TOGGLE_ID = "error-good-mode-toggle"

#: Badge reporting the verified-good object count (verified mode only).
ERROR_VERIFIED_COUNT_ID = "error-verified-count"

# -- ranked table + figure ---------------------------------------------------

#: ``dash_table.DataTable`` of the ranked :meth:`ErrorCutoffFinder.analyze`
#: result (measurement, AUC, cutoff, recall, specificity, BH-p, …).
ERROR_TABLE_ID = "error-cutoff-table"

#: ``dcc.Graph`` hosting the good-vs-error distribution with the editable
#: cutoff line (drag emits ``relayoutData``).
ERROR_FIGURE_ID = "error-distribution-figure"

# -- cutoff input + readout + filter spec ------------------------------------

#: Numeric ``dcc.Input`` mirroring the dragged cutoff (precise / accessible
#: alternative to dragging the line).
ERROR_CUTOFF_INPUT_ID = "error-cutoff-input"

#: Recall / specificity readout pills, recomputed at the current cutoff.
ERROR_READOUT_ID = "error-readout"

#: Read-only ``dcc.Textarea`` holding the copy-able filter spec (JSON +
#: human query).
ERROR_FILTER_SPEC_ID = "error-filter-spec"

#: ``dcc.Clipboard`` copying the filter-spec textarea content.
ERROR_CLIPBOARD_ID = "error-filter-spec-clipboard"

# -- save report + chrome ----------------------------------------------------

#: Explicit all-category publication button.
ERROR_PUBLISH_BTN_ID = "error-publish-all-btn"

#: Toast reporting the exact publication outcome.
ERROR_PUBLISH_TOAST_ID = "error-publish-toast"

#: Banner shown when stored labels re-keyed/dropped against the current
#: master (from ``filtered_state.rekey_report`` / ``.stale``).
ERROR_STALE_BANNER_ID = "error-stale-banner"

#: "Need more labels" empty-state card (shown when ``enough_data`` is
#: ``False`` or the verified-good count is below ``min_good_n``).
ERROR_EMPTY_STATE_ID = "error-empty-state"

#: Wrapper around the table+figure content, hidden in the empty state.
ERROR_CONTENT_ID = "error-content"

# -- per-tab stores ----------------------------------------------------------

#: ``dcc.Store`` of the focused-measurement context:
#: ``{category, measurement, direction, cutoff, good_values, error_values}``.
#: The drag/numeric readout reads the value arrays from here so it never
#: re-reads parquet.
STORE_ERROR_FOCUS_ID = "store-error-focus"

#: ``dcc.Store`` echoing the focused-category token for the recompute path
#: (written by a chip click).
STORE_ERROR_CATEGORY_ID = "store-error-category"

#: The mode-aware message paragraph inside the empty-state card (swapped to
#: a "review more QC groups" prompt when verified mode is the limiting class).
ERROR_EMPTY_STATE_MSG_ID = "error-empty-state-msg"


__all__ = [
    "ERROR_CATEGORY_CHIPS_ID",
    "ERROR_GOOD_MODE_TOGGLE_ID",
    "ERROR_VERIFIED_COUNT_ID",
    "ERROR_TABLE_ID",
    "ERROR_FIGURE_ID",
    "ERROR_CUTOFF_INPUT_ID",
    "ERROR_READOUT_ID",
    "ERROR_FILTER_SPEC_ID",
    "ERROR_CLIPBOARD_ID",
    "ERROR_PUBLISH_BTN_ID",
    "ERROR_PUBLISH_TOAST_ID",
    "ERROR_STALE_BANNER_ID",
    "ERROR_EMPTY_STATE_ID",
    "ERROR_EMPTY_STATE_MSG_ID",
    "ERROR_CONTENT_ID",
    "STORE_ERROR_FOCUS_ID",
    "STORE_ERROR_CATEGORY_ID",
]
