"""Single source of truth for component IDs in the results viewer.

This module exposes plain string constants for static (non-pattern-matching)
component ids, plus small helpers for the pattern-matching ids used by the
dynamic filter rows and viewer cards. Layout and callback modules both
import from here so the contract between them is stable and grep-able.

Notes:
    - Pattern-matching ids returned by helpers are plain dicts; Dash hashes
      them at registration time, so callbacks can use ``MATCH`` / ``ALL``
      against the ``type`` key documented on each helper.
    - Constants and pattern type-strings use kebab-case to match Dash
      convention and the existing ids in
      :mod:`phenotypic.gui.builder._ids`.
"""

from __future__ import annotations

from typing import Dict


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

#: Holds the FilterSpec store payload — a list of
#: ``{"column": str, "values": list[str]}`` dicts (see
#: :class:`phenotypic.gui.results_viewer._filter_state.FilterSpec`).
STORE_FILTER_SPEC = "store-filter-spec"

#: List of unique ``[dataset, stem]`` pairs surviving the active filter.
STORE_IMAGE_PAIRS = "store-image-pairs"

#: Ordered list of card ids currently rendered in the cards column.
STORE_CARD_LIST = "store-card-list"

#: Bool: when true, OpenSeadragon viewers across all cards are linked so
#: pan/zoom on one mirrors to the others.
STORE_LOCK_VIEWS = "store-lock-views"


# ---------------------------------------------------------------------------
# Static buttons
# ---------------------------------------------------------------------------

#: Adds an empty filter row to the sidebar.
BTN_ADD_FILTER_ROW = "btn-add-filter-row"

#: Adds a new viewer card to the cards column.
BTN_ADD_CARD = "btn-add-card"

#: Toggles the cross-card viewport-locking behaviour.
BTN_LOCK_VIEWS_TOGGLE = "btn-lock-views-toggle"


# ---------------------------------------------------------------------------
# Static layout anchors used by the JS layer
# ---------------------------------------------------------------------------

#: Parent ``<div>`` wrapping every viewer card. The clientside
#: ``MutationObserver`` watches this node so it can dispose OSD viewers
#: when their card is removed from the DOM.
CARDS_CONTAINER_ID = "cards-container"

#: One-line pipeline label rendered in the header bar (text from
#: :attr:`OutputRoot.pipeline_summary`).
HEADER_PIPELINE_CHIP_ID = "header-pipeline-chip"

#: Outer pattern-matching root for the dynamic filter rows. The filter
#: rows are rendered into this ``html.Div`` by a callback whenever
#: ``STORE_FILTER_SPEC`` changes.
FILTER_ROWS_CONTAINER_ID = "filter-rows-container"

#: Read-only chip showing how many image pairs survive the active
#: filter (text reflects the size of ``STORE_IMAGE_PAIRS``).
FILTER_MATCH_COUNT_ID = "filter-match-count"


# ---------------------------------------------------------------------------
# Pattern-matching id-builders — filter rows
# ---------------------------------------------------------------------------


def filter_row_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a filter-row container.

    Args:
        idx: Stable per-row identifier (typically a UUID4 hex string).

    Returns:
        Dict of shape ``{"type": "filter-row", "index": idx}``.
    """
    return {"type": "filter-row", "index": idx}


def filter_row_column_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a filter-row's column dropdown.

    Args:
        idx: Owning filter row's ``index``.

    Returns:
        Dict of shape ``{"type": "filter-row-column", "index": idx}``.
    """
    return {"type": "filter-row-column", "index": idx}


def filter_row_values_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a filter-row's multi-value dropdown.

    Args:
        idx: Owning filter row's ``index``.

    Returns:
        Dict of shape ``{"type": "filter-row-values", "index": idx}``.
    """
    return {"type": "filter-row-values", "index": idx}


def filter_row_paste_btn_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a filter-row's bulk-paste button.

    Args:
        idx: Owning filter row's ``index``.

    Returns:
        Dict of shape ``{"type": "filter-row-paste-btn", "index": idx}``.
    """
    return {"type": "filter-row-paste-btn", "index": idx}


def filter_row_paste_popover_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a filter-row's bulk-paste popover.

    Args:
        idx: Owning filter row's ``index``.

    Returns:
        Dict of shape ``{"type": "filter-row-paste-popover", "index": idx}``.
    """
    return {"type": "filter-row-paste-popover", "index": idx}


def filter_row_paste_textarea_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for the textarea inside the paste popover.

    Args:
        idx: Owning filter row's ``index``.

    Returns:
        Dict of shape ``{"type": "filter-row-paste-textarea", "index": idx}``.
    """
    return {"type": "filter-row-paste-textarea", "index": idx}


def filter_row_paste_apply_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for the Apply button inside the paste popover.

    Args:
        idx: Owning filter row's ``index``.

    Returns:
        Dict of shape ``{"type": "filter-row-paste-apply", "index": idx}``.
    """
    return {"type": "filter-row-paste-apply", "index": idx}


def filter_row_paste_chips_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for the chip-list preview inside the popover.

    Args:
        idx: Owning filter row's ``index``.

    Returns:
        Dict of shape ``{"type": "filter-row-paste-chips", "index": idx}``.
    """
    return {"type": "filter-row-paste-chips", "index": idx}


def filter_row_remove_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a filter-row's remove button.

    Args:
        idx: Owning filter row's ``index``.

    Returns:
        Dict of shape ``{"type": "filter-row-remove", "index": idx}``.
    """
    return {"type": "filter-row-remove", "index": idx}


# ---------------------------------------------------------------------------
# Pattern-matching id-builders — viewer cards
# ---------------------------------------------------------------------------


def card_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a viewer-card container.

    Args:
        idx: Stable per-card identifier (typically a UUID4 hex string).

    Returns:
        Dict of shape ``{"type": "card", "index": idx}``.
    """
    return {"type": "card", "index": idx}


def card_picker_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's image picker dropdown.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-picker", "index": idx}``.
    """
    return {"type": "card-picker", "index": idx}


def card_osd_div_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's OpenSeadragon container.

    The clientside JS reads the ``index`` to key the
    ``Map<divId, OSD.Viewer>`` so each card has exactly one viewer at any
    time.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-osd-div", "index": idx}``.
    """
    return {"type": "card-osd-div", "index": idx}


def card_details_toggle_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's details-toggle button.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-details-toggle", "index": idx}``.
    """
    return {"type": "card-details-toggle", "index": idx}


def card_details_table_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's per-object DataTable.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-details-table", "index": idx}``.
    """
    return {"type": "card-details-table", "index": idx}


def card_remove_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's remove button.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-remove", "index": idx}``.
    """
    return {"type": "card-remove", "index": idx}


def card_state_store_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's per-card state store.

    The store payload is a ``{"dataset": str, "stem": str}`` dict (or
    ``None`` for an unselected card); a clientside callback reacts to
    changes here to (re-)mount the OpenSeadragon viewer.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-state", "index": idx}``.
    """
    return {"type": "card-state", "index": idx}


def card_details_collapse_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's details ``dbc.Collapse``.

    The collapse wraps the per-object ``DataTable`` and is toggled by the
    sibling :func:`card_details_toggle_id` button.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-details-collapse", "index": idx}``.
    """
    return {"type": "card-details-collapse", "index": idx}


def card_info_chip_dataset_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's "dataset" badge.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-info-dataset", "index": idx}``.
    """
    return {"type": "card-info-dataset", "index": idx}


def card_info_chip_stem_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's "image stem" badge.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-info-stem", "index": idx}``.
    """
    return {"type": "card-info-stem", "index": idx}


def card_info_chip_count_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's "n objects" badge.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-info-count", "index": idx}``.
    """
    return {"type": "card-info-count", "index": idx}


# ---------------------------------------------------------------------------
# One-shot triggers
# ---------------------------------------------------------------------------

#: ``dcc.Interval`` fired once on first load to seed an initial card so the
#: layout isn't empty when the user first opens the viewer.
INITIAL_CARD_TRIGGER_ID = "initial-card-trigger"

#: Hidden ``dcc.Store`` written by the clientside callback that bridges
#: per-card image selections to ``window.__phenotypicResultsViewer
#: .applyImageSelection``. Carries a millisecond timestamp purely as a
#: change-trigger; the data itself is unused on the Python side.
OSD_MOUNT_TRIGGER_ID = "osd-mount-trigger"

#: Hidden ``dcc.Store`` written by the clientside callback that bridges
#: ``STORE_LOCK_VIEWS`` to ``window.__phenotypicResultsViewer.setLockViews``.
#: Same trigger-only semantics as :data:`OSD_MOUNT_TRIGGER_ID`.
LOCK_VIEWS_EFFECT_ID = "lock-views-effect"


__all__ = [
    "STORE_FILTER_SPEC",
    "STORE_IMAGE_PAIRS",
    "STORE_CARD_LIST",
    "STORE_LOCK_VIEWS",
    "BTN_ADD_FILTER_ROW",
    "BTN_ADD_CARD",
    "BTN_LOCK_VIEWS_TOGGLE",
    "CARDS_CONTAINER_ID",
    "HEADER_PIPELINE_CHIP_ID",
    "FILTER_ROWS_CONTAINER_ID",
    "FILTER_MATCH_COUNT_ID",
    "filter_row_id",
    "filter_row_column_id",
    "filter_row_values_id",
    "filter_row_paste_btn_id",
    "filter_row_paste_popover_id",
    "filter_row_paste_textarea_id",
    "filter_row_paste_apply_id",
    "filter_row_paste_chips_id",
    "filter_row_remove_id",
    "card_id",
    "card_picker_id",
    "card_osd_div_id",
    "card_details_toggle_id",
    "card_details_table_id",
    "card_remove_id",
    "card_state_store_id",
    "card_details_collapse_id",
    "card_info_chip_dataset_id",
    "card_info_chip_stem_id",
    "card_info_chip_count_id",
    "INITIAL_CARD_TRIGGER_ID",
    "OSD_MOUNT_TRIGGER_ID",
    "LOCK_VIEWS_EFFECT_ID",
]
