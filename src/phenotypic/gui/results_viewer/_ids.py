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
# Empty-state hand-off (only mounted when ``output_root is None``)
# ---------------------------------------------------------------------------

#: Container for the empty-state hand-off banner. Visibility toggled by
#: a callback that mirrors :data:`SHELL_SIDEBAR_SELECTION_STORE`.
EMPTY_HANDOFF_BANNER = "results-viewer-empty-handoff-banner"

#: Read-only label echoing the rel-path of the currently-selected sidebar
#: entry (or ``"(none)"`` when nothing is selected).
EMPTY_HANDOFF_LABEL = "results-viewer-empty-handoff-label"

#: "Open in viewer" button. Disabled when the selection is missing or the
#: clicked path lacks the ``is_cli_output`` capability.
EMPTY_HANDOFF_OPEN_BUTTON = "results-viewer-empty-handoff-open"

#: Inline error slot rendered inside the empty-state card when the POST to
#: ``/sandbox/api/viewer/output-root`` returns 4xx (e.g. layout invalid).
EMPTY_HANDOFF_ERROR = "results-viewer-empty-handoff-error"


# ---------------------------------------------------------------------------
# Static buttons
# ---------------------------------------------------------------------------

#: Adds an empty filter row to the sidebar.
BTN_ADD_FILTER_ROW = "btn-add-filter-row"

#: Adds a new viewer card to the cards column.
BTN_ADD_CARD = "btn-add-card"

#: Toggles the cross-card viewport-locking behaviour.
BTN_LOCK_VIEWS_TOGGLE = "btn-lock-views-toggle"

#: Top-bar button that opens/closes the right-docked filter offcanvas.
BTN_FILTERS_TOGGLE = "btn-filters-toggle"

#: Count badge on the Filters toggle showing the number of active filter rows.
FILTER_TOGGLE_BADGE_ID = "filter-toggle-badge"


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

#: Right-docked ``dbc.Offcanvas`` hosting the filter panel; its ``is_open``
#: is driven by :data:`BTN_FILTERS_TOGGLE`.
OFFCANVAS_FILTER_ID = "filter-offcanvas"


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

#: Hidden ``dcc.Store`` written by the clientside callback that toggles
#: ``.is-selected`` on colony cells in response to
#: :data:`STORE_COLONY_SELECTION` changes. Carries a millisecond
#: timestamp purely as a change-trigger; the data itself is unused on
#: the Python side. Keeping selection styling in JS lets shift-click
#: avoid a full server round-trip + grid re-render.
COLONY_SELECTION_EFFECT_ID = "colony-selection-effect"


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

#: Top-level ``dbc.Tabs`` container switching between the plate-level cards
#: view and the colony-level grid view.
TABS_ID = "results-viewer-tabs"

#: ``dbc.Tab`` value for the plate (per-image cards) view.
TAB_PLATE_ID = "tab-plate"

#: ``dbc.Tab`` value for the colony (per-object grid) view.
TAB_COLONY_ID = "tab-colony"

#: ``dbc.Tab`` value for the QC checks view. The tab body itself is
#: built by Wave E; this constant is mounted now so the Heatmap tab's
#: callbacks (which reference the recipe-revision store) can land
#: ahead of the QC tab implementation without an import cycle.
TAB_QC_ID = "tab-qc"

#: ``dbc.Tab`` value for the heatmap view (per-image grid view).
TAB_HEATMAP_ID = "tab-heatmap"


# ---------------------------------------------------------------------------
# QC stores (Wave D mounts the stores; Wave E writes to them)
# ---------------------------------------------------------------------------

#: ``dcc.Store`` carrying the active :class:`QcRecipe` revision counter.
#: Bumped by Wave E's ``add``/``remove``/``update`` callbacks. The
#: Heatmap tab subscribes to it so its picker options refresh whenever
#: a check is added or removed.
STORE_QC_RECIPE_REVISION = "store-qc-recipe-revision"

#: ``dcc.Store`` bumped by Wave E's QC tab callback **after** it has
#: finished writing :data:`phenotypic.gui._config.CFG_QC_AUGMENTED_FRAME`.
#: The Heatmap render callback subscribes to it (in addition to
#: ``STORE_REMOVED_KEYS``) so the augmented-frame read is deterministic
#: rather than racing the QC writer. See the spec's "shared
#: augmented-frame cache" section (lines 775-798).
STORE_QC_AUGMENTED_REVISION = "store-qc-augmented-revision"


# ---------------------------------------------------------------------------
# Colony-view static
# ---------------------------------------------------------------------------

#: Dropdown selecting the measurement column plotted on the colony-grid
#: x-axis (or used as the primary sort key when laying out the grid).
COLONY_X_AXIS_DROPDOWN_ID = "colony-x-axis-dropdown"

#: Dropdown selecting the measurement column plotted on the colony-grid
#: y-axis (or used as the secondary sort key when laying out the grid).
COLONY_Y_AXIS_DROPDOWN_ID = "colony-y-axis-dropdown"

#: Parent ``<div>`` wrapping every colony-cell tile. The clientside layer
#: queries this node to wire up checkbox / drag-select behaviour.
COLONY_GRID_CONTAINER_ID = "colony-grid-container"

#: Toolbar above the colony grid hosting axis dropdowns, refresh, overlay
#: toggle and the crop-size info chip.
COLONY_TOOLBAR_ID = "colony-toolbar"

#: Read-only chip showing the per-cell crop size (e.g. ``128x128 px``).
COLONY_CROP_SIZE_INFO_ID = "colony-crop-size-info"

#: Button forcing a re-render of the colony grid (re-reads stores and
#: re-builds the tile DOM).
COLONY_BTN_REFRESH_ID = "colony-btn-refresh"

#: Slider controlling the rendered tile size in pixels (lets the user
#: shrink to fit narrow screens or enlarge to inspect detail). The
#: server-side crop is still produced at the full bbox resolution; this
#: slider only scales the on-screen ``<img>`` width/height via CSS.
COLONY_TILE_SIZE_SLIDER_ID = "colony-tile-size-slider"

#: Toggle that turns the per-cell objmap/contour overlay on or off.
COLONY_OVERLAY_TOGGLE_ID = "colony-overlay-toggle"


# ---------------------------------------------------------------------------
# Bulk action bar
# ---------------------------------------------------------------------------

#: Container for the bulk-selection action bar. Hidden when the
#: selection store is empty; revealed once one or more cells are checked.
COLONY_BULK_BAR_ID = "colony-bulk-bar"

#: Label inside the bulk bar reporting how many colonies are currently
#: selected (``"N selected"``).
COLONY_BULK_COUNT_LABEL_ID = "colony-bulk-count-label"

#: Button removing every currently-selected colony from the curated set
#: (writes their keys into ``STORE_REMOVED_KEYS``).
COLONY_BULK_REMOVE_BTN_ID = "colony-bulk-remove-btn"

#: Button restoring every currently-selected colony — i.e. dropping their
#: keys from ``STORE_REMOVED_KEYS``.
COLONY_BULK_RESTORE_BTN_ID = "colony-bulk-restore-btn"

#: Button clearing the active selection without touching the curated set.
COLONY_BULK_CLEAR_BTN_ID = "colony-bulk-clear-btn"


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------

#: ``dcc.Store`` holding the curated removed-colony keys as a list of
#: ``"<image_file>::<label>"`` strings. Persisted across reloads so manual
#: curation survives a page refresh.
STORE_REMOVED_KEYS = "store-removed-keys"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

#: ``dcc.Store`` holding the current colony multi-select as a list of
#: ``"<image_file>::<label>"`` keys. Owned by the clientside layer; the
#: Python side only reads it to drive the bulk action bar.
STORE_COLONY_SELECTION = "store-colony-selection"

#: ``dcc.Store`` written by the clientside layer carrying the most recent
#: selection delta (``{"added": [...], "removed": [...]}``). Trigger-only
#: store used to keep large full-selection diffs out of the callback graph.
STORE_COLONY_SELECTION_DELTA = "store-colony-selection-delta"

#: ``dcc.Store`` holding the current visual order of the colony grid as a
#: list of ``"<image_file>::<label>"`` keys, so range-selection (shift-click)
#: in JS can resolve "everything between A and B" without re-querying Dash.
STORE_COLONY_GRID_ORDER = "store-colony-grid-order"


# ---------------------------------------------------------------------------
# Pattern-matching id-builders — colony cells
# ---------------------------------------------------------------------------


def colony_cell_id(image_file: str, label: int) -> Dict[str, str | int]:
    """Build the pattern-matching id for a colony-grid cell container.

    Args:
        image_file: ``Metadata_ImageFile`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape ``{"type": "colony-cell", "image_file": image_file, "label": label}``.
    """
    return {"type": "colony-cell", "image_file": image_file, "label": label}


def colony_cell_remove_btn_id(image_file: str, label: int) -> Dict[str, str | int]:
    """Build the pattern-matching id for a colony-cell single-action remove button.

    Args:
        image_file: ``Metadata_ImageFile`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape
        ``{"type": "colony-cell-remove-btn", "image_file": image_file, "label": label}``.
    """
    return {"type": "colony-cell-remove-btn", "image_file": image_file, "label": label}


def colony_cell_count_badge_id(image_file: str, label: int) -> Dict[str, str | int]:
    """Build the pattern-matching id for a colony-cell N=k badge.

    The badge reports how many colonies are aggregated behind a given tile
    (relevant when the grid bins by axis values rather than rendering one
    tile per object).

    Args:
        image_file: ``Metadata_ImageFile`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape
        ``{"type": "colony-cell-count-badge", "image_file": image_file, "label": label}``.
    """
    return {"type": "colony-cell-count-badge", "image_file": image_file, "label": label}


def colony_cell_popover_body_id(image_file: str, label: int) -> Dict[str, str | int]:
    """Build the pattern-matching id for a colony-cell popover body.

    The body is rendered empty at grid build time and populated on first
    badge click via a ``MATCH`` callback. Defers fetching of every
    member's crop until the user actually opens the stack.

    Args:
        image_file: ``Metadata_ImageFile`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape
        ``{"type": "colony-cell-popover-body", "image_file": image_file, "label": label}``.
    """
    return {"type": "colony-cell-popover-body", "image_file": image_file, "label": label}


def colony_cell_popover_data_id(image_file: str, label: int) -> Dict[str, str | int]:
    """Build the pattern-matching id for a colony-cell popover's data store.

    Each multi-colony cell carries a co-located ``dcc.Store`` holding the
    members list and per-grid sizes; the populate-on-click callback reads
    it as State (matched to the firing badge) so the cell context never
    has to be re-derived on click.

    Args:
        image_file: ``Metadata_ImageFile`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape
        ``{"type": "colony-cell-popover-data", "image_file": image_file, "label": label}``.
    """
    return {"type": "colony-cell-popover-data", "image_file": image_file, "label": label}


def colony_cell_expand_btn_id(image_file: str, label: int) -> Dict[str, str | int]:
    """Build the pattern-matching id for a colony-cell expand-on-click trigger.

    Clicking the trigger opens a detailed view of the cell's underlying
    colonies (e.g. when the tile aggregates multiple objects into a single
    representative thumbnail).

    Args:
        image_file: ``Metadata_ImageFile`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape
        ``{"type": "colony-cell-expand-btn", "image_file": image_file, "label": label}``.
    """
    return {"type": "colony-cell-expand-btn", "image_file": image_file, "label": label}


__all__ = [
    "STORE_FILTER_SPEC",
    "STORE_IMAGE_PAIRS",
    "STORE_CARD_LIST",
    "STORE_LOCK_VIEWS",
    "BTN_ADD_FILTER_ROW",
    "BTN_ADD_CARD",
    "BTN_LOCK_VIEWS_TOGGLE",
    "BTN_FILTERS_TOGGLE",
    "FILTER_TOGGLE_BADGE_ID",
    "CARDS_CONTAINER_ID",
    "HEADER_PIPELINE_CHIP_ID",
    "FILTER_ROWS_CONTAINER_ID",
    "FILTER_MATCH_COUNT_ID",
    "OFFCANVAS_FILTER_ID",
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
    "COLONY_SELECTION_EFFECT_ID",
    "TABS_ID",
    "TAB_PLATE_ID",
    "TAB_COLONY_ID",
    "TAB_QC_ID",
    "TAB_HEATMAP_ID",
    "STORE_QC_RECIPE_REVISION",
    "STORE_QC_AUGMENTED_REVISION",
    "COLONY_X_AXIS_DROPDOWN_ID",
    "COLONY_Y_AXIS_DROPDOWN_ID",
    "COLONY_GRID_CONTAINER_ID",
    "COLONY_TOOLBAR_ID",
    "COLONY_CROP_SIZE_INFO_ID",
    "COLONY_BTN_REFRESH_ID",
    "COLONY_TILE_SIZE_SLIDER_ID",
    "COLONY_OVERLAY_TOGGLE_ID",
    "COLONY_BULK_BAR_ID",
    "COLONY_BULK_COUNT_LABEL_ID",
    "COLONY_BULK_REMOVE_BTN_ID",
    "COLONY_BULK_RESTORE_BTN_ID",
    "COLONY_BULK_CLEAR_BTN_ID",
    "STORE_REMOVED_KEYS",
    "STORE_COLONY_SELECTION",
    "STORE_COLONY_SELECTION_DELTA",
    "STORE_COLONY_GRID_ORDER",
    "colony_cell_id",
    "colony_cell_remove_btn_id",
    "colony_cell_count_badge_id",
    "colony_cell_popover_body_id",
    "colony_cell_popover_data_id",
    "colony_cell_expand_btn_id",
]
