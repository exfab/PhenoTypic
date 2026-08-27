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

#: Bool: when true, the Viv stages across all cards are linked so
#: pan/zoom on one mirrors to the others.
STORE_LOCK_VIEWS = "store-lock-views"

#: Monotonic shell binding generation carried by this rendered page.
STORE_BINDING_GENERATION = "results-binding-generation"


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
#: ``MutationObserver`` watches this node so it can destroy Viv stages
#: when their card is removed from the DOM.
CARDS_CONTAINER_ID = "cards-container"

#: One-line pipeline label rendered in the header bar (text from
#: :attr:`OutputRoot.pipeline_summary`).
HEADER_PIPELINE_CHIP_ID = "header-pipeline-chip"

#: Mode badge in the header bar — "Full run" (per-image ``results/`` present)
#: vs "Standalone bundle" (deliverables-only). Reads ``OutputRoot.has_results``.
HEADER_MODE_BADGE_ID = "header-mode-badge"

#: Snapshot freshness badge updated by a lightweight fingerprint check.
HEADER_SNAPSHOT_STATUS_ID = "header-snapshot-status"

#: Explicit shared Results/Analysis snapshot refresh action.
BTN_REFRESH_SNAPSHOT = "btn-refresh-snapshot"

#: Inline error surfaced when an explicit snapshot refresh is refused.
HEADER_REFRESH_ERROR_ID = "header-refresh-error"

#: Persistent diagnostic shown when discovery succeeded but completion
#: evidence does not authorize any Results/Analysis mutation.
READ_ONLY_DIAGNOSTIC_ID = "results-read-only-diagnostic"

#: Poll trigger for status-only freshness checks. It never refreshes data.
SNAPSHOT_STATUS_INTERVAL_ID = "snapshot-status-interval"

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


def filter_row_method_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's method dropdown.

    Returns ``{"type": "filter-row-method", "index": idx}``.
    """
    return {"type": "filter-row-method", "index": idx}


def filter_row_range_min_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's range-min numeric input."""
    return {"type": "filter-row-range-min", "index": idx}


def filter_row_range_max_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's range-max numeric input."""
    return {"type": "filter-row-range-max", "index": idx}


def filter_row_compare_op_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's compare-operator dropdown."""
    return {"type": "filter-row-compare-op", "index": idx}


def filter_row_compare_value_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's compare-threshold numeric input."""
    return {"type": "filter-row-compare-value", "index": idx}


def filter_row_text_pattern_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's contains text input."""
    return {"type": "filter-row-text-pattern", "index": idx}


def filter_row_text_regex_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's contains regex checkbox."""
    return {"type": "filter-row-text-regex", "index": idx}


def filter_row_text_case_id(idx: str) -> Dict[str, str]:
    """Pattern id for a filter-row's contains case-sensitive checkbox."""
    return {"type": "filter-row-text-case", "index": idx}


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


def card_picker_prev_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's previous-image button.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-picker-prev", "index": idx}``.
    """
    return {"type": "card-picker-prev", "index": idx}


def card_picker_next_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's next-image button.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-picker-next", "index": idx}``.
    """
    return {"type": "card-picker-next", "index": idx}


def card_stage_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's Viv canvas container.

    The clientside bridge reads the ``index`` to key its
    ``Map<containerId, instance>`` so each card has exactly one deck.gl
    viewer at any time. Named for the *stage*, not for a library: the
    element is a full-canvas image surface, and the pixel client mounted
    into it changed once already.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-viv-stage", "index": idx}``.
    """
    return {"type": "card-viv-stage", "index": idx}


def card_source_store_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's resolved source spec.

    Payload is :func:`~phenotypic.gui.results_viewer._store_source
    .build_source_spec`'s dict, which crosses to
    ``window.phenotypicViv.setSource`` unmodified, or ``None`` when the
    card holds no store-backed image.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-source-spec", "index": idx}``.
    """
    return {"type": "card-source-spec", "index": idx}


def card_display_state_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's layer display state.

    Payload is ``{"seriesPath": str, "labelVisible": bool, "opacity":
    {"image": float, "labels": float}}`` -- what the Layers panel has been
    set to, kept apart from the store-derived spec so a re-source does not
    silently reset it.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-display-state", "index": idx}``.
    """
    return {"type": "card-display-state", "index": idx}


def card_layers_panel_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's floating Layers panel body.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-layers-panel", "index": idx}``.
    """
    return {"type": "card-layers-panel", "index": idx}


def card_layer_eye_id(idx: str, layer: str) -> Dict[str, str]:
    """Build the pattern-matching id for one Layers-panel visibility button.

    Carries the layer name as a third key so one MATCH/ALL callback covers
    every row of one card, whatever series the store turned out to hold.

    Args:
        idx: Owning card's ``index``.
        layer: Series name, or the objmap label name.

    Returns:
        Dict of shape ``{"type": "card-layer-eye", "index": idx,
        "layer": layer}``.
    """
    return {"type": "card-layer-eye", "index": idx, "layer": layer}


def card_layer_opacity_id(idx: str, layer: str) -> Dict[str, str]:
    """Build the pattern-matching id for one Layers-panel opacity slider.

    Args:
        idx: Owning card's ``index``.
        layer: Series name, or the objmap label name.

    Returns:
        Dict of shape ``{"type": "card-layer-opacity", "index": idx,
        "layer": layer}``.
    """
    return {"type": "card-layer-opacity", "index": idx, "layer": layer}


def card_pyramid_readout_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's served-level readout.

    Written by the clientside bridge from the facade's ``onLevelChange``,
    never from a server-side level computation: the level in use is
    deck.gl's per-frame choice, and a readout labelled "the level actually
    being served" is trusted exactly when diagnosing the bug a
    server-side number would misreport.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-pyramid-readout", "index": idx}``.
    """
    return {"type": "card-pyramid-readout", "index": idx}


def card_zoom_readout_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's zoom readout.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-zoom-readout", "index": idx}``.
    """
    return {"type": "card-zoom-readout", "index": idx}


def card_source_note_id(idx: str) -> Dict[str, str]:
    """Build the pattern-matching id for a card's provenance footer note.

    Args:
        idx: Owning card's ``index``.

    Returns:
        Dict of shape ``{"type": "card-source-note", "index": idx}``.
    """
    return {"type": "card-source-note", "index": idx}


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
    changes here to resolve the card's Viv source spec.

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
#: per-card source specs to ``window.__phenotypicResultsViewer
#: .applyPlateSources``. Carries a millisecond timestamp purely as a
#: change-trigger; the data itself is unused on the Python side.
VIV_MOUNT_TRIGGER_ID = "viv-mount-trigger"

#: Hidden ``dcc.Store`` written by the clientside callback that bridges
#: ``STORE_LOCK_VIEWS`` to ``window.__phenotypicResultsViewer.setLockViews``.
#: Same trigger-only semantics as :data:`VIV_MOUNT_TRIGGER_ID`.
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

#: ``dbc.Tab`` value for the Error-analysis view. The 5th tab; its
#: recompute callback gates on ``active_tab == TAB_ERROR_ID`` so the
#: cutoff finder never runs while the user is curating on another tab.
TAB_ERROR_ID = "tab-error"


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

#: Store carrying the rendered colony tile size in CSS pixels. The
#: server-side crop is still produced at the full bbox resolution; this
#: value only scales the on-screen ``<img>`` width/height via CSS.
STORE_COLONY_TILE_SIZE = "store-colony-tile-size"

#: ``−`` button for the rendered colony tile size stepper.
COLONY_TILE_SIZE_MINUS = "colony-tile-size-minus"

#: ``+`` button for the rendered colony tile size stepper.
COLONY_TILE_SIZE_PLUS = "colony-tile-size-plus"

#: Read-only ``150 px`` tile-size readout between the stepper buttons.
COLONY_TILE_SIZE_READOUT = "colony-tile-size-readout"

#: Toggle that turns the per-cell objmap/contour overlay on or off.
COLONY_OVERLAY_TOGGLE_ID = "colony-overlay-toggle"

#: ``−`` button of the colony-view tile-spotlight ``dim`` stepper. Each
#: click steps the shared :data:`STORE_TILE_DIM_ALPHA` strength down by
#: :data:`phenotypic.gui._config.TILE_DIM_STEP` (clamped at
#: :data:`TILE_DIM_MIN`).
COLONY_DIM_MINUS = "colony-dim-minus"

#: ``+`` button of the colony-view tile-spotlight ``dim`` stepper. Steps
#: the shared strength up by ``TILE_DIM_STEP`` (clamped at ``TILE_DIM_MAX``).
COLONY_DIM_PLUS = "colony-dim-plus"

#: Read-only ``dim 0.60`` readout between the colony stepper's buttons.
#: Synced from :data:`STORE_TILE_DIM_ALPHA` by the shared readout callback.
COLONY_DIM_READOUT = "colony-dim-readout"

#: Segmented control (``dbc.RadioItems``, button-group style) choosing which
#: image layer the colony crops source — ``rgb`` / ``detect_mat`` / ``objmap``
#: (labelled "RGB" / "Enhanced" / "Labels"). Rendered into the colony toolbar
#: only when per-image ``results/`` HDFs are available
#: (:attr:`OutputRoot.has_results`); a standalone deliverables bundle hides it
#: (overlays are pre-baked RGB, so the layer choice is moot there).
LAYER_TOGGLE = "colony-layer-toggle"

#: ``dcc.Store`` mirroring the active pixel layer (one of ``rgb`` /
#: ``detect_mat`` / ``objmap``; default ``rgb``). Mounted **unconditionally**
#: in the colony tab body so the grid-render callback's Input always resolves
#: — even in a standalone bundle where the visible :data:`LAYER_TOGGLE` is
#: hidden. The render callback threads its value onto every crop URL as
#: ``&layer=<layer>``.
STORE_ACTIVE_LAYER = "store-active-layer"


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

#: Category dropdown in the bulk bar — "Mark N selected as ▾". Options are
#: ``filtered_state.categories()`` (core + custom); selecting one marks the
#: active selection via ``mark_many(selected, category)``. Distinct from the
#: explicit Remove(=other)/Restore buttons, which stay.
COLONY_BULK_MARK_DROPDOWN_ID = "colony-bulk-mark-dropdown"

#: ``dcc.Store`` ticked whenever the category vocabulary changes (a custom
#: category is registered), so the bulk-mark dropdowns + open radial wheels
#: refresh their options/body. Bumped by the custom-add callbacks (Task 7).
STORE_CATEGORY_VOCAB_REVISION = "store-category-vocab-revision"


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------

#: ``dcc.Store`` holding the curated removed-colony keys as a list of
#: ``"<image_file>::<label>"`` strings. Persisted across reloads so manual
#: curation survives a page refresh.
STORE_REMOVED_KEYS = "store-removed-keys"

#: Hidden revision store bumped after configured ``PlotMeas`` outputs refresh
#: from a GUI curation update. It serializes the side effect behind
#: ``STORE_REMOVED_KEYS`` without coupling any visible component to file writes.
STORE_PLOT_REFRESH_REVISION = "store-plot-refresh-revision"

#: ``dcc.Store`` holding the tile-spotlight ``dim`` strength (a float in
#: ``[TILE_DIM_MIN, TILE_DIM_MAX]``) shared by **both** the colony-view and
#: QC-Review tile galleries. The ``−``/``+`` steppers in each toolbar write
#: it (via :func:`phenotypic.gui._config.step_dim_alpha`); both galleries'
#: render callbacks read it and thread it onto each crop URL as ``&dim=``.
#: ``storage_type="local"`` so the chosen strength survives reloads.
STORE_TILE_DIM_ALPHA = "store-tile-dim-alpha"


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
        image_file: ``Metadata_ImageName`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape ``{"type": "colony-cell", "image_file": image_file, "label": label}``.
    """
    return {"type": "colony-cell", "image_file": image_file, "label": label}


def colony_cell_remove_btn_id(
    image_file: str, label: int
) -> Dict[str, str | int]:
    """Build the pattern-matching id for a colony-cell single-action remove button.

    Args:
        image_file: ``Metadata_ImageName`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape
        ``{"type": "colony-cell-remove-btn", "image_file": image_file, "label": label}``.
    """
    return {
        "type": "colony-cell-remove-btn",
        "image_file": image_file,
        "label": label,
    }


def colony_cell_count_badge_id(
    image_file: str, label: int
) -> Dict[str, str | int]:
    """Build the pattern-matching id for a colony-cell N=k badge.

    The badge reports how many colonies are aggregated behind a given tile
    (relevant when the grid bins by axis values rather than rendering one
    tile per object).

    Args:
        image_file: ``Metadata_ImageName`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape
        ``{"type": "colony-cell-count-badge", "image_file": image_file, "label": label}``.
    """
    return {
        "type": "colony-cell-count-badge",
        "image_file": image_file,
        "label": label,
    }


def colony_cell_popover_body_id(
    image_file: str, label: int
) -> Dict[str, str | int]:
    """Build the pattern-matching id for a colony-cell popover body.

    The body is rendered empty at grid build time and populated on first
    badge click via a ``MATCH`` callback. Defers fetching of every
    member's crop until the user actually opens the stack.

    Args:
        image_file: ``Metadata_ImageName`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape
        ``{"type": "colony-cell-popover-body", "image_file": image_file, "label": label}``.
    """
    return {
        "type": "colony-cell-popover-body",
        "image_file": image_file,
        "label": label,
    }


def colony_cell_popover_data_id(
    image_file: str, label: int
) -> Dict[str, str | int]:
    """Build the pattern-matching id for a colony-cell popover's data store.

    Each multi-colony cell carries a co-located ``dcc.Store`` holding the
    members list and per-grid sizes; the populate-on-click callback reads
    it as State (matched to the firing badge) so the cell context never
    has to be re-derived on click.

    Args:
        image_file: ``Metadata_ImageName`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape
        ``{"type": "colony-cell-popover-data", "image_file": image_file, "label": label}``.
    """
    return {
        "type": "colony-cell-popover-data",
        "image_file": image_file,
        "label": label,
    }


def colony_cell_expand_btn_id(
    image_file: str, label: int
) -> Dict[str, str | int]:
    """Build the pattern-matching id for a colony-cell expand-on-click trigger.

    Clicking the trigger opens a detailed view of the cell's underlying
    colonies (e.g. when the tile aggregates multiple objects into a single
    representative thumbnail).

    Args:
        image_file: ``Metadata_ImageName`` for the cell's representative colony.
        label: ``Object_Label`` for the cell's representative colony.

    Returns:
        Dict of shape
        ``{"type": "colony-cell-expand-btn", "image_file": image_file, "label": label}``.
    """
    return {
        "type": "colony-cell-expand-btn",
        "image_file": image_file,
        "label": label,
    }


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
    "HEADER_MODE_BADGE_ID",
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
    "filter_row_method_id",
    "filter_row_range_min_id",
    "filter_row_range_max_id",
    "filter_row_compare_op_id",
    "filter_row_compare_value_id",
    "filter_row_text_pattern_id",
    "filter_row_text_regex_id",
    "filter_row_text_case_id",
    "card_id",
    "card_picker_id",
    "card_picker_prev_id",
    "card_picker_next_id",
    "card_stage_id",
    "card_source_store_id",
    "card_display_state_id",
    "card_layers_panel_id",
    "card_layer_eye_id",
    "card_layer_opacity_id",
    "card_pyramid_readout_id",
    "card_zoom_readout_id",
    "card_source_note_id",
    "card_details_toggle_id",
    "card_details_table_id",
    "card_remove_id",
    "card_state_store_id",
    "card_details_collapse_id",
    "card_info_chip_dataset_id",
    "card_info_chip_stem_id",
    "card_info_chip_count_id",
    "INITIAL_CARD_TRIGGER_ID",
    "VIV_MOUNT_TRIGGER_ID",
    "LOCK_VIEWS_EFFECT_ID",
    "COLONY_SELECTION_EFFECT_ID",
    "TABS_ID",
    "TAB_PLATE_ID",
    "TAB_COLONY_ID",
    "TAB_QC_ID",
    "TAB_HEATMAP_ID",
    "TAB_ERROR_ID",
    "STORE_QC_RECIPE_REVISION",
    "STORE_QC_AUGMENTED_REVISION",
    "COLONY_X_AXIS_DROPDOWN_ID",
    "COLONY_Y_AXIS_DROPDOWN_ID",
    "COLONY_GRID_CONTAINER_ID",
    "COLONY_TOOLBAR_ID",
    "COLONY_CROP_SIZE_INFO_ID",
    "COLONY_BTN_REFRESH_ID",
    "STORE_COLONY_TILE_SIZE",
    "COLONY_TILE_SIZE_MINUS",
    "COLONY_TILE_SIZE_PLUS",
    "COLONY_TILE_SIZE_READOUT",
    "COLONY_OVERLAY_TOGGLE_ID",
    "COLONY_DIM_MINUS",
    "COLONY_DIM_PLUS",
    "COLONY_DIM_READOUT",
    "LAYER_TOGGLE",
    "STORE_ACTIVE_LAYER",
    "COLONY_BULK_BAR_ID",
    "COLONY_BULK_COUNT_LABEL_ID",
    "COLONY_BULK_REMOVE_BTN_ID",
    "COLONY_BULK_RESTORE_BTN_ID",
    "COLONY_BULK_CLEAR_BTN_ID",
    "COLONY_BULK_MARK_DROPDOWN_ID",
    "STORE_CATEGORY_VOCAB_REVISION",
    "STORE_REMOVED_KEYS",
    "STORE_PLOT_REFRESH_REVISION",
    "STORE_TILE_DIM_ALPHA",
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
