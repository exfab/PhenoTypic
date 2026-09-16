"""Static Dash component ids for the Browse tab (single source of truth)."""

from __future__ import annotations

BROWSE_DATASET_ROW = "browse-dataset-row"  # wrapper, hidden when flat
BROWSE_DATASET_PICKER = "browse-dataset-picker"
BROWSE_IMAGE_PICKER = "browse-image-picker"
BROWSE_PREV_BTN = "browse-prev-btn"
BROWSE_NEXT_BTN = "browse-next-btn"
BROWSE_POSITION = "browse-position"  # current picker position: "N of M"
BROWSE_NAV_EVENT_STORE = (
    "browse-nav-event-store"  # JS keyboard event: {delta, sequence, source}
)
BROWSE_KEEP_POSITION = "browse-keep-position"  # local-only viewport preference
BROWSE_OSD_DIV = "browse-osd-div"  # OSD mounts here by id (.browse-osd-canvas)
BROWSE_PREVIEW_IMG = "browse-preview-img"  # progressive preview behind OSD
BROWSE_OSD_LOADING = "browse-osd-loading"  # spinner overlay, toggled by browse.js on OSD open/open-failed
BROWSE_LOADING_TEXT = (
    "browse-loading-text"  # loading/error caption inside the overlay
)
BROWSE_FILMSTRIP = (
    "browse-filmstrip"  # bounded nearby-image strip, rendered by browse.js
)
BROWSE_PREPARE_BTN = "browse-prepare-btn"
BROWSE_STOP_PREPARE_BTN = "browse-stop-prepare-btn"
BROWSE_CLEAR_CACHE_BTN = "browse-clear-cache-btn"
BROWSE_PREPARATION_STATUS = "browse-preparation-status"
BROWSE_PREPARATION_PROGRESS = "browse-preparation-progress"
BROWSE_CACHE_USAGE = "browse-cache-usage"
BROWSE_BACKEND_DETAILS = "browse-backend-details"
BROWSE_PREPARATION_STATUS_STORE = "browse-preparation-status-store"
BROWSE_PREPARATION_POLL = "browse-preparation-poll"
BROWSE_CURRENT_IMAGE_STORE = "browse-current-image-store"  # {token, label}
BROWSE_DATASETS_STORE = (
    "browse-datasets-store"  # {dataset_rel: [filename,...]}
)
BROWSE_OSD_SYNC = "browse-osd-sync"  # dummy clientside-callback sink
BROWSE_META_IMAGE_NAME = "browse-meta-image-name"
BROWSE_META_DIMS = "browse-meta-dims"
BROWSE_META_SIZE = "browse-meta-size"
BROWSE_META_CAPTURED = "browse-meta-captured"
BROWSE_META_CAMERA = "browse-meta-camera"
BROWSE_CSV_METADATA_PANEL = "browse-csv-metadata-panel"
BROWSE_EMPTY_HINT = "browse-empty-hint"  # shown when no source root

BROWSE_SINGLE_BODY = "browse-single-body"  # OSD pane wrapper

__all__ = [
    "BROWSE_DATASET_ROW",
    "BROWSE_DATASET_PICKER",
    "BROWSE_IMAGE_PICKER",
    "BROWSE_PREV_BTN",
    "BROWSE_NEXT_BTN",
    "BROWSE_POSITION",
    "BROWSE_NAV_EVENT_STORE",
    "BROWSE_KEEP_POSITION",
    "BROWSE_OSD_DIV",
    "BROWSE_PREVIEW_IMG",
    "BROWSE_OSD_LOADING",
    "BROWSE_LOADING_TEXT",
    "BROWSE_FILMSTRIP",
    "BROWSE_PREPARE_BTN",
    "BROWSE_STOP_PREPARE_BTN",
    "BROWSE_CLEAR_CACHE_BTN",
    "BROWSE_PREPARATION_STATUS",
    "BROWSE_PREPARATION_PROGRESS",
    "BROWSE_CACHE_USAGE",
    "BROWSE_BACKEND_DETAILS",
    "BROWSE_PREPARATION_STATUS_STORE",
    "BROWSE_PREPARATION_POLL",
    "BROWSE_CURRENT_IMAGE_STORE",
    "BROWSE_DATASETS_STORE",
    "BROWSE_OSD_SYNC",
    "BROWSE_META_IMAGE_NAME",
    "BROWSE_META_DIMS",
    "BROWSE_META_SIZE",
    "BROWSE_META_CAPTURED",
    "BROWSE_META_CAMERA",
    "BROWSE_CSV_METADATA_PANEL",
    "BROWSE_EMPTY_HINT",
    "BROWSE_SINGLE_BODY",
]
