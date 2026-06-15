"""Static Dash component ids for the Browse tab (single source of truth)."""
from __future__ import annotations

BROWSE_DATASET_ROW = "browse-dataset-row"          # wrapper, hidden when flat
BROWSE_DATASET_PICKER = "browse-dataset-picker"
BROWSE_IMAGE_PICKER = "browse-image-picker"
BROWSE_PREV_BTN = "browse-prev-btn"
BROWSE_NEXT_BTN = "browse-next-btn"
BROWSE_OSD_DIV = "browse-osd-div"                  # OSD mounts here by id (.browse-osd-canvas)
BROWSE_OSD_LOADING = "browse-osd-loading"          # spinner overlay, toggled by browse.js on OSD open/open-failed
BROWSE_LOADING_TEXT = "browse-loading-text"        # loading/error caption inside the overlay
BROWSE_CURRENT_IMAGE_STORE = "browse-current-image-store"  # {token, label}
BROWSE_DATASETS_STORE = "browse-datasets-store"    # {dataset_rel: [filename,...]}
BROWSE_OSD_SYNC = "browse-osd-sync"                # dummy clientside-callback sink
BROWSE_META_DIMS = "browse-meta-dims"
BROWSE_META_SIZE = "browse-meta-size"
BROWSE_META_CAPTURED = "browse-meta-captured"
BROWSE_META_CAMERA = "browse-meta-camera"
BROWSE_EMPTY_HINT = "browse-empty-hint"            # shown when no source root

__all__ = [
    "BROWSE_DATASET_ROW",
    "BROWSE_DATASET_PICKER",
    "BROWSE_IMAGE_PICKER",
    "BROWSE_PREV_BTN",
    "BROWSE_NEXT_BTN",
    "BROWSE_OSD_DIV",
    "BROWSE_OSD_LOADING",
    "BROWSE_LOADING_TEXT",
    "BROWSE_CURRENT_IMAGE_STORE",
    "BROWSE_DATASETS_STORE",
    "BROWSE_OSD_SYNC",
    "BROWSE_META_DIMS",
    "BROWSE_META_SIZE",
    "BROWSE_META_CAPTURED",
    "BROWSE_META_CAMERA",
    "BROWSE_EMPTY_HINT",
]
