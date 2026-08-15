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

# --- Timeline view (Phase 2) ---------------------------------------------
BROWSE_VIEW_MODE_TOGGLE = "browse-view-mode-toggle"  # "single" | "timeline"
BROWSE_SINGLE_BODY = "browse-single-body"  # existing OSD pane wrapper
BROWSE_TIMELINE_BODY = "browse-timeline-body"  # timeline matrix wrapper
BROWSE_TL_ROW_SOURCE = "browse-tl-row-source"  # folder|pattern|csv
BROWSE_TL_TIME_SOURCE = "browse-tl-time-source"  # exif|pattern|csv
BROWSE_TL_ROW_CSV_COL = "browse-tl-row-csv-col"
BROWSE_TL_TIME_CSV_COL = "browse-tl-time-csv-col"
BROWSE_TL_CSV_IMAGE_COL = "browse-tl-csv-image-col"
BROWSE_TL_PATTERN_INPUT = "browse-tl-pattern-input"
BROWSE_TL_PATTERN_ADVANCED = "browse-tl-pattern-advanced"
BROWSE_TL_PATTERN_PREVIEW = "browse-tl-pattern-preview"
BROWSE_TL_TILE_SIZE_MINUS = "browse-tl-tile-size-minus"
BROWSE_TL_TILE_SIZE_PLUS = "browse-tl-tile-size-plus"
BROWSE_TL_TILE_SIZE_READOUT = "browse-tl-tile-size-readout"
# Focus-and-navigate (spec §16). The four on-edge directional buttons and the
# focused-cell position readout are DOM targets driven by timeline.js — they
# need NO Dash callbacks (the controller wires clicks + keyboard in JS).
BROWSE_TL_NAV_UP = "browse-tl-nav-up"  # ▲ move focus up a row
BROWSE_TL_NAV_DOWN = "browse-tl-nav-down"  # ▼ move focus down a row
BROWSE_TL_NAV_LEFT = "browse-tl-nav-left"  # ◀ move focus back in time
BROWSE_TL_NAV_RIGHT = "browse-tl-nav-right"  # ▶ move focus forward in time
BROWSE_TL_POSITION = "browse-tl-position"  # "row 1/74 · time 1/24" readout
BROWSE_TL_NUDGE = "browse-tl-nudge"  # "add a CSV" banner
BROWSE_TL_GRID = "browse-tl-grid"  # grid container (timeline.js target)
BROWSE_TL_STORE_TILE_SIZE = "browse-tl-store-tile-size"
BROWSE_TL_STORE_WARNINGS = "browse-tl-store-warnings"
BROWSE_TL_WARNINGS_ALERT = (
    "browse-tl-warnings-alert"  # surfaces CSV-join warnings
)
BROWSE_TL_POPOUT_MODAL = "browse-tl-popout-modal"
BROWSE_TL_POPOUT_TITLE = "browse-tl-popout-title"
BROWSE_TL_POPOUT_OSD = "browse-tl-popout-osd"
BROWSE_TL_POPOUT_STORE = (
    "browse-tl-popout-store"  # {token,label} for the pop-out
)
BROWSE_TL_POPOUT_EVENT = (
    "browse-tl-popout-event"  # revision-bound JS→Dash event
)
BROWSE_TL_POPOUT_APPROVED = "browse-tl-popout-approved"
BROWSE_TL_SOURCE_REVISION = "browse-tl-source-revision"
BROWSE_TL_SESSION = "browse-tl-session"
BROWSE_TL_REVISION_CANDIDATE = "browse-tl-revision-candidate"
BROWSE_TL_REVISION_AUTHORIZED = "browse-tl-revision-authorized"

# --- Compare strip (Phase 4) ---------------------------------------------
# "Compare selected" button: a timeline.js DOM target (no Dash callback). The
# strip is built entirely client-side in plain DOM appended to document.body,
# so the v1 triggers are pure-JS — no Dash host div / JS→Dash bridge input.
BROWSE_TL_COMPARE_BTN = "browse-tl-compare-btn"

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
    "BROWSE_VIEW_MODE_TOGGLE",
    "BROWSE_SINGLE_BODY",
    "BROWSE_TIMELINE_BODY",
    "BROWSE_TL_ROW_SOURCE",
    "BROWSE_TL_TIME_SOURCE",
    "BROWSE_TL_ROW_CSV_COL",
    "BROWSE_TL_TIME_CSV_COL",
    "BROWSE_TL_CSV_IMAGE_COL",
    "BROWSE_TL_PATTERN_INPUT",
    "BROWSE_TL_PATTERN_ADVANCED",
    "BROWSE_TL_PATTERN_PREVIEW",
    "BROWSE_TL_TILE_SIZE_MINUS",
    "BROWSE_TL_TILE_SIZE_PLUS",
    "BROWSE_TL_TILE_SIZE_READOUT",
    "BROWSE_TL_NAV_UP",
    "BROWSE_TL_NAV_DOWN",
    "BROWSE_TL_NAV_LEFT",
    "BROWSE_TL_NAV_RIGHT",
    "BROWSE_TL_POSITION",
    "BROWSE_TL_NUDGE",
    "BROWSE_TL_GRID",
    "BROWSE_TL_STORE_TILE_SIZE",
    "BROWSE_TL_STORE_WARNINGS",
    "BROWSE_TL_WARNINGS_ALERT",
    "BROWSE_TL_POPOUT_MODAL",
    "BROWSE_TL_POPOUT_TITLE",
    "BROWSE_TL_POPOUT_OSD",
    "BROWSE_TL_POPOUT_STORE",
    "BROWSE_TL_POPOUT_EVENT",
    "BROWSE_TL_POPOUT_APPROVED",
    "BROWSE_TL_SOURCE_REVISION",
    "BROWSE_TL_SESSION",
    "BROWSE_TL_REVISION_CANDIDATE",
    "BROWSE_TL_REVISION_AUTHORIZED",
    "BROWSE_TL_COMPARE_BTN",
]
