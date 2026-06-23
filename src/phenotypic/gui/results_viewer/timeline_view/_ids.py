"""Component ids for the results-viewer Timeline tab.

All ids are kebab-case and prefixed ``timeline-`` so the timeline's
pattern/clientside callbacks never collide with the colony/QC surfaces
(spec §15.9). The four ``TIMELINE_NAV_*`` edge buttons and the
``TIMELINE_POSITION`` readout are DOM targets driven by ``timeline.js`` —
they carry NO Dash callback (the controller wires clicks + keyboard in JS).
"""
from __future__ import annotations

#: Grid container (the focus-navigate controller's attach target).
TIMELINE_GRID = "timeline-grid"
#: The whole Timeline tab body wrapper.
TIMELINE_BODY = "timeline-body"

#: Y (row) axis dropdown — uncapped selectable_axis_columns.
TIMELINE_Y_DROPDOWN = "timeline-y-dropdown"
#: X (time) axis dropdown — selectable_time_columns.
TIMELINE_X_DROPDOWN = "timeline-x-dropdown"

#: Colony-style tile-size stepper.
TIMELINE_TILE_SIZE_MINUS = "timeline-tile-size-minus"
TIMELINE_TILE_SIZE_PLUS = "timeline-tile-size-plus"
TIMELINE_TILE_SIZE_READOUT = "timeline-tile-size-readout"
TIMELINE_STORE_TILE_SIZE = "timeline-store-tile-size"

#: Focus-and-navigate edge buttons + position readout (spec §16) — JS targets.
TIMELINE_NAV_UP = "timeline-nav-up"
TIMELINE_NAV_DOWN = "timeline-nav-down"
TIMELINE_NAV_LEFT = "timeline-nav-left"
TIMELINE_NAV_RIGHT = "timeline-nav-right"
TIMELINE_POSITION = "timeline-position"

#: Bucketing-warning banner for a very long time axis (spec §15.2).
TIMELINE_LARGE_AXIS_WARNING = "timeline-large-axis-warning"
#: Guided empty state shown when no eligible time column exists (D9).
TIMELINE_EMPTY_STATE = "timeline-empty-state"

#: Deep-zoom pop-out (reuses the viewer's /tiles DZI route + OSD).
TIMELINE_POPOUT_MODAL = "timeline-popout-modal"
TIMELINE_POPOUT_OSD = "timeline-popout-osd"
TIMELINE_POPOUT_STORE = "timeline-popout-store"        # {dataset, stem}
TIMELINE_POPOUT_INPUT = "timeline-popout-input"        # hidden JS→Dash bridge
TIMELINE_POPOUT_OSD_SYNC = "timeline-popout-osd-sync"  # clientside-callback sink

__all__ = [
    "TIMELINE_GRID",
    "TIMELINE_BODY",
    "TIMELINE_Y_DROPDOWN",
    "TIMELINE_X_DROPDOWN",
    "TIMELINE_TILE_SIZE_MINUS",
    "TIMELINE_TILE_SIZE_PLUS",
    "TIMELINE_TILE_SIZE_READOUT",
    "TIMELINE_STORE_TILE_SIZE",
    "TIMELINE_NAV_UP",
    "TIMELINE_NAV_DOWN",
    "TIMELINE_NAV_LEFT",
    "TIMELINE_NAV_RIGHT",
    "TIMELINE_POSITION",
    "TIMELINE_LARGE_AXIS_WARNING",
    "TIMELINE_EMPTY_STATE",
    "TIMELINE_POPOUT_MODAL",
    "TIMELINE_POPOUT_OSD",
    "TIMELINE_POPOUT_STORE",
    "TIMELINE_POPOUT_INPUT",
    "TIMELINE_POPOUT_OSD_SYNC",
]
