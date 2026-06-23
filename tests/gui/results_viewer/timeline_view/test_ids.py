"""Timeline-view component ids: present, unique, namespaced."""
from __future__ import annotations

from phenotypic.gui.results_viewer.timeline_view import _ids


def test_timeline_ids_present_unique_and_namespaced() -> None:
    timeline_ids = [
        _ids.TIMELINE_GRID,
        _ids.TIMELINE_Y_DROPDOWN,
        _ids.TIMELINE_X_DROPDOWN,
        _ids.TIMELINE_TILE_SIZE_MINUS,
        _ids.TIMELINE_TILE_SIZE_PLUS,
        _ids.TIMELINE_TILE_SIZE_READOUT,
        _ids.TIMELINE_NAV_UP,
        _ids.TIMELINE_NAV_DOWN,
        _ids.TIMELINE_NAV_LEFT,
        _ids.TIMELINE_NAV_RIGHT,
        _ids.TIMELINE_POSITION,
        _ids.TIMELINE_LARGE_AXIS_WARNING,
        _ids.TIMELINE_EMPTY_STATE,
        _ids.TIMELINE_BODY,
        _ids.TIMELINE_STORE_TILE_SIZE,
        _ids.TIMELINE_POPOUT_MODAL,
        _ids.TIMELINE_POPOUT_OSD,
        _ids.TIMELINE_POPOUT_STORE,
        _ids.TIMELINE_POPOUT_INPUT,
        _ids.TIMELINE_POPOUT_OSD_SYNC,
    ]
    assert len(timeline_ids) == len(set(timeline_ids))
    assert all(isinstance(i, str) and i for i in timeline_ids)
    # Namespaced so colony/QC pattern callbacks never cross-fire (spec §15.9).
    assert all(i.startswith("timeline-") for i in timeline_ids)
