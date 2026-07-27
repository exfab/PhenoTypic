from phenotypic.gui.browse import _ids


def test_ids_are_unique_nonempty_strings():
    values = [
        _ids.BROWSE_DATASET_ROW,
        _ids.BROWSE_DATASET_PICKER,
        _ids.BROWSE_IMAGE_PICKER,
        _ids.BROWSE_PREV_BTN,
        _ids.BROWSE_NEXT_BTN,
        _ids.BROWSE_OSD_DIV,
        _ids.BROWSE_OSD_LOADING,
        _ids.BROWSE_LOADING_TEXT,
        _ids.BROWSE_CURRENT_IMAGE_STORE,
        _ids.BROWSE_DATASETS_STORE,
        _ids.BROWSE_OSD_SYNC,
        _ids.BROWSE_META_DIMS,
        _ids.BROWSE_META_SIZE,
        _ids.BROWSE_META_CAPTURED,
        _ids.BROWSE_META_CAMERA,
        _ids.BROWSE_EMPTY_HINT,
    ]
    assert all(isinstance(v, str) and v for v in values)
    assert len(set(values)) == len(values)


def test_timeline_ids_present_and_unique() -> None:
    from phenotypic.gui.browse import _ids

    timeline_ids = [
        _ids.BROWSE_VIEW_MODE_TOGGLE,
        _ids.BROWSE_SINGLE_BODY,
        _ids.BROWSE_TIMELINE_BODY,
        _ids.BROWSE_TL_ROW_SOURCE,
        _ids.BROWSE_TL_TIME_SOURCE,
        _ids.BROWSE_TL_ROW_CSV_COL,
        _ids.BROWSE_TL_TIME_CSV_COL,
        _ids.BROWSE_TL_CSV_IMAGE_COL,
        _ids.BROWSE_TL_PATTERN_INPUT,
        _ids.BROWSE_TL_PATTERN_ADVANCED,
        _ids.BROWSE_TL_PATTERN_PREVIEW,
        _ids.BROWSE_TL_TILE_SIZE_MINUS,
        _ids.BROWSE_TL_TILE_SIZE_PLUS,
        _ids.BROWSE_TL_TILE_SIZE_READOUT,
        _ids.BROWSE_TL_NAV_UP,
        _ids.BROWSE_TL_NAV_DOWN,
        _ids.BROWSE_TL_NAV_LEFT,
        _ids.BROWSE_TL_NAV_RIGHT,
        _ids.BROWSE_TL_POSITION,
        _ids.BROWSE_TL_NUDGE,
        _ids.BROWSE_TL_GRID,
        _ids.BROWSE_TL_STORE_TILE_SIZE,
        _ids.BROWSE_TL_STORE_WARNINGS,
        _ids.BROWSE_TL_WARNINGS_ALERT,
        _ids.BROWSE_TL_POPOUT_MODAL,
        _ids.BROWSE_TL_POPOUT_TITLE,
        _ids.BROWSE_TL_POPOUT_OSD,
        _ids.BROWSE_TL_POPOUT_STORE,
        _ids.BROWSE_TL_POPOUT_EVENT,
        _ids.BROWSE_TL_POPOUT_APPROVED,
        _ids.BROWSE_TL_SOURCE_REVISION,
        _ids.BROWSE_TL_SESSION,
        _ids.BROWSE_TL_REVISION_CANDIDATE,
        _ids.BROWSE_TL_REVISION_AUTHORIZED,
    ]
    assert len(timeline_ids) == len(set(timeline_ids))  # all unique
    assert all(isinstance(i, str) and i for i in timeline_ids)
