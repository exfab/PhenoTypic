from phenotypic.gui.browse import _ids


def test_ids_are_unique_nonempty_strings():
    values = [
        _ids.BROWSE_DATASET_ROW,
        _ids.BROWSE_DATASET_PICKER,
        _ids.BROWSE_IMAGE_PICKER,
        _ids.BROWSE_PREV_BTN,
        _ids.BROWSE_NEXT_BTN,
        _ids.BROWSE_POSITION,
        _ids.BROWSE_NAV_EVENT_STORE,
        _ids.BROWSE_KEEP_POSITION,
        _ids.BROWSE_OSD_DIV,
        _ids.BROWSE_PREVIEW_IMG,
        _ids.BROWSE_OSD_LOADING,
        _ids.BROWSE_LOADING_TEXT,
        _ids.BROWSE_FILMSTRIP,
        _ids.BROWSE_PREPARE_BTN,
        _ids.BROWSE_STOP_PREPARE_BTN,
        _ids.BROWSE_CLEAR_CACHE_BTN,
        _ids.BROWSE_PREPARATION_STATUS,
        _ids.BROWSE_PREPARATION_PROGRESS,
        _ids.BROWSE_CACHE_USAGE,
        _ids.BROWSE_BACKEND_DETAILS,
        _ids.BROWSE_PREPARATION_STATUS_STORE,
        _ids.BROWSE_PREPARATION_POLL,
        _ids.BROWSE_CURRENT_IMAGE_STORE,
        _ids.BROWSE_DATASETS_STORE,
        _ids.BROWSE_OSD_SYNC,
        _ids.BROWSE_META_IMAGE_NAME,
        _ids.BROWSE_META_DIMS,
        _ids.BROWSE_META_SIZE,
        _ids.BROWSE_META_CAPTURED,
        _ids.BROWSE_META_CAMERA,
        _ids.BROWSE_EMPTY_HINT,
        _ids.BROWSE_SINGLE_BODY,
    ]
    assert all(isinstance(v, str) and v for v in values)
    assert len(set(values)) == len(values)
