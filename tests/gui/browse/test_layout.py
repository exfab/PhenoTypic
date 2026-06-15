from phenotypic.gui.browse import _ids as ids
from phenotypic.gui.browse._layout import build_browse_layout


def _ids_in_tree(node, found):
    cid = getattr(node, "id", None)
    if cid:
        found.add(cid)
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for c in children:
            _ids_in_tree(c, found)
    elif children is not None:
        _ids_in_tree(children, found)


def test_layout_contains_core_ids():
    found: set[str] = set()
    _ids_in_tree(build_browse_layout(), found)
    for required in (
        ids.BROWSE_DATASET_PICKER,
        ids.BROWSE_IMAGE_PICKER,
        ids.BROWSE_PREV_BTN,
        ids.BROWSE_NEXT_BTN,
        ids.BROWSE_OSD_DIV,
        ids.BROWSE_OSD_LOADING,
        ids.BROWSE_LOADING_TEXT,
        ids.BROWSE_CURRENT_IMAGE_STORE,
        ids.BROWSE_DATASETS_STORE,
        ids.BROWSE_OSD_SYNC,
        ids.BROWSE_META_DIMS,
        ids.BROWSE_EMPTY_HINT,
    ):
        assert required in found
