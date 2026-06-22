"""Preview modal mounts in the layout with the expected sub-components."""
from pathlib import Path

from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._layout import build_node_preview_modal


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for c in children:
        yield from _walk(c)


def test_modal_has_blocking_props_and_children():
    modal = build_node_preview_modal()
    assert modal.id == ids.MODAL_NODE_PREVIEW
    assert modal.backdrop == "static"
    assert modal.is_open is False
    found = {getattr(n, "id", None) for n in _walk(modal)}
    assert ids.PREVIEW_OSD_DIV in found
    assert ids.PREVIEW_LAYER_RADIO in found
    assert ids.PREVIEW_CAPTION in found
    assert ids.PREVIEW_DZI_URL_STORE in found


def test_preview_js_asset_exists():
    js = Path("src/phenotypic/gui/builder/assets/preview.js")
    assert js.exists()
    text = js.read_text()
    assert "__phenotypicNodePreview" in text
    assert "mountViewer" in text and "disposeViewer" in text
