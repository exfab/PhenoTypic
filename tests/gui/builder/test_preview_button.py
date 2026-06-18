"""Image-producing node cards carry a preview action button; measure nodes don't."""
from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._linear_layout import _preview_button


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for c in children:
        yield from _walk(c)


def test_preview_button_has_preview_action_id():
    btn = _preview_button(scope_path=[], block_id="b" * 32)
    assert btn.id == ids.linear_node_action_id(
        action="preview", scope_path=[], block_id="b" * 32
    )


def test_preview_action_id_shape():
    pid = ids.linear_node_action_id(action="preview", scope_path=[], block_id="b" * 32)
    assert pid["type"] == ids.LINEAR_NODE_ACTION
    assert pid["action"] == "preview"
    assert pid["block_id"] == "b" * 32
