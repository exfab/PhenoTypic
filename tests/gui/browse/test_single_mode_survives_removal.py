"""Browse keeps Single mode and loses the mode toggle.

The point of this file is the *second* test: a removal that hides the surviving
mode instead of unhiding it is the failure mode with the widest blast radius,
and it is invisible in a unit test that only checks for absence.
"""

from phenotypic.gui.browse import _ids as ids
from phenotypic.gui.browse._layout import build_browse_layout


def _ids_in(component) -> set[str]:
    found: set[str] = set()

    def walk(node) -> None:
        node_id = getattr(node, "id", None)
        if isinstance(node_id, str):
            found.add(node_id)
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                walk(child)
        elif children is not None:
            walk(children)

    walk(component)
    return found


def test_browse_layout_has_no_view_mode_toggle():
    present = _ids_in(build_browse_layout())
    assert ids.BROWSE_SINGLE_BODY in present
    assert not any(name.startswith("browse-tl-") for name in present)
    assert "browse-view-mode-toggle" not in present
    assert "browse-timeline-body" not in present


def test_browse_single_body_is_unconditional():
    """Single must not be hidden by a leftover ``display: none``."""

    def find(node, target):
        if getattr(node, "id", None) == target:
            return node
        children = getattr(node, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                hit = find(child, target)
                if hit is not None:
                    return hit
        elif children is not None:
            return find(children, target)
        return None

    body = find(build_browse_layout(), ids.BROWSE_SINGLE_BODY)
    assert body is not None
    assert (getattr(body, "style", None) or {}).get("display") != "none"
