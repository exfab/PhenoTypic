"""The "Shared camera" lock is a visible affordance, not hidden behaviour.

Spec section 6.2 asks for the lock in the chrome now, while it has exactly
one state, so the eventual unlock-one-cell mode has somewhere to live.
Retrofitting an affordance onto a mode that already shipped as invisible
behaviour is the expensive order, and a user who cannot see that the cells
share a camera has no way to guess why zooming one moved all of them.

The lock reads ON and disabled. That pair is the point: ON because
``setGridViews`` gives every cell view one shared ``zoom`` and nothing else
is reachable, disabled because there is no second state to select yet.
Shipping it enabled would offer a control that does nothing.
"""

from __future__ import annotations

from typing import Iterator

from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer.colony_view._layout import _build_toolbar


def _walk(component: object) -> Iterator[object]:
    """Yield ``component`` and every descendant component, depth-first."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def _find(component: object, component_id: str) -> object | None:
    for node in _walk(component):
        if getattr(node, "id", None) == component_id:
            return node
    return None


def test_the_colony_toolbar_carries_a_shared_camera_lock() -> None:
    toolbar = _build_toolbar()
    lock = _find(toolbar, ids.COLONY_SHARED_CAMERA_TOGGLE_ID)
    assert lock is not None, (
        "no shared-camera lock in the colony toolbar -- the grid's one "
        "shared zoom would be invisible behaviour"
    )
    assert lock.label == "Shared camera"


def test_the_lock_reads_on_and_is_not_selectable_yet() -> None:
    lock = _find(_build_toolbar(), ids.COLONY_SHARED_CAMERA_TOGGLE_ID)
    assert lock is not None
    assert lock.value is True, (
        "the lock must read ON: every cell view merges its own target over "
        "ONE shared zoom, so the locked state is the only one that exists"
    )
    assert lock.disabled is True, (
        "the lock must be disabled while unlock-one-cell does not exist -- "
        "an enabled control that changes nothing is worse than none"
    )
