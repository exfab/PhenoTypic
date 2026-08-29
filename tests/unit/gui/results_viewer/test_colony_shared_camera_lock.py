"""The Colony toolbar exposes one linked, bounded microscope-stage camera."""

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


def test_colony_toolbar_carries_the_complete_linked_camera() -> None:
    """Pan, reset, fit, zoom, and actual-pixel controls are all reachable."""
    toolbar = _build_toolbar()
    expected = {
        ids.COLONY_CAMERA_TOOLBAR_ID,
        ids.COLONY_CAMERA_PAN_UP,
        ids.COLONY_CAMERA_PAN_DOWN,
        ids.COLONY_CAMERA_PAN_LEFT,
        ids.COLONY_CAMERA_PAN_RIGHT,
        ids.COLONY_CAMERA_CENTER,
        ids.COLONY_CAMERA_FIT,
        ids.COLONY_CAMERA_ZOOM_OUT,
        ids.COLONY_CAMERA_ZOOM_IN,
        ids.COLONY_CAMERA_ONE_TO_ONE,
        ids.COLONY_CAMERA_ZOOM_READOUT,
        ids.COLONY_CAMERA_LINKED_STATUS,
    }
    missing = {
        component_id
        for component_id in expected
        if _find(toolbar, component_id) is None
    }
    assert not missing


def test_colony_grid_redraw_uses_an_unambiguous_label() -> None:
    """The grid redraw must not be confused with snapshot refresh."""
    toolbar = _build_toolbar()
    redraw = _find(toolbar, ids.COLONY_BTN_REFRESH_ID)

    assert redraw is not None
    assert redraw.children == "⟳ Redraw grid"


def test_camera_contract_is_visible_and_keyboard_focusable() -> None:
    """The shared-camera behavior is stated rather than hidden."""
    toolbar = _build_toolbar()
    camera = _find(toolbar, ids.COLONY_CAMERA_TOOLBAR_ID)
    status = _find(toolbar, ids.COLONY_CAMERA_LINKED_STATUS)
    readout = _find(toolbar, ids.COLONY_CAMERA_ZOOM_READOUT)

    assert camera is not None
    assert camera.tabIndex == 0
    assert getattr(camera, "aria-label") == "Linked colony image controls"
    assert status is not None
    assert "All tiles linked" in status.children
    assert readout is not None
    assert readout.children == "Fit"


def test_camera_buttons_have_action_specific_accessible_names() -> None:
    """Symbol-only controls remain intelligible to assistive technology."""
    toolbar = _build_toolbar()
    expected = {
        ids.COLONY_CAMERA_PAN_UP: "Pan colony crops up",
        ids.COLONY_CAMERA_PAN_DOWN: "Pan colony crops down",
        ids.COLONY_CAMERA_PAN_LEFT: "Pan colony crops left",
        ids.COLONY_CAMERA_PAN_RIGHT: "Pan colony crops right",
        ids.COLONY_CAMERA_CENTER: "Center colony crops",
        ids.COLONY_CAMERA_ZOOM_OUT: "Zoom colony crops out",
        ids.COLONY_CAMERA_ZOOM_IN: "Zoom colony crops in",
    }
    for component_id, aria_label in expected.items():
        control = _find(toolbar, component_id)
        assert control is not None
        assert getattr(control, "aria-label") == aria_label
