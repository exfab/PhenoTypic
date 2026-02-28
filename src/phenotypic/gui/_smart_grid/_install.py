"""Visibility-aware napari grid view."""
from __future__ import annotations
import weakref
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)
_INSTALLED_FLAG = "_phenotypic_smart_grid"


def install_smart_grid(viewer: napari.Viewer) -> None:
    """Patch viewer.grid to only allocate cells for visible layers."""
    # Idempotency
    if getattr(viewer, _INSTALLED_FLAG, False):
        return

    grid = viewer.grid
    viewer_ref = weakref.ref(viewer)

    # Store original class methods (unbound)
    _cls = type(grid)
    _orig_actual_shape = _cls.actual_shape
    _orig_position = _cls.position

    # --- Cached visibility map (rebuilt once per scenegraph update) ---
    _vis_map: dict[int, int] = {}  # {layer_index: visible_position}

    def _rebuild_vis_map():
        nonlocal _vis_map
        v = viewer_ref()
        if v is None:
            _vis_map = {}
            return
        _vis_map = {
            i: pos for pos, (i, _layer) in enumerate(
                (i, layer) for i, layer in enumerate(v.layers) if layer.visible
            )
        }

    def patched_actual_shape(nlayers=1):
        if not grid.enabled:
            return (1, 1)
        _rebuild_vis_map()
        if not _vis_map:
            return (1, 1)
        return _orig_actual_shape(grid, len(_vis_map))

    def patched_position(index, nlayers):
        if not grid.enabled:
            return (0, 0)
        vis_pos = _vis_map.get(index)
        if vis_pos is None:
            return (-1, -1)
        return _orig_position(grid, vis_pos, len(_vis_map))

    # Shadow class methods on instance
    grid.__dict__["actual_shape"] = patched_actual_shape
    grid.__dict__["position"] = patched_position

    # --- Deferred canvas access and scenegraph wrapping ---
    _canvas = None
    _scenegraph_wrapped = False

    def _get_canvas():
        nonlocal _canvas
        if _canvas is None:
            try:
                _canvas = viewer_ref().window._qt_viewer.canvas
            except (AttributeError, TypeError):
                pass
        return _canvas

    def _ensure_scenegraph_wrapped():
        nonlocal _scenegraph_wrapped
        if _scenegraph_wrapped:
            return
        c = _get_canvas()
        if c is None:
            return
        _scenegraph_wrapped = True

        from phenotypic.gui._smart_grid._grid_labels import add_grid_labels

        _orig_update_scenegraph = c._update_scenegraph

        def _smart_update_scenegraph(event=None):
            _orig_update_scenegraph(event)
            v = viewer_ref()
            if v is not None:
                try:
                    add_grid_labels(c, v)
                except Exception:
                    logger.debug("Grid label update failed", exc_info=True)

        c._update_scenegraph = _smart_update_scenegraph

    # Connect visibility events to trigger grid rebuild
    def _on_visibility_change(event=None):
        _ensure_scenegraph_wrapped()
        c = _get_canvas()
        if grid.enabled and c is not None:
            c._update_scenegraph()

    def _connect_layer(layer):
        layer.events.visible.connect(_on_visibility_change)
        layer.events.name.connect(_on_visibility_change)  # Name change → relabel

    def _on_layer_inserted(event):
        _connect_layer(event.value)

    def _on_layer_removed(event):
        layer = event.value
        layer.events.visible.disconnect(_on_visibility_change)
        layer.events.name.disconnect(_on_visibility_change)
        _on_visibility_change()  # Rebuild grid after removal

    # Connect existing layers
    for layer in viewer.layers:
        _connect_layer(layer)

    # Connect future layers
    viewer.layers.events.inserted.connect(_on_layer_inserted)
    viewer.layers.events.removed.connect(_on_layer_removed)

    # Mark as installed
    viewer.__dict__[_INSTALLED_FLAG] = True
