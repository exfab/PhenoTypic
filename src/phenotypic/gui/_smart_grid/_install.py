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

    def _visible_indices():
        v = viewer_ref()
        if v is None:
            return []
        return [i for i, layer in enumerate(v.layers) if layer.visible]

    def patched_actual_shape(nlayers=1):
        if not grid.enabled:
            return (1, 1)
        n_visible = len(_visible_indices())
        if n_visible == 0:
            return (1, 1)
        return _orig_actual_shape(grid, n_visible)

    def patched_position(index, nlayers):
        if not grid.enabled:
            return (0, 0)
        vis = _visible_indices()
        if index not in vis:
            return (-1, -1)
        vis_idx = vis.index(index)
        return _orig_position(grid, vis_idx, len(vis))

    # Shadow class methods on instance
    grid.__dict__["actual_shape"] = patched_actual_shape
    grid.__dict__["position"] = patched_position

    # Connect visibility events to trigger grid rebuild
    try:
        canvas = viewer.window._qt_viewer.canvas
    except AttributeError:
        logger.warning("Cannot access napari canvas; smart grid features limited")
        canvas = None

    # Wrap _update_scenegraph to add grid labels after each rebuild
    if canvas is not None:
        from phenotypic.gui._smart_grid._grid_labels import add_grid_labels

        _orig_update_scenegraph = canvas._update_scenegraph

        def _smart_update_scenegraph(event=None):
            _orig_update_scenegraph(event)
            v = viewer_ref()
            if v is not None:
                try:
                    add_grid_labels(canvas, v)
                except Exception:
                    logger.debug("Grid label update failed", exc_info=True)

        canvas._update_scenegraph = _smart_update_scenegraph

    def _on_visibility_change(event=None):
        if grid.enabled and canvas is not None:
            canvas._update_scenegraph()

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
