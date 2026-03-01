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

    # --- Cached visibility map and overlay state ---
    _vis_map: dict[int, int] = {}  # {layer_index: visible_position}
    _overlay_enabled = False
    _overlay_clones: list = []

    def _get_overlay_enabled():
        return _overlay_enabled

    def _set_overlay_enabled(enabled):
        nonlocal _overlay_enabled
        _overlay_enabled = enabled
        _ensure_scenegraph_wrapped()
        c = _get_canvas()
        if c is not None:
            c._update_scenegraph()

    def _rebuild_vis_map():
        nonlocal _vis_map
        v = viewer_ref()
        if v is None:
            _vis_map = {}
            return
        from phenotypic.gui._smart_grid._overlay_visuals import is_overlay_layer

        _vis_map = {
            i: pos for pos, (i, _layer) in enumerate(
                (i, layer) for i, layer in enumerate(v.layers)
                if layer.visible
                and not (_overlay_enabled and is_overlay_layer(layer))
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
        _rebuild_vis_map()
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
        from phenotypic.gui._smart_grid._overlay_visuals import (
            cleanup_clones,
            create_overlay_clones,
            is_overlay_layer,
        )

        _orig_update_scenegraph = c._update_scenegraph

        def _smart_update_scenegraph(event=None):
            cleanup_clones(_overlay_clones, c)
            _orig_update_scenegraph(event)
            v = viewer_ref()
            if v is not None:
                if _overlay_enabled and grid.enabled:
                    # Detach original overlay visuals — they'd otherwise remain
                    # parented to the canvas-wide self.view (order=100, drawn on
                    # top) because patched_position returns (-1,-1) and napari's
                    # _setup_layer_views_in_grid never re-parents them.
                    for layer in v.layers:
                        if is_overlay_layer(layer) and layer in c.layer_to_visual:
                            c.layer_to_visual[layer].node.parent = None
                    _overlay_clones[:] = create_overlay_clones(c, v)
                try:
                    add_grid_labels(c, v)
                except Exception:
                    logger.debug("Grid label update failed", exc_info=True)

        c._update_scenegraph = _smart_update_scenegraph

        from phenotypic.gui._smart_grid._grid_popup import patch_grid_popup

        v = viewer_ref()
        if v is not None:
            patch_grid_popup(v, _get_overlay_enabled, _set_overlay_enabled)

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
        _ensure_scenegraph_wrapped()

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

    # Eagerly wrap scenegraph if canvas is already available
    _ensure_scenegraph_wrapped()

    # Mark as installed
    viewer.__dict__[_INSTALLED_FLAG] = True
