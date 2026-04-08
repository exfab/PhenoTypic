"""Vispy visual cloning for grid overlay mode."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

logger = logging.getLogger(__name__)


def is_overlay_layer(layer) -> bool:
    """Check if a layer should be treated as an overlay (Labels, Points, or Shapes)."""
    from napari.layers import Labels, Points, Shapes

    return isinstance(layer, (Labels, Points, Shapes))


def create_overlay_clones(canvas, viewer: napari.Viewer) -> list:
    """Create vispy visual clones for overlay layers in each image viewbox."""
    from napari._vispy.utils.visual import create_vispy_layer

    clones = []
    overlay_layers = [
        layer for layer in viewer.layers
        if layer.visible and is_overlay_layer(layer)
    ]

    if not overlay_layers or not canvas.grid_views:
        return clones

    draw_order = len(viewer.layers) + 10

    for viewbox in canvas.grid_views:
        for layer in overlay_layers:
            clone = create_vispy_layer(layer)
            clone.node.parent = viewbox.scene
            clone.node.order = draw_order
            clones.append(clone)

    return clones


def cleanup_clones(clones: list, canvas=None) -> None:
    """Close all vispy clones and free GPU resources."""
    had_clones = len(clones) > 0
    for clone in clones:
        try:
            clone.close()
        except Exception:
            logger.debug("Clone cleanup failed", exc_info=True)
    clones.clear()
    if had_clones and canvas is not None:
        try:
            canvas._scene_canvas.context.finish()
        except Exception:
            pass
