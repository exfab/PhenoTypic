"""Text labels for napari grid viewboxes."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

_ELLIPSIS = "\u2026"  # Unicode ellipsis character


def truncate_middle(text: str, max_len: int = 20) -> str:
    """Truncate text with ellipsis in the middle."""
    if len(text) <= max_len:
        return text
    keep = max_len - 1  # 1 char for ellipsis
    front = keep // 2
    back = keep - front
    return f"{text[:front]}{_ELLIPSIS}{text[-back:]}"


def add_grid_labels(canvas, viewer: napari.Viewer) -> None:
    """Add layer name labels to each grid viewbox."""
    if not viewer.grid.enabled or not canvas.grid_views:
        return

    from vispy.visuals.transforms import STTransform
    from vispy.scene.visuals import Text

    padding = 6

    for viewbox, (_, layer_indices) in zip(
        canvas.grid_views,
        viewer.grid.iter_viewboxes(len(viewer.layers)),
        strict=False,
    ):
        if not layer_indices:
            continue
        names = [viewer.layers[i].name for i in layer_indices]
        label = truncate_middle(", ".join(names), 20)

        text_node = Text(
            text=label,
            pos=(0, 0),
            color="white",
            font_size=10,
            parent=viewbox,
        )
        text_node.anchors = ("left", "top")
        text_node.transform = STTransform(
            translate=(padding, padding, 0, 0),
        )
