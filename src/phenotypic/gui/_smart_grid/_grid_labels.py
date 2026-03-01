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
    from vispy.scene.visuals import Rectangle, Text

    padding = 6
    font_size = 10

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
            font_size=font_size,
            parent=viewbox,
        )
        text_node.anchors = ("left", "top")
        text_node.transform = STTransform(
            translate=(padding, padding * 3, 0, 0),
        )

        # Semi-transparent background sized from font metrics
        # (vispy Text.bounds() returns (0,0) — it uses pos, not glyphs)
        # ~0.75 em avg character width for proportional font (OpenSans)
        tw = len(label) * font_size * 0.85
        th = font_size * 1.8
        inner_pad = 4
        bg = Rectangle(
            center=(tw / 2 + inner_pad, -th / 2),
            width=tw + inner_pad * 2,
            height=th + inner_pad * 2,
            color=(0, 0, 0, 0.3),
            border_color=None,
            parent=viewbox,
        )
        bg.transform = STTransform(
            translate=(padding, padding * 3, 0, 0),
        )
        bg.order = 0       # drawn first (behind)
        text_node.order = 1  # drawn second (in front)
