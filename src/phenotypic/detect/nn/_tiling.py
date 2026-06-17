"""Shared fixed-geometric tiling for GPU detectors (Spec 2b, Task 3).

Extracted from ``_sam3_detector.py`` so the instance detector (SAM3, IoU-NMS
merge) and the semantic detectors (INSID3, FSSDINO, union stitch) share one
tiling implementation. A ``GpuDetector`` runs before ``GridFinder`` and only
ever sees a raw ``input_layer`` array, so tiling is **grid-unaware**: fixed
~``tile_px`` crops with fractional overlap whose union always covers the image.

Instance detectors merge cross-tile duplicates by IoU-NMS (kept in
``_sam3_detector.py``); semantic detectors just OR the per-tile boolean masks
(:func:`stitch_semantic_tiles`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class _Tile:
    """One axis-aligned crop rectangle in full-image coordinates.

    Attributes:
        y0: Top row (inclusive).
        x0: Left column (inclusive).
        y1: Bottom row (exclusive).
        x1: Right column (exclusive).
    """

    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def h(self) -> int:
        """Tile height in pixels (``y1 - y0``)."""
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        """Tile width in pixels (``x1 - x0``)."""
        return self.x1 - self.x0


def _tile_starts(extent: int, tile_px: int, stride: int) -> list[int]:
    """Return tile start offsets along one axis, covering ``[0, extent)``.

    The final start is clamped so the last tile ends exactly at ``extent``
    (overlapping the previous tile rather than spilling past the edge).

    Args:
        extent: Axis length in pixels.
        tile_px: Nominal tile size along this axis.
        stride: Step between consecutive tile starts (``tile_px`` minus overlap).

    Returns:
        Sorted, de-duplicated list of start offsets.
    """
    if extent <= tile_px:
        return [0]
    starts: list[int] = []
    pos = 0
    last_start = extent - tile_px
    while pos < last_start:
        starts.append(pos)
        pos += stride
    starts.append(last_start)
    # De-dup while preserving order (the clamp can coincide with a step).
    seen: set[int] = set()
    unique: list[int] = []
    for s in starts:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _plan_tiles(
    shape: tuple[int, int], tile_px: int, overlap: float
) -> list[_Tile]:
    """Plan fixed ~``tile_px`` tiles with fractional ``overlap`` over an image.

    The union of the returned tiles always covers the full image; tiles never
    exceed ``tile_px`` on either axis and never spill past the image bounds.
    An image that already fits one tile yields a single full-image tile.

    Args:
        shape: ``(H, W)`` of the full image.
        tile_px: Nominal tile size in pixels.
        overlap: Fractional overlap between neighbouring tiles, in ``[0, 1)``.

    Returns:
        List of :class:`_Tile` rectangles in full-image coordinates.
    """
    h, w = int(shape[0]), int(shape[1])
    stride = max(1, int(round(tile_px * (1.0 - overlap))))
    tiles: list[_Tile] = []
    for y0 in _tile_starts(h, tile_px, stride):
        for x0 in _tile_starts(w, tile_px, stride):
            y1 = min(y0 + tile_px, h)
            x1 = min(x0 + tile_px, w)
            tiles.append(_Tile(y0, x0, y1, x1))
    return tiles


def stitch_semantic_tiles(
    tiles: List[_Tile],
    tile_masks: List["np.ndarray"],
    out_shape: tuple[int, int],
) -> "np.ndarray":
    """Union per-tile boolean masks back into one full-image ``objmask``.

    Semantic output has no instance identity, so overlaps simply OR — no NMS is
    needed (contrast the instance path's IoU-NMS in ``_sam3_detector.py``).

    Args:
        tiles: Crop rectangles in full-image coordinates (from
            :func:`_plan_tiles`), aligned with *tile_masks*.
        tile_masks: Per-tile boolean masks, each ``(tile.h, tile.w)``.
        out_shape: ``(H, W)`` of the full image.

    Returns:
        A full-image boolean ``objmask`` (the union of the tile masks).

    Raises:
        ValueError: If *tiles* and *tile_masks* differ in length.
    """
    import numpy as np

    if len(tiles) != len(tile_masks):
        raise ValueError(
            f"stitch_semantic_tiles: {len(tiles)} tiles vs "
            f"{len(tile_masks)} masks"
        )
    full = np.zeros(out_shape, dtype=bool)
    for tile, mask in zip(tiles, tile_masks):
        full[tile.y0:tile.y1, tile.x0:tile.x1] |= np.asarray(mask, dtype=bool)
    return full
