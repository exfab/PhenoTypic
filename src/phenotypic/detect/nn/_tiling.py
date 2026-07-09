"""Shared fixed-geometric tiling for GPU detectors (Spec 2b, Task 3).

Extracted from ``_sam3_detector.py`` so the instance detector (SAM3) and the
semantic detectors (INSID3, FSSDINO, union stitch) share one tiling
implementation. A ``GpuDetector`` runs before ``GridFinder`` and only ever sees
a raw ``input_layer`` array, so tiling is **grid-unaware**: fixed ~``tile_px``
crops with fractional overlap whose union always covers the image.

This module owns the whole tiling policy, including both cross-tile merges:

* **Instance** detectors merge with :func:`assign_by_centroid_core`, which keeps
  each instance in exactly the one tile whose *core* (Voronoi cell of the tile
  centres) contains its centroid. Duplicates and fragments are impossible by
  construction. :func:`_merge_tiles_iou_nms` is the older IoU-NMS merge, kept
  for the single-tile relabel path and for back-compat.
* **Semantic** detectors just OR the per-tile boolean masks
  (:func:`stitch_semantic_tiles`) — no instance identity, so no merge policy.
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

    Semantic output has no instance identity, so overlaps simply OR — no merge
    policy is needed (contrast the instance path's
    :func:`assign_by_centroid_core`).

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


def _iou(mask_a: "np.ndarray", mask_b: "np.ndarray") -> float:
    """Intersection-over-union of two boolean masks (0.0 if both empty)."""
    inter = int((mask_a & mask_b).sum())
    if inter == 0:
        return 0.0
    union = int((mask_a | mask_b).sum())
    return inter / union if union else 0.0


def _merge_tiles_iou_nms(
    objmaps: List["np.ndarray"], iou_thresh: float
) -> "np.ndarray":
    """Greedy IoU-NMS merge of per-tile objmaps into one contiguous objmap.

    Each input objmap is already offset into full-image coordinates (same
    shape). Instances are collected across all tiles, sorted largest-first,
    and greedily kept unless they overlap an already-kept instance by more than
    ``iou_thresh`` (a cross-tile duplicate). Survivors are relabelled ``1..N``
    largest-first so smaller colonies overwrite at overlaps, preserving
    small-colony identity (mirrors ``Sam2Detector``'s painting order).

    This only suppresses *duplicates* — instances a neighbouring tile saw
    whole. It cannot suppress a **fragment**: IoU between a whole colony and
    the cleaved copy a neighbouring tile saw equals the fragment's area
    fraction, so every fragment covering ``<= iou_thresh`` of its parent
    survives. Prefer :func:`assign_by_centroid_core` for tiled instance
    detection; this remains for the single-tile relabel path.

    Args:
        objmaps: Per-tile uint16 objmaps, each in full-image coordinates.
        iou_thresh: IoU above which two instances are treated as duplicates.

    Returns:
        A single uint16 objmap with contiguous labels ``1..N``.
    """
    import numpy as np

    if not objmaps:
        raise ValueError("_merge_tiles_iou_nms requires at least one objmap")
    shape = objmaps[0].shape

    masks: list[np.ndarray] = []
    for objmap in objmaps:
        for label in np.unique(objmap):
            if label == 0:
                continue
            masks.append(objmap == label)
    if not masks:
        return np.zeros(shape, dtype=np.uint16)

    masks.sort(key=lambda m: int(m.sum()), reverse=True)
    kept: list[np.ndarray] = []
    for cand in masks:
        if any(_iou(cand, k) > iou_thresh for k in kept):
            continue
        kept.append(cand)

    max_labels = int(np.iinfo(np.uint16).max)
    if len(kept) > max_labels:
        import warnings

        warnings.warn(
            f"Tiled merge kept {len(kept)} instances, exceeding uint16 range. "
            f"Only the first {max_labels} (largest) will be labeled.",
            UserWarning,
            stacklevel=2,
        )
        kept = kept[:max_labels]

    objmap = np.zeros(shape, dtype=np.uint16)
    for idx, mask in enumerate(kept, start=1):
        objmap[mask] = idx
    return objmap


def tile_overlap_px(tiles: List[_Tile]) -> int:
    """Smallest overlap in pixels between any two overlapping tiles.

    Zero when a single tile covers the image. This is the bound that decides
    whether a colony can be lost: an instance wider than the overlap is cleaved
    in every tile that contains it.

    Args:
        tiles: Crop rectangles from :func:`_plan_tiles`.

    Returns:
        Minimum positive pairwise overlap along either axis, or ``0``.
    """
    if len(tiles) < 2:
        return 0
    best: int | None = None
    for i, a in enumerate(tiles):
        for b in tiles[i + 1:]:
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            if oy > 0 and ox > 0:  # genuinely overlapping, not merely abutting
                cand = min(oy, ox)
                best = cand if best is None else min(best, cand)
    return int(best) if best is not None else 0


def owning_tile_index(tiles: List[_Tile], centroid_yx: tuple[float, float]) -> int:
    """Index of the tile whose *core* contains ``centroid_yx``.

    A tile's core is the region closer to its centre than to any other tile's
    centre — a Voronoi partition of the tile centres, intersected with the tile.
    Since :func:`_plan_tiles` guarantees the tiles cover the image, every point
    lies in at least one tile, and the nearest-centre rule picks exactly one.
    Border tiles' cores therefore reach the image edge with no gap.

    This is what makes cross-tile duplicates impossible: a colony fully inside
    one tile is claimed by whichever core holds its true centroid; the same
    colony's *fragment* in a neighbouring tile has a centroid pushed within
    ``d/2`` of that tile's edge, while the core begins ``overlap_px / 2`` inside
    it — so when ``overlap_px >= d`` the fragment is never claimed.

    Args:
        tiles: Crop rectangles from :func:`_plan_tiles`.
        centroid_yx: ``(y, x)`` in full-image coordinates.

    Returns:
        Index into *tiles*.
    """
    cy, cx = float(centroid_yx[0]), float(centroid_yx[1])
    best_i = 0
    best_d: float | None = None
    for i, t in enumerate(tiles):
        if not (t.y0 <= cy < t.y1 and t.x0 <= cx < t.x1):
            continue
        ty = (t.y0 + t.y1) / 2.0
        tx = (t.x0 + t.x1) / 2.0
        d = (cy - ty) ** 2 + (cx - tx) ** 2
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i


def assign_by_centroid_core(
    tiles: List[_Tile],
    tile_objmaps: List["np.ndarray"],
    out_shape: tuple[int, int],
) -> "np.ndarray":
    """Merge tile-local instance maps by centroid-in-core assignment.

    Each instance is kept by exactly the one tile whose core contains its
    centroid (:func:`owning_tile_index`); every other copy is discarded. No NMS,
    no edge tolerance, no duplicates by construction. Fragments are dropped
    because nobody claims them.

    Contrast :func:`_merge_tiles_iou_nms`, whose IoU between a whole colony and
    its cross-tile fragment equals the fragment's area fraction ``f`` — so every
    fragment with ``f <= iou_thresh`` survives and, being painted later in the
    largest-first order, overwrites the colony it came from.

    The alternative fix — rejecting instances near a crop edge, as SAM2 does —
    is not viable here. Edge rejection needs ``overlap_px >= d + 2 * atol`` for
    a colony of diameter ``d``; violate it and the colony is within ``atol`` of
    an edge in *every* tile containing it, so it is not fragmented but silently
    **deleted**. SAM2 tolerates that because its crop pyramid always runs a
    full-image layer and its ``1 / box_area`` NMS outvotes the coarse copy;
    uniform tiles have neither defense.

    Survivors are relabelled ``1..N`` largest-first, matching
    :func:`_merge_tiles_iou_nms`.

    Args:
        tiles: Crop rectangles in full-image coordinates.
        tile_objmaps: Per-tile uint16 objmaps, each ``(tile.h, tile.w)``,
            **tile-local** (not offset into the full image).
        out_shape: ``(H, W)`` of the full image.

    Returns:
        A full-image uint16 objmap with contiguous labels ``1..N``.

    Raises:
        ValueError: If *tiles* and *tile_objmaps* differ in length.

    Warns:
        UserWarning: When the largest retained instance is wider than
            :func:`tile_overlap_px` — the condition under which a colony can be
            cleaved in every tile and lost.
    """
    import warnings

    import numpy as np

    if len(tiles) != len(tile_objmaps):
        raise ValueError(
            f"assign_by_centroid_core: {len(tiles)} tiles vs "
            f"{len(tile_objmaps)} objmaps"
        )

    # Survivors are recorded as (area, diameter, tile_index, label) — never as
    # full-image masks. Materializing one (H, W) boolean per instance costs
    # ~12 MB on a 4000x3000 plate, and every survivor is alive at once for the
    # largest-first sort: ~12 GB for a 1000-colony plate (measured). Painting
    # instead from each survivor's tile-local slice keeps the working set at one
    # tile plus the output objmap.
    kept: list[tuple[int, int, int, int]] = []
    for i, (tile, om) in enumerate(zip(tiles, tile_objmaps)):
        om = np.asarray(om)
        for label in np.unique(om):
            if label == 0:
                continue
            ys, xs = np.nonzero(om == label)
            cy = ys.mean() + tile.y0
            cx = xs.mean() + tile.x0
            if owning_tile_index(tiles, (cy, cx)) != i:
                continue
            diameter = max(
                int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1
            )
            kept.append((int(ys.size), diameter, i, int(label)))

    if not kept:
        return np.zeros(out_shape, dtype=np.uint16)

    kept.sort(key=lambda rec: rec[0], reverse=True)

    overlap = tile_overlap_px(tiles)
    if overlap:
        d = kept[0][1]
        if d > overlap:
            warnings.warn(
                f"Largest instance is {d} px across but tiles overlap by only "
                f"{overlap} px; an instance wider than the overlap can be "
                f"cleaved in every tile and lost. Raise tile_overlap.",
                UserWarning,
                stacklevel=2,
            )

    max_labels = int(np.iinfo(np.uint16).max)
    if len(kept) > max_labels:
        warnings.warn(
            f"Tiled merge kept {len(kept)} instances, exceeding uint16 range. "
            f"Only the first {max_labels} (largest) will be labeled.",
            UserWarning,
            stacklevel=2,
        )
        kept = kept[:max_labels]

    # Paint largest-first, so smaller colonies overwrite at overlaps and keep
    # their identity. Each survivor is painted through a view onto its own tile
    # rectangle; no full-image mask is ever built.
    objmap = np.zeros(out_shape, dtype=np.uint16)
    for idx, (_area, _diameter, i, label) in enumerate(kept, start=1):
        tile = tiles[i]
        om = np.asarray(tile_objmaps[i])
        objmap[tile.y0:tile.y1, tile.x0:tile.x1][om == label] = idx
    return objmap
