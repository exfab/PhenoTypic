"""Shared tile-imaging primitives for every GUI tab that renders colony crops.

Two GUI surfaces render fixed-size, centroid-aligned crops of the
overlay PNGs the CLI writes at
``<root>/deliverables/overlays/<dataset>/<stem>.png``:

* the **colony-view** tab — a 2D axis-header grid (see
  :mod:`phenotypic.gui.results_viewer.colony_view._grid`), and
* the **QC review** tab — a flat, faceted gallery (Phase D).

Rather than duplicating the cropper, the path-traversal guard, the
crop-serving Flask route, and the per-tile chrome across both, this
module owns the single implementation and both consumers import it:

* :func:`crop_overlay` / :func:`_load_overlay_rgb` — slice a square crop
  out of an overlay PNG, padded so the result is always ``size`` x
  ``size``, with a small LRU cache over the decoded source.
* :func:`is_safe_path_component` — the URL-capture path-traversal guard.
* :func:`register_crop_route` — a factory that mounts the centered-crop
  Flask route under any URL segment (``crops`` for the colony view, a QC
  segment later), so both tabs serve crops the same way.
* :func:`build_tile_cell` — the per-tile chrome (img/placeholder,
  multi-select checkbox, remove/restore button) — **the dedup boundary**.
* :func:`build_tile_grid` — a flat gallery of tiles plus the row-major
  key list both tabs reuse for shift+click range selection.
* :func:`expand_range` — resolve a shift+click slice over a row-major key
  list (re-exported from ``colony_view/_grid.py`` for back-compat).
"""

from __future__ import annotations

import functools
import io
import logging
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

import dash
import numpy as np
from dash import html
from dash.development.base_component import Component
from flask import Blueprint, Response
from PIL import Image as PILImage
from werkzeug.exceptions import BadRequest, NotFound

from phenotypic.gui._design import TILE_DIM_RGB

if TYPE_CHECKING:
    from phenotypic.gui.results_viewer._output_root import OutputRoot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Overlay cropping
# ---------------------------------------------------------------------------

#: Number of distinct overlay PNGs to keep decoded in memory. A typical
#: grid pulls from 1–8 plates per render, so this comfortably covers a
#: full grid without holding more than a few hundred MB of pixel data.
_OVERLAY_CACHE_SIZE = 8


@functools.lru_cache(maxsize=_OVERLAY_CACHE_SIZE)
def _load_overlay_rgb(path: str, mtime_ns: int) -> PILImage.Image:
    """Decode an overlay PNG to RGB and cache the result.

    Args:
        path: Absolute path to the overlay PNG, as a string so the
            cache key is hashable.
        mtime_ns: ``st_mtime_ns`` at lookup time. Including it in the
            cache key invalidates the cached frame when the overlay is
            regenerated under a running viewer.
    """
    del mtime_ns  # Cache-key only.
    with PILImage.open(path) as img:
        return img.convert("RGB")


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp ``value`` to the inclusive ``[low, high]`` integer range."""
    return max(low, min(high, value))


def _dim_outside_bbox(
    canvas: np.ndarray,
    keep: tuple[int, int, int, int],
    *,
    alpha: float,
    bg: tuple[int, int, int] = TILE_DIM_RGB,
) -> np.ndarray:
    """Blend the pixels outside the keep-rectangle toward ``bg`` by ``alpha``.

    The tile-spotlight effect: leave the target colony's bbox rectangle at
    full opacity and fade everything around it toward black so the measured
    colony is unambiguous on a crowded plate. The keep-rectangle is an
    axis-aligned hard boundary — inside is untouched, outside is dimmed at
    full strength, with a crisp edge.

    Args:
        canvas: The assembled crop, an ``(size, size, 3)`` uint8 array.
        keep: ``(top, left, bottom, right)`` of the keep-rectangle in
            canvas pixels. A degenerate rectangle (``bottom <= top`` or
            ``right <= left``) keeps nothing, so the whole canvas dims.
        alpha: Dim strength in ``[0, 1]``. ``alpha <= 0.0`` is a no-op and
            returns ``canvas`` unchanged (the regression-safe fast path).
        bg: RGB colour the dimmed pixels blend toward. Defaults to
            :data:`phenotypic.gui._design.TILE_DIM_RGB` (black).

    Returns:
        A new ``(size, size, 3)`` uint8 array with the surroundings dimmed,
        or ``canvas`` itself when ``alpha <= 0.0``.
    """
    if alpha <= 0.0:
        return canvas
    out: np.ndarray = canvas.astype(np.float32)
    mask = np.ones(canvas.shape[:2], dtype=bool)
    top, left, bottom, right = keep
    if bottom > top and right > left:
        mask[top:bottom, left:right] = False  # keep-rect = not dimmed
    bgv = np.asarray(bg, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + bgv * alpha
    return out.astype(np.uint8)


def crop_overlay(
    png_path: Path,
    center_rr: float,
    center_cc: float,
    size: int,
    pad_value: tuple[int, int, int] = (0, 0, 0),
    *,
    dim_alpha: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
) -> bytes:
    """Crop an overlay PNG to a fixed ``size`` x ``size`` window centered on a colony.

    Computes ``(top, left) = (round(center_rr) - size // 2,
    round(center_cc) - size // 2)``, clamps the requested window to the
    image bounds, and pastes the clamped region onto a freshly-allocated
    canvas filled with ``pad_value``. The output therefore always has
    the exact requested dimensions, even when the colony sits near an
    edge of the source image.

    Args:
        png_path: Path to the overlay PNG written by the CLI (typically
            ``<root>/deliverables/overlays/<dataset>/<stem>.png``). RGB and
            RGBA sources are both accepted; the result is always RGB.
        center_rr: Row coordinate (Y) of the colony centroid, in image
            pixels. Read from ``master_measurements.parquet``.
        center_cc: Column coordinate (X) of the colony centroid, in
            image pixels.
        size: Side length of the square crop, in pixels. Must be
            positive.
        pad_value: RGB fill colour used for any portion of the crop that
            falls outside the source image. Defaults to black.
        dim_alpha: Tile-spotlight strength. When ``> 0`` (and ``bbox`` is
            given), pixels outside the target colony's bbox rectangle are
            blended toward :data:`phenotypic.gui._design.TILE_DIM_RGB` by
            this fraction, leaving the bbox interior untouched. ``0.0``
            (the default) disables the effect — the output is then
            byte-for-byte identical to the pre-feature crop.
        bbox: ``(min_rr, max_rr, min_cc, max_cc)`` of the target colony in
            image pixels (from ``master_measurements.parquet``). Used only
            when ``dim_alpha > 0`` to locate the keep-rectangle. ``None``
            (the default) disables dimming regardless of ``dim_alpha``, so
            older masters lacking the ``Bbox_Min/Max`` columns degrade to a
            plain crop.

    Returns:
        PNG-encoded bytes of the ``size`` x ``size`` crop in RGB mode.

    """
    mtime_ns = os.stat(png_path).st_mtime_ns
    source = _load_overlay_rgb(str(png_path), mtime_ns)
    return _crop_pil_source(
        source,
        center_rr,
        center_cc,
        size,
        pad_value,
        dim_alpha=dim_alpha,
        bbox=bbox,
    )


def _crop_pil_source(
    source: PILImage.Image,
    center_rr: float,
    center_cc: float,
    size: int,
    pad_value: tuple[int, int, int] = (0, 0, 0),
    *,
    dim_alpha: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
) -> bytes:
    """Crop an already-decoded RGB source to a centered ``size`` x ``size`` window.

    The shared geometry body behind both :func:`crop_overlay` (source = a baked
    overlay PNG) and :func:`crop_store_rgb` (source = one store layer). Computes
    ``(top, left) = (round(center_rr) - size // 2, round(center_cc) - size //
    2)``, clamps the requested window to the image bounds, and pastes the
    clamped region onto a freshly-allocated canvas filled with ``pad_value`` so
    the output always has the exact requested dimensions even near an edge.

    Args:
        source: The decoded RGB :class:`PIL.Image.Image` to slice from.
        center_rr: Row coordinate (Y) of the colony centroid, in image pixels.
        center_cc: Column coordinate (X) of the colony centroid, in image pixels.
        size: Side length of the square crop, in pixels. Must be positive.
        pad_value: RGB fill colour for any portion of the crop that falls
            outside the source image. Defaults to black.
        dim_alpha: Tile-spotlight strength; see :func:`crop_overlay`.
        bbox: ``(min_rr, max_rr, min_cc, max_cc)`` keep-rectangle; see
            :func:`crop_overlay`.

    Returns:
        PNG-encoded bytes of the ``size`` x ``size`` crop in RGB mode.
    """
    src_width, src_height = source.size

    half = size // 2
    left_unclamped = round(center_cc) - half
    top_unclamped = round(center_rr) - half
    right_unclamped = left_unclamped + size
    bottom_unclamped = top_unclamped + size

    left_clamped = max(0, left_unclamped)
    top_clamped = max(0, top_unclamped)
    right_clamped = min(src_width, right_unclamped)
    bottom_clamped = min(src_height, bottom_unclamped)

    result = PILImage.new("RGB", (size, size), pad_value)

    # If the clamped window has positive area, paste it onto the padded
    # canvas at the offset that re-aligns it with the unclamped origin.
    if right_clamped > left_clamped and bottom_clamped > top_clamped:
        region = source.crop(
            (left_clamped, top_clamped, right_clamped, bottom_clamped)
        )
        paste_x = max(0, -left_unclamped)
        paste_y = max(0, -top_unclamped)
        result.paste(region, (paste_x, paste_y))

    # Tile-spotlight dim pass. Skipped entirely (no PIL<->ndarray round
    # trip) when disabled, so the disabled path is byte-for-byte identical
    # to the pre-feature output.
    if dim_alpha > 0.0 and bbox is not None:
        min_rr, max_rr, min_cc, max_cc = bbox
        # Keep-rectangle in canvas pixels, computed from the SAME unclamped
        # origin that drives the paste so it tracks the bbox even when the
        # colony sits near an image edge (negative unclamped origin).
        keep_top = _clamp(round(min_rr) - top_unclamped, 0, size)
        keep_bottom = _clamp(round(max_rr) - top_unclamped, 0, size)
        keep_left = _clamp(round(min_cc) - left_unclamped, 0, size)
        keep_right = _clamp(round(max_cc) - left_unclamped, 0, size)
        dimmed = _dim_outside_bbox(
            np.asarray(result),
            (keep_top, keep_left, keep_bottom, keep_right),
            alpha=dim_alpha,
        )
        result = PILImage.fromarray(dimmed)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Full-resolution store-layer cropping and pyramid-level tile reads
# ---------------------------------------------------------------------------

#: One of the displayable store layer names a crop can source.
LayerName = Literal["rgb", "detect_mat", "objmap"]

#: Default layer every crop/tile surface sources when none is requested. The
#: finished RGB plate is the most legible default; the other layers are opt-in.
#: Single-sourced here so the colony-view toggle, its store seed, and the crop
#: route default can never drift apart.
DEFAULT_LAYER: LayerName = "rgb"

#: Number of decoded store layers to keep in memory. Sized by the DATA, not by
#: request variety: the cache key carries the RESOLVED pyramid level rather
#: than the caller's ``target_px``, so distinct targets selecting the same
#: level share one entry (ledger FLOW-10). Bound is level-count x layer-count;
#: a 4000x6000 plate has 5 levels and there are 4 layers.
_STORE_LAYER_CACHE_SIZE = 24


class StoreUnreadable(RuntimeError):
    """A store exists but this build of PhenoTypic cannot decode it.

    ``require_readable_store`` gates ``store_schema_version`` **by value** and
    raises a bare :class:`ValueError`. Bare, that reaches the crop route's
    blanket handler and the user is told "internal error: crop generation
    failed" while the actionable message -- which names both versions and the
    remedy -- reaches only the log. Re-raised as this type, both the crop and
    the DZI route answer ``422`` and pass the message through.
    """


def _readable_block(store_path: Path | str) -> dict:
    """Read ``attributes.phenotypic``, refusing a store this build can't decode.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.

    Returns:
        The ``phenotypic`` block.

    Raises:
        StoreUnreadable: If ``store_schema_version`` is not this build's.
    """
    from phenotypic.sdk_ import ngff_

    try:
        return ngff_.require_readable_store(Path(store_path))
    except ValueError as exc:
        raise StoreUnreadable(str(exc)) from exc


def _store_member_path(block: dict, store_path: Path | str, layer: str) -> str:
    """Resolve one layer to its store-relative group path.

    ``objmap`` is resolved through ``phenotypic.labels.objmap``, never by a
    hard-coded ``rgb/labels/objmap``: an rgb-less store puts the label under
    ``gray``.

    Args:
        block: The ``attributes.phenotypic`` block.
        store_path: Store root, for the error message only.
        layer: ``"rgb"``, ``"gray"``, ``"detect_mat"``, or ``"objmap"``.

    Returns:
        The store-relative group path of the layer.

    Raises:
        KeyError: If *layer* is not present in the store.
    """
    from phenotypic.sdk_ import ngff_

    # ``.get`` on LABELS: a label-less store omits the key entirely, so
    # indexing it would raise before the ``is None`` branch below.
    member = block[ngff_.PhenotypicAttr.SERIES].get(layer) or block.get(
        ngff_.PhenotypicAttr.LABELS, {}
    ).get(layer)
    if member is None:
        raise KeyError(f"Store {store_path} has no layer {layer!r}")
    return member


def _level_shape(
    store_path: Path | str, member: str, level: int
) -> tuple[int, ...]:
    """Return one pyramid level's array shape, from its own array metadata.

    Args:
        store_path: Store root.
        member: Store-relative group path of the layer.
        level: Pyramid level index.

    Returns:
        The level's shape.

    Raises:
        FileNotFoundError: If the level the store DECLARES is not on disk.
    """
    import json

    from phenotypic.sdk_ import ngff_

    meta = Path(store_path) / member / str(level) / ngff_.STORE_ROOT_JSON
    return tuple(json.loads(meta.read_text(encoding="utf-8"))["shape"])


def select_pyramid_level(
    store_path: Path | str, layer: str, target_px: int
) -> int:
    """Return the coarsest pyramid level that still covers ``target_px``.

    "Covers" means the level's longest spatial edge is at least *target_px*.
    Reading a finer level than that is the pre-pyramid behaviour and wastes
    the whole point of the change; reading a coarser one renders a visibly
    soft tile. When even level 0 is smaller than the request, level 0 is
    returned -- there is nothing better to offer.

    The level count comes from ``phenotypic.pyramid.levels`` and each level's
    shape from that level's own array metadata. **Never from a directory
    listing:** a ``.part`` sweep or a partially written store would make a
    listing report a truncated pyramid as the whole pyramid, and every request
    would then silently resolve to level 0. A declared level that is missing
    on disk raises instead.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.
        layer: ``"rgb"``, ``"gray"``, ``"detect_mat"``, or ``"objmap"``.
        target_px: Longest edge, in pixels, the caller intends to render.

    Returns:
        A pyramid level index; ``0`` is full resolution.

    Raises:
        KeyError: If *layer* is not present in the store.
        FileNotFoundError: If the store declares a level it does not hold.
        StoreUnreadable: If this build cannot decode the store.
    """
    from phenotypic.sdk_ import ngff_

    block = _readable_block(store_path)
    member = _store_member_path(block, store_path, layer)
    levels = int(block[ngff_.PhenotypicAttr.PYRAMID]["levels"])
    chosen = 0
    for level in range(levels):
        shape = _level_shape(store_path, member, level)
        if max(shape[-2:]) >= target_px:
            chosen = level
    return chosen


@functools.lru_cache(maxsize=_STORE_LAYER_CACHE_SIZE)
def _load_zarr_level_rgb(
    path: str, content_token: str, layer: LayerName, level: int
) -> PILImage.Image:
    """Decode ONE pyramid level of one store layer to an RGB PIL image.

    Memory discipline: reads only the requested layer's level array -- never
    ``load_image_from_store`` / ``Image.load_zarr``, which materialise every
    layer (hundreds of MB for a plate) to discard all but one.

    Args:
        path: Absolute store path, as a string so the cache key is hashable.
        content_token: Identity of the store's published content. Including
            it in the key invalidates the cached frame when the store is
            republished under a running viewer.
        layer: Layer to decode.
        level: Pyramid level index, already resolved by the caller. The key
            carries the LEVEL rather than the requested pixel size so the
            key space is bounded by the data, not by request variety.

    Returns:
        The decoded level as an RGB :class:`PIL.Image.Image`.

    Raises:
        KeyError: If *layer* is absent from the store.
        StoreUnreadable: If this build cannot decode the store.
    """
    del content_token  # Cache-key only.
    arr = _read_store_level(path, layer, level)
    return PILImage.fromarray(_store_layer_array_to_rgb(arr, layer), mode="RGB")


def _load_zarr_layer_rgb(
    path: str, content_token: str, layer: LayerName, target_px: int
) -> PILImage.Image:
    """Decode the coarsest store level covering ``target_px``, cached by level.

    Thin resolver over :func:`_load_zarr_level_rgb`. The split is what keeps
    the LRU useful: several distinct ``target_px`` values routinely select the
    same level, and keying the cache on the request size instead would thrash
    a 4-entry cache on exactly the path the pyramid exists to accelerate.

    Args:
        path: Absolute store path, as a string.
        content_token: Identity of the store's published content.
        layer: Layer to decode.
        target_px: Longest edge, in pixels, the caller intends to render.

    Returns:
        The decoded level as an RGB :class:`PIL.Image.Image`.
    """
    level = select_pyramid_level(path, layer, target_px)
    return _load_zarr_level_rgb(path, content_token, layer, level)


def _read_store_level(
    store_path: Path | str,
    layer: str,
    level: int,
    window: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Read one layer's level array, optionally only a ``(t, b, l, r)`` window.

    ``rgb`` is stored ``(C, Y, X)`` and is returned channel-last, matching
    ``Image.load_layer_zarr``.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.
        layer: Layer to read.
        level: Pyramid level index.
        window: ``(top, bottom, left, right)`` in level pixels, or ``None``
            for the whole level.

    Returns:
        The array, channel-last for ``rgb``.

    Raises:
        KeyError: If *layer* is absent from the store.
        StoreUnreadable: If this build cannot decode the store.
    """
    import zarr

    from phenotypic.sdk_ import ngff_

    block = _readable_block(store_path)
    member = _store_member_path(block, store_path, layer)
    array = zarr.open_array(
        store=ngff_.long_path(Path(store_path) / member / str(level)),
        mode="r",
    )
    if window is None:
        raw = array[...]
    else:
        top, bottom, left, right = window
        # Windowed: zarr pulls the covering shards/chunks only. A 64x64 crop
        # from a sharded level 0 costs a shard-index read plus one full
        # 1024x1024 inner chunk -- cheap, but not free, and not "the same as
        # h5py", which an earlier draft implied.
        raw = (
            array[:, top:bottom, left:right]
            if len(array.shape) == 3
            else array[top:bottom, left:right]
        )
    data: np.ndarray = np.asarray(raw)
    return np.moveaxis(data, 0, -1) if layer == "rgb" else data


def _store_layer_array_to_rgb(arr: np.ndarray, layer: str) -> np.ndarray:
    """Convert a decoded store layer array to an RGB uint8 array."""
    from phenotypic.gui.builder._image_renderer import (
        _label_map_to_rgb,
        _normalize_to_uint8,
    )

    if layer == "rgb":
        return arr.astype(np.uint8)
    if layer == "objmap":
        return _label_map_to_rgb(arr)
    # detect_mat / gray-like float layer
    gray = _normalize_to_uint8(arr)
    return np.stack([gray] * 3, axis=-1)


def crop_store_rgb(
    store_path: Path,
    layer: LayerName,
    center_rr: float,
    center_cc: float,
    size: int,
    mtime_ns: int,
    *,
    dim_alpha: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
) -> bytes:
    """Full-resolution sibling of :func:`crop_overlay`, sourcing a store layer.

    Same centering / padding / dimming contract as :func:`crop_overlay`; the
    only difference is the pixel source (a windowed read of one layer's
    level-0 array, decoded to RGB, instead of a baked overlay PNG). Geometry
    is byte-identical because both croppers share :func:`_crop_pil_source`.

    Level 0 always: a crop is a full-resolution inspection view, so there is
    no pyramid level to select.

    Args:
        store_path: Path to the per-image store written under
            ``results/<dataset>/zarr/<stem>.ome.zarr``.
        layer: The store layer to render (``"rgb"``, ``"detect_mat"``,
            ``"objmap"``).
        center_rr: Row coordinate (Y) of the colony centroid, in image pixels.
        center_cc: Column coordinate (X) of the colony centroid, in image pixels.
        size: Side length of the square crop, in pixels.
        mtime_ns: Accepted for caller/API compatibility; crop reads are
            windowed and not full-layer cached, so nothing keys on it.
        dim_alpha: Tile-spotlight strength; see :func:`crop_overlay`.
        bbox: ``(min_rr, max_rr, min_cc, max_cc)`` keep-rectangle; see
            :func:`crop_overlay`.

    Returns:
        PNG-encoded bytes of the ``size`` x ``size`` crop in RGB mode.

    Raises:
        KeyError: If *layer* is absent from the store.
        StoreUnreadable: If this build cannot decode the store.
    """
    del mtime_ns
    return _crop_store_layer_window(
        store_path,
        layer,
        center_rr,
        center_cc,
        size,
        dim_alpha=dim_alpha,
        bbox=bbox,
    )


def _crop_store_layer_window(
    store_path: Path,
    layer: LayerName,
    center_rr: float,
    center_cc: float,
    size: int,
    pad_value: tuple[int, int, int] = (0, 0, 0),
    *,
    dim_alpha: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
) -> bytes:
    """Crop by reading only the requested level-0 window out of the store."""
    half = size // 2
    left_unclamped = round(center_cc) - half
    top_unclamped = round(center_rr) - half
    right_unclamped = left_unclamped + size
    bottom_unclamped = top_unclamped + size

    block = _readable_block(store_path)
    member = _store_member_path(block, store_path, layer)
    # ``rgb`` is stored (C, Y, X), so the spatial extent is the LAST two axes
    # on every layer -- ``shape[:2]`` would read (C, Y) and clamp the window
    # to three columns.
    level0 = _level_shape(store_path, member, 0)
    src_height, src_width = level0[-2:]
    left_clamped = max(0, left_unclamped)
    top_clamped = max(0, top_unclamped)
    right_clamped = min(src_width, right_unclamped)
    bottom_clamped = min(src_height, bottom_unclamped)

    arr: np.ndarray | None = None
    if right_clamped > left_clamped and bottom_clamped > top_clamped:
        arr = _read_store_level(
            store_path,
            layer,
            0,
            window=(top_clamped, bottom_clamped, left_clamped, right_clamped),
        )

    result = PILImage.new("RGB", (size, size), pad_value)
    if arr is not None:
        region = PILImage.fromarray(
            _store_layer_array_to_rgb(arr, layer), mode="RGB"
        )
        paste_x = max(0, -left_unclamped)
        paste_y = max(0, -top_unclamped)
        result.paste(region, (paste_x, paste_y))

    if dim_alpha > 0.0 and bbox is not None:
        min_rr, max_rr, min_cc, max_cc = bbox
        keep_top = _clamp(round(min_rr) - top_unclamped, 0, size)
        keep_bottom = _clamp(round(max_rr) - top_unclamped, 0, size)
        keep_left = _clamp(round(min_cc) - left_unclamped, 0, size)
        keep_right = _clamp(round(max_cc) - left_unclamped, 0, size)
        dimmed = _dim_outside_bbox(
            np.asarray(result),
            (keep_top, keep_left, keep_bottom, keep_right),
            alpha=dim_alpha,
        )
        result = PILImage.fromarray(dimmed)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


def crop_colony(
    output_root: OutputRoot,
    dataset: str,
    stem: str,
    layer: LayerName,
    center_rr: float,
    center_cc: float,
    size: int,
    *,
    dim_alpha: float = 0.0,
    bbox: tuple[float, float, float, float] | None = None,
) -> bytes | None:
    """Tier the crop source per-image: the store layer when available, else overlay.

    The single entry point both the colony-view ``/crops`` route and the QC
    review gallery use to fetch a centered colony crop. It prefers the
    per-image OME-Zarr store (via :func:`crop_store_rgb`) and falls back to
    the baked overlay PNG (via :func:`crop_overlay`) for a standalone
    deliverables bundle that ships overlays but no ``results/`` stores.

    A :class:`StoreUnreadable` is deliberately **not** caught: falling back to
    the overlay would show plausible pixels while hiding a run-wide,
    actionable condition (a store this build cannot decode). The caller turns
    it into a ``422`` carrying the store's own message.

    Args:
        output_root: Validated handle on the CLI output directory; supplies
            :meth:`store_path`, :meth:`has_overlay`, and :meth:`overlay_path`.
        dataset: Dataset name (matches ``Metadata_Dataset``).
        stem: Image stem (matches ``Metadata_ImageName`` minus its extension).
        layer: Store layer to render when a store is the source (e.g.
            ``"rgb"``); ignored for the overlay fallback (overlays are
            pre-baked RGB).
        center_rr: Row coordinate (Y) of the colony centroid, in image pixels.
        center_cc: Column coordinate (X) of the colony centroid, in image pixels.
        size: Side length of the square crop, in pixels.
        dim_alpha: Tile-spotlight strength; see :func:`crop_overlay`.
        bbox: ``(min_rr, max_rr, min_cc, max_cc)`` keep-rectangle; see
            :func:`crop_overlay`.

    Returns:
        PNG-encoded bytes of the ``size`` x ``size`` crop, or ``None`` when
        neither a store nor an overlay exists (the caller serves a 404).

    Raises:
        StoreUnreadable: If the store exists but this build cannot decode it.
    """
    store = output_root.store_path(dataset, stem)
    if store is not None:
        try:
            return crop_store_rgb(
                store,
                layer,
                center_rr,
                center_cc,
                size,
                os.stat(store).st_mtime_ns,
                dim_alpha=dim_alpha,
                bbox=bbox,
            )
        except KeyError:
            # The store exists but carries no such layer (e.g. a grayscale
            # pipeline writes no ``rgb`` series). Degrade to the baked
            # overlay PNG rather than 500; fall through to the overlay
            # branch below (else -> None -> 404).
            logger.debug(
                "Store %s missing layer %r; falling back to overlay for %s/%s",
                store,
                layer,
                dataset,
                stem,
            )
    if output_root.has_overlay(dataset, stem):
        png = output_root.overlay_path(dataset, stem)
        return crop_overlay(
            png, center_rr, center_cc, size, dim_alpha=dim_alpha, bbox=bbox
        )
    return None


# ---------------------------------------------------------------------------
# Path-traversal guard
# ---------------------------------------------------------------------------

#: Allow only filesystem-safe identifiers (alphanumeric, dot, underscore,
#: dash) — same character class :func:`werkzeug.utils.secure_filename` is
#: comfortable with, but applied before any path math.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def is_safe_path_component(name: str) -> bool:
    """Return ``True`` if ``name`` is safe to use as a single path component.

    Validates both the ``<dataset>`` and ``<stem>`` URL captures before
    feeding them into filesystem paths.

    Args:
        name: Candidate identifier from the URL.

    Returns:
        ``True`` only if ``name`` is non-empty, contains no path
        separators or parent-directory tokens, does not start with a
        leading dot, and matches ``[A-Za-z0-9._-]+``.
    """
    if not name or name.startswith("."):
        return False
    if "/" in name or "\\" in name or ".." in name:
        return False
    return bool(_NAME_RE.match(name))


def resolve_within_root(
    root: Path,
    tail: str,
    *,
    allowed_roots: frozenset[str],
) -> Path:
    """Resolve a client-controlled ``tail`` to a file inside ``root``.

    The single path-escape guard for every route that serves bytes out of a
    store directory. Two properties are load-bearing and easy to get wrong:

    * Segments are validated INDIVIDUALLY by
      :func:`is_safe_path_component`. The traversal surface here is wider
      than a two-component route's because the tail is arbitrary depth.
    * ``allowed_roots`` is tested on the RESOLVED path, not on the URL
      segments. Testing the unresolved head lets a symlink inside a readable
      root (``<root>/rgb/x -> ../tables/measurements/table.parquet``) satisfy
      both the head check and containment, and the file is served.

    Only the FIRST resolved component is restricted. Restricting every
    component would reject ``labels``, ``objmap`` and every level index,
    which would kill the label layer. The store's own root ``zarr.json`` is
    exempt at depth 1 only -- a pixel client bootstraps from it, and it is
    the one file that belongs to no series.

    Args:
        root: Directory the result must live inside.
        tail: Client-controlled path, ``/``-separated.
        allowed_roots: First-component allow-list. **Required, and there is
            no permissive value.** A security primitive whose default is "no
            restriction" is one forgotten keyword from serving
            ``tables/measurements/table.parquet``, and the omission would
            read as ordinary code at review. An empty ``frozenset()`` rejects
            everything, which is the correct fail-closed shape.

    Returns:
        The resolved file path.

    Raises:
        BadRequest: A segment is unsafe, or the resolved path escapes
            ``root``.
        NotFound: The path does not exist, is not a file, or its first
            resolved component is not in ``allowed_roots``.
    """
    from phenotypic.sdk_ import ngff_

    segments = [segment for segment in tail.split("/") if segment]
    if not segments:
        raise NotFound()
    for segment in segments:
        if not is_safe_path_component(segment):
            raise BadRequest()

    # BOTH resolves inside the try. ``root`` itself can vanish mid-request:
    # ``promote_store`` republishes by renaming the whole store directory
    # (``sdk_/ngff_.py``), so this is the routine path, not an exotic race --
    # it is the very event the generation token exists to handle. Left
    # outside, a promote during a pan raises ``FileNotFoundError`` and the
    # client gets a 500 where 404 is meant.
    try:
        root_resolved = root.resolve(strict=True)
        resolved = root.joinpath(*segments).resolve(strict=True)
    except (OSError, RuntimeError):
        raise NotFound() from None
    if not resolved.is_relative_to(root_resolved):
        raise BadRequest()
    if not resolved.is_file():
        raise NotFound()

    rel = resolved.relative_to(root_resolved)
    head = rel.parts[0]
    if head not in allowed_roots and not (
        len(rel.parts) == 1 and head == ngff_.STORE_ROOT_JSON
    ):
        raise NotFound()
    return resolved


# ---------------------------------------------------------------------------
# Crop-serving Flask route factory
# ---------------------------------------------------------------------------

#: Lower bound on the ``?size=`` query parameter. Smaller crops would not
#: hold a useful colony preview; rejecting them early avoids confused
#: callers.
_MIN_CROP_SIZE = 16

#: Upper bound on the ``?size=`` query parameter. Anything larger is
#: almost certainly the result of a bug in the caller (the colony-view
#: grid picks crops on the order of 64-512 px); cap it to avoid a 4k+
#: PNG allocation per request.
_MAX_CROP_SIZE = 4096

#: Sanity ceiling on the parsed ``<label>`` URL component. Real
#: ``Object_Label`` values are dense small integers; anything beyond a
#: billion is almost certainly malformed input.
_MAX_OBJECT_LABEL = 10**9


def register_crop_route(
    app: dash.Dash, output_root: OutputRoot, segment: str
) -> None:
    """Mount a per-colony centered-crop route under ``/<segment>`` on ``app.server``.

    Exposes one route under the ``/<segment>`` URL prefix:

    * ``GET /<segment>/<dataset>/<stem>/<label>.png?size=<int>`` — returns
      a PNG-encoded ``size`` x ``size`` crop of the dataset's overlay PNG
      centered on the colony with ``Object_Label == label`` in image
      ``stem``.

    The route never mutates state and never writes to disk; the on-disk
    cache used by the DZI tile route is intentionally not reused here
    because per-colony crops are tiny and inexpensive to recompute.

    Both the colony-view tab (``segment="crops"``) and the QC review tab
    mount this factory so they serve crops identically; the ``segment``
    keeps their blueprint names and URL namespaces distinct.

    Args:
        app: The Dash application whose Flask server should be extended.
        output_root: Validated handle on the CLI output directory.
            Captured by closure and used to resolve overlay PNGs and the
            master measurements DataFrame.
        segment: URL path segment the route mounts under (e.g. ``crops``).
            Also seeds the blueprint name so multiple segments can coexist
            on one server.
    """
    # Imported here rather than at module scope:
    #  * ``flask.request`` is a Werkzeug ``LocalProxy``; exposing it as a
    #    module attribute makes ``phenotypic.gui._shared.tiles.request``
    #    raise ``RuntimeError: Working outside of request context`` the
    #    moment anything reads ``__name__`` off it — which the public
    #    namespace walk in ``tests/unit/test_pickleable.py`` does during
    #    parametrize-ID generation, aborting collection of all of
    #    ``tests/unit``. Keeping it function-local hides it from that walk.
    #  * the ``results_viewer`` submodules would otherwise form an import
    #    cycle (``results_viewer.__init__`` eagerly imports ``_app`` →
    #    ``_tile_routes`` → this module for the path guard). They are only
    #    needed once a route is actually mounted, well after both packages
    #    have finished importing.
    import polars as pl
    from flask import request

    from phenotypic.gui._config import TILE_DIM_MAX, TILE_DIM_MIN
    from phenotypic.gui.results_viewer._filtered_state import (
        KEY_DATASET,
        KEY_IMAGE_FILE,
        KEY_OBJECT_LABEL,
    )

    # Bbox extent columns used to locate the spotlight keep-rectangle.
    # Older masters may lack them — handled gracefully below (bbox=None).
    _BBOX_EXTENT_COLS = (
        "Bbox_MinRR",
        "Bbox_MaxRR",
        "Bbox_MinCC",
        "Bbox_MaxCC",
    )

    bp = Blueprint(
        f"results_viewer_crops_{segment}", __name__, url_prefix=f"/{segment}"
    )

    @bp.route("/<dataset>/<stem>/<label>.png")
    def crop_endpoint(
        dataset: str, stem: str, label: str
    ) -> Response | tuple[str, int]:
        """Serve a single PNG colony crop for ``(dataset, stem, label)``."""
        # --- 1. Path-component validation --------------------------------
        if not is_safe_path_component(dataset) or not is_safe_path_component(
            stem
        ):
            logger.warning(
                "Rejected crop request with unsafe identifiers: "
                "dataset=%r stem=%r",
                dataset,
                stem,
            )
            return ("bad request: invalid dataset or stem", 400)

        # --- 2. Label parsing --------------------------------------------
        try:
            label_int = int(label)
        except (TypeError, ValueError):
            logger.warning(
                "Rejected crop request with non-numeric label: %r", label
            )
            return ("bad request: label must be an integer", 400)
        if label_int < 0 or label_int > _MAX_OBJECT_LABEL:
            logger.warning(
                "Rejected crop request with out-of-range label: %d", label_int
            )
            return ("bad request: label out of range", 400)

        # --- 3. Size parsing ---------------------------------------------
        size = request.args.get("size", type=int)
        if size is None:
            return ("bad request: missing required ?size=<int>", 400)
        if size < _MIN_CROP_SIZE or size > _MAX_CROP_SIZE:
            return (
                f"bad request: size must be between {_MIN_CROP_SIZE} and "
                f"{_MAX_CROP_SIZE} (got {size})",
                400,
            )

        # --- 3b. Dim-strength parsing ------------------------------------
        # The tile-spotlight strength rides the URL as ``?dim=`` exactly
        # like ``?size=``. Unlike size it is *clamped*, never rejected — a
        # stray value should soften (or disable) the spotlight, not break
        # the tile. Omitted ⇒ 0.0 ⇒ undimmed, so legacy/cached URLs without
        # ``dim`` keep their full-context crop.
        dim = request.args.get("dim", type=float, default=0.0)
        dim = min(TILE_DIM_MAX, max(TILE_DIM_MIN, dim))

        # --- 3c. Layer selection -----------------------------------------
        # Which image layer to source the crop from when a full-res HDF is
        # available (``rgb`` / ``detect_mat`` / ``objmap``). Defaults to the
        # finished RGB plate. Ignored on the overlay fallback (overlays are
        # pre-baked RGB). Validate against ``LayerName`` at the boundary so an
        # unknown layer 404s here rather than surfacing later as a ``KeyError``
        # (missing HDF dataset) → 500 inside ``crop_colony``.
        layer_raw = request.args.get(
            "layer", type=str, default=cast(str, DEFAULT_LAYER)
        )
        if layer_raw not in get_args(LayerName):
            return (f"not found: unsupported layer {layer_raw!r}", 404)
        layer = cast(LayerName, layer_raw)

        # --- 4. Lookup ----------------------------------------------------
        # Cast key columns explicitly so the comparison still matches when
        # the master frame stores Metadata_ImageName as Categorical or
        # Object_Label as a narrower int type. Pull the bbox extent columns
        # alongside the centroid when present; older masters lacking them
        # fall through to bbox=None (no dimming).
        has_bbox_cols = all(
            col in output_root.master_df.columns for col in _BBOX_EXTENT_COLS
        )
        select_cols = ["Bbox_CenterRR", "Bbox_CenterCC"]
        if has_bbox_cols:
            select_cols.extend(_BBOX_EXTENT_COLS)
        try:
            row = (
                output_root.master_df.filter(
                    (pl.col(KEY_DATASET).cast(pl.String) == dataset)
                    & (pl.col(KEY_IMAGE_FILE).cast(pl.String) == stem)
                    & (pl.col(KEY_OBJECT_LABEL).cast(pl.Int64) == label_int)
                )
                .select(select_cols)
                .head(1)
            )
        except Exception:
            logger.exception(
                "Master DataFrame lookup failed for dataset=%s stem=%s label=%d",
                dataset,
                stem,
                label_int,
            )
            return ("internal error: master measurements lookup failed", 500)

        if row.is_empty():
            return (
                f"not found: no row for stem={stem!r} label={label_int}",
                404,
            )

        center_rr = float(row.get_column("Bbox_CenterRR")[0])
        center_cc = float(row.get_column("Bbox_CenterCC")[0])

        # Spotlight keep-rectangle source. ``(min_rr, max_rr, min_cc,
        # max_cc)`` when the columns are present, else None (graceful
        # degrade — crop_overlay then skips the dim pass regardless of dim).
        bbox: tuple[float, float, float, float] | None = None
        if has_bbox_cols:
            bbox = (
                float(row.get_column("Bbox_MinRR")[0]),
                float(row.get_column("Bbox_MaxRR")[0]),
                float(row.get_column("Bbox_MinCC")[0]),
                float(row.get_column("Bbox_MaxCC")[0]),
            )

        # --- 5+6. Crop (full-res HDF layer, overlay fallback) ------------
        # ``crop_colony`` tiers the pixel source per-image: the per-image
        # full-resolution HDF layer when ``results/`` is present, else the
        # baked overlay PNG (standalone deliverables bundle). ``None`` means
        # neither source exists -> 404.
        if not output_root.snapshot_is_current():
            return (
                "conflict: processing sources changed; refresh the snapshot",
                409,
            )
        source_token = output_root.bound_image_source_token(dataset, stem)
        try:
            png_bytes = crop_colony(
                output_root,
                dataset,
                stem,
                layer,
                center_rr,
                center_cc,
                size,
                dim_alpha=dim,
                bbox=bbox,
            )
        except Exception:
            logger.exception(
                "Crop generation failed for dataset=%s stem=%s label=%d size=%d "
                "layer=%s",
                dataset,
                stem,
                label_int,
                size,
                layer,
            )
            return ("internal error: crop generation failed", 500)

        if (
            not output_root.snapshot_is_current()
            or not output_root.image_source_token_is_current(
                dataset,
                stem,
                source_token,
            )
        ):
            return (
                "conflict: processing sources changed during crop read; "
                "refresh the snapshot",
                409,
            )
        if png_bytes is None:
            return (
                f"not found: no image source for {dataset!r}/{stem!r}",
                404,
            )

        # --- 7. Response --------------------------------------------------
        response = Response(png_bytes, mimetype="image/png")
        response.headers["Cache-Control"] = "no-cache"
        return response

    app.server.register_blueprint(bp)
    logger.debug(
        "Registered results viewer crop routes under /%s for root=%s",
        segment,
        output_root.root,
    )


# ---------------------------------------------------------------------------
# Per-tile chrome — the dedup boundary
# ---------------------------------------------------------------------------


def build_tile_cell(
    *,
    image_file: str,
    label: int,
    dataset: str,
    crop_size: int,
    display_size: int,
    has_image_source: bool,
    is_removed: bool,
    is_selected: bool,
    url_builder: Callable[[str, str, int, int], str],
    remove_button: Component | list[Component],
    extra_children: Iterable[Component] | None = None,
    outer_height: int | None = None,
) -> Component:
    """Render the chrome + crop for a single tile — shared across tabs.

    Builds the framed ``display_size`` x ``display_size`` card carrying

    * the crop ``<img>`` (or a striped placeholder when the overlay is
      missing),
    * a CSS-styled multi-select checkbox carrying a
      ``data-key="<image_file>::<label>"`` attribute (the JS layer reads
      it to drive shift+click selection), and
    * the caller-supplied ``remove_button`` (each tab owns its own
      pattern-matched id so the remove/restore callback resolves
      correctly).

    Any tab-specific siblings (e.g. the colony view's ``N=k`` stack badge
    and popover) are appended after the frame via ``extra_children``.

    Args:
        image_file: ``Metadata_ImageName`` of the represented colony.
        label: ``Object_Label`` of the represented colony.
        dataset: ``Metadata_Dataset`` of the represented colony.
        crop_size: Server crop side length, in pixels — passed to
            ``url_builder`` so the PNG is generated at full resolution
            covering the colony's bbox.
        display_size: CSS render size, in pixels. The browser scales the
            ``<img>`` to this size; ``object-fit: cover`` keeps colonies
            centered without distortion.
        has_image_source: Whether an HDF layer or overlay PNG exists on disk;
            if not, a striped placeholder is rendered instead of an ``<img>``.
        is_removed: Whether the colony is in the curated removal set.
            Dims the crop and toggles the ``is-removed`` modifier.
        is_selected: Whether the tile is in the active multi-select.
            Toggles the ``is-selected`` modifier + the checkbox's
            ``is-checked`` state.
        url_builder: Callable ``(dataset, image_file, label, crop_size) ->
            str`` returning the crop ``<img>`` src. Each tab supplies a
            builder bound to its own crop-route segment + url prefix.
        remove_button: The tab's per-tile curation affordance. Either a
            single ``dbc.Button`` (the legacy remove/restore button) or a
            list of sibling components (e.g. the radial trigger's
            ``[trigger, popover, store]`` triple). A list is spliced into
            the frame as separate children.
        extra_children: Optional siblings appended after the frame inside
            the outer cell ``<div>`` (e.g. a stack badge + popover).
        outer_height: Outer cell height, in pixels. Defaults to
            ``display_size`` (no extra vertical room). Pass a larger value
            to reserve space for a sibling that peeks out below the frame.

    Returns:
        A component ready to drop into a tile grid or gallery container.
    """
    classes = ["colony-cell"]
    if is_selected:
        classes.append("is-selected")
    if is_removed:
        classes.append("is-removed")

    if has_image_source:
        crop_url = url_builder(dataset, image_file, label, crop_size)
        crop_node: Component = html.Img(
            src=crop_url,
            className="colony-cell-img",
            style={
                "width": f"{display_size}px",
                "height": f"{display_size}px",
                "display": "block",
                "opacity": "0.3" if is_removed else "1",
                "objectFit": "cover",
            },
        )
    else:
        crop_node = html.Div(
            className="colony-cell-placeholder",
            style={
                "width": f"{display_size}px",
                "height": f"{display_size}px",
                "backgroundImage": (
                    "repeating-linear-gradient(45deg, "
                    "rgba(0,54,96,0.05) 0px, rgba(0,54,96,0.05) 8px, "
                    "rgba(0,54,96,0.10) 8px, rgba(0,54,96,0.10) 16px)"
                ),
                "border": "1px dashed rgba(0,54,96,0.25)",
            },
        )

    # CSS-styled span playing the role of a checkbox. We don't use a real
    # <input> because Dash 4 doesn't expose html.Input, and the JS layer
    # only needs `data-key` + the class name to wire up the click event.
    # Visual checked state is driven by the `is-checked` modifier class.
    checkbox_class = "colony-cell-checkbox"
    if is_selected:
        checkbox_class += " is-checked"
    # `data-*` HTML attributes can't be expressed in Dash's typed Span
    # kwargs, so unpack via an Any-typed dict to bypass the stub mismatch.
    extra_props: dict[str, Any] = {"data-key": f"{image_file}::{label}"}
    checkbox_inner = html.Span(
        "",
        className=checkbox_class,
        **extra_props,
    )
    checkbox = html.Span(
        checkbox_inner,
        className="colony-cell-checkbox-wrap",
        style={
            "position": "absolute",
            "top": "4px",
            "left": "4px",
            "zIndex": "2",
        },
    )

    # Image card: the framed display_size×display_size area carrying the
    # crop, checkbox, and remove button. Sits in front of any sibling
    # (e.g. the stack tab) via z-index so the sibling can peek out from
    # beneath the bottom edge.
    remove_children: list[Component] = (
        list(remove_button)
        if isinstance(remove_button, list)
        else [remove_button]
    )
    frame = html.Div(
        [crop_node, checkbox, *remove_children],
        className="colony-cell-frame",
    )

    children: list[Component] = [frame]
    if extra_children is not None:
        children.extend(extra_children)

    if outer_height is None:
        outer_height = display_size

    return html.Div(
        children,
        className=" ".join(classes),
        style={
            "position": "relative",
            "width": f"{display_size}px",
            "height": f"{outer_height}px",
            "overflow": "visible",
        },
    )


# ---------------------------------------------------------------------------
# Flat tile gallery
# ---------------------------------------------------------------------------

#: Default CSS gap between tiles in a flat gallery, in pixels.
_GALLERY_GAP_PX = 8


def build_tile_grid(
    keys: list[tuple[str, str, int]],
    url_builder: Callable[[str, str, int, int], str],
    *,
    selected: set[tuple[str, int]],
    removed: set[tuple[str, int]],
    crop_size: int,
    display_size: int,
    has_image_source: Callable[[str, str], bool],
    remove_button_builder: Callable[
        [str, int, bool], Component | list[Component]
    ],
    gap_px: int = _GALLERY_GAP_PX,
) -> tuple[Component, list[tuple[str, int]]]:
    """Render a flat gallery of tiles and its row-major key order.

    Unlike :func:`phenotypic.gui.results_viewer.colony_view._grid.build_grid`
    (a 2D axis-header table), this lays the tiles out left-to-right in a
    wrapping flex row in the iteration order of ``keys`` — the layout the
    QC review gallery uses. The returned ``grid_order`` mirrors that
    visible order so :func:`expand_range` can resolve shift+click slices.

    Args:
        keys: ``(dataset, image_file, label)`` tuples in the order tiles
            should appear.
        url_builder: Callable ``(dataset, image_file, label, crop_size) ->
            str`` returning each tile's crop ``<img>`` src.
        selected: ``(image_file, label)`` keys currently multi-selected.
        removed: ``(image_file, label)`` keys currently in the curated
            removal set.
        crop_size: Server crop side length, in pixels.
        display_size: CSS render size, in pixels, for each tile.
        has_image_source: Callable ``(dataset, image_file) -> bool`` answering
            whether an HDF layer or overlay PNG exists (typically
            :meth:`OutputRoot.has_image_source`).
        remove_button_builder: Callable ``(image_file, label, is_removed)
            -> Component | list[Component]`` returning the tile's curation
            affordance so the gallery owner controls the button id + styling.
            May return a single button (legacy ✕) or a list of sibling
            components (e.g. the radial trigger's ``[trigger, popover,
            store]`` triple), which :func:`build_tile_cell` splices in.
        gap_px: CSS gap between tiles, in pixels. Defaults to 8.

    Returns:
        A tuple ``(component, grid_order)``. ``grid_order`` is the
        row-major flat list of ``(image_file, label)`` keys in the same
        order as the rendered tiles.
    """
    children: list[Component] = []
    grid_order: list[tuple[str, int]] = []
    for dataset, image_file, label in keys:
        key = (image_file, label)
        grid_order.append(key)
        is_removed = key in removed
        children.append(
            build_tile_cell(
                image_file=image_file,
                label=label,
                dataset=dataset,
                crop_size=crop_size,
                display_size=display_size,
                has_image_source=has_image_source(dataset, image_file),
                is_removed=is_removed,
                is_selected=key in selected,
                url_builder=url_builder,
                remove_button=remove_button_builder(
                    image_file, label, is_removed
                ),
            )
        )

    # Full-width responsive grid: ``auto-fill`` packs as many
    # ``display_size``-wide columns as the container affords (more on wide
    # screens, fewer when narrow) and ``1fr`` lets each column stretch to
    # fill the row, so the gallery uses ALL the horizontal space instead
    # of a fixed-width strip. The grid grows with its content (no fixed
    # height / internal scroll) so tiles flow down the page.
    gallery = html.Div(
        children,
        className="colony-grid tile-gallery",
        style={
            "display": "grid",
            "gridTemplateColumns": (
                f"repeat(auto-fill, minmax({display_size}px, 1fr))"
            ),
            "gap": f"{gap_px}px",
            "padding": "0.5rem",
            "alignItems": "flex-start",
            "justifyItems": "center",
        },
    )
    return gallery, grid_order


# ---------------------------------------------------------------------------
# Range expansion
# ---------------------------------------------------------------------------


def expand_range(
    grid_order: list[tuple[str, int]],
    anchor: tuple[str, int],
    target: tuple[str, int],
) -> list[tuple[str, int]]:
    """Return the contiguous slice of ``grid_order`` between two keys.

    Direction-agnostic: the slice always runs from the lower of the two
    indices to the higher (inclusive) regardless of which key was
    originally clicked first.

    Args:
        grid_order: Row-major flat list of cell keys, as returned by
            :func:`build_tile_grid` or
            :func:`phenotypic.gui.results_viewer.colony_view._grid.build_grid`.
        anchor: Key of the cell that started the range (the most recent
            non-shift click).
        target: Key of the shift-clicked cell.

    Returns:
        Inclusive slice of ``grid_order`` between the two keys.

    Raises:
        ValueError: If either ``anchor`` or ``target`` is not in
            ``grid_order``.
    """
    try:
        a = grid_order.index(anchor)
    except ValueError as exc:
        raise ValueError(f"anchor {anchor!r} is not in grid_order") from exc
    try:
        b = grid_order.index(target)
    except ValueError as exc:
        raise ValueError(f"target {target!r} is not in grid_order") from exc
    lo, hi = (a, b) if a <= b else (b, a)
    return grid_order[lo : hi + 1]


__all__ = [
    "LayerName",
    "DEFAULT_LAYER",
    "crop_overlay",
    "crop_store_rgb",
    "select_pyramid_level",
    "StoreUnreadable",
    "crop_colony",
    "is_safe_path_component",
    "register_crop_route",
    "build_tile_cell",
    "build_tile_grid",
    "expand_range",
]
