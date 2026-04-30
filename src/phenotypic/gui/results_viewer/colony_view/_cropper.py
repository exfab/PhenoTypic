"""Per-colony overlay cropping for the results viewer's colony-view tab.

Slices a fixed-size, centroid-aligned square crop out of the overlay PNG
that the CLI writes at ``<root>/results/<dataset>/overlays/<stem>.png``
and returns it as PNG-encoded bytes. The colony-view grid uses these
crops to render uniformly sized thumbnails; the caller is responsible
for picking ``size`` (typically the max bounding-box dimension across
the filtered colony set, plus padding) so every tile in the displayed
grid shares the same canvas size.

Crops that spill past the image edge are padded with ``pad_value`` so
the returned image is always exactly ``size`` x ``size``. The decoded
overlay is held in a small LRU cache keyed on
``(path, mtime_ns)`` because a typical grid render hits the same handful
of plate scans dozens of times in quick succession; without the cache
each cell would re-decode the full overlay.
"""

from __future__ import annotations

import functools
import io
import os
from pathlib import Path

from PIL import Image as PILImage

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


def crop_overlay(
    png_path: Path,
    center_rr: float,
    center_cc: float,
    size: int,
    pad_value: tuple[int, int, int] = (0, 0, 0),
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
            ``<root>/results/<dataset>/overlays/<stem>.png``). RGB and
            RGBA sources are both accepted; the result is always RGB.
        center_rr: Row coordinate (Y) of the colony centroid, in image
            pixels. Read from ``master_measurements.parquet``.
        center_cc: Column coordinate (X) of the colony centroid, in
            image pixels.
        size: Side length of the square crop, in pixels. Must be
            positive.
        pad_value: RGB fill colour used for any portion of the crop that
            falls outside the source image. Defaults to black.

    Returns:
        PNG-encoded bytes of the ``size`` x ``size`` crop in RGB mode.

    """
    # TODO(future): mirror this with crop_hdf_rgb(h5_path, ...) that
    # loads the raw RGB layer via Image.load_hdf5 (see
    # src/phenotypic/_core/_image_parts/_image_io_handler.py:944) for
    # overlay-free crops.
    mtime_ns = os.stat(png_path).st_mtime_ns
    source = _load_overlay_rgb(str(png_path), mtime_ns)

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

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()
