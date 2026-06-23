"""Thumbnail cache naming, downscaling, and the route factory."""
from __future__ import annotations

import re

from phenotypic.gui._shared.timeline._thumbnail import (
    ThumbUnavailable,
    thumb_cache_name,
)


def test_thumb_cache_name_is_flat_safe_and_self_invalidating() -> None:
    name = thumb_cache_name("d1/img-1", 128, 1234567890)
    # No path separators; ends in the bucket + mtime + .png.
    assert "/" not in name and "\\" not in name
    assert name.endswith("_128_1234567890.png")
    assert re.fullmatch(r"[A-Za-z0-9_-]+\.png", name)


def test_thumb_cache_name_distinguishes_mtime() -> None:
    a = thumb_cache_name("d1/img-1", 128, 111)
    b = thumb_cache_name("d1/img-1", 128, 222)
    assert a != b  # a regenerated source yields a fresh cache file


def test_thumb_unavailable_is_runtime_error() -> None:
    assert issubclass(ThumbUnavailable, RuntimeError)


import io
from pathlib import Path

from PIL import Image as PILImage

from phenotypic.gui._shared.timeline._thumbnail import downscale_to_thumb


def test_downscale_preserves_aspect_with_longest_edge(tmp_path: Path) -> None:
    src = tmp_path / "wide.png"
    PILImage.new("RGB", (200, 100), (255, 0, 0)).save(src, format="PNG")

    data = downscale_to_thumb(src, 64)

    out = PILImage.open(io.BytesIO(data))
    assert out.format == "PNG"
    assert out.size == (64, 32)  # 200x100 → longest edge 64


def test_downscale_outputs_rgb(tmp_path: Path) -> None:
    src = tmp_path / "rgba.png"
    PILImage.new("RGBA", (50, 50), (0, 255, 0, 128)).save(src, format="PNG")

    out = PILImage.open(io.BytesIO(downscale_to_thumb(src, 32)))
    assert out.mode == "RGB"
