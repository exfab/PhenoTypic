"""Unit tests for :mod:`phenotypic.gui.results_viewer.colony_view._cropper`.

Exercises the PNG-bytes-returning crop function:

- centred crop in a uniformly-coloured image returns the expected size,
- a crop near the corner of the image gets edge-padded so the output is
  still exactly ``size×size``,
- a crop entirely outside the image returns a fully-padded canvas.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from phenotypic.gui._shared import tiles
from phenotypic.gui.results_viewer.colony_view._cropper import crop_overlay


def _write_solid_png(
    path: Path, w: int, h: int, color: tuple[int, int, int]
) -> None:
    """Helper: write a solid-colour RGB PNG of size *w*×*h*."""
    PILImage.new("RGB", (w, h), color).save(path, format="PNG")


def test_centered_crop_returns_solid_color(tmp_path: Path) -> None:
    """A crop in the interior of a solid white image is solid white."""
    src = tmp_path / "src.png"
    _write_solid_png(src, 100, 100, (255, 255, 255))

    png_bytes = crop_overlay(src, center_rr=50, center_cc=50, size=20)
    img = PILImage.open(io.BytesIO(png_bytes))

    assert img.size == (20, 20)
    assert img.mode == "RGB"
    # All pixels are white.
    assert img.getextrema() == ((255, 255), (255, 255), (255, 255))


def test_corner_crop_is_padded(tmp_path: Path) -> None:
    """A crop near the (0, 0) corner is padded with the pad colour.

    With center=(5, 5) and size=20, the top-left 5 px of the crop fall
    outside the source image and must be filled with ``pad_value``. The
    bottom-right portion is the (clamped) source pixels.
    """
    src = tmp_path / "src.png"
    _write_solid_png(src, 100, 100, (255, 255, 255))  # white inside

    png_bytes = crop_overlay(
        src, center_rr=5, center_cc=5, size=20, pad_value=(0, 0, 0)
    )
    img = PILImage.open(io.BytesIO(png_bytes))

    assert img.size == (20, 20)
    # Top-left pixel must be black (padding); a pixel deep inside the
    # crop must be white (source).
    assert img.getpixel((0, 0)) == (0, 0, 0)
    assert img.getpixel((19, 19)) == (255, 255, 255)


def test_fully_outside_crop_is_pad_only(tmp_path: Path) -> None:
    """A crop centred well outside the image is fully padded.

    Defensive case: if a buggy caller supplies a centre far outside the
    image, the function must still return a valid PNG of the requested
    size rather than crashing.
    """
    src = tmp_path / "src.png"
    _write_solid_png(src, 50, 50, (255, 255, 255))

    png_bytes = crop_overlay(
        src,
        center_rr=200,
        center_cc=200,
        size=10,
        pad_value=(7, 8, 9),
    )
    img = PILImage.open(io.BytesIO(png_bytes))

    assert img.size == (10, 10)
    # Every pixel is the pad colour.
    assert img.getextrema() == ((7, 7), (8, 8), (9, 9))


def test_rgba_source_is_normalised_to_rgb(tmp_path: Path) -> None:
    """An RGBA source PNG is flattened to RGB on load (no alpha leak)."""
    src = tmp_path / "src.png"
    PILImage.new("RGBA", (40, 40), (200, 100, 50, 255)).save(src, format="PNG")

    png_bytes = crop_overlay(src, center_rr=20, center_cc=20, size=8)
    img = PILImage.open(io.BytesIO(png_bytes))

    assert img.mode == "RGB"
    assert img.getpixel((4, 4)) == (200, 100, 50)


def test_hdf_crop_reads_window_without_full_layer_loader(
    tmp_path: Path, monkeypatch
) -> None:
    import h5py

    h5_path = tmp_path / "plate.h5"
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[24:40, 24:40] = (10, 80, 160)
    with h5py.File(h5_path, "w") as fh:
        layers = fh.create_group("layers")
        layers.create_dataset("rgb", data=rgb)

    def _raise_full_layer(*_args, **_kwargs):
        raise AssertionError(
            "crop_hdf_rgb should not decode the full HDF layer"
        )

    monkeypatch.setattr(tiles, "_load_hdf_layer_rgb", _raise_full_layer)

    png_bytes = tiles.crop_hdf_rgb(
        h5_path,
        "rgb",
        center_rr=32,
        center_cc=32,
        size=16,
        mtime_ns=h5_path.stat().st_mtime_ns,
    )
    img = PILImage.open(io.BytesIO(png_bytes))

    assert img.size == (16, 16)
    assert img.getpixel((8, 8)) == (10, 80, 160)
