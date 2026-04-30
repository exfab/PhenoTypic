"""Unit tests for :mod:`phenotypic.gui.results_viewer._dzi_tiler`.

Covers the public ``tile()`` contract: manifest generation, DZI XML
schema, tile-pyramid level count, top-level tile fan-out, mtime-based
cache invalidation, concurrent calls, and stale-tree wiping. Tests
generate small fixture PNGs with NumPy/Pillow and write everything
under ``tmp_path`` so nothing leaks between runs.
"""

from __future__ import annotations

import math
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from phenotypic.gui.results_viewer._dzi_tiler import tile

_DZI_NS = "http://schemas.microsoft.com/deepzoom/2008"


def _make_fixture_png(path: Path, size: tuple[int, int] = (1024, 1024)) -> Path:
    """Render a deterministic noise PNG at ``path`` and return the path.

    Args:
        path: Destination file path.
        size: ``(width, height)`` in pixels.

    Returns:
        ``path`` (so callers can chain).
    """

    arr = (
        np.random.default_rng(42).random((size[1], size[0], 3)) * 255
    ).astype("uint8")
    PILImage.fromarray(arr).save(path)
    return path


def test_tile_basic_generation_returns_valid_manifest(tmp_path: Path) -> None:
    """``tile()`` returns the manifest path; the file is well-formed XML."""

    png = _make_fixture_png(tmp_path / "img.png")
    out_dir = tmp_path / "tiles"

    manifest = tile(png, out_dir)
    assert isinstance(manifest, Path)
    assert manifest.is_file()
    head = manifest.read_text(encoding="utf-8")
    assert head.lstrip().startswith("<?xml")


def test_tile_manifest_xml_schema(tmp_path: Path) -> None:
    """Manifest root is ``<Image>`` with expected attrs and ``<Size>`` child."""

    png = _make_fixture_png(tmp_path / "img.png", size=(1024, 768))
    out_dir = tmp_path / "tiles"

    manifest = tile(png, out_dir, tile_size=254, overlap=1)
    root = ET.parse(manifest).getroot()

    assert root.tag == f"{{{_DZI_NS}}}Image"
    assert root.attrib["Format"] == "png"
    assert int(root.attrib["TileSize"]) == 254
    assert int(root.attrib["Overlap"]) == 1

    size_el = root.find(f"{{{_DZI_NS}}}Size")
    assert size_el is not None
    assert int(size_el.attrib["Width"]) == 1024
    assert int(size_el.attrib["Height"]) == 768


def test_tile_pyramid_level_count_and_base_level(tmp_path: Path) -> None:
    """Pyramid has ``ceil(log2(max(W, H))) + 1`` levels; level 0 is 1x1.

    For a 1024-pixel-wide image: ``log2(1024) = 10`` so levels 0..10
    inclusive must exist (11 levels total). Level 0 is the single
    1x1 root tile.
    """

    width, height = 1024, 1024
    png = _make_fixture_png(tmp_path / "img.png", size=(width, height))
    out_dir = tmp_path / "tiles"

    tile(png, out_dir)
    files_dir = out_dir / f"{png.stem}_files"
    assert files_dir.is_dir()

    expected_max_level = int(math.ceil(math.log2(max(width, height))))
    levels = sorted(int(p.name) for p in files_dir.iterdir() if p.is_dir())
    assert levels == list(range(expected_max_level + 1))

    level0 = files_dir / "0"
    level0_tiles = list(level0.glob("*.png"))
    assert len(level0_tiles) == 1
    with PILImage.open(level0_tiles[0]) as img:
        assert img.size == (1, 1)


def test_top_level_tile_count_matches_grid(tmp_path: Path) -> None:
    """Top level for a 1024x1024 image at ``tile_size=254`` has 5*5 = 25 tiles."""

    width, height = 1024, 1024
    tile_size = 254
    png = _make_fixture_png(tmp_path / "img.png", size=(width, height))
    out_dir = tmp_path / "tiles"

    tile(png, out_dir, tile_size=tile_size, overlap=1)
    top_level = int(math.ceil(math.log2(max(width, height))))
    top_dir = out_dir / f"{png.stem}_files" / str(top_level)
    tiles = list(top_dir.glob("*.png"))

    expected_per_axis = math.ceil(width / tile_size)
    assert expected_per_axis == 5
    assert len(tiles) == expected_per_axis * expected_per_axis == 25


def test_cache_hit_is_no_op(tmp_path: Path) -> None:
    """A second ``tile()`` call returns the same path with unchanged mtime."""

    png = _make_fixture_png(tmp_path / "img.png", size=(512, 512))
    out_dir = tmp_path / "tiles"

    first = tile(png, out_dir)
    first_mtime = first.stat().st_mtime
    # Sleep briefly so any new write would have a strictly larger mtime.
    time.sleep(0.05)
    second = tile(png, out_dir)
    assert second == first
    assert second.stat().st_mtime == first_mtime


def test_cache_invalidation_when_png_mtime_advances(tmp_path: Path) -> None:
    """Bumping the source PNG mtime forces tile regeneration."""

    png = _make_fixture_png(tmp_path / "img.png", size=(512, 512))
    out_dir = tmp_path / "tiles"

    manifest = tile(png, out_dir)
    initial_mtime = manifest.stat().st_mtime

    # Make the PNG strictly newer than the manifest.
    future = initial_mtime + 5
    os.utime(png, (future, future))

    tile(png, out_dir)
    new_mtime = manifest.stat().st_mtime
    assert new_mtime > initial_mtime, (
        f"manifest mtime should advance after PNG bump "
        f"(initial={initial_mtime}, new={new_mtime})"
    )


def test_concurrent_tile_calls_serialise_via_lock(tmp_path: Path) -> None:
    """Four concurrent ``tile()`` calls return the same manifest without errors.

    The per-image lock should serialise the writers so the cache never
    ends up half-written. The exact count of tiles isn't asserted — the
    invariant under test is that no exception escapes the lock and the
    resulting manifest is valid XML.
    """

    png = _make_fixture_png(tmp_path / "img.png", size=(512, 512))
    out_dir = tmp_path / "tiles"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: tile(png, out_dir), range(4)))

    assert len(results) == 4
    assert all(r == results[0] for r in results)
    # Manifest still parses after the dust settles.
    root = ET.parse(results[0]).getroot()
    assert root.tag == f"{{{_DZI_NS}}}Image"


def test_stale_files_dir_is_wiped_on_regen(tmp_path: Path) -> None:
    """Pre-existing junk in ``<stem>_files/`` is removed when regenerating."""

    png = _make_fixture_png(tmp_path / "img.png", size=(256, 256))
    out_dir = tmp_path / "tiles"
    files_dir = out_dir / f"{png.stem}_files"
    files_dir.mkdir(parents=True, exist_ok=True)
    junk = files_dir / "junk.txt"
    junk.write_text("stale tile from a previous interrupted run")

    # Force regeneration by ensuring no manifest exists yet (we never
    # wrote one) — ``tile()`` will rmtree the dir before re-tiling.
    tile(png, out_dir)

    assert not junk.exists(), (
        "stale junk file should have been cleared by shutil.rmtree on regen"
    )
    # And a real tile pyramid was written in its place.
    assert (files_dir / "0").is_dir()
