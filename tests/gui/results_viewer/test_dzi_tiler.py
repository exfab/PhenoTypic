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
import pytest
from PIL import Image as PILImage

from phenotypic.gui.results_viewer import _dzi_tiler
from phenotypic.gui.results_viewer._dzi_tiler import (
    DZI_BACKEND_INFO,
    DziBackendInfo,
    resolve_dzi_backend,
    tile,
)

_DZI_NS = "http://schemas.microsoft.com/deepzoom/2008"


def test_backend_info_is_immutable_and_startup_inspectable() -> None:
    """The selected backend is exposed as immutable structured state."""

    assert DZI_BACKEND_INFO.name in {"pillow", "pyvips"}
    assert DZI_BACKEND_INFO == resolve_dzi_backend()
    with pytest.raises((AttributeError, TypeError)):
        DZI_BACKEND_INFO.name = "pillow"  # type: ignore[misc]


def test_resolve_dzi_backend_can_force_pillow() -> None:
    """Forced Pillow selection does not depend on pyvips availability."""

    assert resolve_dzi_backend("pillow") == DziBackendInfo(
        name="pillow", version=None, fallback_reason=None
    )


def test_resolve_dzi_backend_rejects_unknown_mode() -> None:
    """Misspelled backend modes fail before any image work starts."""

    with pytest.raises(ValueError, match="Unsupported DZI backend"):
        resolve_dzi_backend("gpu")  # type: ignore[arg-type]


def test_resolve_dzi_backend_reports_sanitized_unavailable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Automatic fallback reports the error type without loader paths."""

    loader_error = OSError("/secret/location/libvips.42.dylib")
    monkeypatch.setattr(_dzi_tiler, "pyvips", None)
    monkeypatch.setattr(_dzi_tiler, "_PYVIPS_IMPORT_ERROR", loader_error)

    info = resolve_dzi_backend("auto")

    assert info.name == "pillow"
    assert info.fallback_reason == "OSError: libvips unavailable"
    assert "/secret/location" not in info.fallback_reason
    with pytest.raises(RuntimeError, match="pyvips backend requested"):
        resolve_dzi_backend("pyvips")


def test_resolve_dzi_backend_reports_native_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A working pyvips binding exposes the loaded native library version."""

    class _FakePygvips:
        @staticmethod
        def version(part: int) -> int:
            return (8, 18, 5)[part]

    monkeypatch.setattr(_dzi_tiler, "pyvips", _FakePygvips())

    assert resolve_dzi_backend("pyvips") == DziBackendInfo(
        name="pyvips", version="8.18.5", fallback_reason=None
    )


def _make_fixture_png(
    path: Path, size: tuple[int, int] = (1024, 1024)
) -> Path:
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


def test_forced_pillow_never_calls_pyvips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The forced-Pillow seam remains usable when pyvips is installed."""

    png = _make_fixture_png(tmp_path / "img.png", size=(64, 48))

    def _unexpected_pyvips(*args: object) -> None:
        raise AssertionError("pyvips must not run in forced-Pillow mode")

    monkeypatch.setattr(_dzi_tiler, "_tile_with_pyvips", _unexpected_pyvips)

    manifest = tile(png, tmp_path / "tiles", backend="pillow")

    assert manifest.is_file()


def test_pyvips_dzsave_error_cleans_partial_output_and_retries_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the marked dzsave failure gets one clean Pillow retry."""

    png = _make_fixture_png(tmp_path / "img.png", size=(64, 48))
    output_dir = tmp_path / "tiles"
    files_dir = output_dir / "img_files"
    manifest = output_dir / "img.dzi"
    pillow_calls = 0
    original_pillow = _dzi_tiler._tile_with_pillow

    monkeypatch.setattr(
        _dzi_tiler,
        "resolve_dzi_backend",
        lambda mode="auto": DziBackendInfo("pyvips", "8.18.5", None),
    )

    def _failed_pyvips(*args: object) -> None:
        files_dir.mkdir(parents=True)
        (files_dir / "partial.png").write_bytes(b"partial")
        manifest.write_text("partial", encoding="utf-8")
        raise _dzi_tiler._PyvipsDzsaveError

    def _record_pillow(*args: object) -> None:
        nonlocal pillow_calls
        pillow_calls += 1
        assert not files_dir.exists()
        assert not manifest.exists()
        original_pillow(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(_dzi_tiler, "_tile_with_pyvips", _failed_pyvips)
    monkeypatch.setattr(_dzi_tiler, "_tile_with_pillow", _record_pillow)

    result = tile(png, output_dir, backend="pyvips")

    assert result == manifest
    assert result.is_file()
    assert pillow_calls == 1


def test_non_dzsave_error_does_not_retry_with_pillow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filesystem errors from the pyvips path propagate without retry."""

    png = _make_fixture_png(tmp_path / "img.png", size=(64, 48))
    pillow_called = False
    monkeypatch.setattr(
        _dzi_tiler,
        "resolve_dzi_backend",
        lambda mode="auto": DziBackendInfo("pyvips", "8.18.5", None),
    )

    def _permission_error(*args: object) -> None:
        raise PermissionError("read-only destination")

    def _record_pillow(*args: object) -> None:
        nonlocal pillow_called
        pillow_called = True

    monkeypatch.setattr(_dzi_tiler, "_tile_with_pyvips", _permission_error)
    monkeypatch.setattr(_dzi_tiler, "_tile_with_pillow", _record_pillow)

    with pytest.raises(PermissionError, match="read-only destination"):
        tile(png, tmp_path / "tiles", backend="pyvips")
    assert not pillow_called


def test_pyvips_wrapper_marks_only_native_dzsave_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry marker is limited to pyvips errors raised by dzsave."""

    class _FakeVipsError(Exception):
        pass

    class _FailedImage:
        def dzsave(self, *args: object, **kwargs: object) -> None:
            raise _FakeVipsError("native operation failed")

    class _FakeImageFactory:
        @staticmethod
        def new_from_file(*args: object, **kwargs: object) -> _FailedImage:
            return _FailedImage()

    class _FakePygvips:
        Error = _FakeVipsError
        Image = _FakeImageFactory

    png = _make_fixture_png(tmp_path / "img.png", size=(16, 16))
    monkeypatch.setattr(_dzi_tiler, "pyvips", _FakePygvips())

    with pytest.raises(_dzi_tiler._PyvipsDzsaveError):
        _dzi_tiler._tile_with_pyvips(png, tmp_path / "tiles", 254, 1)


def test_pyvips_wrapper_propagates_non_native_dzsave_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regular Python filesystem error is not converted into retry state."""

    class _FakeVipsError(Exception):
        pass

    class _FailedImage:
        def dzsave(self, *args: object, **kwargs: object) -> None:
            raise PermissionError("read-only destination")

    class _FakeImageFactory:
        @staticmethod
        def new_from_file(*args: object, **kwargs: object) -> _FailedImage:
            return _FailedImage()

    class _FakePygvips:
        Error = _FakeVipsError
        Image = _FakeImageFactory

    png = _make_fixture_png(tmp_path / "img.png", size=(16, 16))
    monkeypatch.setattr(_dzi_tiler, "pyvips", _FakePygvips())

    with pytest.raises(PermissionError, match="read-only destination"):
        _dzi_tiler._tile_with_pyvips(png, tmp_path / "tiles", 254, 1)


@pytest.mark.parametrize(
    "message",
    [
        "unable to create directory: Permission denied",
        "unable to write tile: No space left on device",
    ],
)
def test_pyvips_wrapper_does_not_retry_native_storage_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    message: str,
) -> None:
    """Native libvips storage failures remain visible to the caller."""

    class _FakeVipsError(Exception):
        pass

    class _FailedImage:
        def dzsave(self, *args: object, **kwargs: object) -> None:
            raise _FakeVipsError(message)

    class _FakeImageFactory:
        @staticmethod
        def new_from_file(*args: object, **kwargs: object) -> _FailedImage:
            return _FailedImage()

    class _FakePygvips:
        Error = _FakeVipsError
        Image = _FakeImageFactory

    png = _make_fixture_png(tmp_path / "img.png", size=(16, 16))
    monkeypatch.setattr(_dzi_tiler, "pyvips", _FakePygvips())

    with pytest.raises(_FakeVipsError, match=message):
        _dzi_tiler._tile_with_pyvips(png, tmp_path / "tiles", 254, 1)
