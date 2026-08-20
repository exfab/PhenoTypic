"""Tile reads select a pyramid level instead of decoding the whole layer.

``select_pyramid_level`` returns the **coarsest** level that still covers the
request. Reading finer wastes the pyramid; reading coarser renders visibly
soft. The level count comes from ``phenotypic.pyramid.levels`` plus each
level's own array metadata -- never from a directory listing, which a ``.part``
sweep or a partially written store would make lie.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.gui._shared.tiles import (
    StoreUnreadable,
    crop_store_rgb,
    select_pyramid_level,
)


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    """One promoted store for the whole module: ``save2zarr`` is ~2 s."""
    tmp_path = tmp_path_factory.mktemp("store")
    return Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")


def test_full_size_request_selects_level_zero(store: Path) -> None:
    level0 = Image.load_layer_zarr(store, "gray", level=0)
    assert select_pyramid_level(store, "gray", max(level0.shape)) == 0


def test_small_request_selects_a_coarse_level(store: Path) -> None:
    assert select_pyramid_level(store, "gray", 64) > 0


def test_a_coarse_level_really_is_fewer_bytes(store: Path) -> None:
    """The point of the exercise, asserted on bytes rather than on an index.

    A level index is only a proxy. This reads both arrays and compares their
    materialised sizes, so a selection that returned a coarse index while the
    read still pulled level 0 would not pass.
    """
    level = select_pyramid_level(store, "gray", 64)
    coarse = Image.load_layer_zarr(store, "gray", level=level)
    full = Image.load_layer_zarr(store, "gray", level=0)
    assert coarse.nbytes < full.nbytes


def test_selected_level_still_covers_the_request(store: Path) -> None:
    """Coarser than the request would render visibly soft."""
    for target in (64, 128, 256, 512, 1024):
        level = select_pyramid_level(store, "gray", target)
        shape = Image.load_layer_zarr(store, "gray", level=level).shape
        assert max(shape) >= target or level == 0


def test_selection_never_reads_finer_than_necessary(store: Path) -> None:
    level = select_pyramid_level(store, "gray", 256)
    if level > 0:
        finer = Image.load_layer_zarr(store, "gray", level=level - 1).shape
        assert max(finer) > 256


def test_level_count_comes_from_metadata_not_directory_listing(
    store: Path, tmp_path: Path
) -> None:
    """A ``.part`` sweep or a partial write would make a listing lie.

    ``levels`` is read from ``phenotypic.pyramid``, so a store missing a
    level it declares is an ERROR. A listing-derived count would silently
    report the truncated pyramid as the whole pyramid and serve a level-0
    tile for every request.
    """
    truncated = tmp_path / "truncated.ome.zarr"
    shutil.copytree(store, truncated)
    shutil.rmtree(truncated / "gray" / "1")
    with pytest.raises(FileNotFoundError):
        select_pyramid_level(truncated, "gray", 64)


def test_single_level_store_always_selects_zero(tmp_path: Path) -> None:
    """The ``levels=1`` path: builder node previews."""
    flat = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "f.ome.zarr", layers=("gray",)
    )
    assert select_pyramid_level(flat, "gray", 32) == 0


def test_unknown_layer_raises_key_error(store: Path) -> None:
    with pytest.raises(KeyError):
        select_pyramid_level(store, "not-a-layer", 64)


def test_a_future_store_is_refused_rather_than_decoded(tmp_path: Path) -> None:
    """``store_schema_version`` is gated by VALUE, and the GUI must say so.

    ``require_readable_store`` raises a bare ``ValueError`` per read. Left
    bare it reaches the crop route's blanket handler and the user is told
    "internal error", with the real, actionable message only in a log. It is
    re-raised as ``StoreUnreadable`` so both routes can answer 422 and pass
    the message through.
    """
    import json

    from phenotypic.sdk_.ngff_ import PhenotypicAttr, STORE_ROOT_JSON

    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    root = store / STORE_ROOT_JSON
    payload = json.loads(root.read_text(encoding="utf-8"))
    payload["attributes"][PhenotypicAttr.ROOT][
        PhenotypicAttr.STORE_SCHEMA_VERSION
    ] = 999
    root.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StoreUnreadable) as excinfo:
        select_pyramid_level(store, "gray", 64)
    assert "999" in str(excinfo.value)
    assert "upgrade" in str(excinfo.value)


# ---------------------------------------------------------------------------
# crop_store_rgb — windowed full-resolution reads
# ---------------------------------------------------------------------------


def _decode_rgb(png_bytes: bytes) -> np.ndarray:
    import io

    from PIL import Image as PILImage

    return np.asarray(PILImage.open(io.BytesIO(png_bytes)).convert("RGB"))


def test_crop_matches_the_full_resolution_slice(store: Path) -> None:
    """A crop is a windowed read of LEVEL 0, not of a selected level."""
    from phenotypic.gui.builder._image_renderer import _normalize_to_uint8

    full = Image.load_layer_zarr(store, "detect_mat", level=0)
    png = crop_store_rgb(
        store,
        "detect_mat",
        center_rr=42,
        center_cc=42,
        size=64,
        mtime_ns=0,
    )
    crop = _decode_rgb(png)
    assert crop.shape == (64, 64, 3)
    expected = _normalize_to_uint8(full)[10:74, 10:74]
    np.testing.assert_array_equal(crop[..., 0], expected)


def test_crop_reads_only_the_window_not_the_whole_layer(
    store: Path, monkeypatch
) -> None:
    """Read amplification is a shard index plus one inner chunk, not a layer.

    Asserted on the number of ELEMENTS pulled out of the zarr array: a
    ``[...]`` full-array read would be orders of magnitude larger than the
    window, and is exactly the regression a rename-only port produces.
    """
    import zarr

    pulled: list[int] = []
    real_getitem = zarr.Array.__getitem__

    def _counting(self, selection):
        out = real_getitem(self, selection)
        pulled.append(int(np.asarray(out).size))
        return out

    monkeypatch.setattr(zarr.Array, "__getitem__", _counting)
    crop_store_rgb(store, "detect_mat", 42, 42, 64, 0)
    assert pulled, "no zarr read happened at all"
    assert max(pulled) <= 64 * 64, pulled


def test_crop_of_a_missing_layer_raises_key_error(tmp_path: Path) -> None:
    """A grayscale-only store has no ``rgb``; ``crop_colony`` needs the KeyError."""
    flat = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "f.ome.zarr", layers=("gray",)
    )
    with pytest.raises(KeyError):
        crop_store_rgb(flat, "rgb", 20, 20, 16, 0)


def test_crop_pads_at_the_edge(store: Path) -> None:
    """The shared ``_crop_pil_source`` padding contract survives the port."""
    png = crop_store_rgb(store, "detect_mat", 0, 0, 32, 0)
    crop = _decode_rgb(png)
    assert crop.shape == (32, 32, 3)
    # The top-left quadrant falls outside the image and is pad-filled black.
    assert crop[:16, :16].max() == 0


def test_rgb_crop_returns_channel_last_pixels(store: Path) -> None:
    """``rgb`` is stored ``(C, Y, X)``; a naive window read transposes it."""
    rgb = Image.load_layer_zarr(store, "rgb", level=0)
    png = crop_store_rgb(store, "rgb", 42, 42, 64, 0)
    np.testing.assert_array_equal(_decode_rgb(png), rgb[10:74, 10:74, :3])


# ---------------------------------------------------------------------------
# _load_zarr_layer_rgb — level resolution and the cache-key contract
# ---------------------------------------------------------------------------


def test_layer_loader_actually_resolves_the_level_from_target_px(
    store: Path,
) -> None:
    """The resolver must USE ``target_px``, not merely accept it.

    Every production caller today asks for the level-0 edge, so a loader that
    ignored ``target_px`` and always read level 0 would give every existing
    test the right answer -- while the pyramid did nothing at all. This is
    the only assertion that separates the two.
    """
    from phenotypic.gui._shared.tiles import _load_zarr_layer_rgb

    coarse = _load_zarr_layer_rgb(str(store), "tok", "detect_mat", 64)
    finest = _load_zarr_layer_rgb(str(store), "tok", "detect_mat", 10**6)
    assert max(coarse.size) < max(finest.size)


def test_two_targets_selecting_one_level_share_a_cache_entry(
    store: Path,
) -> None:
    """Ledger FLOW-10: the LRU key is the resolved LEVEL, not the request size.

    Keyed on ``target_px``, a handful of distinct request sizes thrash the
    cache on exactly the path the pyramid exists to accelerate. Keyed on the
    level, the key space is bounded by the data.
    """
    from phenotypic.gui._shared.tiles import (
        _load_zarr_layer_rgb,
        _load_zarr_level_rgb,
    )

    _load_zarr_level_rgb.cache_clear()
    assert select_pyramid_level(store, "detect_mat", 700) == select_pyramid_level(
        store, "detect_mat", 799
    )
    _load_zarr_layer_rgb(str(store), "tok", "detect_mat", 700)
    _load_zarr_layer_rgb(str(store), "tok", "detect_mat", 799)
    assert _load_zarr_level_rgb.cache_info().hits == 1


def test_a_changed_content_token_busts_the_cache(store: Path) -> None:
    """The token is what invalidates a republished store under a live viewer."""
    from phenotypic.gui._shared.tiles import (
        _load_zarr_layer_rgb,
        _load_zarr_level_rgb,
    )

    _load_zarr_level_rgb.cache_clear()
    _load_zarr_layer_rgb(str(store), "tok-a", "detect_mat", 700)
    _load_zarr_layer_rgb(str(store), "tok-b", "detect_mat", 700)
    assert _load_zarr_level_rgb.cache_info().hits == 0
