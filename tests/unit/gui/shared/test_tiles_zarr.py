"""Colony crops are windowed level-0 reads of the store.

`select_pyramid_level` and the `_load_zarr_*` full-layer loaders that used
to be exercised here are gone: their only caller chain was the results
Plate's DZI path, and the Plate now reads store chunks in the browser, where
deck.gl picks the level per frame. What the writer RECORDED is still pinned,
in ``tests/unit/gui/results_viewer/test_level_selection.py``; what the
browser CHOOSES is phase 5's check. A test pinning a function no path calls
reads as maintained, so those went with the function.

What survives here is the crop path, which never selected a level:
``crop_store_rgb`` is a windowed read of level 0, and it is the one place
``StoreUnreadable`` still has to escape rather than degrade to plausible
pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.gui._shared.tiles import StoreUnreadable, crop_store_rgb


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    """One promoted store for the whole module: ``save2zarr`` is ~2 s."""
    tmp_path = tmp_path_factory.mktemp("store")
    return Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")


def test_a_future_store_is_refused_rather_than_decoded(tmp_path: Path) -> None:
    """``store_schema_version`` is gated by VALUE, and the GUI must say so.

    ``require_readable_store`` raises a bare ``ValueError`` per read. Left
    bare it reaches the crop route's blanket handler and the user is told
    "internal error", with the real, actionable message only in a log. It is
    re-raised as ``StoreUnreadable`` so both the crop route and the byte
    route answer 422 and pass the message through.
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
        crop_store_rgb(store, "gray", 20, 20, 16, 0)
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
