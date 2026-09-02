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

#: One real migrated store from the verification run. Read-only, and absent
#: on any machine but the cluster -- every test that touches it skips when
#: it is not there rather than failing.
FIXTURE_STORE = Path(
    "/rhome/anguy344/bigdata_exfab/projects/ucr_029_e_d_Maresca/data/results"
    "/2026-08-11-migration-test/results/7-24-26_redo_full/zarr"
    "/d000466_280_003_2026-07-26_06-34-47.ome.zarr"
)


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
        crop_store_rgb(store, "gray", 20, 20, 16)
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
    crop_store_rgb(store, "detect_mat", 42, 42, 64)
    assert pulled, "no zarr read happened at all"
    assert max(pulled) <= 64 * 64, pulled


def test_crop_of_a_missing_layer_raises_key_error(tmp_path: Path) -> None:
    """A grayscale-only store has no ``rgb``; ``crop_colony`` needs the KeyError."""
    flat = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "f.ome.zarr", layers=("gray",)
    )
    with pytest.raises(KeyError):
        crop_store_rgb(flat, "rgb", 20, 20, 16)


def test_crop_pads_at_the_edge(store: Path) -> None:
    """The shared ``_crop_pil_source`` padding contract survives the port."""
    png = crop_store_rgb(store, "detect_mat", 0, 0, 32)
    crop = _decode_rgb(png)
    assert crop.shape == (32, 32, 3)
    # The top-left quadrant falls outside the image and is pad-filled black.
    assert crop[:16, :16].max() == 0


def test_rgb_crop_returns_channel_last_pixels(store: Path) -> None:
    """``rgb`` is stored ``(C, Y, X)``; a naive window read transposes it."""
    rgb = Image.load_layer_zarr(store, "rgb", level=0)
    png = crop_store_rgb(store, "rgb", 42, 42, 64)
    np.testing.assert_array_equal(_decode_rgb(png), rgb[10:74, 10:74, :3])


# ---------------------------------------------------------------------------
# uint16 -> uint8 display scaling
# ---------------------------------------------------------------------------


def test_a_uint16_ramp_renders_monotonically(tmp_path: Path) -> None:
    """The regression pin for the mod-256 truncation.

    ``arr.astype(np.uint8)`` is a modular reduction: 18175 -> 255 and
    18176 -> 0, so a monotonic source becomes a sawtooth. Any correct
    scaling is monotonic non-decreasing. Measured on a real store, the
    truncated path produced 75 descending steps where scaling produces 0.
    """
    from phenotypic.gui._shared.tiles import scale_to_uint8

    ramp = np.arange(19061, 38171, dtype=np.uint16)
    out = scale_to_uint8(ramp, 20511, 44047).astype(np.int16)

    assert (np.diff(out) >= 0).all(), "scaling must never descend"
    assert out.max() > out.min(), "the ramp must not collapse to one value"


def test_values_above_the_range_clip_rather_than_wrap() -> None:
    """Clipping is what makes a per-image range safe for a crop window."""
    from phenotypic.gui._shared.tiles import scale_to_uint8

    over = np.array([44047 + 5000], dtype=np.uint16)
    under = np.array([20511 - 5000], dtype=np.uint16)

    assert int(scale_to_uint8(over, 20511, 44047)[0]) == 255
    assert int(scale_to_uint8(under, 20511, 44047)[0]) == 0


def test_uint8_stores_are_passed_through_unchanged() -> None:
    """An 8-bit store must not be contrast-stretched by the new path."""
    from phenotypic.gui._shared.tiles import scale_to_uint8

    arr = np.array([0, 7, 128, 255], dtype=np.uint8)
    assert np.array_equal(scale_to_uint8(arr, 0, 255), arr)


def test_the_display_range_works_for_every_layer_not_just_rgb() -> None:
    """``objmap``'s member is ``rgb/labels/objmap``, not ``objmap``.

    Passing a member path where the reader wants a layer name appears to
    work for ``rgb`` -- the series map is the identity -- and raises
    KeyError for ``objmap`` while returning a silent ``(0, 0)`` for
    ``detect_mat`` and ``gray``. That asymmetry is why this asserts on
    every layer the store carries.
    """
    from phenotypic.gui._shared.tiles import image_display_range

    if not FIXTURE_STORE.exists():
        pytest.skip("migration-test fixture absent")

    for layer in ("rgb", "objmap", "detect_mat", "gray"):
        lo, hi = image_display_range(FIXTURE_STORE, layer)
        assert hi > lo, f"{layer} produced a degenerate range ({lo}, {hi})"


@pytest.mark.skipif(
    not FIXTURE_STORE.exists(), reason="migration-test fixture absent"
)
def test_a_brighter_window_renders_brighter_than_a_dim_one() -> None:
    """The pin against a PER-CROP stretch, which is the naive fix.

    Under a per-crop min-max stretch every window is expanded to fill
    0-255, so a bright region and a dim region render with the SAME mean
    and the ordering is destroyed. Under one per-image range the ordering
    survives. ``f(x) == f(x)`` would prove nothing; this asserts a property
    only the correct implementation has.

    Runs on the REAL uint16 store, not the synthetic one. The synthetic
    plate is uint8, so ``scale_to_uint8`` returns it untouched and the
    scaling path this test exists to pin is never entered -- against that
    fixture the test passed before ``scale_to_uint8`` was written at all.

    Candidate windows are located on the SMALLEST pyramid level and their
    centres scaled back up, so the search costs one 196x318 read instead of
    a 3132x5086 one.
    """
    from phenotypic.gui._shared.tiles import _readable_block
    from phenotypic.sdk_.ngff_ import PhenotypicAttr

    size = 256

    block = _readable_block(FIXTURE_STORE)
    top = int(block[PhenotypicAttr.PYRAMID]["levels"]) - 1
    factor = 2**top
    # The level-`top` window whose footprint matches a `size`-px level-0 crop.
    proxy = max(2, size // factor)
    half = proxy // 2

    # ``load_layer_zarr`` returns ``rgb`` CHANNEL-LAST -- ``(H, W, 3)``, the
    # same convention ``_read_store_level`` documents. Collapsing ``axis=0``
    # would average the ROWS and leave a ``(W, 3)`` array whose ``shape[1]``
    # is 3 rather than the image width.
    small = Image.load_layer_zarr(FIXTURE_STORE, "rgb", level=top)
    plane = small.mean(axis=-1) if small.ndim == 3 else small
    height, width = plane.shape

    # Compare WINDOWS, not rows: two extreme ROWS can be a few pixels apart,
    # so their crops overlap almost entirely and the means differ by too
    # little to tell the fix from noise. Centres stay a half-window inside
    # every edge, so the level-0 crops they map to carry no black padding.
    candidates = [
        (rr, cc, float(plane[rr - half : rr + half, cc - half : cc + half].mean()))
        for rr in range(half, height - half, max(1, half))
        for cc in range(half, width - half, max(1, half))
    ]
    bright_rr, bright_cc, bright_level = max(candidates, key=lambda item: item[2])
    dim_rr, dim_cc, dim_level = min(candidates, key=lambda item: item[2])
    assert bright_level > dim_level, "the search found no brightness spread"

    bright = _decode_rgb(
        crop_store_rgb(
            FIXTURE_STORE,
            "rgb",
            float(bright_rr * factor),
            float(bright_cc * factor),
            size,
        )
    ).mean()
    dim = _decode_rgb(
        crop_store_rgb(
            FIXTURE_STORE,
            "rgb",
            float(dim_rr * factor),
            float(dim_cc * factor),
            size,
        )
    ).mean()

    assert bright > dim, (
        f"a brighter region rendered no brighter ({bright:.1f} vs {dim:.1f}) "
        "-- the range is being taken from the crop, not the image"
    )


def test_a_uint8_crop_never_pays_for_a_display_range(
    store: Path, monkeypatch
) -> None:
    """An 8-bit crop must not read a pyramid level it cannot use.

    ``scale_to_uint8`` returns a uint8 array untouched whatever range it is
    handed, so resolving the range first is pure waste -- and on a store
    below ``PYRAMID_STOP_PX`` the "smallest level" IS level 0, making that
    waste a FULL-LAYER read on every crop. Every store this project wrote
    before the OME-Zarr migration is uint8, so this is the common path, not
    an edge case.

    Asserted on elements pulled out of zarr, not on a call count, because
    the cost being guarded is the read.
    """
    import zarr

    size = 64
    pulled: list[int] = []
    real_getitem = zarr.Array.__getitem__

    def _counting(self, selection):
        out = real_getitem(self, selection)
        pulled.append(int(np.asarray(out).size))
        return out

    monkeypatch.setattr(zarr.Array, "__getitem__", _counting)
    crop_store_rgb(store, "rgb", 42, 42, size)

    assert pulled, "no zarr read happened at all"
    assert max(pulled) <= 3 * size * size, (
        f"a uint8 crop pulled {max(pulled)} elements for a {size}x{size} "
        f"window -- the display range is being read and then discarded"
    )


@pytest.mark.skipif(
    not FIXTURE_STORE.exists(), reason="migration-test fixture absent"
)
def test_a_uint16_crop_still_resolves_the_display_range(monkeypatch) -> None:
    """The other half, and the one that stops the fix being optimised away.

    The uint8 skip above is a pure win, and the tempting next step is to
    drop the range read altogether. On a uint16 store that reinstates the
    mod-256 truncation exactly. So this pins that a 16-bit crop DOES
    consult the range -- without it, the sibling test above is a
    one-directional guard that a regression can satisfy by doing nothing.
    """
    from phenotypic.gui._shared import tiles

    seen: list[tuple[int, int]] = []
    real = tiles.image_display_range

    def _recording(store_path, layer):
        result = real(store_path, layer)
        seen.append(result)
        return result

    monkeypatch.setattr(tiles, "image_display_range", _recording)
    crop_store_rgb(
        FIXTURE_STORE,
        "rgb",
        1783.158135,
        342.748203,
        64,
    )

    assert seen, "a uint16 crop resolved no display range -- it truncates"
    lo, hi = seen[0]
    assert hi > lo, f"degenerate range {(lo, hi)} would render solid black"
    assert hi > 255, (
        f"range {(lo, hi)} fits in 8 bits, so this fixture is not the "
        "uint16 store the test needs"
    )


@pytest.mark.skipif(
    not FIXTURE_STORE.exists(), reason="migration-test fixture absent"
)
def test_a_real_colony_crop_is_smooth_not_noise() -> None:
    """The end-to-end pin: a crop of a real colony must read as an image.

    Object 24 is the largest in this image (9,182 px) at (1783.2, 342.7).
    Truncation gave a mean horizontal neighbour delta of 85.3; smooth
    imagery reads 0-5. The threshold is 20 -- far from both, so this is
    not a delicate test.
    """
    png = crop_store_rgb(
        FIXTURE_STORE,
        "rgb",
        1783.158135,
        342.748203,
        256,
    )
    a = _decode_rgb(png).astype(np.int16)
    delta = np.abs(np.diff(a[:, :, 0], axis=1))

    assert delta.mean() < 20.0, f"crop reads as noise: mean delta {delta.mean():.1f}"
    assert (delta > 100).mean() < 0.01


# ---------------------------------------------------------------------------
# objmap contour compositing
# ---------------------------------------------------------------------------


def test_contours_draw_a_boundary_around_the_focal_label() -> None:
    """Boundaries are drawn for the focal label and dimmed for neighbours."""
    from phenotypic.gui._shared.tiles import composite_contours

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[8:16, 8:16] = 7  # focal
    labels[20:28, 20:28] = 9  # neighbour

    out = composite_contours(rgb, labels, focal=7)

    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert (out != 0).any(), "no contour was drawn"
    assert not np.array_equal(out, rgb)


def test_contours_are_a_no_op_when_no_label_is_present() -> None:
    from phenotypic.gui._shared.tiles import composite_contours

    rgb = np.full((16, 16, 3), 40, dtype=np.uint8)
    labels = np.zeros((16, 16), dtype=np.uint16)
    assert np.array_equal(composite_contours(rgb, labels, focal=3), rgb)


def test_the_focal_and_neighbour_contours_are_different_colours() -> None:
    """A crowded crop must say WHICH colony the point refers to.

    One tint for every boundary would satisfy the two tests above while
    leaving the focal colony indistinguishable from its neighbours, which
    is the whole reason the crop is drawn.
    """
    from phenotypic.gui._design import OI_ORANGE, OI_SKY
    from phenotypic.gui._shared.tiles import composite_contours

    def _rgb(hex_: str) -> tuple[int, int, int]:
        h = hex_.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[8:16, 8:16] = 7
    labels[20:28, 20:28] = 9

    out = composite_contours(rgb, labels, focal=7)
    painted = {tuple(px) for px in out.reshape(-1, 3) if tuple(px) != (0, 0, 0)}

    assert _rgb(OI_ORANGE) in painted, "focal boundary missing"
    assert _rgb(OI_SKY) in painted, "neighbour boundary missing"


def test_contours_are_skipped_rather_than_fatal_on_a_label_less_store(
    tmp_path: Path,
) -> None:
    """A store with no ``objmap`` must still serve the plain rgb crop.

    Letting the objmap ``KeyError`` escape would reach ``crop_colony``'s
    missing-layer handler, which falls back to the baked overlay -- so a
    request for contours on a label-less store would silently change the
    PIXEL SOURCE rather than just omitting the boundaries.
    """
    flat = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "nolabels.ome.zarr", layers=("rgb",)
    )
    with_contours = crop_store_rgb(flat, "rgb", 42, 42, 64, contours=1)
    without = crop_store_rgb(flat, "rgb", 42, 42, 64)
    assert _decode_rgb(with_contours).tolist() == _decode_rgb(without).tolist()


def test_a_real_store_crop_gains_contours_only_when_asked(store: Path) -> None:
    """The default is off, and the flag is what turns it on.

    Both halves matter. Asserting only that ``contours=N`` differs would
    pass against an implementation that always draws them -- which would
    change the Colony grid's appearance and owe a ``FEATURES.md`` update.
    Asserting only the default would pass against one that never draws.

    Centred on a REAL object rather than an arbitrary coordinate: an empty
    corner of the plate has no boundary to draw, so a crop there would read
    as "contours are broken" when they are merely absent.
    """
    objmap = Image.load_layer_zarr(store, "objmap", level=0)
    focal = int(objmap.max())
    assert focal > 0, "the fixture store carries no labelled objects"
    rows, cols = np.nonzero(objmap == focal)
    center_rr, center_cc = float(rows.mean()), float(cols.mean())

    plain = _decode_rgb(
        crop_store_rgb(store, "rgb", center_rr, center_cc, 64)
    )
    drawn = _decode_rgb(
        crop_store_rgb(store, "rgb", center_rr, center_cc, 64, contours=focal)
    )

    assert not np.array_equal(plain, drawn), "contours=N drew nothing"
    default = _decode_rgb(
        crop_store_rgb(store, "rgb", center_rr, center_cc, 64)
    )
    np.testing.assert_array_equal(plain, default)
