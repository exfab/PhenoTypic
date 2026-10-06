"""Single-channel inputs honour the float ``[0, 1]`` gray / detect_mat contract.

An integer 2-D (or ``H x W x 1``) array is divided by its dtype's maximum at
construction -- the divisor ``rgb2gray`` already applies to RGB input -- so
``gray`` and ``detect_mat`` are float32 in ``[0, 1]`` whatever the scan's
channel count. A 2-D float array carries no scale, so one outside ``[0, 1]`` is
refused, as a 3-D float array already was. The retained ``original`` stays the
decoded integers, and stores written before the change (integer gray and
detect_mat) are normalised on load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import skimage.io as skio
import tifffile
import zarr

from phenotypic import GridImage, Image

_MAX = {np.uint8: 255, np.uint16: 65535}


def _plate(dtype=np.uint8, agar=100, colony=200) -> np.ndarray:
    """A 64x64 single-channel scan: flat agar with one bright 20x20 colony."""
    arr = np.full((64, 64), agar, dtype=dtype)
    arr[20:40, 20:40] = colony
    return arr


def _unit(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / np.float32(np.iinfo(arr.dtype).max)


def _assert_unit_float(img: Image, expected: np.ndarray) -> None:
    for layer in (img.gray[:], img.detect_mat[:]):
        assert layer.dtype == np.float32
        assert float(layer.min()) >= 0.0 and float(layer.max()) <= 1.0
        np.testing.assert_allclose(layer, expected, rtol=0, atol=1e-7)


def _seeded(arr: np.ndarray) -> Image:
    img = Image(arr=arr)
    objmap = np.zeros(arr.shape[:2], dtype=np.uint16)
    objmap[20:40, 20:40] = 1
    img.objmap[:] = objmap
    return img


# --------------------------------------------------------------- construction


@pytest.mark.parametrize(
    "dtype,agar,colony", [(np.uint8, 100, 200), (np.uint16, 10000, 30000)]
)
@pytest.mark.parametrize("trailing_channel", [False, True], ids=["HxW", "HxWx1"])
def test_integer_single_channel_is_normalised_by_the_dtype_max(
    dtype, agar, colony, trailing_channel
) -> None:
    arr = _plate(dtype, agar, colony)
    img = Image(arr=arr[..., None] if trailing_channel else arr)

    assert img.bit_depth == (8 if dtype is np.uint8 else 16)
    assert img.rgb.isempty()
    _assert_unit_float(img, _unit(arr))


def test_explicit_bit_depth_is_kept_as_a_label() -> None:
    img = Image(arr=_plate(np.uint16, 100, 200), bit_depth=8)
    assert img.bit_depth == 8
    _assert_unit_float(img, _unit(_plate(np.uint16, 100, 200)))


def test_in_range_float_single_channel_is_unchanged() -> None:
    arr = _plate(np.float32, 0.4, 0.8)
    _assert_unit_float(Image(arr=arr), arr)


@pytest.mark.parametrize("agar,colony", [(2.0, 200.0), (-0.5, 0.5)])
def test_out_of_range_float_single_channel_is_refused(agar, colony) -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        Image(arr=_plate(np.float32, agar, colony))


def test_rgb_input_is_unchanged() -> None:
    from skimage.color import rgb2gray

    rgb = np.dstack([_plate(), _plate(agar=90, colony=210), _plate(agar=80)])
    img = Image(arr=rgb)
    np.testing.assert_array_equal(img.rgb[:], rgb)
    np.testing.assert_allclose(img.gray[:], rgb2gray(rgb).astype(np.float32), atol=0)


def test_gray_setter_keeps_fractional_values() -> None:
    img = Image(arr=_plate())
    img.gray[0:2, 0:2] = 0.5
    assert float(img.gray[:][0, 0]) == pytest.approx(0.5)


@pytest.mark.parametrize("cls", [Image, GridImage])
def test_a_crop_keeps_the_bit_depth(cls) -> None:
    crop = cls(arr=_plate())[0:32, 0:32]
    assert crop.bit_depth == 8
    _assert_unit_float(crop, _unit(_plate())[0:32, 0:32])


# ---------------------------------------------------------------------- imread


@pytest.mark.parametrize(
    "filename,arr",
    [
        ("g8.png", _plate()),
        ("g16.png", _plate(np.uint16, 10000, 30000)),
        ("g16.tif", _plate(np.uint16, 10000, 30000)),
        ("g8.jpg", _plate()),
    ],
)
def test_imread_of_a_single_channel_file_is_normalised(tmp_path, filename, arr) -> None:
    path = tmp_path / filename
    if path.suffix == ".tif":
        tifffile.imwrite(path, arr)
    else:
        skio.imsave(path, arr, check_contrast=False)
    decoded = skio.imread(path)  # JPEG is lossy: compare to what was decoded
    assert decoded.ndim == 2 and decoded.dtype == arr.dtype

    _assert_unit_float(Image.imread(path), _unit(decoded))


def test_imread_of_an_out_of_range_float_tiff_is_refused(tmp_path) -> None:
    path = tmp_path / "gf.tif"
    tifffile.imwrite(path, _plate(np.float32, 2.0, 1000.0))
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        Image.imread(path)


# -------------------------------------------------------------------- original


@pytest.mark.parametrize("dtype,agar,colony", [(np.uint8, 3, 254), (np.uint16, 7, 65534)])
def test_retained_original_is_the_decoded_integers(dtype, agar, colony) -> None:
    arr = _plate(dtype, agar, colony)
    img = Image(arr=arr)
    img._retain_original()
    assert img._original.dtype == dtype
    np.testing.assert_array_equal(img._original, arr)


def test_retained_original_is_exact_for_every_uint16_value() -> None:
    arr = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    img = Image(arr=arr)
    img._retain_original()
    np.testing.assert_array_equal(img._original, arr)


def test_retained_original_uses_the_source_dtype_not_the_bit_depth_label() -> None:
    arr = _plate(np.uint16, 100, 200)
    img = Image(arr=arr, bit_depth=8)
    img._retain_original()
    assert img._original.dtype == np.uint16
    np.testing.assert_array_equal(img._original, arr)


def test_retained_original_of_a_copy_is_the_decoded_integers() -> None:
    arr = _plate()
    duplicate = Image(arr=arr).copy()
    duplicate._retain_original()
    assert duplicate._original.dtype == np.uint8
    np.testing.assert_array_equal(duplicate._original, arr)


def test_retained_original_of_a_float_input_stays_float() -> None:
    arr = _plate(np.float32, 0.4, 0.8)
    img = Image(arr=arr)
    img._retain_original()
    assert img._original.dtype == np.float32
    np.testing.assert_array_equal(img._original, arr)


def test_store_original_series_stays_integer(tmp_path) -> None:
    arr = _plate()
    img = Image(arr=arr, name="g8")
    img._retain_original()
    store = img.save2zarr(tmp_path / "g8.ome.zarr")
    original = np.asarray(zarr.open_array(store=str(store / "original" / "0"), mode="r"))
    assert original.dtype == np.uint8
    np.testing.assert_array_equal(original, arr)


# ------------------------------------------------------------- legacy stores


def _legacy_integer_image(arr: np.ndarray, name: str = "legacy") -> Image:
    """What a single-channel image looked like before normalisation: raw integers."""
    img = Image(arr=arr, name=name)
    img._data.gray = arr.copy()
    img._data.detect_mat = arr.copy()
    assert img._data.gray.dtype == arr.dtype
    return img


@pytest.mark.parametrize("dtype,agar,colony", [(np.uint8, 100, 200), (np.uint16, 10000, 30000)])
def test_load_zarr_normalises_a_legacy_integer_store(tmp_path, dtype, agar, colony) -> None:
    arr = _plate(dtype, agar, colony)
    store = _legacy_integer_image(arr).save2zarr(tmp_path / "legacy.ome.zarr")
    stored = np.asarray(zarr.open_array(store=str(store / "detect_mat" / "0"), mode="r"))
    assert stored.dtype == dtype, "fixture did not write an integer detect_mat"

    loaded = Image.load_zarr(store)
    assert loaded.bit_depth == (8 if dtype is np.uint8 else 16)
    _assert_unit_float(loaded, _unit(arr))


def test_imread_of_a_legacy_integer_store_is_normalised(tmp_path) -> None:
    arr = _plate()
    store = _legacy_integer_image(arr).save2zarr(tmp_path / "legacy.ome.zarr")
    plain = Image.imread(store)
    np.testing.assert_allclose(plain.gray[:], _unit(arr), atol=1e-7)
    assert plain.gray[:].dtype == np.float32


def _write_v1_flat_single_channel(path: Path, arr: np.ndarray) -> Path:
    """The legacy flat layout for a gray-only image (no ``rgb`` dataset).

    ``_generate.write_v1_flat`` always writes ``rgb``, which a gray-only
    image does not have.
    """
    import h5py

    with h5py.File(path, mode="w") as handle:
        handle.attrs["schema_version"] = 1
        handle.attrs["bit_depth"] = 8
        handle.create_dataset("gray", data=arr)
        handle.create_dataset("detect_mat", data=arr)
        handle["detect_mat"].attrs["detect_mode"] = "gray"
        handle.create_dataset("objmap", data=np.zeros(arr.shape, dtype=np.uint16))
    return path


@pytest.mark.parametrize("layout", ["v2_grouped", "v1_flat"])
def test_legacy_hdf_with_integer_layers_is_normalised(tmp_path, layout) -> None:
    from tests.fixtures.legacy_hdf import _generate

    arr = _plate()
    if layout == "v2_grouped":
        path = _generate.write_v2_grouped(
            tmp_path / "img.h5", _legacy_integer_image(arr, name="img")
        )
    else:
        path = _write_v1_flat_single_channel(tmp_path / "img.h5", arr)

    loaded = Image._load_hdf5_for_migration(path)
    _assert_unit_float(loaded, _unit(arr))


def test_load_pickle_normalises_a_legacy_integer_detect_mat(tmp_path) -> None:
    arr = _plate()
    path = tmp_path / "legacy.pkl"
    _legacy_integer_image(arr).save2pickle(str(path))

    loaded = Image.load_pickle(path)
    _assert_unit_float(loaded, _unit(arr))


# ------------------------------------------------- consumers that assume [0, 1]


@pytest.mark.parametrize("detector", ["OtsuDetector", "UserThreshold"])
def test_detectors_find_the_colony_on_an_integer_scan(detector) -> None:
    import phenotypic.detect as detect

    out = getattr(detect, detector)().apply(Image(arr=_plate()))
    assert out.num_objects == 1
    assert float(out.objmask[:].astype(bool).mean()) == pytest.approx(400 / 4096)


def test_subtract_gaussian_is_not_truncated_on_an_integer_scan() -> None:
    from phenotypic.enhance import SubtractGaussian

    out = SubtractGaussian(sigma=10).apply(Image(arr=_plate()))
    assert out.detect_mat[:].dtype == np.float32
    assert np.unique(out.detect_mat[:]).size > 2


@pytest.mark.parametrize("dtype,agar,colony", [(np.uint8, 100, 200), (np.uint16, 10000, 30000)])
def test_measure_intensity_matches_the_equivalent_float_scan(dtype, agar, colony) -> None:
    from phenotypic.measure import MeasureIntensity

    arr = _plate(dtype, agar, colony)
    as_int = MeasureIntensity().measure(_seeded(arr))
    as_float = MeasureIntensity().measure(_seeded(_unit(arr)))
    cols = as_float.select_dtypes("number").columns
    np.testing.assert_allclose(
        as_int[cols].to_numpy(float), as_float[cols].to_numpy(float), rtol=1e-6
    )


def test_measure_texture_runs_on_an_integer_scan() -> None:
    from phenotypic.measure import MeasureTexture

    as_int = MeasureTexture().measure(_seeded(_plate()))
    as_float = MeasureTexture().measure(_seeded(_unit(_plate())))
    cols = as_float.select_dtypes("number").columns
    np.testing.assert_allclose(
        as_int[cols].to_numpy(float), as_float[cols].to_numpy(float), rtol=1e-6,
        equal_nan=True,
    )


def test_integrated_intensity_is_in_normalised_units() -> None:
    from phenotypic.measure import MeasureSize

    df = MeasureSize().measure(_seeded(_plate()))
    col = next(c for c in df.columns if "IntegratedIntensity" in str(c))
    assert float(df[col].iloc[0]) == pytest.approx(400 * 200 / 255, rel=1e-5)


# ------------------------------------------- other integer dtypes, bool, NaN


@pytest.mark.parametrize(
    "dtype,agar,colony,bits",
    [
        (np.int64, 100, 200, 8),
        (np.int32, 10000, 30000, 16),
        (np.uint32, 10000, 30000, 16),
        (np.int16, 100, 200, 8),
        (np.int8, 10, 100, 8),
        (np.uint64, 300, 65535, 16),
    ],
)
def test_other_integer_dtypes_take_the_narrowest_width_their_values_fit(
    dtype, agar, colony, bits
) -> None:
    """Not their own dtype's maximum: an int64 plate divided by 2**63 - 1 is ~1e-17."""
    arr = _plate(dtype, agar, colony)
    img = Image(arr=arr)
    full_scale = 2**bits - 1
    assert img.bit_depth == bits
    _assert_unit_float(img, arr.astype(np.float32) / np.float32(full_scale))


def test_a_python_literal_int64_plate_is_read_as_8_bit() -> None:
    arr = np.array([[100, 200], [150, 250]])
    assert arr.dtype == np.int64
    img = Image(arr=arr)
    assert img.bit_depth == 8
    _assert_unit_float(img, arr.astype(np.float32) / 255)


def test_other_integer_dtypes_raise_no_unknown_dtype_warning() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Image(arr=_plate(np.int64, 100, 200))


def test_an_explicit_bit_depth_chooses_the_width_for_other_integer_dtypes() -> None:
    arr = _plate(np.int64, 100, 200)
    img = Image(arr=arr, bit_depth=16)
    assert img.bit_depth == 16
    _assert_unit_float(img, arr.astype(np.float32) / 65535)


def test_negative_signed_integers_are_refused_as_integers() -> None:
    arr = _plate(np.int16, 100, 200)
    arr[0, 0] = -5
    with pytest.raises(ValueError, match="negative") as info:
        Image(arr=arr)
    assert "int16" in str(info.value)
    assert "float" not in str(info.value)


def test_integers_wider_than_16_bit_are_refused_with_their_range() -> None:
    with pytest.raises(ValueError, match=r"int32.*\[100, 70000\]"):
        Image(arr=_plate(np.int32, 100, 70000))


def test_integers_outside_an_explicit_bit_depth_are_refused() -> None:
    with pytest.raises(ValueError, match="8-bit"):
        Image(arr=_plate(np.int64, 100, 300), bit_depth=8)


def test_retained_original_of_a_uint32_scan_is_its_16_bit_values() -> None:
    arr = _plate(np.uint32, 7, 65534)
    img = Image(arr=arr)
    img._retain_original()
    assert img._original.dtype == np.uint16
    np.testing.assert_array_equal(img._original, arr)


def test_bool_single_channel_becomes_float_zero_one() -> None:
    import warnings

    mask = _plate(np.uint8, 0, 1).astype(bool)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        img = Image(arr=mask)
    assert img.bit_depth == 8
    _assert_unit_float(img, mask.astype(np.float32))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_single_channel_float_is_refused(bad) -> None:
    arr = _plate(np.float32, 0.4, 0.8)
    arr[3, 3] = bad
    with pytest.raises(ValueError, match="non-finite"):
        Image(arr=arr)


def test_set_image_with_float_resets_the_integer_source_dtype() -> None:
    """A stale uint8 source dtype would rebuild float data as 'decoded integers'."""
    img = Image(arr=_plate())
    replacement = _plate(np.float32, 0.25, 0.75)
    img.set_image(replacement)
    img._retain_original()
    assert img._original.dtype == np.float32
    np.testing.assert_array_equal(img._original, replacement)


# ------------------------------------- stored and derived state is restored


def _with_float_gray(img: Image, gray: np.ndarray) -> Image:
    """Stand in for a store written before the [0, 1] refusal: float gray in counts."""
    img._data.gray = gray.copy()
    img._data.detect_mat = gray.copy()
    return img


@pytest.mark.parametrize("dtype,fill", [(np.uint8, 1.0), (np.uint16, 255 / 65535)])
def test_pad_image_fills_float_layers_on_the_unit_scale(dtype, fill) -> None:
    from phenotypic.correction import PadImage

    arr = _plate(dtype, 100, 200) if dtype is np.uint8 else _plate(dtype, 10000, 30000)
    padded = PadImage(left=4, right=4, top=4, bottom=4, constant_value=255).apply(
        Image(arr=arr)
    )
    for layer in (padded.gray[:], padded.detect_mat[:]):
        np.testing.assert_allclose(layer[0:4, :], fill, rtol=1e-6)
        assert float(layer.max()) <= 1.0


def test_pad_image_keeps_the_integer_fill_on_rgb_and_scales_gray() -> None:
    from phenotypic.correction import PadImage

    rgb = np.dstack([_plate()] * 3)
    padded = PadImage(left=4, right=4, top=4, bottom=4, constant_value=255).apply(
        Image(arr=rgb)
    )
    assert np.all(padded.rgb[:][0:4] == 255)
    np.testing.assert_allclose(padded.gray[:][0:4], 1.0, rtol=1e-6)


@pytest.mark.parametrize("cls", [Image, GridImage])
def test_gray_only_pad_255_then_crop_and_reload(tmp_path, cls) -> None:
    from phenotypic.correction import PadImage

    padded = PadImage(left=4, right=4, top=4, bottom=4, constant_value=255).apply(
        cls(arr=_plate(), name="padded")
    )
    crop = padded[0:32, 0:32]
    np.testing.assert_allclose(crop.gray[:], padded.gray[:][0:32, 0:32])

    loaded = Image.load_zarr(padded.save2zarr(tmp_path / "padded.ome.zarr"))
    np.testing.assert_array_equal(loaded.gray[:], padded.gray[:])
    np.testing.assert_array_equal(loaded.detect_mat[:], padded.detect_mat[:])


@pytest.mark.parametrize("cls", [Image, GridImage])
def test_a_crop_takes_out_of_range_gray_as_it_is(cls) -> None:
    """A crop copies derived state; it does not re-validate it as user input."""
    counts = _plate(np.float32, 100.0, 200.0)
    img = _with_float_gray(cls(arr=_plate()), counts)
    crop = img[0:32, 0:32]
    np.testing.assert_array_equal(crop.gray[:], counts[0:32, 0:32])
    assert not np.shares_memory(crop._data.gray, img._data.gray)


def test_load_zarr_loads_a_legacy_float_store_in_counts_as_stored(tmp_path) -> None:
    counts = _plate(np.float32, 100.0, 200.0)
    store = _with_float_gray(Image(arr=_plate(), name="counts"), counts).save2zarr(
        tmp_path / "counts.ome.zarr"
    )
    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        loaded = Image.load_zarr(store)
    np.testing.assert_array_equal(loaded.gray[:], counts)
    np.testing.assert_array_equal(loaded.detect_mat[:], counts)


def test_legacy_hdf_with_float_gray_in_counts_still_migrates(tmp_path) -> None:
    from tests.fixtures.legacy_hdf import _generate

    counts = _plate(np.float32, 100.0, 200.0)
    path = _generate.write_v2_grouped(
        tmp_path / "img.h5", _with_float_gray(Image(arr=_plate(), name="img"), counts)
    )
    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        loaded = Image._load_hdf5_for_migration(path)
    np.testing.assert_array_equal(loaded.gray[:], counts)


def test_load_pickle_loads_float_gray_in_counts_as_stored(tmp_path) -> None:
    counts = _plate(np.float32, 100.0, 200.0)
    path = tmp_path / "counts.pkl"
    _with_float_gray(Image(arr=_plate()), counts).save2pickle(str(path))
    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        loaded = Image.load_pickle(path)
    np.testing.assert_array_equal(loaded.gray[:], counts)


def test_load_zarr_restores_the_integer_source_dtype(tmp_path) -> None:
    arr = _plate()
    img = Image(arr=arr, name="g8")
    img._retain_original()
    loaded = Image.load_zarr(img.save2zarr(tmp_path / "g8.ome.zarr"))
    loaded._retain_original()
    assert loaded._original.dtype == np.uint8
    np.testing.assert_array_equal(loaded._original, arr)


@pytest.mark.parametrize("cls", [Image, GridImage])
def test_a_crop_keeps_the_integer_source_dtype(cls) -> None:
    arr = _plate()
    crop = cls(arr=arr)[0:32, 0:32]
    crop._retain_original()
    assert crop._original.dtype == np.uint8
    np.testing.assert_array_equal(crop._original, arr[0:32, 0:32])


def test_load_pickle_keeps_the_colour_configuration(tmp_path) -> None:
    from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS

    rng = np.random.default_rng(3)
    rgb = rng.integers(30, 220, size=(32, 32, 3), dtype=np.uint8)
    img = Image(arr=rgb, illuminant="D50", gamma=None)
    img._observer = "CIE 1964 10 Degree Standard Observer"
    img.set_detect_mode("LabL")
    path = tmp_path / "d50.pkl"
    img.save2pickle(str(path))

    loaded = Image.load_pickle(path)
    assert loaded.illuminant == "D50"
    assert loaded.gamma == GAMMA_ENCODINGS.LINEAR
    assert loaded._observer == "CIE 1964 10 Degree Standard Observer"
    np.testing.assert_allclose(loaded.detect_mat[:], img.detect_mat[:], atol=1e-6)
