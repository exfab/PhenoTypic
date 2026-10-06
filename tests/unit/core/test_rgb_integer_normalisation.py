"""Integer RGB input is narrowed to 8 or 16 bits, and stored RGB gray is restored.

An RGB array whose dtype is not ``uint8``/``uint16`` (an ``int64`` from a
Python literal, an ``int32`` TIFF) carries no bit depth. ``rgb2gray`` divides
integer RGB by its dtype's maximum, so an ``int64`` plate came out ~1e-17. Its
values now choose the width, as they do for a single-channel array: the
explicit ``bit_depth`` when given, else the narrowest of 8 and 16 bits they
fit, and the RGB layer is stored in that unsigned dtype so ``bit_depth``,
``rgb`` and ``gray`` agree. Negative values and values wider than 16 bits are
refused.

A store, HDF file or pickle of an RGB image is stored state, not user input:
its gray layer is restored as written (with a warning when outside
``[0, 1]``, as an old ``PadImage(constant_value=255)`` wrote it) instead of
going through the public gray setter, which refuses such values.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from skimage.color import rgb2gray

from phenotypic import Image
from phenotypic.sdk_.funcs_ import normalize_rgb_bitdepth


def _rgb(dtype, agar: int, colony: int) -> np.ndarray:
    """A 32x32 RGB scan: flat agar with one brighter 12x12 colony, channels offset."""
    arr = np.empty((32, 32, 3), dtype=dtype)
    for c, offset in enumerate((0, 5, 10)):
        arr[..., c] = agar - offset
        arr[10:22, 10:22, c] = colony - offset
    return arr


_WIDE_DTYPES = [np.int16, np.int32, np.uint32, np.int64]
_RANGES = [
    pytest.param(100, 200, 8, id="8-bit-values"),
    pytest.param(10000, 30000, 16, id="16-bit-values"),
]
_UNSIGNED = {8: np.uint8, 16: np.uint16}


# ---------------------------------------------------------------- narrowing


@pytest.mark.parametrize("agar,colony,bits", _RANGES)
@pytest.mark.parametrize("dtype", _WIDE_DTYPES)
def test_other_integer_rgb_takes_the_narrowest_width_its_values_fit(
    dtype, agar, colony, bits
) -> None:
    arr = _rgb(dtype, agar, colony)
    narrowed = arr.astype(_UNSIGNED[bits])
    img = Image(arr=arr)

    assert img.bit_depth == bits
    assert img.rgb[:].dtype == _UNSIGNED[bits]
    np.testing.assert_array_equal(img.rgb[:], arr)
    expected_gray = rgb2gray(narrowed).astype(np.float32)
    np.testing.assert_array_equal(img.gray[:], expected_gray)
    assert 0.0 <= float(img.gray[:].min()) and float(img.gray[:].max()) <= 1.0


@pytest.mark.parametrize("dtype", _WIDE_DTYPES)
def test_other_integer_rgb_raises_no_unknown_dtype_warning(dtype) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Image(arr=_rgb(dtype, 100, 200))


def test_a_python_literal_int64_rgb_is_read_as_8_bit() -> None:
    arr = np.array([[[100, 120, 140], [200, 210, 220]]] * 2)
    assert arr.dtype == np.int64
    img = Image(arr=arr)
    assert img.bit_depth == 8
    np.testing.assert_array_equal(img.gray[:], rgb2gray(arr.astype(np.uint8)).astype(np.float32))


def test_an_explicit_bit_depth_chooses_the_width_for_other_integer_rgb() -> None:
    arr = _rgb(np.int64, 100, 200)
    img = Image(arr=arr, bit_depth=16)
    assert img.bit_depth == 16
    assert img.rgb[:].dtype == np.uint16
    np.testing.assert_array_equal(
        img.gray[:], rgb2gray(arr.astype(np.uint16)).astype(np.float32)
    )


def test_normalize_rgb_bitdepth_agrees_with_the_narrowed_width() -> None:
    arr = _rgb(np.int64, 100, 200)
    img = Image(arr=arr)
    np.testing.assert_allclose(img.rgb.normed(), arr / 255.0, rtol=1e-12)
    np.testing.assert_allclose(normalize_rgb_bitdepth(img.rgb[:]), arr / 255.0, rtol=1e-12)


def test_other_integer_rgba_is_narrowed_before_compositing() -> None:
    rgba = np.dstack([_rgb(np.int64, 100, 200), np.full((32, 32), 255, np.int64)])
    expected = Image(arr=rgba.astype(np.uint8))
    img = Image(arr=rgba)
    assert img.bit_depth == 8
    np.testing.assert_array_equal(img.gray[:], expected.gray[:])


def test_retained_original_of_an_int64_rgb_scan_is_its_values() -> None:
    arr = _rgb(np.int64, 10000, 30000)
    img = Image(arr=arr)
    img._retain_original()
    assert img._original.dtype == np.uint16
    np.testing.assert_array_equal(img._original, arr)


# ------------------------------------------------------------------ refusal


def test_negative_rgb_values_are_refused_as_integers() -> None:
    arr = _rgb(np.int16, 100, 200)
    arr[0, 0, 1] = -5
    with pytest.raises(ValueError, match="negative") as info:
        Image(arr=arr)
    assert "int16" in str(info.value)
    assert "RGB" in str(info.value)
    assert "float" not in str(info.value)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int64])
def test_rgb_values_wider_than_16_bit_are_refused_with_their_range(dtype) -> None:
    arr = _rgb(dtype, 100, 70000)
    with pytest.raises(ValueError, match=rf"{np.dtype(dtype).name}.*\[90, 70000\]"):
        Image(arr=arr)


def test_rgb_values_outside_an_explicit_bit_depth_are_refused() -> None:
    with pytest.raises(ValueError, match="8-bit"):
        Image(arr=_rgb(np.int64, 100, 300), bit_depth=8)


@pytest.mark.parametrize(
    "peak,expected",
    [(255, np.uint8), (256, np.uint16), (65535, np.uint16)],
)
def test_the_width_boundary_is_the_largest_representable_value(peak, expected) -> None:
    """255 is the last 8-bit value and 65535 the last 16-bit one.

    An off-by-one here stores 256 as uint8, where it wraps to 0 -- a colony
    that silently becomes agar.
    """
    arr = np.zeros((4, 4, 3), dtype=np.int64)
    arr[0, 0] = peak
    image = Image(arr=arr)
    assert image.rgb[:].dtype == expected
    assert int(image.rgb[:].max()) == peak


def test_one_past_the_16_bit_range_is_refused() -> None:
    arr = np.zeros((4, 4, 3), dtype=np.int64)
    arr[0, 0] = 65536
    with pytest.raises(ValueError, match=r"\[0, 65536\]"):
        Image(arr=arr)


# ------------------------------------------------- uint8 / uint16 unchanged


@pytest.mark.parametrize(
    "dtype,agar,colony", [(np.uint8, 100, 200), (np.uint16, 10000, 30000)]
)
def test_uint8_and_uint16_rgb_are_byte_identical(dtype, agar, colony) -> None:
    arr = _rgb(dtype, agar, colony)
    img = Image(arr=arr)
    assert img.bit_depth == (8 if dtype is np.uint8 else 16)
    assert img.rgb[:].dtype == dtype
    assert img.rgb[:].tobytes() == arr.tobytes()
    assert img.gray[:].tobytes() == rgb2gray(arr).astype(np.float32).tobytes()


# ------------------------------------------ stored RGB gray is restored as is


def _rgb_with_gray_in_counts() -> tuple[Image, np.ndarray]:
    """An RGB image as an old ``PadImage(constant_value=255)`` left it.

    The RGB border holds the integer fill, and the float gray / detect_mat
    border holds the same raw 255 instead of 1.0.
    """
    arr = _rgb(np.uint8, 100, 200)
    arr[:4] = 255
    img = Image(arr=arr, name="padded")
    gray = img.gray[:].copy()
    gray[:4] = 255.0
    img._data.gray = gray
    img._data.detect_mat = gray.copy()
    return img, gray


def test_load_zarr_restores_an_rgb_store_gray_outside_unit_range(tmp_path) -> None:
    img, gray = _rgb_with_gray_in_counts()
    store = img.save2zarr(tmp_path / "padded.ome.zarr")

    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        loaded = Image.load_zarr(store)
    np.testing.assert_array_equal(loaded.rgb[:], img.rgb[:])
    np.testing.assert_array_equal(loaded.gray[:], gray)
    np.testing.assert_array_equal(loaded.detect_mat[:], gray)


@pytest.mark.parametrize("layout", ["v2_grouped", "v1_flat"])
def test_legacy_hdf_rgb_gray_outside_unit_range_still_migrates(tmp_path, layout) -> None:
    from tests.fixtures.legacy_hdf import _generate

    img, gray = _rgb_with_gray_in_counts()
    write = _generate.write_v2_grouped if layout == "v2_grouped" else _generate.write_v1_flat
    path = write(tmp_path / "img.h5", img)

    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        loaded = Image._load_hdf5_for_migration(path)
    np.testing.assert_array_equal(loaded.rgb[:], img.rgb[:])
    np.testing.assert_array_equal(loaded.gray[:], gray)


def test_an_in_range_rgb_store_loads_without_the_warning(tmp_path) -> None:
    img = Image(arr=_rgb(np.uint8, 100, 200), name="plain")
    store = img.save2zarr(tmp_path / "plain.ome.zarr")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = Image.load_zarr(store)
    assert not [w for w in caught if "outside [0, 1]" in str(w.message)]
    np.testing.assert_array_equal(loaded.gray[:], img.gray[:])


def test_the_public_gray_setter_stays_strict() -> None:
    img, gray = _rgb_with_gray_in_counts()
    target = Image(arr=img.rgb[:])
    with pytest.raises(AssertionError, match="between 0 and 1"):
        target.gray[:] = gray
