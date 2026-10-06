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
