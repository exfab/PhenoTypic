"""Colour configuration survives every construct-from-image path.

``gamma``, ``illuminant`` and ``_observer`` describe how an image's RGB pixels
are encoded, so a derived image (``copy()``, ``Image(other)``, a crop, a grid
section, a single colony) must keep them. An argument the caller passes
explicitly still wins over the source image's value — including one equal to
the array-input default, which is why the constructor defaults are a sentinel.
"""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import GridImage, Image
from phenotypic.correction import CropImage
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureColor
from phenotypic.sdk_.constants_ import GAMMA_ENCODINGS

_OBSERVER_10 = "CIE 1964 10 Degree Standard Observer"
_OBSERVER_2 = "CIE 1931 2 Degree Standard Observer"


def _rgb_plate() -> np.ndarray:
    """Pale agar with four brighter, cream-coloured colonies and mild noise."""
    rng = np.random.default_rng(7)
    arr = np.empty((96, 96, 3), dtype=np.uint8)
    arr[:] = (150, 140, 120)
    for r, c in ((24, 24), (24, 72), (72, 24), (72, 72)):
        yy, xx = np.ogrid[:96, :96]
        arr[(yy - r) ** 2 + (xx - c) ** 2 <= 81] = (235, 215, 170)
    noise = rng.integers(-6, 7, arr.shape)
    return np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)


def _d50_linear(cls=Image, **kwargs) -> Image:
    return cls(arr=_rgb_plate(), name="plate", illuminant="D50", gamma=None, **kwargs)


def _assert_d50_linear(img: Image) -> None:
    assert img.illuminant == "D50"
    assert img.gamma == GAMMA_ENCODINGS.LINEAR


# --------------------------------------------------------------------- copy


@pytest.mark.parametrize("cls", [Image, GridImage])
def test_copy_keeps_illuminant_and_gamma(cls) -> None:
    _assert_d50_linear(_d50_linear(cls).copy())


def test_copy_keeps_observer() -> None:
    img = _d50_linear()
    img._observer = _OBSERVER_10
    assert img.copy()._observer == _OBSERVER_10


def test_image_of_image_keeps_illuminant_and_gamma() -> None:
    _assert_d50_linear(Image(_d50_linear()))


def test_grid_image_of_image_keeps_illuminant_and_gamma() -> None:
    _assert_d50_linear(GridImage(_d50_linear()))


def test_array_input_still_defaults_to_d65_srgb() -> None:
    img = Image(arr=_rgb_plate())
    assert img.illuminant == "D65"
    assert img.gamma == GAMMA_ENCODINGS.SRGB
    assert img._observer == _OBSERVER_2


@pytest.mark.parametrize("cls", [Image, GridImage])
def test_explicit_arguments_override_the_source_config(cls) -> None:
    """An explicit D65/sRGB is honoured even though it equals the array default."""
    out = cls(_d50_linear(), illuminant="D65", gamma=GAMMA_ENCODINGS.SRGB)
    assert out.illuminant == "D65"
    assert out.gamma == GAMMA_ENCODINGS.SRGB


def test_one_explicit_argument_overrides_only_itself() -> None:
    out = Image(_d50_linear(), illuminant="D65")
    assert out.illuminant == "D65"
    assert out.gamma == GAMMA_ENCODINGS.LINEAR


def test_an_invalid_explicit_illuminant_is_still_refused() -> None:
    with pytest.raises(ValueError, match="illuminant"):
        Image(_d50_linear(), illuminant="A")


# --------------------------------------------------------------- operations


def test_operation_apply_on_a_copy_keeps_the_config() -> None:
    out = OtsuDetector().apply(_d50_linear())
    _assert_d50_linear(out)


def test_measure_color_on_a_copy_equals_measure_color_in_place() -> None:
    """The copy used by ``op.apply`` must measure Lab/XYZ under the source config.

    Before the fix the copy fell back to D65/sRGB and ``ColorLab_L*`` moved by
    ~16 units on a D50/linear image.
    """
    img = _d50_linear()
    OtsuDetector().apply(img, inplace=True)
    assert img.num_objects > 0

    in_place = MeasureColor().measure(img)
    via_copy = MeasureColor().measure(img.copy())

    numeric = in_place.select_dtypes("number").columns
    cols = [
        c for c in numeric
        if str(c).startswith(("ColorLab_", "ColorXYZ_", "Colorxy_"))
    ]
    assert any(str(c).startswith("ColorLab_L*") for c in cols), cols
    np.testing.assert_allclose(
        via_copy[cols].to_numpy(float),
        in_place[cols].to_numpy(float),
        rtol=1e-6,
        equal_nan=True,
    )


@pytest.mark.parametrize("cls", [Image, GridImage])
def test_crop_image_keeps_the_config(cls) -> None:
    out = CropImage(left=4, right=4, top=4, bottom=4).apply(_d50_linear(cls))
    _assert_d50_linear(out)


# -------------------------------------------------------------------- crops


@pytest.mark.parametrize("cls", [Image, GridImage])
def test_crop_keeps_illuminant_and_gamma(cls) -> None:
    _assert_d50_linear(_d50_linear(cls)[0:48, 0:48])


@pytest.mark.parametrize("cls", [Image, GridImage])
def test_crop_keeps_detect_mode(cls) -> None:
    img = _d50_linear(cls)
    img.set_detect_mode("LabL")
    crop = img[0:48, 0:48]
    assert crop.detect_mode == "LabL"
    np.testing.assert_array_equal(crop.detect_mat[:], img.detect_mat[0:48, 0:48])
    # Recomputing on the crop needs its D50/linear config, not only the mode.
    crop.detect_mat.reset()
    np.testing.assert_allclose(crop.detect_mat[:], img.detect_mat[0:48, 0:48], atol=1e-6)


def test_single_colony_keeps_illuminant_and_gamma() -> None:
    img = _d50_linear()
    OtsuDetector().apply(img, inplace=True)
    _assert_d50_linear(img.objects[0])


def test_grid_section_keeps_illuminant_and_gamma(synth_plate_detected) -> None:
    grid = GridImage(synth_plate_detected, illuminant="D50", gamma=None)
    _assert_d50_linear(grid.grid[0])


# ----------------------------------------------------------- grid finder


def test_grid_image_copy_does_not_share_the_grid_finder() -> None:
    original = _d50_linear(GridImage, nrows=8, ncols=12)
    duplicate = original.copy()

    assert duplicate.grid_finder is not original.grid_finder
    duplicate.nrows = 4
    assert original.nrows == 8
    assert duplicate.nrows == 4


def test_grid_image_copy_keeps_the_grid_finder_settings() -> None:
    original = _d50_linear(GridImage, nrows=16, ncols=24)
    duplicate = original.copy()

    assert type(duplicate.grid_finder) is type(original.grid_finder)
    assert (duplicate.nrows, duplicate.ncols) == (16, 24)
