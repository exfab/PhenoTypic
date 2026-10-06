"""A single-element index on an array accessor returns the scalar.

``__getitem__`` hands back a read-only view of the image's layer so a caller
cannot write past the setter. A key that selects one element makes numpy return
a scalar rather than a view, and a scalar has no flags to set, so ``gray[0, 0]``
used to raise ``ValueError: Cannot set flags on array scalars``. Slices must
stay read-only views.
"""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image


def _rgb_plate() -> np.ndarray:
    """Pale agar with one brighter, cream-coloured colony."""
    arr = np.empty((16, 16, 3), dtype=np.uint8)
    arr[:] = (150, 140, 120)
    arr[4:12, 4:12] = (230, 220, 190)
    return arr


def _gray_plate() -> np.ndarray:
    return np.full((8, 8), 0.5, dtype=np.float32)


@pytest.mark.parametrize("layer", ["gray", "detect_mat"])
def test_a_pixel_of_a_gray_only_image_is_its_scalar(layer) -> None:
    value = getattr(Image(arr=_gray_plate()), layer)[0, 0]
    assert isinstance(value, np.float32)
    assert value == np.float32(0.5)


@pytest.mark.parametrize("layer", ["gray", "detect_mat"])
def test_a_pixel_of_an_rgb_image_is_its_scalar(layer) -> None:
    img = Image(arr=_rgb_plate())
    value = getattr(img, layer)[5, 5]
    assert isinstance(value, np.float32)
    assert value == getattr(img, layer)[:][5, 5]


def test_an_rgb_channel_value_is_its_scalar() -> None:
    value = Image(arr=_rgb_plate()).rgb[5, 5, 0]
    assert isinstance(value, np.uint8)
    assert value == 230


@pytest.mark.parametrize("space", ["XYZ", "XYZ_D65", "xy", "Lab", "hsv"])
def test_a_colour_space_component_is_its_scalar(space) -> None:
    accessor = getattr(Image(arr=_rgb_plate()).color, space)
    value = accessor[5, 5, 0]
    assert np.isscalar(value)
    assert value == accessor[:][5, 5, 0]


@pytest.mark.parametrize(
    "layer,key",
    [
        ("gray", (slice(0, 1), slice(0, 1))),
        ("gray", 3),
        ("detect_mat", (slice(0, 2), slice(0, 2))),
        ("rgb", (5, 5)),
        ("rgb", (slice(None), slice(None), 0)),
    ],
)
def test_a_slice_is_still_a_read_only_view(layer, key) -> None:
    view = getattr(Image(arr=_rgb_plate()), layer)[key]
    assert isinstance(view, np.ndarray)
    assert not view.flags.writeable


def test_a_colour_space_slice_is_still_a_read_only_view() -> None:
    view = Image(arr=_rgb_plate()).color.Lab[0:2, 0:2]
    assert isinstance(view, np.ndarray)
    assert not view.flags.writeable


def test_reading_the_whole_rgb_layer_leaves_it_writable_through_the_setter() -> None:
    """``rgb._subject_arr`` used to mark the image's own array read-only.

    Every whole-layer reader (``vmax``, ``normed``, ``show``) goes through it, so
    the next ``image.rgb[...] = value`` failed with "assignment destination is
    read-only".
    """
    img = Image(arr=_rgb_plate())
    assert img.rgb.vmax() == 255
    img.rgb[0:2, 0:2] = 10
    assert np.all(img.rgb[:][0:2, 0:2] == 10)
