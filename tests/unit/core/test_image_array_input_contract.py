"""What ``Image(arr=...)`` and ``set_image`` do with the array they are given.

- **Adoption.** A ``uint8``/``uint16`` RGB array, and a ``float32``
  single-channel array, is adopted by reference, not copied: images are large
  and a copy costs memory the project deliberately avoids. The caller must not
  mutate the array afterwards, since that would desynchronise ``rgb`` from the
  ``gray`` / ``detect_mat`` derived from it.
- **Bit depth.** An explicit ``bit_depth`` is a constraint every later array
  must fit. An inferred one only describes the array it was inferred from, so
  a new integer array re-infers it rather than being refused for not fitting.
- **Shape first.** An unsupported shape is refused as a shape, before any
  integer narrowing reads (or copies) the values.
"""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import GridImage, Image


def _rgb(dtype, agar: int, colony: int) -> np.ndarray:
    """A 32x32 RGB scan: flat agar with one brighter 12x12 colony."""
    arr = np.full((32, 32, 3), agar, dtype=dtype)
    arr[10:22, 10:22] = colony
    return arr


def _gray(dtype, agar, colony) -> np.ndarray:
    arr = np.full((32, 32), agar, dtype=dtype)
    arr[10:22, 10:22] = colony
    return arr


# ------------------------------------------------------------------ adoption


@pytest.mark.parametrize(
    "dtype,agar,colony", [(np.uint8, 100, 200), (np.uint16, 10000, 30000)]
)
def test_a_uint8_or_uint16_rgb_array_is_adopted_not_copied(dtype, agar, colony) -> None:
    """The contract: the image holds the caller's array; the caller must not mutate it."""
    arr = _rgb(dtype, agar, colony)
    img = Image(arr=arr)
    assert np.shares_memory(img._data.rgb, arr)


def test_a_float32_single_channel_array_is_adopted_not_copied() -> None:
    arr = _gray(np.float32, 0.4, 0.8)
    img = Image(arr=arr)
    assert np.shares_memory(img._data.gray, arr)


def test_whole_layer_reads_do_not_lock_the_adopted_array() -> None:
    arr = _rgb(np.uint8, 100, 200)
    img = Image(arr=arr)
    np.asarray(img.rgb)
    img.rgb.vmax()
    img.rgb.normed()
    assert arr.flags.writeable
    assert img._data.rgb.flags.writeable


def test_a_narrowed_integer_rgb_array_is_a_copy() -> None:
    arr = _rgb(np.int64, 100, 200)
    img = Image(arr=arr)
    assert not np.shares_memory(img._data.rgb, arr)


# ----------------------------------------------------------------- bit depth


@pytest.mark.parametrize("make", [_rgb, _gray], ids=["rgb", "gray"])
def test_set_image_reinfers_an_inferred_bit_depth_for_a_wider_frame(make) -> None:
    img = Image(arr=make(np.int64, 100, 200))
    assert img.bit_depth == 8
    wide = make(np.int64, 100, 1000)
    img.set_image(wide)
    assert img.bit_depth == 16
    if make is _rgb:
        assert img.rgb[:].dtype == np.uint16
        np.testing.assert_array_equal(img.rgb[:], wide)
    else:
        np.testing.assert_allclose(img.gray[:], wide / 65535, rtol=1e-6)


def test_set_image_reinfers_an_inferred_bit_depth_for_a_uint16_frame() -> None:
    """An inferred 8 used to stick to a uint16 frame: rgb uint16, bit_depth 8."""
    img = Image(arr=_rgb(np.uint8, 100, 200))
    img.set_image(_rgb(np.uint16, 10000, 30000))
    assert img.bit_depth == 16


def test_set_image_with_a_float_frame_keeps_an_inferred_bit_depth() -> None:
    """A float array carries no width; it is quantised at the image's bit depth."""
    img = Image(arr=_rgb(np.uint8, 100, 200))
    img.set_image(_rgb(np.float64, 0.25, 0.75))
    assert img.bit_depth == 8
    assert img.rgb[:].dtype == np.uint8


@pytest.mark.parametrize(
    "derive",
    [
        pytest.param(lambda img: img.copy(), id="copy"),
        pytest.param(lambda img: Image(img), id="Image(img)"),
        pytest.param(lambda img: img[0:16, 0:16], id="crop"),
        pytest.param(lambda img: GridImage(img)[0:16, 0:16], id="grid-crop"),
    ],
)
def test_a_derived_image_keeps_its_bit_depth_inferred(derive) -> None:
    derived = derive(Image(arr=_rgb(np.int64, 100, 200)))
    assert derived.bit_depth == 8
    derived.set_image(_rgb(np.int64, 100, 1000))
    assert derived.bit_depth == 16


@pytest.mark.parametrize(
    "derive",
    [
        pytest.param(lambda img: img, id="self"),
        pytest.param(lambda img: img.copy(), id="copy"),
        pytest.param(lambda img: img[0:16, 0:16], id="crop"),
    ],
)
def test_an_explicit_bit_depth_still_refuses_a_wider_frame(derive) -> None:
    img = derive(Image(arr=_rgb(np.int64, 100, 200), bit_depth=8))
    with pytest.raises(ValueError, match="declared 8-bit"):
        img.set_image(_rgb(np.int64, 100, 1000))
    assert img.bit_depth == 8


# --------------------------------------------------------------- shape first


@pytest.mark.parametrize("peak", [70000, -5], ids=["too-wide", "negative"])
def test_a_4d_integer_stack_is_refused_as_a_shape(peak) -> None:
    stack = np.zeros((2, 8, 8, 3), dtype=np.int64)
    stack[0, 0, 0, 0] = peak
    with pytest.raises(ValueError, match="unsupported number of dimensions"):
        Image(arr=stack)


def test_a_2_channel_integer_array_is_refused_as_a_shape() -> None:
    arr = np.zeros((8, 8, 2), dtype=np.int64)
    arr[0, 0, 0] = 70000
    with pytest.raises(ValueError, match="2 channels"):
        Image(arr=arr)
