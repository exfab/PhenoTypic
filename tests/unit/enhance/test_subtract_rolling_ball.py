"""``SubtractRollingBall`` flattens the background and keeps the colonies.

scikit-image's ``rolling_ball`` treats intensity as height and sizes its ball
in the image's own units: ``radius`` pixels wide and ``radius`` intensity
units tall. ``detect_mat`` spans ``[0, 1]``, so before 0.21.0 a radius-100
ball was 100 whole intensity ranges tall; its surface falls by a colony's
height (~0.04) within about 3 px of the apex, so the ball fitted inside every
colony and the "background" it traced was the colonies themselves. On the
bundled 72 h plate a colony's median height above background went from 0.208
to 0.000.

scikit-image's gallery (``plot_rolling_ball.py``, "Image Datatypes")
describes the same problem and scales the ellipsoid by ``/255`` for a
``[0, 1]`` image. Its example doubles the intensity axis (``2 r / 255``);
this operation uses ``r / 255``, which reproduces ``ball_kernel(r)`` exactly
on 8-bit values, as ``test_the_ball_equals_the_textbook_ball_on_8_bit_values``
pins.

The known answer below: flat-topped disks of known height on a tilted plane.
Under a plane a ball touches at one fixed offset, so the plane's residual is
one constant everywhere the ball's tangent point stays inside the image; each
disk must then keep its height above that constant, less the rise of the ball
into a disk of its radius, ``r - sqrt(r**2 - d**2)`` grey levels.
"""

from __future__ import annotations

import numpy as np
import pytest
from skimage.restoration import rolling_ball

from phenotypic import Image
from phenotypic.enhance import SubtractRollingBall

HEIGHT, WIDTH = 160, 200
RADIUS = 30
DISK_RADIUS = 6
#: (row, col, height) of each flat-topped disk.
DISKS = ((50, 60, 0.10), (80, 110, 0.20), (110, 150, 0.30))
#: The plane: 0.2 + 0.001 per column + 0.0005 per row, all inside [0, 1].
RAMP = (0.2, 0.0005, 0.001)

#: How far the ball can rise into a flat-topped disk of DISK_RADIUS px: its
#: surface at that offset from the apex, in detect_mat units (0.0024).
MAX_RISE = (RADIUS - np.sqrt(RADIUS**2 - DISK_RADIUS**2)) / 255


def _disk_mask(row: int, col: int) -> np.ndarray:
    rr, cc = np.ogrid[:HEIGHT, :WIDTH]
    return (rr - row) ** 2 + (cc - col) ** 2 <= DISK_RADIUS**2


def _plate() -> tuple[np.ndarray, np.ndarray]:
    """The test landscape, and a mask of pixels far from disks and edges."""
    offset, per_row, per_col = RAMP
    rr, cc = np.mgrid[:HEIGHT, :WIDTH].astype(np.float32)
    landscape = (offset + per_row * rr + per_col * cc).astype(np.float32)
    near_disk = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for row, col, height in DISKS:
        disk = _disk_mask(row, col)
        landscape[disk] += height
        reach = 2 * DISK_RADIUS
        near_disk[row - reach: row + reach + 1, col - reach: col + reach + 1] = True
    interior = np.zeros((HEIGHT, WIDTH), dtype=bool)
    interior[RADIUS:-RADIUS, RADIUS:-RADIUS] = True
    return landscape, interior & ~near_disk


def _subtracted(landscape: np.ndarray, **params) -> np.ndarray:
    image = Image(landscape.copy())
    SubtractRollingBall(**params).apply(image, inplace=True)
    return image.detect_mat[:]


@pytest.fixture(scope="module")
def plate():
    return _plate()


@pytest.fixture(scope="module")
def result(plate):
    landscape, _ = plate
    return _subtracted(landscape, radius=RADIUS)


def test_the_plane_is_flattened_to_one_constant(plate, result) -> None:
    _, flat = plate
    residual = result[flat]

    # The ball sits on the pixel lattice, so its tangent point moves in
    # whole-pixel steps: the constant wobbles by ~3e-5 (0.007 grey levels).
    assert residual.max() - residual.min() < 1e-4
    # The ball's apex sits below a tilted plane by r(sqrt(1 + s**2) - 1)
    # grey levels, s the plane's slope in grey levels per pixel.
    slope = np.hypot(RAMP[1], RAMP[2]) * 255
    expected = RADIUS * (np.sqrt(1 + slope**2) - 1) / 255
    assert residual.mean() == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(("row", "col", "height"), DISKS)
def test_each_disk_keeps_its_height(plate, result, row, col, height) -> None:
    _, flat = plate
    background = float(np.median(result[flat]))
    disk = _disk_mask(row, col)

    kept = result[disk] - background

    # Never more than the disk was, and never less than the ball's rise.
    assert kept.max() <= height + 1e-5
    assert kept.min() >= height - MAX_RISE - 1e-5
    assert np.median(kept) == pytest.approx(height, abs=MAX_RISE)


def test_the_ball_equals_the_textbook_ball_on_8_bit_values(plate) -> None:
    """The kernel is ``ball_kernel(radius)`` with heights in 1/255 steps."""
    landscape, _ = plate
    textbook = landscape - rolling_ball(landscape * 255, radius=RADIUS) / 255

    np.testing.assert_allclose(
        _subtracted(landscape, radius=RADIUS), textbook, atol=1e-6
    )


def test_a_custom_kernel_is_used_as_given(plate) -> None:
    landscape, _ = plate
    kernel = np.zeros((3, 3), dtype=np.float32)

    expected = landscape - rolling_ball(landscape, kernel=kernel)

    np.testing.assert_allclose(
        _subtracted(landscape, radius=RADIUS, kernel=kernel), expected, atol=1e-7
    )
