"""``SmallObjectRemover`` keeps the ``min_size`` boundary across scikit-image.

scikit-image 0.26 deprecated ``remove_small_objects(min_size=...)``, which
removes objects with **fewer than** ``min_size`` pixels, in favour of
``max_size=...``, which removes objects with ``max_size`` pixels **or fewer**,
and warns with a ``FutureWarning`` on every ``min_size`` call. The lock pins
0.25.2, which has no ``max_size``, while a fresh PyPI install resolves 0.26.
These tests pin the boundary (an object of exactly ``min_size`` pixels is
kept, one of ``min_size - 1`` is removed) with warnings raised as errors,
against the scikit-image that is installed -- the lock's, in CI. Run them
under 0.26 with ``uv run --with scikit-image==0.26.0 python -m pytest``
(``pytest`` alone runs the project interpreter and ignores ``--with``).
``tests/unit/sdk_/test_skimage_compat.py`` drives both branches of the
helper with stubs, on any release.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import ndimage as ndi

from phenotypic import Image
from phenotypic.refine import SmallObjectRemover
from phenotypic.sdk_._skimage_compat import (
    _max_size_for,
    remove_objects_smaller_than,
)

MIN_SIZE = 12


def _image_with_objects(sizes: list[int]) -> Image:
    """A blank plate with one horizontal bar of each size, rows apart."""
    objmap = np.zeros((4 * len(sizes) + 2, MIN_SIZE + 4), dtype=np.uint16)
    for index, size in enumerate(sizes):
        objmap[2 + 4 * index, 2 : 2 + size] = index + 1
    image = Image(arr=np.zeros(objmap.shape, dtype=np.float32))
    image.objmap[:] = objmap
    return image


def _surviving_sizes(image: Image) -> list[int]:
    labels, counts = np.unique(image.objmap[:], return_counts=True)
    return sorted(counts[labels != 0].tolist())


def _apply_with_warnings_as_errors(image: Image, min_size: int) -> Image:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return SmallObjectRemover(min_size=min_size).apply(image)


def test_label_image_keeps_min_size_and_drops_one_pixel_smaller() -> None:
    image = _image_with_objects([MIN_SIZE, MIN_SIZE - 1])
    assert _surviving_sizes(image) == [MIN_SIZE - 1, MIN_SIZE]

    refined = _apply_with_warnings_as_errors(image, MIN_SIZE)

    assert _surviving_sizes(refined) == [MIN_SIZE]


@pytest.mark.parametrize(
    ("size", "survives"), [(MIN_SIZE, True), (MIN_SIZE - 1, False)]
)
def test_single_object_boundary_on_the_boolean_path(size, survives) -> None:
    """One label takes the boolean branch (``objmap.max() <= 1``)."""
    image = _image_with_objects([size])
    assert image.objmap[:].max() == 1

    refined = _apply_with_warnings_as_errors(image, MIN_SIZE)

    assert _surviving_sizes(refined) == ([size] if survives else [])


@pytest.mark.parametrize(
    "min_size", [0, 1, 2, 3, 7, 30, 0.5, 2.5, 6.01, 29.999]
)
@pytest.mark.parametrize("as_bool", [False, True])
def test_helper_matches_the_strictly_smaller_rule(min_size, as_bool) -> None:
    """Oracle: zero every component whose pixel count is ``< min_size``.

    Computed with numpy/scipy only, independent of scikit-image, over a
    random label image whose components span sizes 1..~40 pixels. The
    fractional values are where ``min_size - 1`` (``2.5`` -> ``1.5``) or a
    floor (``2.5`` -> ``1``) would drop the wrong components on 0.26.
    """
    rng = np.random.default_rng(20261104)
    labels, _ = ndi.label(rng.random((48, 48)) > 0.55)
    sizes = np.bincount(labels.ravel())
    expected = labels.copy()
    small = sizes < min_size
    small[0] = False
    expected[small[labels]] = 0
    ar = labels > 0 if as_bool else labels

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = remove_objects_smaller_than(ar, min_size)

    np.testing.assert_array_equal(
        result, expected > 0 if as_bool else expected
    )


@pytest.mark.parametrize(
    ("min_size", "max_size"),
    [
        (12, 11),
        (1, 0),
        (0, 0),
        (-3, 0),
        (2.0, 1),
        (2.5, 2),
        (0.5, 0),
        (29.999, 29),
    ],
)
def test_max_size_is_the_largest_size_strictly_below_min_size(
    min_size, max_size
) -> None:
    """The 0.26 argument, pinned whatever scikit-image is installed."""
    assert _max_size_for(min_size) == max_size
