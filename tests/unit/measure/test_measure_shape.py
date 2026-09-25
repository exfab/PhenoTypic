"""Unit tests for MeasureShape after the size/shape split."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.measure import MeasureShape
from phenotypic.schema import OBJECT, SHAPE


def _image_with_objmap(objmap: np.ndarray) -> Image:
    rgb = np.zeros((*objmap.shape, 3), dtype=np.uint8)
    rgb[objmap > 0] = 200
    image = Image(rgb)
    image.objmap[:] = objmap
    return image


def _split_rectangle_objmap() -> np.ndarray:
    """Two labels sharing their full internal edge, as a watershed split of a
    merged colony pair looks: label 1 is 41x20, label 2 is 41x21."""
    objmap = np.zeros((43, 43), dtype=int)
    objmap[1:42, 1:21] = 1
    objmap[1:42, 21:42] = 2
    return objmap


@pytest.fixture
def split_rectangle_image() -> Image:
    return _image_with_objmap(_split_rectangle_objmap())


def test_emits_exactly_the_shape_schema(split_rectangle_image):
    frame = MeasureShape().measure(split_rectangle_image)
    assert list(frame.columns) == [str(OBJECT.LABEL), *SHAPE.get_headers()]


def test_size_magnitudes_and_misnamed_radii_are_gone():
    headers = set(SHAPE.get_headers())
    for retired in ("Area", "Perimeter", "ConvexArea", "BboxArea", "MajorAxisLength",
                    "MinorAxisLength", "MeanRadius", "MedianRadius", "MaxRadius"):
        assert f"Shape_{retired}" not in headers
    assert {"Shape_MeanBoundaryDist", "Shape_MedianBoundaryDist"} <= headers


def test_touching_labels_do_not_inflate_boundary_distances(split_rectangle_image):
    """Branch-verified values for the 41x20 / 41x21 pair measured per object. The
    merged EDT is off by 2.56 px, so abs=1e-3 cannot pass by accident.

    Mutation: compute the EDT over image.objmap[:] -> fails.
    """
    frame = MeasureShape().measure(split_rectangle_image)
    dist = frame[str(SHAPE.MEAN_BOUNDARY_DIST)]
    assert dist.iloc[0] == pytest.approx(4.6951, abs=1e-3)
    assert dist.iloc[1] == pytest.approx(4.8676, abs=1e-3)
    assert frame[str(SHAPE.MEDIAN_BOUNDARY_DIST)].iloc[0] == pytest.approx(4.0, abs=1e-9)


def test_merged_edt_would_fail_this_test():
    """Mutation control: prove the fixture above can detect the bug it guards.

    Reproduce the old whole-objmap EDT and assert it gives the wrong answer.
    If this test ever starts reporting the correct values, the fixture has
    stopped exercising the merge and the test above is no longer load-bearing.
    """
    from scipy.ndimage import distance_transform_edt

    objmap = _split_rectangle_objmap()
    merged = distance_transform_edt(objmap)
    assert merged[objmap == 1].max() == 20.0  # not 10.0
    assert merged[objmap == 2].max() == 21.0  # not 11.0
    # The guarded column is the mean, so the control must move the mean too.
    assert merged[objmap == 1].mean() != pytest.approx(4.6951, abs=1e-3)


def test_mean_boundary_dist_of_a_disk_is_one_third_of_its_radius():
    """Pins the rename: this column is interior thickness, not a radius. The analytic
    value is R/3 = 13.33; rasterisation lifts it slightly, so 0.3 px (under 1/R of R
    at R=40, the validation script's check 01 mechanism)."""
    y, x = np.mgrid[-50:50, -50:50]
    objmap = (x**2 + y**2 <= 40**2).astype(int)
    frame = MeasureShape().measure(_image_with_objmap(objmap))
    assert frame[str(SHAPE.MEAN_BOUNDARY_DIST)].iloc[0] == pytest.approx(40 / 3, abs=0.3)


def test_no_objects_raises_like_every_other_measurer():
    """Amendment A2: main's contract is kept."""
    from phenotypic.sdk_.exceptions_ import OperationFailedError

    with pytest.raises(OperationFailedError, match="NoObjectsError"):
        MeasureShape().measure(_image_with_objmap(np.zeros((20, 20), dtype=int)))


def test_degenerate_objects_give_nan_hull_measures_without_warning():
    objmap = np.zeros((20, 20), dtype=int)
    objmap[2, 2] = 1
    objmap[10, 3:12] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = MeasureShape().measure(_image_with_objmap(objmap))
    for header in (SHAPE.SOLIDITY, SHAPE.MIN_FERET_DIAMETER, SHAPE.MAX_FERET_DIAMETER):
        assert frame[str(header)].isna().all(), header
