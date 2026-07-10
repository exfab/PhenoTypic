"""Unit tests for MeasureShape."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.measure import MeasureShape


@pytest.fixture
def split_rectangle_image() -> Image:
    """A 43x43 image whose objmap holds two labels sharing their full internal edge.

    Label 1 is a 41x20 rectangle, label 2 is a 41x21 rectangle. They touch
    along a 41-pixel edge with no background between them, which is what a
    watershed split of a merged colony pair looks like.
    """
    rgb = np.zeros((43, 43, 3), dtype=np.uint8)
    rgb[1:42, 1:42] = 200
    image = Image(rgb)
    objmap = np.zeros((43, 43), dtype=int)
    objmap[1:42, 1:21] = 1
    objmap[1:42, 21:42] = 2
    image.objmap[:] = objmap
    return image


def test_convex_area_is_an_area_not_a_perimeter(split_rectangle_image):
    """A filled rectangle is its own convex hull: ConvexArea == Area, Solidity == 1."""
    measurements = MeasureShape().measure(split_rectangle_image)

    # Label 1 is a 41x20 rectangle => 820 pixels. It is convex, so its convex
    # hull is itself. Exact equality is correct here: both sides are integer
    # pixel counts from regionprops, not floating-point geometry.
    assert measurements["Shape_Area"].iloc[0] == 820.0
    assert measurements["Shape_ConvexArea"].iloc[0] == 820.0
    assert measurements["Shape_Solidity"].iloc[0] == pytest.approx(1.0, abs=1e-12)


def test_touching_labels_do_not_inflate_each_others_radii(split_rectangle_image):
    """distance_transform_edt binarizes its input, so a whole-objmap EDT merges
    adjacent labels. Each object must be transformed in isolation.

    Label 1 is 41x20, so its largest inscribed circle has radius 20/2 = 10.
    Label 2 is 41x21, so the discrete inscribed radius is 11. Computed over the
    merged 41x41 block instead, the two report 20.0 and 21.0.
    """
    measurements = MeasureShape().measure(split_rectangle_image)

    # Exact: the EDT of an axis-aligned rectangle is exact integer arithmetic
    # (scipy reconstructs distances from an int32 feature transform), so the
    # inscribed radius of a rectangle of even width is exactly half that width.
    assert measurements["Shape_InscribedRadius"].iloc[0] == 10.0
    assert measurements["Shape_InscribedRadius"].iloc[1] == 11.0

    # Mean depth-from-boundary. Tolerance 1e-3 is far below the 2.56-pixel
    # error the merged EDT produces, so this assertion cannot pass by accident.
    assert measurements["Shape_MeanBoundaryDist"].iloc[0] == pytest.approx(4.6951, abs=1e-3)
    assert measurements["Shape_MeanBoundaryDist"].iloc[1] == pytest.approx(4.8676, abs=1e-3)
    assert measurements["Shape_MedianBoundaryDist"].iloc[0] == pytest.approx(4.0, abs=1e-9)


def test_merged_edt_would_fail_this_test():
    """Mutation control: prove the fixture above can detect the bug it guards.

    Reproduce the old whole-objmap EDT and assert it gives the wrong answer.
    If this test ever starts reporting the correct values, the fixture has
    stopped exercising the merge and the test above is no longer load-bearing.
    """
    from scipy.ndimage import distance_transform_edt

    objmap = np.zeros((43, 43), dtype=int)
    objmap[1:42, 1:21] = 1
    objmap[1:42, 21:42] = 2

    merged = distance_transform_edt(objmap)
    assert merged[objmap == 1].max() == 20.0  # not 10.0
    assert merged[objmap == 2].max() == 21.0  # not 11.0


def test_radius_columns_are_named_for_what_they_measure():
    """MeanRadius/MedianRadius were means of the distance transform, i.e. depth
    from the boundary, not radii. MaxRadius was the inscribed radius. All three
    are renamed; the old names must be gone.
    """
    from phenotypic.schema import SHAPE

    headers = set(SHAPE.get_headers())
    assert "Shape_MeanBoundaryDist" in headers
    assert "Shape_MedianBoundaryDist" in headers
    assert "Shape_InscribedRadius" in headers
    assert "Shape_MeanRadius" not in headers
    assert "Shape_MedianRadius" not in headers
    assert "Shape_MaxRadius" not in headers
