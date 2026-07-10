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
