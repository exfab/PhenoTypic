import pytest

from phenotypic.abc_ import ImageOperation

import phenotypic
from phenotypic.data import load_plate_12hr
from phenotypic.detect import RoundPeaksDetector
from phenotypic.abc_ import ImageOperation

from .test_fixtures import walk_package_for_class
from .resources.TestHelper import timeit

ops = walk_package_for_class(pkg=phenotypic, target_class=ImageOperation)


@pytest.mark.parametrize("qualname,obj", ops)
@timeit
def test_operation(qualname, obj):
    """The goal of this test is to ensure that all operations are callable with basic functionality
     and return a valid Image object."""
    image = phenotypic.GridImage(load_plate_12hr())
    image1 = RoundPeaksDetector().apply(image)
    assert obj().apply(image).isempty() is False, "Operation failed"

    image2 = RoundPeaksDetector().apply(image)
    assert image1 == image2, "Operation was not reproducible"
