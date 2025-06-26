import pytest

from phenotypic.abstract import ImageOperation

import phenotypic
from phenotypic.data import load_plate_12hr
from phenotypic.detection import WatershedDetector

from .test_fixtures import _image_operations
from .resources.TestHelper import timeit

@pytest.mark.parametrize("qualname,obj", _image_operations)
@timeit
def test_operation(qualname, obj):
    """The goal of this test is to ensure that all operations are callable with basic functionality
     and return a valid Image object."""
    image = phenotypic.GridImage(load_plate_12hr())
    image = WatershedDetector().apply(image)
    assert obj().apply(image).isempty() is False
