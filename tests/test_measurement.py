import pytest

from phenotypic.abstract import MeasureFeatures

import pandas as pd

import phenotypic
from phenotypic.data import load_plate_12hr
from phenotypic.detection import WatershedDetector

from .test_fixtures import _image_measurements
from .resources.TestHelper import timeit


@pytest.mark.parametrize("qualname,obj", _image_measurements)
@timeit
def test_measurement(qualname, obj):
    """The goal of this test is to ensure that all operations are callable with basic functionality,
     and return a valid Image object."""
    image = phenotypic.GridImage(load_plate_12hr())
    image = WatershedDetector().apply(image)
    assert isinstance(obj().measure(image), pd.DataFrame)
