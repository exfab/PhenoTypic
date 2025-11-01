import pytest

from phenotypic.ABC_ import MeasureFeatures

import pandas as pd

import phenotypic
from phenotypic.data import load_plate_72hr
from phenotypic.detection import OtsuDetector
from phenotypic.objedit import MaskOpener

from .test_fixtures import _image_measurements
from .resources.TestHelper import timeit


@pytest.mark.parametrize("qualname,obj", _image_measurements)
@timeit
def test_measurement(qualname, obj):
    """The goal of this test is to ensure that all operations are callable with basic functionality,
     and return a valid dataframe object. This does not check for accuracy"""
    image = phenotypic.GridImage(load_plate_72hr())
    OtsuDetector(ignore_borders=True).apply(image, inplace=True)
    MaskOpener().apply(image, inplace=True)
    assert isinstance(obj().measure(image), pd.DataFrame)
