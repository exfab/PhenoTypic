import pytest

import pandas as pd

from phenotypic.data import load_synth_yeast_plate
from phenotypic.refine import MaskOpener

from ..test_fixtures import _image_measurements
from ..resources.TestHelper import timeit


@pytest.mark.parametrize("qualname,obj", _image_measurements)
@timeit
def test_measurement(qualname, obj):
    """The goal of this test is to ensure that all operations are callable with basic functionality,
    and return a valid dataframe object. This does not check for accuracy"""
    image = load_synth_yeast_plate()
    MaskOpener().apply(image, inplace=True)
    assert isinstance(obj().measure(image), pd.DataFrame)
