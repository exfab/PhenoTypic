import pytest

import pandas as pd

from phenotypic.data import load_synth_yeast_plate
from phenotypic.refine import MaskOpener

from unit.test_fixtures import _image_measurements
from unit.resources.TestHelper import timeit


@pytest.fixture(scope="session")
def mask_opened_image():
    """Session-scoped image with MaskOpener pre-applied."""
    image = load_synth_yeast_plate()
    MaskOpener().apply(image, inplace=True)
    return image


@pytest.mark.smoke
@pytest.mark.parametrize("qualname,obj", _image_measurements)
@timeit
def test_measurement(qualname, obj, mask_opened_image):
    """The goal of this test is to ensure that all operations are callable with basic functionality,
    and return a valid dataframe object. This does not check for accuracy"""
    image = mask_opened_image.copy()
    assert isinstance(obj().measure(image), pd.DataFrame)
