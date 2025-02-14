import pytest

import numpy as np

from phenoscope.data import (
    load_colony_12_hr,
    load_colony_72hr,
    load_plate_12hr,
    load_plate_72hr,
)

from phenoscope import Image

@pytest.fixture(scope='session')
def sample_image_arrays():
    """Fixture that returns (image_array, format)"""
    return [
        (load_colony_12_hr(), None),    # Test Auto Formatter
        (load_colony_72hr(), 'RGB'),
        (load_plate_12hr(), 'RGB'),
        (load_plate_72hr(), 'RGB'),
        (np.full(shape=(100,100), fill_value=0), 'Grayscale'), # Black Image
        (np.full(shape=(100,100), fill_value=1), 'Grayscale'), # White Image
    ]

def test_empty_image():
    empty_image = Image()
    assert empty_image is not None
    assert empty_image.shape is None

def test_image_array_init(sample_image_arrays):
    for image, format in sample_image_arrays:
        phenoscope_image = Image(input_image=image, input_format=format)
        assert phenoscope_image is not None
        assert phenoscope_image.shape == image.shape