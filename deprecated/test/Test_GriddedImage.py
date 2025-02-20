import pytest

import numpy as np
from phenoscope.grid import GriddedImage

from .Test_Image import sample_data

def test_blank_gridded_image():
    img = GriddedImage()
    assert img.array is None
    assert img.matrix is None
    assert img.enhanced_matrix is None
    assert img.obj_mask is None
    assert img.obj_map is None

def test_image(sample_data):
    img = GriddedImage(sample_data['image'])

    assert img.array is not None
    assert img.matrix is not None
    assert img.enhanced_matrix is not None

    assert np.array_equal(img.obj_mask, np.full(shape=img.shape, fill_value=1))
    assert np.array_equal(img.obj_map, np.full(shape=img.shape, fill_value=1))


