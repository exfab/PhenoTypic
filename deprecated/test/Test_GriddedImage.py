import pytest

import numpy as np
from src.grid import GridImage

from .Test_Image import sample_data

def test_blank_gridded_image():
    img = GridImage()
    assert img.array is None
    assert img.matrix is None
    assert img.enhanced_matrix is None
    assert img.omask is None
    assert img.omap is None

def test_image(sample_data):
    img = GridImage(sample_data['image'])

    assert img.array is not None
    assert img.matrix is not None
    assert img.enhanced_matrix is not None

    assert np.array_equal(img.omask, np.full(shape=img.shape, fill_value=1))
    assert np.array_equal(img.omap, np.full(shape=img.shape, fill_value=1))


