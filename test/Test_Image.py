import pytest

import matplotlib.pyplot as plt

from phenoscope import Image
from phenoscope.data import PlateDay1


def test_image():
    img = Image(PlateDay1())

    assert img.array is not None
    assert img.matrix is not None
    assert img.enhanced_matrix is not None

    assert img.object_mask is None
    assert img.object_map is None

def test_image_show_array():
    img = Image(PlateDay1())
    fig, ax = img.show_array()
    assert fig is not None
    assert ax is not None
    plt.close(fig)

def test_image_show_matrix():
    img = Image(PlateDay1())
    fig, ax = img.show_matrix()
    assert fig is not None
    assert ax is not None
    plt.close(fig)

def test_image_show_enhanced_matrix():
    img = Image(PlateDay1())
    fig, ax = img.show_enhanced()
    assert fig is not None
    assert ax is not None
    plt.close(fig)