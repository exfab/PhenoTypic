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

def test_image_show():
    img = Image(PlateDay1())
    fig, ax = img.show()
    plt.close(fig)