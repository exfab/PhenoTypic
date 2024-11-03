import pytest

from phenoscope import Image
from phenoscope.sample_images import PlateDay1


def test_image():
    img = Image(PlateDay1())
    assert img.array is not None
    assert img.enhanced_array is not None
    assert img.object_mask is None
    assert img.object_map is None
