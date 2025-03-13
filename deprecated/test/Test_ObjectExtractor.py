import pytest

from src import Image
from src.grid import GridImage

from src.detection import OtsuDetector
from src.measure import ObjectImageExtractor

from .Test_Image import sample_data

def test_object_extractor_on_image(sample_data):
    img = Image(sample_data['image'])
    OtsuDetector().apply(img, inplace=True)
    object_images = ObjectImageExtractor().measure(img)
    assert len(object_images.keys()) > 1
    assert object_images is not None

def test_object_extractor_on_gridded_image(sample_data):
    img = GridImage(sample_data['image'])
    OtsuDetector().apply(img, inplace=True)
    object_images = ObjectImageExtractor().measure(img)
    assert len(object_images.keys()) > 1
    assert object_images is not None