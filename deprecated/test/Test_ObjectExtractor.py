import pytest

from phenoscope import Image
from phenoscope.grid import GriddedImage

from phenoscope.detection import OtsuDetector
from phenoscope.features import ObjectImageExtractor

from .Test_Image import sample_data

def test_object_extractor_on_image(sample_data):
    img = Image(sample_data['image'])
    OtsuDetector().detect(img,inplace=True)
    object_images = ObjectImageExtractor().extract(img)
    assert len(object_images.keys()) > 1
    assert object_images is not None

def test_object_extractor_on_gridded_image(sample_data):
    img = GriddedImage(sample_data['image'])
    OtsuDetector().detect(img,inplace=True)
    object_images = ObjectImageExtractor().extract(img)
    assert len(object_images.keys()) > 1
    assert object_images is not None