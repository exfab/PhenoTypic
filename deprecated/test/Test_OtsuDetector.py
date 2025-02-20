import pytest
import numpy as np

from phenoscope import Image
from phenoscope.detection import OtsuDetector

from .Test_Image import sample_data

def test_inplace_OtsuDetector(sample_data):
    img = Image(sample_data['image'])
    original_mask = img.obj_mask
    original_map = img.obj_map

    detector = OtsuDetector()

    detector.detect(img,inplace=True)

    assert not np.array_equal(img.obj_mask, original_mask)
    assert not np.array_equal(img.obj_map, original_map)

def test_OtsuDetector(sample_data):
    img = Image(sample_data['image'])
    original_mask = img.obj_mask
    original_map = img.obj_map

    detector = OtsuDetector()

    img = detector.detect(img)

    assert not np.array_equal(img.obj_mask, original_mask)
    assert not np.array_equal(img.obj_map, original_map)

