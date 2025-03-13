import pytest
import numpy as np

from src import Image
from src.detection import OtsuDetector

from .Test_Image import sample_data

def test_inplace_OtsuDetector(sample_data):
    img = Image(sample_data['image'])
    original_mask = img.omask
    original_map = img.omap

    detector = OtsuDetector()

    detector.apply(img, inplace=True)

    assert not np.array_equal(img.omask, original_mask)
    assert not np.array_equal(img.omap, original_map)

def test_OtsuDetector(sample_data):
    img = Image(sample_data['image'])
    original_mask = img.omask
    original_map = img.omap

    detector = OtsuDetector()

    img = detector.apply(img)

    assert not np.array_equal(img.omask, original_mask)
    assert not np.array_equal(img.omap, original_map)

