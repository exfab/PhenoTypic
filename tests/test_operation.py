import pytest

from debug.checkmem import qualname
from phenotypic.abc_ import ImageOperation

import phenotypic
from phenotypic.data import load_synthetic_detection_image
from phenotypic.detect import RoundPeaksDetector
from phenotypic.abc_ import ImageOperation

from .test_fixtures import walk_package_for_class
from .resources.TestHelper import timeit

ops = walk_package_for_class(pkg=phenotypic, target_class=ImageOperation)

image_ops = [
    (qualname, obj) for qualname, obj in ops
    if "Grid" not in qualname
]


@pytest.mark.parametrize("qualname,obj", image_ops)
@timeit
def test_operation(qualname, obj):
    """The goal of this test is to ensure that all operations are callable with basic functionality
     and return a valid Image object."""
    image = phenotypic.Image(load_synthetic_detection_image())
    image = RoundPeaksDetector().apply(image)

    instance = obj()
    assert isinstance(instance, obj.__class__), "ImageOperation failed to initialize"

    image1 = instance.apply(image)
    assert image1.isempty() is False, "Returned image is empty"

    image2 = instance.apply(image)
    assert image1 == image2, "Operation was not reproducible"


grid_ops = [
    (qualname, obj) for qualname, obj in ops
    if "Grid" in qualname
]


@pytest.mark.parametrize("qualname,obj", grid_ops)
@timeit
def test_operation(qualname, obj):
    """The goal of this test is to ensure that all operations are callable with basic functionality
     and return a valid Image object."""
    image = phenotypic.GridImage(load_synthetic_detection_image())
    image = RoundPeaksDetector().apply(image)

    instance = obj()
    assert isinstance(instance, obj.__class__), "ImageOperation failed to initialize"

    image1 = instance.apply(image)
    assert image1.isempty() is False, "Returned image is empty"

    image2 = instance.apply(image)
    assert image1 == image2, "Operation was not reproducible"
