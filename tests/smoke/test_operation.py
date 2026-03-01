import pytest

import phenotypic
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector

from unit.test_fixtures import walk_package_for_class
from unit.resources.TestHelper import timeit

ops = walk_package_for_class(pkg=phenotypic,
                             target_class=phenotypic.abc_.ImageOperation)

image_ops = [(qualname, obj) for qualname, obj in ops
             if ("Grid" not in qualname) or ("phenotypic.abc_" not in qualname)]


@pytest.fixture(scope="session")
def detected_grid_image():
    """Session-scoped detected GridImage for smoke tests."""
    image = phenotypic.GridImage(load_synth_yeast_plate())
    OtsuDetector().apply(image, inplace=True)
    return image


@pytest.mark.smoke
@pytest.mark.parametrize("qualname,obj", image_ops)
@timeit
def test_operation(qualname, obj, detected_grid_image):
    """The goal of this test is to ensure that all operations are callable with
    basic functionality and return a valid Image object."""
    image = detected_grid_image.copy()

    instance = obj()
    assert isinstance(instance, obj), "Operation did not instantiate with defaults"

    image1 = instance.apply(image)
    assert image1.isempty() is False, "Operation failed"

    image2 = instance.apply(image)

    # bm3d denoiser likely has unintended randomness from precision conversion
    if "BM3D" not in qualname:
        assert image1 == image2, "Operation was not reproducible"


grid_ops = [(qualname, obj) for qualname, obj in ops if
            ("Grid" in qualname) or ("phenotypic.abc_" not in qualname)]
