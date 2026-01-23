import pytest

import phenotypic
from phenotypic.data import load_synth_plate
from phenotypic.detect import RoundPeaksDetector

from ..test_fixtures import walk_package_for_class
from ..resources.TestHelper import timeit

ops = walk_package_for_class(pkg=phenotypic,
                             target_class=phenotypic.abc_.ImageOperation)

image_ops = [(qualname, obj) for qualname, obj in ops
             if ("Grid" not in qualname) or ("phenotypic.abc_" not in qualname)]


@pytest.mark.parametrize("qualname,obj", image_ops)
@timeit
def test_operation(qualname, obj):
    """The goal of this test is to ensure that all operations are callable with
    basic functionality and return a valid Image object."""
    image = phenotypic.Image(load_synth_plate())

    RoundPeaksDetector().apply(image, inplace=True)

    instance = obj()
    assert isinstance(instance, obj), "Operation did not instantiate with defaults"

    image1 = instance.apply(image)
    assert image1.isempty() is False, "Operation failed"

    image2 = instance.apply(image)

    # bm3d denoiser likely has unintended randomness from precision conversion
    if "BM3D" not in qualname:
        assert image1 == image2, "Operation was not reproducible"


@pytest.mark.parametrize("qualname,obj", image_ops)
@timeit
def test_operation_compatibility_with_grid_image(qualname, obj):
    image = phenotypic.GridImage(load_synth_plate())

    RoundPeaksDetector().apply(image, inplace=True)

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


@pytest.mark.parametrize("qualname,obj", grid_ops)
@timeit
def test_operation(qualname, obj):
    """The goal of this test is to ensure that all operations are callable with
    basic functionality and return a valid Image object."""
    image = phenotypic.GridImage(load_synth_plate(), nrows=8, ncols=12)

    RoundPeaksDetector().apply(image, inplace=True)

    instance = obj()
    assert isinstance(instance, obj), "Operation did not instantiate with defaults"

    image1 = instance.apply(image)
    assert image1.isempty() is False, "Operation failed"

    image2 = instance.apply(image)
    assert image1 == image2, "Operation was not reproducible"
