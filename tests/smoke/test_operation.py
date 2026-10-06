import contextlib

import numpy as np
import pandas as pd
import pytest

import phenotypic
from phenotypic.abc_ import ObjectDetector, RefMetadata
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector

from unit.test_fixtures import walk_package_for_class
from unit.resources.TestHelper import timeit

ops = walk_package_for_class(
    pkg=phenotypic, target_class=phenotypic.abc_.ImageOperation
)

image_ops = [
    (qualname, obj)
    for qualname, obj in ops
    if (("Grid" not in qualname) or ("phenotypic.abc_" not in qualname))
    and "ColorCorrector" not in qualname
    # CalibrateColorRpcc requires `rois` -- the rectangles where the colour
    # chart appears -- and there is no sensible default, so like
    # ColorCorrector it cannot be bare-constructed.
    and "CalibrateColorRpcc" not in qualname
    # GridApply requires an `image_op` (the operation run on each grid
    # section); like ColorCorrector it cannot be bare-constructed, so
    # it is excluded from the defaults-only smoke contract.
    and "GridApply" not in qualname
]

# Filter image_ops down to ObjectDetector subclasses for the objmap-consistency
# contract. detector_ops inherits the `Grid*` and `ColorCorrector` exclusions
# from image_ops automatically.
detector_ops = [
    (qualname, obj)
    for qualname, obj in image_ops
    if issubclass(obj, ObjectDetector)
]


@pytest.fixture(scope="session")
def detected_grid_image():
    """Session-scoped detected GridImage for smoke tests."""
    image = phenotypic.GridImage(load_synth_yeast_plate())
    OtsuDetector().apply(image, inplace=True)
    return image


def _reference_context(obj, image):
    """The ambient table a ``RefMetadata`` op reads; nothing for any other op.

    Such an op holds no table and refuses to run without one, so the
    defaults-only contract supplies a media blank: an unprocessed copy of the
    same synthetic plate, named in a one-row table.
    """
    if not issubclass(obj, RefMetadata):
        return contextlib.nullcontext()
    table = pd.DataFrame({"ImageName": [image.name], "BlankImage": ["blank"]})
    blank = phenotypic.GridImage(load_synth_yeast_plate())
    # Same plate, so it would otherwise carry the target's name and be
    # refused as the image naming itself.
    blank.name = "blank"
    return phenotypic.ReferenceContext(table, images={"blank": blank})


@pytest.mark.smoke
@pytest.mark.parametrize("qualname,obj", image_ops)
@timeit
def test_operation(qualname, obj, detected_grid_image):
    """The goal of this test is to ensure that all operations are callable with
    basic functionality and return a valid Image object."""
    image = detected_grid_image.copy()

    instance = obj()
    assert isinstance(instance, obj), (
        "Operation did not instantiate with defaults"
    )

    with _reference_context(obj, image):
        image1 = instance.apply(image)
        assert image1.isempty() is False, "Operation failed"

        image2 = instance.apply(image)

    # bm3d denoiser likely has unintended randomness from precision conversion
    # (ColorDenoise + DenoiseBlockMatch + EnhanceBlockMatch + BM3D all wrap
    # bm3d.bm3d_rgb / bm3d).
    if (
        ("BM3D" not in qualname)
        and ("DenoiseBlockMatch" not in qualname)
        and ("EnhanceBlockMatch" not in qualname)
        and ("ColorDenoise" not in qualname)
    ):
        assert image1 == image2, "Operation was not reproducible"


@pytest.mark.smoke
@pytest.mark.parametrize("qualname,obj", image_ops)
@timeit
def test_inplace_contract(qualname, obj, detected_grid_image):
    """ABC contract: inplace=False returns a new object; inplace=True mutates
    and returns the same input.

    Replaces per-detector test_inplace_false_preserves_original /
    test_inplace_true_modifies_original copies that used to live in each
    detector test file.
    """
    # inplace=False: must return a different object than the input.
    snapshot = detected_grid_image.copy()
    with _reference_context(obj, snapshot):
        out = obj().apply(snapshot, inplace=False)
    assert out is not snapshot, (
        f"{qualname} returned the same object with inplace=False"
    )

    # inplace=True: must mutate and return the input image itself.
    # PadImage / CropImage legitimately change image dimensions and have
    # to allocate a new image even with inplace=True; they're excluded here.
    target = detected_grid_image.copy()
    with _reference_context(obj, target):
        ret = obj().apply(target, inplace=True)
    if ("PadImage" not in qualname) and ("CropImage" not in qualname):
        assert ret is target, (
            f"{qualname} did not return the input with inplace=True"
        )


@pytest.mark.smoke
@pytest.mark.parametrize("qualname,obj", detector_ops)
@timeit
def test_detector_objmap_objmask_consistency(
    qualname, obj, detected_grid_image
):
    """ABC contract: any ObjectDetector must satisfy `objmap > 0 == objmask`.

    Replaces per-detector test_objmask_objmap_consistency copies.
    """
    image = detected_grid_image.copy()
    with _reference_context(obj, image):
        obj().apply(image, inplace=True)
    np.testing.assert_array_equal(
        image.objmap[:] > 0,
        image.objmask[:],
        err_msg=f"{qualname}: objmap > 0 must equal objmask",
    )


grid_ops = [
    (qualname, obj)
    for qualname, obj in ops
    if ("Grid" in qualname) or ("phenotypic.abc_" not in qualname)
]
