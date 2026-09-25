"""Unit tests for MeasureSize."""

from __future__ import annotations

import warnings

import numpy as np
import pydantic
import pytest

from phenotypic import Image
from phenotypic.measure import MeasureSize
from phenotypic.schema import OBJECT, SIZE


def _image_with_objmap(objmap: np.ndarray) -> Image:
    rgb = np.zeros((*objmap.shape, 3), dtype=np.uint8)
    rgb[objmap > 0] = 200
    image = Image(rgb)
    image.objmap[:] = objmap
    return image


@pytest.fixture
def split_rectangle_image() -> Image:
    """Two labels sharing their full internal edge (a watershed-split pair).

    Label 1 is 41x20 and label 2 is 41x21, touching along 41 pixels with no
    background between them.
    """
    objmap = np.zeros((43, 43), dtype=int)
    objmap[1:42, 1:21] = 1
    objmap[1:42, 21:42] = 2
    return _image_with_objmap(objmap)


def test_emits_exactly_the_size_schema(split_rectangle_image):
    frame = MeasureSize().measure(split_rectangle_image)
    assert list(frame.columns) == [str(OBJECT.LABEL), *SIZE.get_headers()]


def test_area_equals_regionprops_area(split_rectangle_image):
    """Shape's ratios and Intensity's density divide by props.area; the published
    Size_Area must be the same number. Integer pixel counts, compared exactly."""
    frame = MeasureSize().measure(split_rectangle_image)
    props = split_rectangle_image.objects.props
    assert frame[str(SIZE.AREA)].tolist() == [float(p.area) for p in props]
    assert frame[str(SIZE.AREA)].tolist() == [820.0, 861.0]


def test_convex_area_is_the_scipy_hull_volume(split_rectangle_image):
    """Label 1's pixel centres span a 40 x 19 hull: area 760 (the pixel count is 820).

    Mutation: switch the helper to `.area` -> 118.0 (the hull perimeter); switch
    MeasureSize to `props.area_convex` -> 820.0. Both fail.
    """
    frame = MeasureSize().measure(split_rectangle_image)
    assert frame[str(SIZE.CONVEX_AREA)].iloc[0] == pytest.approx(760.0, abs=1e-9)


def test_touching_labels_do_not_inflate_each_others_inscribed_radius(split_rectangle_image):
    """A whole-objmap EDT merges the pair and reports 20/21; per object it is 10/11.
    The EDT of an axis-aligned rectangle is exact integer arithmetic.

    Mutation: compute the EDT over image.objmap[:] -> 20.0/21.0, and this fails.
    """
    frame = MeasureSize().measure(split_rectangle_image)
    assert frame[str(SIZE.INSCRIBED_RADIUS)].tolist() == [10.0, 11.0]


def test_border_touching_colony_counts_the_border_as_an_edge():
    """Review Focus 3. A 10-row band spanning the full width of the top edge: the
    padded crop gives 5; the old whole-image EDT gave 10 (scipy measures only to
    zero pixels inside the array)."""
    objmap = np.zeros((20, 30), dtype=int)
    objmap[0:10, :] = 1
    frame = MeasureSize().measure(_image_with_objmap(objmap))
    assert frame[str(SIZE.INSCRIBED_RADIUS)].iloc[0] == 5.0


def test_no_objects_raises_like_every_other_measurer():
    """Review Focus 1 / amendment A2: main's contract is kept, not changed.
    MeasureFeatures.measure re-raises without chaining, but names the original
    type in the message (abc_/_measure_features.py, the `except Exception` arm)."""
    from phenotypic.sdk_.exceptions_ import OperationFailedError

    with pytest.raises(OperationFailedError, match="NoObjectsError"):
        MeasureSize().measure(_image_with_objmap(np.zeros((20, 20), dtype=int)))


def test_degenerate_objects_measure_without_raising_or_warning():
    """Review Focus 2. A single pixel and a 1-pixel-wide line: Qhull fails, so
    ConvexArea is NaN; the radii come from the EDT and contour and stay finite."""
    objmap = np.zeros((20, 20), dtype=int)
    objmap[2, 2] = 1
    objmap[10, 3:12] = 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = MeasureSize().measure(_image_with_objmap(objmap))
    assert frame[str(SIZE.CONVEX_AREA)].isna().all()
    assert np.isfinite(frame[str(SIZE.INSCRIBED_RADIUS)]).all()
    assert np.isfinite(frame[str(SIZE.MAX_RADIUS)]).all()


def test_trim_proportion_zero_makes_robust_mean_the_plain_mean(split_rectangle_image):
    """With no trim, the trimmed mean is the plain mean. Both average the same
    signature, so they agree to summation-order rounding: 360 terms x 1 ulp
    of ~20 px is ~1.6e-12, so abs=1e-9 has headroom and still catches any real
    wiring slip (a swapped statistic moves the value by more than 1 px here)."""
    frame = MeasureSize(trim_proportion=0.0).measure(split_rectangle_image)
    assert frame[str(SIZE.ROBUST_MEAN_RADIUS)].to_numpy() == pytest.approx(
        frame[str(SIZE.MEAN_RADIUS)].to_numpy(), abs=1e-9
    )


@pytest.mark.parametrize(
    "kwargs",
    [{"angular_bins": 4}, {"trim_proportion": 0.5}, {"plateau_tolerance": 0.0}],
)
def test_field_bounds_are_enforced(kwargs):
    with pytest.raises(pydantic.ValidationError):
        MeasureSize(**kwargs)


def test_non_default_fields_round_trip_through_json():
    """Review Focus 5."""
    op = MeasureSize(angular_bins=90, trim_proportion=0.1, plateau_tolerance=0.02)
    loaded = MeasureSize.from_json(op.to_json())
    assert (loaded.angular_bins, loaded.trim_proportion, loaded.plateau_tolerance) == (
        90, 0.1, 0.02
    )
