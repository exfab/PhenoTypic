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


def test_a_hole_counts_as_background_for_the_inscribed_radius():
    """A ring (outer radius 30, hole radius 12) touching nothing. MeasureSize reads
    `props.image`, the unfilled mask, so the widest inscribed circle fits in the
    ring's band: sqrt(85) = 9.2195, not the filled disk's ~30.

    Oracle: main's whole-image EDT on the objmap, exact for an isolated object.

    Mutation: `props.image` -> `props.image_filled` in MeasureSize -> 30.02, fails.
    """
    from scipy.ndimage import distance_transform_edt

    y, x = np.mgrid[0:100, 0:100]
    d2 = (y - 50) ** 2 + (x - 50) ** 2
    objmap = ((d2 <= 30**2) & (d2 > 12**2)).astype(int)
    oracle = float(distance_transform_edt(objmap).max())
    frame = MeasureSize().measure(_image_with_objmap(objmap))
    assert oracle == pytest.approx(np.sqrt(85), abs=1e-12)  # guard the oracle
    assert frame[str(SIZE.INSCRIBED_RADIUS)].iloc[0] == pytest.approx(oracle, abs=1e-12)


def test_rows_align_with_labels_that_are_non_contiguous_and_out_of_raster_order():
    """Review LOW-5. MeasureSize and MeasureIntensity index per-object arrays by
    position and rely on `np.unique(objmap)`, `regionprops` and
    `labels2series()` all being in sorted label order. Labels 2, 9 and 300 are
    placed in reverse raster order (300 at the top, 2 at the bottom), with
    distinct areas, depths and gray levels, so any positional slip puts one
    object's value on another's row.

    Oracle per label: the pixel count, main's whole-image EDT maximum (exact:
    the objects touch nothing), and the mean gray value over the label.

    Mutation: iterate `sorted(image.objects.props, key=lambda p: p.bbox)`
    (raster order) in either measurer -> the rows cross, and this fails.
    """
    from scipy.ndimage import distance_transform_edt

    from phenotypic.measure import MeasureIntensity
    from phenotypic.schema import INTENSITY

    y, x = np.mgrid[0:120, 0:120]
    objmap = np.zeros((120, 120), dtype=int)
    objmap[10:20, 60:100] = 300
    objmap[(y - 50) ** 2 + (x - 40) ** 2 <= 10**2] = 9
    objmap[(y - 90) ** 2 + (x - 80) ** 2 <= 20**2] = 2
    rgb = np.zeros((120, 120, 3), dtype=np.uint8)
    for label, level in ((300, 60), (9, 140), (2, 230)):
        rgb[objmap == label] = level
    image = Image(rgb)
    image.objmap[:] = objmap

    size = MeasureSize().measure(image).set_index(str(OBJECT.LABEL))
    intensity = MeasureIntensity().measure(image).set_index(str(OBJECT.LABEL))
    depth = distance_transform_edt(objmap > 0)
    gray = image.gray[:]
    assert list(size.index) == [2, 9, 300]
    for label in (2, 9, 300):
        mask = objmap == label
        assert size.loc[label, str(SIZE.AREA)] == mask.sum()
        assert size.loc[label, str(SIZE.INSCRIBED_RADIUS)] == pytest.approx(
            depth[mask].max(), abs=1e-12
        )
        assert intensity.loc[label, str(INTENSITY.DENSITY)] == pytest.approx(
            gray[mask].sum() / mask.sum(), rel=1e-6
        )


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
