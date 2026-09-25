"""Unit tests for MeasureSize's radial-signature geometry.

Every expected value here is checked against an analytic shape. Tolerances are
derived from two error sources:

* Angular binning: taking the outermost contour crossing per bin overestimates
  by at most about (1/2) * r * dtheta^2. At K=360, dtheta = 2*pi/360 = 0.01745,
  so for r ~ 40 that is under 0.01 px.
* The marching-squares contour sits on the 0.5 iso-level, which displaces the
  boundary by up to half a pixel.

The half-pixel term dominates, so 0.6 px is the working tolerance: roughly
1.2x the dominant error. It is tight enough to catch the failure this code
exists to prevent -- boundary-pixel sampling of a colony with a runner is off
by 2.03 px -- which `test_boundary_pixel_sampling_would_break_down` proves.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import trim_mean

from phenotypic.measure import MeasureSize
from phenotypic.measure._object_geometry import object_edt

TOL = 0.6  # pixels; see module docstring


def _crop(mask: np.ndarray) -> np.ndarray:
    """Crop a mask to its bounding box, mimicking regionprops.image.

    Only for a mask with no all-background row or column inside its box:
    `np.ix_` drops such rows, which would pull separate pieces together. The
    multi-piece tests below go through `MeasureSize().measure` instead.
    """
    return mask[np.ix_(mask.any(axis=1), mask.any(axis=0))]


def _disk(radius: int, half: int = 130) -> np.ndarray:
    y, x = np.mgrid[-half:half, -half:half]
    return x**2 + y**2 <= radius**2


def _disk_with_runner() -> np.ndarray:
    """Radius-40 colony with a half-width-3 runner reaching to r = 90.

    The runner subtends 2*arctan(3/40) = 8.6 degrees, i.e. 2.4% of all
    directions, but it is roughly 2 * 50 = 100 pixels of boundary arc length.
    """
    y, x = np.mgrid[-130:130, -130:130]
    return (x**2 + y**2 <= 40**2) | ((np.abs(y) <= 3) & (x >= 0) & (x <= 90))


def test_signature_of_a_disk_is_its_radius():
    op = MeasureSize()
    mask = _crop(_disk(40))
    profile = op._measure_radial_profile(mask)

    assert profile["Size_InscribedRadius"] == pytest.approx(40.0, abs=TOL)
    assert profile["Size_RobustMeanRadius"] == pytest.approx(40.0, abs=TOL)
    # MaxRadius lands on the 0.5 iso-level, half a pixel outside the disk.
    assert profile["Size_MaxRadius"] == pytest.approx(40.5, abs=TOL)


def test_angular_sampling_matches_the_analytic_mean_radius_of_an_ellipse():
    """The mean of r(theta) over an ellipse, computed analytically, is 39.2188.

    This is deliberately not the arc-length mean (45.51) nor the
    equivalent-circle radius (41.83). Getting 39.22 proves the signature is
    sampled uniformly in angle.
    """
    op = MeasureSize()
    a, b = 70.0, 25.0
    y, x = np.mgrid[-200:200, -200:200]
    mask = _crop((x / a) ** 2 + (y / b) ** 2 <= 1)
    signature = op._trace_radial_signature(mask, object_edt(mask))

    theta = np.linspace(0, 2 * np.pi, 200_001)[:-1]
    analytic = (a * b / np.hypot(b * np.cos(theta), a * np.sin(theta))).mean()

    assert analytic == pytest.approx(39.2188, abs=1e-3)  # guard the guard
    assert signature.mean() == pytest.approx(analytic, abs=0.25)


def test_runner_does_not_break_down_the_robust_mean():
    """A 20% trimmed mean tolerates up to 20% contamination. Under angular
    sampling the runner occupies its 2.4% angular width, so the estimate holds.
    """
    op = MeasureSize()
    profile = op._measure_radial_profile(_crop(_disk_with_runner()))

    assert profile["Size_RobustMeanRadius"] == pytest.approx(40.0, abs=TOL)
    assert profile["Size_MaxRadius"] == pytest.approx(90.5, abs=TOL)
    assert profile["Size_InscribedRadius"] == pytest.approx(40.0, abs=TOL)


def test_boundary_pixel_sampling_would_break_down():
    """Mutation control. Proves the tolerance in the test above does real work.

    Sample the same colony's boundary by pixel instead of by angle. The runner
    then supplies ~30% of the samples, exceeding the trimmed mean's 20%
    breakdown point, and the estimate lands 2.03 px away from the truth --
    well outside TOL. If this ever falls inside TOL, the test above has
    stopped discriminating.
    """
    from skimage.segmentation import find_boundaries

    mask = _crop(_disk_with_runner())
    op = MeasureSize()
    edt = object_edt(mask)
    center = np.argwhere(edt >= 0.99 * edt.max()).mean(axis=0)

    pixels = np.argwhere(find_boundaries(mask, mode="inner")).astype(float)
    radii = np.hypot(pixels[:, 0] - center[0], pixels[:, 1] - center[1])

    contamination = float((radii > 45).mean())
    assert contamination > 0.20, "runner must exceed the 20% breakdown point"
    assert abs(trim_mean(radii, 0.2) - 40.0) > TOL

    # And the angular scheme keeps the runner under the breakdown point.
    signature = op._trace_radial_signature(mask, edt)
    assert float((signature > 45).mean()) < 0.05


def test_reach_uses_the_outermost_crossing_per_bin_not_the_mean():
    """Mutation control for the max-per-bin aggregation.

    Each angular bin must report the *outermost* contour crossing, not the mean
    of the crossings that fall in it. The two agree once every bin is narrow
    enough to hold a single boundary feature, so this test forces them apart
    with a deliberately coarse bin count: at angular_bins=16 the runner's tip
    (r=90.5) shares a ~22.5-degree bin with the disk edge (r=40). The outermost
    crossing keeps the tip; a mean-per-bin would average it down toward ~61.

    If _trace_radial_signature is switched to mean-per-bin, MaxRadius here
    drops from ~90.5 to ~61 -- 29 px, far outside TOL -- and this fails.
    """
    op = MeasureSize(angular_bins=16)
    profile = op._measure_radial_profile(_crop(_disk_with_runner()))

    # Max-per-bin preserves the runner tip even at coarse resolution.
    assert profile["Size_MaxRadius"] == pytest.approx(90.5, abs=TOL)
    # The mean-per-bin value (~61.5) must be excluded by a wide margin, so the
    # assertion above cannot pass under that mutation.
    assert profile["Size_MaxRadius"] > 61.5 + 10 * TOL


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((1, 1), dtype=bool),
        np.ones((2, 2), dtype=bool),
        np.ones((1, 5), dtype=bool),
    ],
    ids=["single-pixel", "2x2", "thin-line"],
)
def test_degenerate_objects_do_not_raise(mask):
    """Specks survive detection. They must not crash or warn the measurer."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        profile = MeasureSize()._measure_radial_profile(mask)

    assert np.isfinite(profile["Size_InscribedRadius"])
    assert np.isfinite(profile["Size_RobustMeanRadius"])
    assert np.isfinite(profile["Size_MedianRadius"])
    assert np.isfinite(profile["Size_MeanRadius"])
    assert np.isfinite(profile["Size_MaxRadius"])


def test_disk_median_and_mean_radius_equal_its_radius():
    profile = MeasureSize()._measure_radial_profile(_crop(_disk(40)))
    # The contour sits on the 0.5 iso-level: within TOL of both 40 and 40.5.
    assert profile["Size_MedianRadius"] == pytest.approx(40.0, abs=TOL)
    assert profile["Size_MeanRadius"] == pytest.approx(40.0, abs=TOL)


def test_elongated_colony_matches_the_analytic_rectangle():
    """A 20-row x 100-column pixel block. Its 0.5-iso contour is exactly a 100 x 20
    rectangle about the EDT plateau centroid, so the analytic values from the
    logic-validation script (check 04) apply: 10.0 / 14.1 / 21.0 / 16.2 / 50.9.
    Two rasterisation effects move the measured values, both inside TOL:
    marching squares cuts each corner diagonally, lowering MaxRadius by under
    0.1 px; and max-per-bin takes the extreme vertex in each 1-degree bin, which
    where r(theta) is steep (~4.5 px/degree near the corners) exceeds the
    bin-centre value by up to half a bin x |r'|. Measured on main's code
    (plan-review probe P3): 14.147 / 21.126 / 16.196 / 50.895, largest
    deviation 0.126 px against TOL = 0.6.
    """
    profile = MeasureSize()._measure_radial_profile(np.ones((20, 100), dtype=bool))
    assert profile["Size_InscribedRadius"] == 10.0  # exact: integer EDT
    assert profile["Size_MedianRadius"] == pytest.approx(14.1, abs=TOL)
    assert profile["Size_MeanRadius"] == pytest.approx(21.0, abs=TOL)
    assert profile["Size_RobustMeanRadius"] == pytest.approx(16.2, abs=TOL)
    assert profile["Size_MaxRadius"] == pytest.approx(50.9, abs=TOL)


def _disk_with_wide_runner() -> np.ndarray:
    """Radius-40 colony with a half-width-8 runner reaching to x = 100.

    Analytic (logic-validation script check 06): runner covers 5.6% of directions,
    MeanRadius 42.35, RobustMeanRadius 40.00, MedianRadius 40.00.
    """
    y, x = np.mgrid[-130:130, -130:130]
    return (x**2 + y**2 <= 40**2) | ((np.abs(y) <= 8) & (x >= 0) & (x <= 100))


def test_runner_pulls_the_mean_but_not_the_robust_mean():
    """The analytic gap is 2.35 px, about twice 2 x TOL, so a swap of the two
    estimators cannot pass: each value may drift by at most TOL.

    Mutation: return trim_mean for MEAN_RADIUS and the plain mean for
    ROBUST_MEAN_RADIUS -> the first assertion fails.
    """
    profile = MeasureSize()._measure_radial_profile(_crop(_disk_with_wide_runner()))
    assert profile["Size_RobustMeanRadius"] == pytest.approx(40.0, abs=TOL)
    assert profile["Size_MeanRadius"] - profile["Size_RobustMeanRadius"] > 2 * TOL
    assert profile["Size_MedianRadius"] == pytest.approx(40.0, abs=TOL)


def test_crescent_keeps_the_radius_ordering():
    """Review Focus 4. A concave colony (disk r=40 minus an offset disk r=30) is not
    star-shaped from its center, so several bins cross the boundary more than
    once. The ordering invariants must still hold."""
    y, x = np.mgrid[-60:60, -60:60]
    mask = _crop((x**2 + y**2 <= 40**2) & ((x - 20) ** 2 + y**2 > 30**2))
    p = MeasureSize()._measure_radial_profile(mask)
    assert p["Size_InscribedRadius"] <= p["Size_MedianRadius"] <= p["Size_MaxRadius"]
    assert p["Size_InscribedRadius"] <= p["Size_RobustMeanRadius"] <= p["Size_MaxRadius"]
    assert p["Size_InscribedRadius"] <= p["Size_MeanRadius"] <= p["Size_MaxRadius"]


# ---------------------------------------------------------------------------
# Labels that are not one 4-connected piece (phase-1 review HIGH-1, option A).
#
# The signature pools every contour of the label, so each bin keeps the
# outermost crossing of any piece. These go through MeasureSize().measure so
# that the crop is regionprops' own `props.image`.
#
# Mutation: restore `max(contours, key=len)` in _trace_radial_signature. With
# the default 4-connected tracing, all three tests fail; with
# fully_connected="high" kept, the two diagonal cases become one contour and
# only the separate-line case fails. Dropping fully_connected="high" alone is
# an equivalent mutant once contours are pooled: connectivity only decides how
# a saddle cell's four edge crossings pair into contours, never which exist.
# ---------------------------------------------------------------------------


def _measure_one_label(mask: np.ndarray):
    from phenotypic import Image

    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[mask] = 200
    image = Image(rgb)
    image.objmap[:] = mask.astype(int)
    assert image.num_objects == 1
    return MeasureSize().measure(image).iloc[0]


def test_a_separate_piece_under_the_same_label_is_measured_with_the_body():
    """Disk of radius 15 centred at (50, 50), plus a 1-px line on row 150 from
    column 20 to 239, all one label, as a merging refiner leaves fragments.

    Mechanism, from the disk centre (the plateau is the disk's, symmetric, so
    the centre is exactly (50, 50)):

    * MaxRadius: the farthest crossing is the line's end cap, the 0.5-iso
      vertex half a pixel past its last pixel, at offset (100, 189.5). Exact.
    * The line's crossings lie at angles 27.8 to 107.0 degrees, so they own
      bins 207..286: 80 of 360, each at least 99.5 px (the near edge of the
      line). The other 280 bins are the disk's, within TOL of 15.
    * MedianRadius: 80 < 180, so the median is a disk bin.
    * The trim drops 72 bins per end, so 8 line bins survive into
      RobustMeanRadius and all 80 into MeanRadius; lower bounds follow.
    * InscribedRadius is the disk's: nearest background at offset (15, 1).

    The old longest-contour rule measured the line from the disk's centre
    (MedianRadius 149.9, review HIGH-1 row 1).
    """
    y, x = np.mgrid[0:200, 0:260]
    mask = (y - 50) ** 2 + (x - 50) ** 2 <= 15**2
    mask[150, 20:240] = True
    row = _measure_one_label(mask)

    assert row["Size_InscribedRadius"] == pytest.approx(np.sqrt(226), abs=1e-12)
    assert row["Size_MaxRadius"] == pytest.approx(np.hypot(100.0, 189.5), abs=1e-9)
    assert row["Size_MedianRadius"] == pytest.approx(15.0, abs=TOL)
    disk_floor = 15.0 - TOL
    assert row["Size_RobustMeanRadius"] >= (208 * disk_floor + 8 * 99.5) / 216
    assert row["Size_MeanRadius"] >= (280 * disk_floor + 80 * 99.5) / 360


def test_a_runner_joined_only_at_a_corner_sets_the_max_radius():
    """Disk of radius 20 centred at (40, 40), plus a 1-px diagonal runner of
    pixels (54 + k, 54 + k), k = 0..39. Pixel (54, 54) is in the disk and
    (55, 55) is not, so the runner touches the disk only at a corner: one
    8-connected label, two 4-connected pieces.

    Mechanism: the runner lies on the diagonal, so the label is symmetric under
    transposition and the plateau centre stays at (40, 40). The runner's
    crossings sit within 0.354 px of the diagonal and at least 20.8 px out, so
    within 0.97 degrees of 45 degrees; 45 degrees is a bin edge, so they own
    bins 224 and 225 and nothing else. Each keeps the tip vertex at offset
    (53.5, 53) or (53, 53.5).

    * MaxRadius = hypot(53.5, 53) exactly.
    * MeanRadius rises by exactly those two bins over the plain disk:
      2 * (reach - s) / 360, where s is the disk's value in each bin, within
      TOL of 20. (Their neighbours hold disk vertices, so no interpolation ramp
      forms.)
    * MedianRadius and RobustMeanRadius stay on the disk.

    The old rule traced 4-connected, left the runner out, and reported the
    disk's 20.5 (review HIGH-1 row 3).
    """
    y, x = np.mgrid[0:120, 0:120]
    disk = (y - 40) ** 2 + (x - 40) ** 2 <= 20**2
    mask = disk.copy()
    for k in range(40):
        mask[54 + k, 54 + k] = True
    row = _measure_one_label(mask)
    plain = _measure_one_label(disk)
    reach = np.hypot(53.5, 53.0)

    assert row["Size_MaxRadius"] == pytest.approx(reach, abs=1e-9)
    lift = row["Size_MeanRadius"] - plain["Size_MeanRadius"]
    assert 2 * (reach - (20.0 + TOL)) / 360 <= lift <= 2 * (reach - (20.0 - TOL)) / 360
    assert row["Size_MedianRadius"] == pytest.approx(20.0, abs=TOL)
    assert row["Size_RobustMeanRadius"] == pytest.approx(20.0, abs=TOL)
    assert row["Size_InscribedRadius"] == plain["Size_InscribedRadius"]


def test_a_diagonal_line_is_one_object_for_the_signature():
    """A 15-px line of pixels (5 + k, 5 + k): one 8-connected label whose
    pixels share no edge, so 4-connected tracing sees 15 separate squares.

    Mechanism: every pixel's distance-transform value is 1, so the whole line
    is the plateau. Taken 8-connected, it is one component, and the centre is
    its centroid, the middle pixel (7, 7) in the crop.

    * MaxRadius: the end pixels' outer crossings, e.g. (14.5, 14), sit at
      offset (7.5, 7) from the centre, so hypot(7.5, 7) exactly.
    * MedianRadius: the middle pixel's own crossings, 0.5 px out on the
      axes, fill the bins at 0, 90, 180 and 270 degrees (bin indices 180,
      270, 0 and 90). Every other crossing, at offsets (j, j +- 0.5) and
      (j +- 0.5, j), lies in the first or third quadrant. The other two
      quadrants therefore lie between two 0.5-valued bins, and
      interpolation holds their 182 bins (0, 90..180, 270..359) at exactly
      0.5. Every other bin is larger, so the median is 0.5.

    Mutations: the old rule traced 4-connected and kept one square, 0.5 for
    every radius (review HIGH-1 row 4). A 4-connected plateau makes the first
    pixel alone the plateau, puts the centre on the line's end, and gives
    MaxRadius hypot(14.5, 14) = 20.16.
    """
    mask = np.zeros((30, 30), dtype=bool)
    for k in range(15):
        mask[5 + k, 5 + k] = True
    row = _measure_one_label(mask)

    assert row["Size_InscribedRadius"] == 1.0
    assert row["Size_MaxRadius"] == pytest.approx(np.hypot(7.5, 7.0), abs=1e-9)
    assert row["Size_MedianRadius"] == 0.5


def test_a_hole_never_supplies_a_bin():
    """A disk of radius 12 with a centred hole of radius 4, as a colony with
    central lysis looks. The signature pools the outlines of the hole-filled
    label, so the hole contributes nothing: from the same centre the ring's
    signature is the disk's, bin for bin.

    Why the fill is needed: an outline of radius R has about 8R vertices, so at
    R = 12, 8R + 4 = 100 vertices spread over 360 bins. Without it, the hole's
    vertices (about 4 px out) land in bins the outer outline left empty, and
    those bins read the hole's radius where they should interpolate the outer
    one (measured: 64 of 360 bins differ, minimum 3.64).

    The centre is pinned by passing the disk's own distance transform, whose
    plateau is symmetric about the disk centre. The contract takes `edt` as an
    argument, which makes this possible.

    Mutation: trace `obj_mask` instead of `binary_fill_holes(obj_mask)` -> the
    signatures differ and the minimum drops to the hole's radius.
    """
    y, x = np.mgrid[-15:16, -15:16]
    disk = x**2 + y**2 <= 12**2
    ring = disk & (x**2 + y**2 > 4**2)
    disk, ring = _crop(disk), _crop(ring)
    edt = object_edt(disk)
    op = MeasureSize()

    ring_signature = op._trace_radial_signature(ring, edt)
    disk_signature = op._trace_radial_signature(disk, edt)
    assert np.array_equal(ring_signature, disk_signature)
    assert ring_signature.min() >= 12.0 - TOL
