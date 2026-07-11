"""Unit tests for MeasureShape's radial-signature geometry.

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

import numpy as np
import pytest
from scipy.stats import trim_mean

from phenotypic.measure import MeasureShape

TOL = 0.6  # pixels; see module docstring


def _crop(mask: np.ndarray) -> np.ndarray:
    """Crop a mask to its bounding box, mimicking regionprops.image."""
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
    op = MeasureShape()
    mask = _crop(_disk(40))
    profile = op._measure_radial_profile(mask)

    assert profile["Shape_InscribedRadius"] == pytest.approx(40.0, abs=TOL)
    assert profile["Shape_RobustMeanRadius"] == pytest.approx(40.0, abs=TOL)
    # ReachRadius lands on the 0.5 iso-level, half a pixel outside the disk.
    assert profile["Shape_ReachRadius"] == pytest.approx(40.5, abs=TOL)


def test_mean_boundary_dist_of_a_disk_is_one_third_of_its_radius():
    """Pins the interpretation the rename encodes: this column is not a radius."""
    profile = MeasureShape()._measure_radial_profile(_crop(_disk(40)))
    # Analytic: mean EDT over a disk of radius R is R/3. Discretisation of the
    # boundary lifts it slightly, so allow 0.3 px on a predicted 13.33.
    assert profile["Shape_MeanBoundaryDist"] == pytest.approx(40.0 / 3.0, abs=0.3)


def test_angular_sampling_matches_the_analytic_mean_radius_of_an_ellipse():
    """The mean of r(theta) over an ellipse, computed analytically, is 39.2188.

    This is deliberately not the arc-length mean (45.51) nor the
    equivalent-circle radius (41.83). Getting 39.22 proves the signature is
    sampled uniformly in angle.
    """
    op = MeasureShape()
    a, b = 70.0, 25.0
    y, x = np.mgrid[-200:200, -200:200]
    mask = _crop((x / a) ** 2 + (y / b) ** 2 <= 1)

    edt = np.pad(mask, 1)
    from scipy.ndimage import distance_transform_edt

    edt = distance_transform_edt(edt)[1:-1, 1:-1]
    signature = op._trace_radial_signature(mask, edt)

    theta = np.linspace(0, 2 * np.pi, 200_001)[:-1]
    analytic = (a * b / np.hypot(b * np.cos(theta), a * np.sin(theta))).mean()

    assert analytic == pytest.approx(39.2188, abs=1e-3)  # guard the guard
    assert signature.mean() == pytest.approx(analytic, abs=0.25)


def test_runner_does_not_break_down_the_robust_mean():
    """A 20% trimmed mean tolerates up to 20% contamination. Under angular
    sampling the runner occupies its 2.4% angular width, so the estimate holds.
    """
    op = MeasureShape()
    profile = op._measure_radial_profile(_crop(_disk_with_runner()))

    assert profile["Shape_RobustMeanRadius"] == pytest.approx(40.0, abs=TOL)
    assert profile["Shape_ReachRadius"] == pytest.approx(90.5, abs=TOL)
    assert profile["Shape_InscribedRadius"] == pytest.approx(40.0, abs=TOL)


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
    op = MeasureShape()
    edt = np.pad(mask, 1)
    from scipy.ndimage import distance_transform_edt

    edt = distance_transform_edt(edt)[1:-1, 1:-1]
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

    If _trace_radial_signature is switched to mean-per-bin, ReachRadius here
    drops from ~90.5 to ~61 -- 29 px, far outside TOL -- and this fails.
    """
    op = MeasureShape(angular_bins=16)
    profile = op._measure_radial_profile(_crop(_disk_with_runner()))

    # Max-per-bin preserves the runner tip even at coarse resolution.
    assert profile["Shape_ReachRadius"] == pytest.approx(90.5, abs=TOL)
    # The mean-per-bin value (~61.5) must be excluded by a wide margin, so the
    # assertion above cannot pass under that mutation.
    assert profile["Shape_ReachRadius"] > 61.5 + 10 * TOL


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
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        profile = MeasureShape()._measure_radial_profile(mask)

    assert np.isfinite(profile["Shape_InscribedRadius"])
    assert np.isfinite(profile["Shape_RobustMeanRadius"])
