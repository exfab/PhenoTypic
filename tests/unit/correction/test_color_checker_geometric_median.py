"""Guards for the colour-checker patch geometric median.

The patch colour used to come from a local Weiszfeld loop in
``_color_correction/_helpers.py`` that spent a single ``eps`` (default
``1e-3``, in sRGB ``[0, 1]``) on three jobs at once: the convergence test on
the update norm, the floor under ``1 / distance``, and a "the estimate has
landed on a data point" test that returned *that data point verbatim*.

Both of the overloaded uses are pinned here, because each one fails
differently and a fix for one does not imply a fix for the other:

* ``test_patch_center_is_not_short_circuited_by_a_coincident_pixel`` pins the
  degenerate branch -- a pixel sitting on the running estimate must not end
  the solve and be returned as the answer. Routing through
  ``robust_color_center`` did not close this branch on its own: the shared
  solver floored the distance at ``1e-10``, which handed that pixel
  ~99.9995% of the weight and pinned the estimate to it anyway. Closed by the
  Vardi-Zhang coincident-point split in ``weiszfeld_median``; see
  ``docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/``
  and the solver-level guards in ``tests/unit/util/test_geometric_median.py``.
* ``test_patch_center_converges_past_a_loose_tolerance`` pins the convergence
  test -- stopping at ``1e-3`` leaves the estimate short of the geometric
  median by more than the tolerance this module actually asks for.

Both are written against ``robust_color_center`` under the very constants
``ColorCheckerProfile`` passes it, so they fail if those constants regress.
"""

from __future__ import annotations

import numpy as np

from phenotypic.correction._color_correction._color_checker_profile import (
    GEOMEDIAN_MAX_ITER,
    GEOMEDIAN_TOL,
)
from phenotypic.util._robust_color_stats import robust_color_center


def _reference_geometric_median(points: np.ndarray) -> np.ndarray:
    """Weiszfeld run to machine precision, independent of the shipped solver.

    Deliberately a separate transcription rather than a call into
    ``phenotypic``: it is the witness the guards are checked against, so it
    must not share a bug with the code under test.
    """
    pts = np.asarray(points, dtype=np.float64)
    guess = np.median(pts, axis=0)
    for _ in range(100_000):
        dist = np.clip(np.linalg.norm(pts - guess, axis=1), 1e-15, None)
        weights = 1.0 / dist
        nxt = (pts * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(nxt - guess) < 1e-15:
            return nxt
        guess = nxt
    raise AssertionError("reference geometric median did not converge")


def _skewed_swatch(seed: int = 11) -> np.ndarray:
    """A checker patch core lit unevenly: tight body + an illumination ramp.

    The skew is the point. On a symmetric cloud the mean and the geometric
    median coincide and every implementation looks correct; it is the skew
    that separates them, and skew is what the geometric median was chosen to
    survive in the first place.
    """
    rng = np.random.default_rng(seed)
    base = np.array([0.35, 0.37, 0.40])
    body = rng.normal(base, 0.004, size=(3000, 3))
    ramp = base + rng.random((900, 1)) * 0.45 + rng.normal(0, 0.004, size=(900, 3))
    pixels = np.clip(np.vstack([body, ramp]), 0.0, 1.0)
    return np.round(pixels * 255.0) / 255.0  # 8-bit, as real swatch pixels are


def test_patch_center_is_not_short_circuited_by_a_coincident_pixel():
    """A pixel lying on the running estimate must not be returned as the answer.

    The first Weiszfeld estimate is the *mean*. On a skewed patch the mean is
    far from the geometric median, so an implementation that returns the
    nearest data point as soon as one falls within ``eps`` of the estimate
    answers with the wrong colour entirely. A patch core is tens of thousands
    of pixels, so a pixel at the quantised centroid is ordinary, not exotic --
    it is placed exactly here to make the branch fire deterministically
    instead of depending on the density of a random draw.
    """
    swatch = _skewed_swatch()
    pixels = np.vstack([swatch, swatch.mean(axis=0)])  # a pixel *at* the mean

    center = robust_color_center(
        pixels, max_iter=GEOMEDIAN_MAX_ITER, tol=GEOMEDIAN_TOL
    )
    reference = _reference_geometric_median(pixels)
    mean = pixels.mean(axis=0)

    # The mean and the geometric median must actually differ here, or the
    # assertions below would hold for a broken solver too.
    assert np.linalg.norm(mean - reference) * 255.0 > 5.0

    # The answer is the geometric median, not the planted pixel / the mean.
    assert np.allclose(center, reference, atol=1e-5)
    assert np.linalg.norm(center - mean) * 255.0 > 5.0

    # And it is not a raw input pixel that happened to be handed back.
    assert not np.any(np.all(pixels == center, axis=1))


def test_patch_center_converges_past_a_loose_tolerance():
    """Convergence must be tighter than the old ``eps=1e-3`` stopping rule.

    Halting once the update norm drops below ``1e-3`` leaves the estimate a
    visible fraction of an 8-bit code value away from the true geometric
    median. The bound below sits between the two: comfortably above what the
    shipped constants achieve, and comfortably below what the loose rule
    settles for, so it is a bound that can actually fail.
    """
    pixels = _skewed_swatch()

    center = robust_color_center(
        pixels, max_iter=GEOMEDIAN_MAX_ITER, tol=GEOMEDIAN_TOL
    )
    reference = _reference_geometric_median(pixels)

    error_code_values = np.linalg.norm(center - reference) * 255.0
    assert error_code_values < 1e-2


def test_profile_geomedian_constants_are_tight_enough():
    """The constants themselves are the contract; pin them against drift."""
    assert GEOMEDIAN_TOL <= 1e-5  # well inside an 8-bit step (1/255 ~= 3.9e-3)
    assert GEOMEDIAN_MAX_ITER >= 100


def test_patch_center_handles_a_uniform_patch():
    """A perfectly flat patch must return that colour, not NaN or a drift."""
    flat = np.tile([0.42, 0.44, 0.46], (500, 1))
    center = robust_color_center(
        flat, max_iter=GEOMEDIAN_MAX_ITER, tol=GEOMEDIAN_TOL
    )
    assert np.allclose(center, [0.42, 0.44, 0.46], atol=1e-9)
