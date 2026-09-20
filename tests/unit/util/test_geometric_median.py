"""Solver-level guards for the Weiszfeld coincident-point singularity.

``weiszfeld_median`` used to floor the per-point distance at ``1e-10`` before
inverting it. A point sitting on the running estimate therefore took ~1e10 of
weight, captured the weighted update, and was returned as the answer with
``converged: True`` after a single iteration. Since the first estimate is the
*mean*, the returned value was the mean -- the non-robust statistic the
geometric median exists to avoid.

These guards sit at the solver, below ``robust_color_center``. The
colony-checker-level consequences are pinned separately in
``tests/unit/correction/test_color_checker_geometric_median.py``; both are
needed, because a fix at one level does not imply a fix at the other.

Spec: docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/
"""

from __future__ import annotations

import warnings

import numpy as np

from phenotypic.util._geometric_median import (
    _coincidence_atol,
    weiszfeld_median,
)


def _reference_median(points: np.ndarray) -> np.ndarray:
    """Vardi-Zhang run to machine precision, transcribed independently.

    This is independent of the *implementation*, not of the *algorithm*: it
    uses the same split, the same radius and the same gamma, differing only in
    initialization and tolerance. A wrong radius rule or a wrong gamma would be
    invisible to it. That is why the guards below also cross-check against
    ``_clipped_classical`` (a genuinely different algorithm) and against the
    objective, which no transcription can fake.
    """
    pts = np.asarray(points, dtype=np.float64)
    y = np.median(pts, axis=0)
    for _ in range(100_000):
        dist = np.linalg.norm(pts - y, axis=1)
        on_estimate = dist <= 1e-12 * max(float(np.linalg.norm(y)), 1.0)
        eta = int(on_estimate.sum())
        far = pts[~on_estimate]
        far_dist = dist[~on_estimate]
        if far.size == 0:
            return y
        weights = 1.0 / far_dist
        reweighted = (far * weights[:, None]).sum(axis=0) / weights.sum()
        if eta == 0:
            nxt = reweighted
        else:
            r = float(np.linalg.norm(((far - y) / far_dist[:, None]).sum(axis=0)))
            if r == 0.0:
                return y
            gamma = min(1.0, eta / r)
            nxt = (1.0 - gamma) * reweighted + gamma * y
        if np.linalg.norm(nxt - y) < 1e-13:
            return nxt
        y = nxt
    raise AssertionError("reference geometric median did not converge")


def _clipped_classical(points: np.ndarray) -> np.ndarray:
    """Classical Weiszfeld from the coordinate-wise median, clipped at 1e-15.

    A different algorithm from ``_reference_median`` -- no split, no damping --
    so agreement between the two is evidence about the answer rather than about
    a shared transcription.
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
    raise AssertionError("clipped classical reference did not converge")


def _old_floored_update(points: np.ndarray, eps: float, max_iter: int) -> np.ndarray:
    """The rule this change replaces, transcribed locally.

    Kept so the clean path can be checked for bit-identity against the code it
    replaced, rather than against a remembered iteration count.
    """
    pts = np.asarray(points, dtype=np.float64)
    x = pts.mean(axis=0)
    for _ in range(max_iter):
        x_old = x.copy()
        dist = np.maximum(np.linalg.norm(pts - x, axis=1), 1e-10)
        weights = 1.0 / dist
        x = np.sum(pts * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(x - x_old) < eps:
            return x
    return x


def _skewed_cloud(seed: int = 11) -> np.ndarray:
    """A checker-patch core lit unevenly: tight body plus a brightness ramp.

    The skew is the point: on a symmetric cloud the mean and the geometric
    median coincide and a broken solver looks correct.
    """
    rng = np.random.default_rng(seed)
    base = np.array([0.35, 0.37, 0.40])
    body = rng.normal(base, 0.004, size=(3000, 3))
    ramp = base + rng.random((900, 1)) * 0.45 + rng.normal(0, 0.004, size=(900, 3))
    pixels = np.clip(np.vstack([body, ramp]), 0.0, 1.0)
    return np.round(pixels * 255.0) / 255.0  # 8-bit, as real swatch pixels are


def _objective(x: np.ndarray, points: np.ndarray) -> float:
    """Sum of Euclidean distances -- the quantity the median minimises."""
    return float(np.linalg.norm(np.asarray(points) - x, axis=1).sum())


def test_a_point_exactly_on_the_estimate_does_not_capture_the_solve():
    """The classical singularity: one pixel at the centroid, out of 3901."""
    cloud = _skewed_cloud()
    points = np.vstack([cloud, cloud.mean(axis=0)])

    # The planted pixel is at distance 0 from the first estimate, so this
    # exercises the degenerate branch rather than depending on luck. Measured
    # exactly 0.0 today; if a numpy change ever makes it ~1e-16 the test still
    # exercises the intended branch, so only this precondition would need
    # relaxing to `<= _coincidence_atol(start)`.
    start = points.mean(axis=0)
    assert np.linalg.norm(points - start, axis=1).min() == 0.0

    got, info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)
    reference = _reference_median(points)

    assert np.allclose(got, reference, atol=1e-5)
    # Cross-check against a different algorithm, and against the objective --
    # the one thing no transcription can fake.
    assert np.allclose(got, _clipped_classical(points), atol=1e-5)
    assert _objective(got, points) < _objective(start, points)

    # The bug returned after exactly one iteration, pinned to the planted pixel.
    assert info["iterations"] > 1
    assert info["converged"] is True
    assert not np.any(np.all(points == got, axis=1))


def test_a_point_just_off_the_estimate_does_not_capture_the_solve():
    """Why the coincidence test is a radius and not an ``== 0`` comparison.

    A pixel 1.11e-16 from the estimate is not equal to it, so an exact-zero
    test lets it through to ``1/d`` with weight ~9e15 and it captures the
    solve just as the old 1e-10 floor did. Measured, that answers 21.896
    8-bit code values from the median -- the same magnitude as the original
    defect.
    """
    cloud = _skewed_cloud()
    points = np.vstack([cloud, cloud.mean(axis=0) + np.array([1e-16, 0.0, 0.0])])

    smallest = np.linalg.norm(points - points.mean(axis=0), axis=1).min()
    assert 0.0 < smallest < 1e-15  # near, but genuinely not, coincident

    got, _info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)
    assert np.allclose(got, _reference_median(points), atol=1e-5)


def test_every_point_identical_returns_that_point():
    """A fully degenerate cloud has no non-coincident points to reweight."""
    points = np.tile([7.0, 7.0, 7.0], (5, 1))

    got, info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)

    assert np.allclose(got, [7.0, 7.0, 7.0], atol=1e-12)
    assert info["converged"] is True


def test_converged_carries_the_subgradient_certificate():
    """``converged: True`` must certify optimality to the accuracy asked for.

    The guarantee the stopping rule actually gives is ``r <= eta + eps*W``
    (see the plan's Property 3), not ``r <= eta``: damping shortens the step by
    ``eta/W``, so the loop can stop while gamma < 1. Asserting the true bound
    rather than the ideal one is what makes this test both correct and able to
    fail -- the pre-fix answer on this input has ``eta = 0`` and ``r = 2317.7``
    against its own bound of ``4.36``, missing it by ~531x.
    """
    cloud = _skewed_cloud()
    points = np.vstack([cloud, cloud.mean(axis=0)])

    # The shipped constants (GEOMEDIAN_TOL / GEOMEDIAN_MAX_ITER, which
    # ColorCheckerProfile passes). This is not incidental: at a tighter
    # tolerance with more iterations the OLD floored rule escapes the capture
    # after its first step and converges correctly, so the guard would pass
    # against the very bug it exists to catch. Measured: old rule at
    # eps=1e-9/5000 lands 0.000 code values from the median; at eps=1e-6/200 it
    # lands 21.896 away.
    eps = 1e-6

    got, info = weiszfeld_median(points, eps=eps, max_iter=200, verbose=False)
    assert info["converged"] is True

    dist = np.linalg.norm(points - got, axis=1)
    on_estimate = dist <= _coincidence_atol(got)
    eta = int(on_estimate.sum())
    far = points[~on_estimate]
    far_dist = dist[~on_estimate]
    r = float(np.linalg.norm(((far - got) / far_dist[:, None]).sum(axis=0)))
    w_total = float((1.0 / far_dist).sum())

    # Measured at these constants: eta=0, r=1.240e-01, W=5.344e+05,
    # bound=5.344e-01 -> passes with ~4.3x margin. The pre-fix answer fails it:
    # r=2317.70 against its own bound of 4.360 (W is larger there, because the
    # planted pixel sits just outside the radius of the captured estimate), so
    # this assertion is not vacuous.
    assert r <= eta + eps * w_total


def test_the_non_degenerate_path_is_bit_identical_to_the_old_rule():
    """With no point in the coincidence radius, the arithmetic is the old one.

    Asserted as bit-identity against a local transcription of the floored
    update rather than as an iteration count: a count is a function of numpy's
    summation blocking and would be "fixed" by editing the number if a numpy
    upgrade moved it.

    This holds because no distance on this cloud falls in the band
    ``(radius, 1e-10]`` where the removed floor and the new rule genuinely
    differ -- the nearest point to any iterate here is ~1.24e-03.
    """
    cloud = _skewed_cloud()

    got, info = weiszfeld_median(cloud, eps=1e-6, max_iter=200, verbose=False)

    assert np.array_equal(got, _old_floored_update(cloud, 1e-6, 200))
    assert np.allclose(got, _reference_median(cloud), atol=1e-5)
    assert info["iterations"] == 12  # secondary; the bit-identity is the guard


def test_gamma_reaching_one_is_an_optimality_certificate():
    """``gamma == 1`` takes a bit-exact zero step, and nothing else did.

    Vardi-Zhang damping is the half of this change that the coincident-point
    split does not cover, and it is invisible on every other cloud in the
    repo: measured, the only non-zero gamma any of them produces is 4.3e-04,
    once in twelve passes, which is numerically indistinguishable from no
    damping at all.

    Why *this* cloud and not an obvious one. On a cloud whose median is merely
    *approached* geometrically the ``change < eps`` rule fires about six
    orders of magnitude before the iterate enters the 1e-12 coincidence
    radius -- measured on ``[[0,0]]*3 + [[5,0],[0,5]]``, which stops at
    ~6.2e-07 after 19 iterations with eta still 0. Putting the multiplicity
    mass *exactly at the mean* is what makes eta > 0 at iteration 0, which is
    the only way to reach the branch at all.

    Here eta = 2 and r = 1, so gamma = min(1, 2/1) = 1 and the update is
    ``0.0 * reweighted + 1.0 * x`` -- exactly x, so ``change`` is exactly 0.0
    and the solver certifies optimality on its first pass. The 1-D median of
    {-1, -1, 0, 0, 2} is 0, so [0, 0] really is the geometric median.

    Bit-exactness is the assertion, not closeness: a mutant that deletes the
    damping (``gamma := 0``) lands at -5.82e-07 after 22 iterations, which an
    approximate comparison would not distinguish from the right answer.
    """
    points = np.array(
        [[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]]
    )
    # The starting estimate is the mean, and it is the answer.
    assert np.array_equal(points.mean(axis=0), np.zeros(2))

    got, info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)

    assert np.array_equal(got, np.zeros(2))
    assert info["iterations"] == 1
    assert info["converged"] is True


def test_gamma_reaching_one_counts_the_full_multiplicity():
    """``eta`` is a count, and collapsing it to 1 steps off the median.

    The companion guard above is satisfied by ``eta := 1`` as well, because
    there r = 1 and min(1, 1/1) is still 1. Multiplicity only becomes
    load-bearing when ``1 < r <= eta``, so this cloud triples the mass at the
    mean: eta = 3 against r = 2, giving gamma = min(1, 3/2) = 1 while a
    solver that discarded the count would get gamma = 0.5 and take half a
    step away.

    The 1-D median of {-2, -1, -1, 0, 0, 0, 4} is 0, so [0, 0] is the
    geometric median. Measured: shipped returns exactly [0, 0] in one
    iteration with a gamma trace of [1.0]; ``eta := 1`` traces [0.5] and
    lands at -1.51e-06 after 33 iterations, and ``gamma := 0`` traces [0.0]
    and lands at -1.60e-06 after 37.
    """
    points = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [4.0, 0.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
            [-2.0, 0.0],
        ]
    )
    assert np.array_equal(points.mean(axis=0), np.zeros(2))

    got, info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)

    assert np.array_equal(got, np.zeros(2))
    assert info["iterations"] == 1
    assert info["converged"] is True


def test_a_zero_subgradient_at_a_data_point_is_an_early_certificate():
    """``r == 0`` with ``eta > 0``: x is optimal, and ``eta / r`` would raise.

    A symmetric pair with a third point between them: the two unit vectors
    from the estimate cancel exactly, so r is 0.0 and 0 lies in the
    subdifferential -- x is the median. ``eta`` is an ``int`` and ``r`` a
    Python ``float``, so without the early return ``eta / r`` raises
    ``ZeroDivisionError`` rather than yielding ``inf``. Measured: the shipped
    solver returns [0, 0] converged in one iteration; with the guard deleted
    the same call raises.

    This input discriminates that one branch and no other -- every other
    mutation of the update block returns [0, 0] in one iteration here too,
    which is why the branch survived the rest of the suite untouched.
    """
    points = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])

    got, info = weiszfeld_median(points, eps=1e-6, max_iter=200, verbose=False)

    assert np.array_equal(got, np.zeros(2))
    assert info["iterations"] == 1
    assert info["converged"] is True


def test_exhausting_the_iteration_budget_reports_not_converged():
    """The other half of the ``converged`` contract, which nothing asserted.

    Spec acceptance criterion 2 is that ``converged: True`` is reported *only*
    when the returned point is a genuine fixed point. Every existing guard
    checks the ``True`` side; a mutant whose exhaustion return reports
    ``converged=True`` passes all of them, because no test in the repo ran the
    solver to exhaustion.

    ``eps=0.0`` makes the ``change < eps`` test unsatisfiable -- a norm is
    never negative -- so the loop can only leave by running out of budget.
    Measured on this cloud at max_iter 1, 3 and 5: ``iterations`` equals
    ``max_iter`` and ``converged`` is False in every case.
    """
    got, info = weiszfeld_median(
        _skewed_cloud(), eps=0.0, max_iter=3, verbose=False
    )

    assert info["iterations"] == 3
    assert info["converged"] is False
    assert got.shape == (3,)


def test_the_coincidence_radius_is_flat_across_srgb_and_relative_above():
    """Bound ``_COINCIDENCE_RTOL`` from above as well as from below.

    ``test_a_point_just_off_the_estimate_does_not_capture_the_solve`` is the
    lower bound: at ``RTOL = 0`` the radius degenerates to an equality test
    and that guard fails. Nothing bounded the other side -- measured, both
    ``RTOL := 1e-9`` (1e3x too large) and ``RTOL := 1e-4`` (1e8x too large)
    passed every guard in this file. An oversized radius swallows real data
    points into the coincident set and returns something closer to the mean,
    which is the original defect wearing a different hat.

    Note what the radius actually does, because the constant's own docstring
    leads with the opposite emphasis. ``max(||x||, 1)`` clamps, so the radius
    is **flat at exactly 1e-12 across the whole sRGB [0, 1] range** -- it is
    not scale-relative there at all, and ``ColorCheckerProfile``, the caller
    with the highest exposure to this code, works entirely in [0, 1]. The
    relative half only begins to bite at L*a*b* magnitudes.

    The upper bound is that the radius stays orders below anything real: an
    8-bit code step is 1/255 ~= 3.9e-03, and L*a*b* nearest-neighbour spacing
    is larger still. Both margins below are measured values, not formulas.
    """
    srgb_mid = np.array([0.5, 0.0, 0.0])
    swatch_base = np.array([0.35, 0.37, 0.40])
    lab = np.array([50.0, 10.0, 20.0])

    # Flat 1e-12 everywhere in sRGB, origin included.
    assert _coincidence_atol(np.zeros(3)) == 1e-12
    assert _coincidence_atol(srgb_mid) == 1e-12
    assert _coincidence_atol(swatch_base) == 1e-12

    # Scale-relative only above unit norm. atol=0.0 matters: np.isclose's
    # default atol of 1e-08 dwarfs every value here and would pass anything.
    assert np.isclose(
        _coincidence_atol(np.ones(3)), 1.732051e-12, rtol=1e-6, atol=0.0
    )
    assert np.isclose(
        _coincidence_atol(lab), 5.477226e-11, rtol=1e-6, atol=0.0
    )

    # Upper bound: a millionfold-inflated radius would still be invisible at
    # 8-bit resolution, and it is still below a Lab spacing anyone can see.
    assert _coincidence_atol(swatch_base) * 1e6 < 1.0 / 255.0
    assert _coincidence_atol(lab) < 1e-6


def test_the_all_coincident_guard_keeps_the_solve_free_of_a_zero_division():
    """The ``far.size == 0`` branch is for arithmetic hygiene, not correctness.

    Say what it is for, because a reader looking for a correctness
    justification will not find one and may delete the branch. It is
    **redundant with the ``r == 0.0`` return**: measured on this cloud with
    the branch deleted, control falls through to ``1.0 / far_distances`` on an
    empty array, ``reweighted`` becomes nan from 0/0, ``eta = n > 0`` sends
    control to the else, ``r`` is the norm of a sum over an empty set and so
    is 0.0, and the ``r == 0.0`` guard returns the same point -- identical
    value, identical ``iterations``, identical ``converged``.

    The one thing that differs is a ``RuntimeWarning: invalid value
    encountered in divide`` from the 0/0, and that is the whole observable
    contribution of the branch. Measured: shipped emits no warning, the
    guard-less transcription emits exactly one. ``pyproject.toml``'s
    ``filterwarnings`` carries a single ``ignore::SyntaxWarning:mahotas`` and
    no ``error`` entry, so nothing else in the suite would go red on it --
    which is why this assertion has to be explicit rather than left to a
    global filter. Should that filter ever be tightened to ``error``, the
    guard-less path would raise rather than warn: this test would still fail,
    but for a reason this docstring does not describe.

    ``simplefilter("always")`` is required, and not defensively. Warnings are
    deduplicated per code location, and
    ``test_every_point_identical_returns_that_point`` above puts the *same*
    ``[7, 7, 7]`` cloud through the *same* ``1.0 / far_distances`` line. Were
    the filters left alone, a guard-less solver would warn there, be silently
    deduplicated here, and this test would pass against the very mutant it
    exists to kill. Mutating the filters is what avoids that -- it bumps the
    filter version and so invalidates the cached ``__warningregistry__``.
    """
    points = np.tile([7.0, 7.0, 7.0], (5, 1))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got, info = weiszfeld_median(
            points, eps=1e-6, max_iter=200, verbose=False
        )

    # Listed rather than counted, so a failure names the warning.
    assert [str(w.message) for w in caught] == []
    assert np.allclose(got, [7.0, 7.0, 7.0], atol=1e-12)
    assert info["iterations"] == 1
    assert info["converged"] is True
