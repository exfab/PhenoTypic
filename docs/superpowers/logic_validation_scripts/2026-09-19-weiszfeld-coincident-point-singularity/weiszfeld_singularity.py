#!/usr/bin/env python
"""Re-derive every numeric claim in the Weiszfeld coincident-point spec.

Subject: ``phenotypic.util._geometric_median.weiszfeld_median``. Before this
change it clamped the per-point distance with ``np.maximum(distances, 1e-10)``.
A point coinciding with the running estimate therefore received weight ``1e10``
and captured the weighted update, pinning the estimate to that point and
reporting convergence. The fix splits coincident points out of the reweighting
(Vardi & Zhang 2000) and removes the floor outright.

Per the project file rule this script **does not import phenotypic**. Every
rule it exercises is re-implemented from scratch, so it is an independent
witness rather than a restatement of the code under test.

``floored_weiszfeld`` below is deliberately the **former** rule, not the
current one. Claims 3, 4 and 6 are claims *about the defect*, so they must keep
describing the code that had it; they stay green after the fix precisely
because nothing here imports the module that changed. Do not "update" them.

Run:  uv run python <this file>
Exits non-zero if any claim fails.
"""

from __future__ import annotations

import sys

import numpy as np

FAILURES: list[str] = []
CLAMP = 1e-10  # the distance floor weiszfeld_median carried before the fix


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Re-implementations (independent of src/)
# ---------------------------------------------------------------------------


def floored_weiszfeld(points, eps=1e-6, max_iter=200):
    """The **former** update rule, transcribed: clamp the distance, reweight.

    This is the rule the fix replaced, not the one that ships now. It is kept
    verbatim because claims 3, 4 and 6 are claims about the defect and need the
    defective arithmetic to demonstrate it. The defaults are
    ``ColorCheckerProfile``'s constants (``GEOMEDIAN_TOL`` = 1e-6,
    ``GEOMEDIAN_MAX_ITER`` = 200); every number this script prints from
    it is at that configuration.
    """
    pts = np.asarray(points, dtype=np.float64)
    x = pts.mean(axis=0)
    for iteration in range(max_iter):
        x_old = x.copy()
        distances = np.linalg.norm(pts - x, axis=1)
        distances = np.maximum(distances, CLAMP)
        weights = 1.0 / distances
        x = (pts * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(x - x_old) < eps:
            return x, iteration + 1, True
    return x, max_iter, False


def vardi_zhang(points, eps=1e-12, max_iter=100_000):
    """Vardi & Zhang (2000) modified Weiszfeld; handles coincident points.

    Splits the estimate's own coincident mass out of the reweighting and
    takes a damped step, so a point sitting on the estimate cannot capture it.
    """
    pts = np.asarray(points, dtype=np.float64)
    y = np.median(pts, axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(pts - y, axis=1)
        coincident = d <= 0.0
        eta = int(coincident.sum())
        far = pts[~coincident]
        d_far = d[~coincident]
        if far.size == 0:
            return y  # every point sits on y
        w = 1.0 / d_far
        T = (far * w[:, None]).sum(axis=0) / w.sum()
        if eta == 0:
            y_next = T
        else:
            r = np.linalg.norm(((far - y) / d_far[:, None]).sum(axis=0))
            if r == 0.0:
                return y
            gamma = min(1.0, eta / r)
            y_next = (1.0 - gamma) * T + gamma * y
        if np.linalg.norm(y_next - y) < eps:
            return y_next
        y = y_next
    raise AssertionError("Vardi-Zhang did not converge")


def vz_mean(points, coincide="radius", eps=1e-12, max_iter=100_000):
    """Vardi-Zhang from the *mean*, with a selectable coincidence test.

    The shipped solver starts at the mean, and the singularity is reachable
    only from there: the planted pixel sits at the centroid. ``vardi_zhang``
    above starts at the coordinate-wise median and is the ground truth claims
    1-7 consume; it must not be used as the subject here, because from a median
    start neither coincidence test ever fires and the comparison is vacuous.

    ``coincide="exact"`` is the ``d == 0`` rule the spec first prescribed;
    ``"radius"`` is the rule that ships, ``1e-12 * max(||y||, 1)``. On this
    sRGB swatch the ``max`` always clamps, so here the radius is a flat
    ``1e-12`` absolute; the relative half only bites above unit norm.
    """
    pts = np.asarray(points, dtype=np.float64)
    y = pts.mean(axis=0)
    for _ in range(max_iter):
        y_old = y.copy()
        d = np.linalg.norm(pts - y, axis=1)
        if coincide == "exact":
            on = d <= 0.0
        else:
            on = d <= 1e-12 * max(float(np.linalg.norm(y)), 1.0)
        eta = int(on.sum())
        far, d_far = pts[~on], d[~on]
        if far.size == 0:
            return y
        w = 1.0 / d_far
        T = (far * w[:, None]).sum(axis=0) / w.sum()
        if eta == 0:
            y = T
        else:
            r = np.linalg.norm(((far - y) / d_far[:, None]).sum(axis=0))
            if r == 0.0:
                return y
            gamma = min(1.0, eta / r)
            y = (1.0 - gamma) * T + gamma * y
        if np.linalg.norm(y - y_old) < eps:
            return y
    raise AssertionError("vz_mean did not converge")


def masked_step(points, y):
    """One no-floor masked update -- the shipped rule's non-degenerate branch."""
    pts = np.asarray(points, dtype=np.float64)
    d = np.linalg.norm(pts - y, axis=1)
    on = d <= 1e-12 * max(float(np.linalg.norm(y)), 1.0)
    far, d_far = pts[~on], d[~on]
    w = 1.0 / d_far
    return (far * w[:, None]).sum(axis=0) / w.sum()


def floored_step(points, y):
    """One floored update -- the rule that was replaced."""
    pts = np.asarray(points, dtype=np.float64)
    w = 1.0 / np.maximum(np.linalg.norm(pts - y, axis=1), CLAMP)
    return (pts * w[:, None]).sum(axis=0) / w.sum()


def objective(x, points):
    """Sum of Euclidean distances -- the quantity the median minimises."""
    return float(np.linalg.norm(np.asarray(points) - x, axis=1).sum())


def skewed_swatch(seed=11):
    """A checker-patch core lit unevenly: tight body plus a brightness ramp."""
    rng = np.random.default_rng(seed)
    base = np.array([0.35, 0.37, 0.40])
    body = rng.normal(base, 0.004, size=(3000, 3))
    ramp = base + rng.random((900, 1)) * 0.45 + rng.normal(0, 0.004, size=(900, 3))
    px = np.clip(np.vstack([body, ramp]), 0.0, 1.0)
    return np.round(px * 255.0) / 255.0  # 8-bit, as real swatch pixels are


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

swatch = skewed_swatch()
mean = swatch.mean(axis=0)
planted = np.vstack([swatch, mean])  # one pixel exactly at the centroid
truth = vardi_zhang(planted)

# Claim 1: on this patch the mean and the geometric median genuinely differ,
# so the test below can distinguish a correct solver from a broken one.
gap = np.linalg.norm(mean - truth) * 255.0
check(
    "1. mean and geometric median differ by >5 8-bit code values",
    gap > 5.0,
    f"gap = {gap:.2f} code values",
)

# Claim 2: the coincident point owns essentially all the weight.
d = np.maximum(np.linalg.norm(planted - mean, axis=1), CLAMP)
w = 1.0 / d
share = 100.0 * w[-1] / w.sum()
check(
    "2. coincident point takes >99.99% of total weight",
    share > 99.99,
    f"weight {w[-1]:.3e} vs {w[:-1].sum():.3e} others -> {share:.4f}%",
)

# Claim 3: the floored rule returns the planted point and calls it converged.
# At ColorCheckerProfile's own constants (eps=1e-6, max_iter=200) it stops
# after ONE iteration -- the budget is never the constraint, the stopping rule
# is: after the capture, the next step is smaller than eps and the loop reads
# that stillness as convergence.
got, iters, converged = floored_weiszfeld(planted)
check(
    "3. the floored rule returns the coincident point, reporting convergence",
    np.allclose(got, mean, atol=1e-6) and converged,
    f"returned mean-pixel={np.allclose(got, mean, atol=1e-6)}, "
    f"converged={converged} after {iters} iteration(s)",
)

# Claim 4: that answer is materially worse than the true median. The
# geometric median minimises this objective, so a larger value is a
# strictly worse answer -- not a matter of tolerance.
f_bad, f_good = objective(got, planted), objective(truth, planted)
check(
    "4. the floored answer has a strictly larger objective than the true median",
    f_bad > f_good,
    f"f(floored)={f_bad:.4f} > f(true)={f_good:.4f} "
    f"(excess {100 * (f_bad - f_good) / f_good:.2f}%)",
)

# Claim 5: Vardi-Zhang is a true fixed point here -- no descent direction left.
probe = 1e-7
worse = [
    objective(truth + step, planted) >= f_good - 1e-12
    for step in (
        np.array([probe, 0, 0]), np.array([-probe, 0, 0]),
        np.array([0, probe, 0]), np.array([0, -probe, 0]),
        np.array([0, 0, probe]), np.array([0, 0, -probe]),
    )
]
check(
    "5. Vardi-Zhang result is a local minimum of the objective",
    all(worse),
    f"{sum(worse)}/6 probe directions do not decrease f",
)

# Claim 6: without a coincident point the floored rule was already correct --
# the defect is specific to the singularity, not to Weiszfeld generally.
plain_got, _, _ = floored_weiszfeld(swatch)
plain_truth = vardi_zhang(swatch)
err = np.linalg.norm(plain_got - plain_truth) * 255.0
check(
    "6. absent a coincident point the floored rule matches the true median",
    err < 1e-2,
    f"error = {err:.2e} code values",
)

# Claim 7: the failure needs only ONE coincident pixel out of ~3900.
check(
    "7. a single coincident pixel among 3901 is sufficient to break it",
    len(planted) == 3901,
    f"n = {len(planted)}, coincident = 1",
)

# Claim 8: an exact `d == 0` coincidence test is NOT sufficient. A pixel
# placed 1.11e-16 from the estimate is not equal to it, so an exact test
# lets it through to 1/d with ~9e15 of weight and it captures the solve --
# the same failure, at the same magnitude, as the 1e-10 floor. The subject is
# mean-initialised on purpose; from a median start neither test ever fires.
nudged = np.vstack([swatch, mean + np.array([1e-16, 0.0, 0.0])])
d_min = np.linalg.norm(nudged - nudged.mean(axis=0), axis=1).min()
truth_nudged = vardi_zhang(nudged)
err_exact = np.linalg.norm(vz_mean(nudged, "exact") - truth_nudged) * 255.0
err_radius = np.linalg.norm(vz_mean(nudged, "radius") - truth_nudged) * 255.0
check(
    "8. an exact d==0 coincidence test still loses the solve",
    d_min > 0.0 and err_exact > 5.0 and err_radius < 1e-6,
    f"d_min = {d_min:.3e} (strictly positive, so `== 0` misses it); "
    f"exact-test error = {err_exact:.3f} code values, "
    f"radius-test error = {err_radius:.3e} code values",
)

# Claim 9: the radius costs nothing where nothing coincides, and does
# something where something does. Both legs are needed: leg A alone would also
# pass if the mask silently never fired, which is the failure mode that makes
# a green check meaningless.
y0 = swatch.mean(axis=0)
clean_masked, clean_floored = masked_step(swatch, y0), floored_step(swatch, y0)
leg_a = np.array_equal(clean_masked, clean_floored)
y1 = nudged.mean(axis=0)
nudged_masked, nudged_floored = masked_step(nudged, y1), floored_step(nudged, y1)
leg_b = not np.allclose(nudged_masked, nudged_floored, atol=1e-9)
check(
    "9. masked update == floored update iff nothing coincides",
    leg_a and leg_b,
    f"clean: bit-identical={leg_a} (first coord {clean_masked[0]!r}); "
    f"nudged: differ={leg_b} by "
    f"{np.linalg.norm(nudged_masked - nudged_floored) * 255.0:.3f} code values",
)

print()
if FAILURES:
    print(f"{len(FAILURES)} claim(s) FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("All claims verified.")
