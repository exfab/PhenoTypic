#!/usr/bin/env python
"""Re-derive every numeric claim in the Weiszfeld coincident-point spec.

Subject: ``phenotypic.util._geometric_median.weiszfeld_median`` clamps the
per-point distance with ``np.maximum(distances, 1e-10)``. A point coinciding
with the running estimate therefore receives weight ``1e10`` and captures the
weighted update, pinning the estimate to that point and reporting convergence.

Per the project file rule this script **does not import phenotypic**. It
re-implements both the shipped update rule and the proposed Vardi-Zhang
replacement from scratch, so it is an independent witness rather than a
restatement of the code under test.

Run:  uv run python <this file>
Exits non-zero if any claim fails.
"""

from __future__ import annotations

import sys

import numpy as np

FAILURES: list[str] = []
CLAMP = 1e-10  # the constant shipped in weiszfeld_median


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Re-implementations (independent of src/)
# ---------------------------------------------------------------------------


def shipped_weiszfeld(points, eps=1e-6, max_iter=200):
    """The shipped update rule, transcribed: clamp the distance, reweight."""
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

# Claim 3: the shipped rule returns the planted point and calls it converged.
got, iters, converged = shipped_weiszfeld(planted)
check(
    "3. shipped rule returns the coincident point, reporting convergence",
    np.allclose(got, mean, atol=1e-6) and converged,
    f"returned mean-pixel={np.allclose(got, mean, atol=1e-6)}, "
    f"converged={converged} after {iters} iteration(s)",
)

# Claim 4: that answer is materially worse than the true median. The
# geometric median minimises this objective, so a larger value is a
# strictly worse answer -- not a matter of tolerance.
f_bad, f_good = objective(got, planted), objective(truth, planted)
check(
    "4. shipped answer has a strictly larger objective than the true median",
    f_bad > f_good,
    f"f(shipped)={f_bad:.4f} > f(true)={f_good:.4f} "
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

# Claim 6: without a coincident point the shipped rule is already correct --
# the defect is specific to the singularity, not to Weiszfeld generally.
plain_got, _, _ = shipped_weiszfeld(swatch)
plain_truth = vardi_zhang(swatch)
err = np.linalg.norm(plain_got - plain_truth) * 255.0
check(
    "6. absent a coincident point the shipped rule matches the true median",
    err < 1e-2,
    f"error = {err:.2e} code values",
)

# Claim 7: the failure needs only ONE coincident pixel out of ~3900.
check(
    "7. a single coincident pixel among 3901 is sufficient to break it",
    len(planted) == 3901,
    f"n = {len(planted)}, coincident = 1",
)

print()
if FAILURES:
    print(f"{len(FAILURES)} claim(s) FAILED: {', '.join(FAILURES)}")
    sys.exit(1)
print("All claims verified.")
