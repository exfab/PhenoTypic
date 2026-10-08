"""Output-level claims behind SubtractPolySurface's per-method level convention.

The spec's behavioural contract (design.md section 4) states, per method, where
the background level ends up. This script re-derives each statement from the
contract's own formulas -- not from any implementation -- so the spec, the
plan, and the tests share one checked set of numbers.

L1  plane: slopes from a least-squares fit of [1, u, v] on normalized
    coordinates convert to per-pixel-index slopes by bx = b_u * 2/(W-1),
    by = b_v * 2/(H-1), and agree with the closed-form per-pixel-index
    least-squares slopes cov(z, j)/var(j), cov(z, i)/var(i).
L2  plane: subtracting only the tilt pivoted at (W/2, H/2) -- not at the true
    centre ((W-1)/2, (H-1)/2) -- leaves mean(out) = mean(z) + (bx + by)/2.
    (The sign is positive. An earlier draft of the design said minus.)
L3  line: fitting each row with a degree-k polynomial on the centred pixel
    coordinate x = j - (W-1)/2, or with a degree-k Legendre basis on the
    normalized coordinate u, gives identical residuals (same span).
L4  line: subtracting each row's fit and adding back the input's global mean
    leaves every row mean equal to mean(z), hence the global mean preserved.
L5  line, degree 0: the row-shift form z - (rowmean_i - mean(rowmeans)) equals
    the degree-0 polynomial form of L4 on a full (unmasked) field.
L6  offset and polynomial: a least-squares fit that contains a constant term
    leaves residuals of mean 0 (to rounding), so the level lands at 0.

Depends only on the stdlib + numpy. Never imports ``phenotypic``.
Exit status is non-zero if any claim fails.
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.polynomial import legendre as leg
from numpy.polynomial import polynomial as mono

FAILURES: list[str] = []
TOL = 1e-11


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def field(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    i, j = np.mgrid[0:height, 0:width].astype(float)
    tilt = 0.4 + 0.0021 * j - 0.0013 * i
    rows = rng.normal(0.0, 0.02, (height, 1)) + rng.normal(0.0, 1e-4, (height, 1)) * j
    return tilt + rows + 0.03 * np.sin(j / 7.0) * np.cos(i / 5.0) + rng.normal(0.0, 0.01, (height, width))


def main() -> int:
    height, width = 83, 120
    z = field(height, width, seed=11)
    i, j = np.mgrid[0:height, 0:width].astype(float)
    v = 2 * i / (height - 1.0) - 1.0
    u = 2 * j / (width - 1.0) - 1.0

    # ---- L1 ------------------------------------------------------------------
    a = np.stack([np.ones(z.size), u.ravel(), v.ravel()], axis=1)
    (_, b_u, b_v), *_ = np.linalg.lstsq(a, z.ravel(), rcond=None)
    bx, by = b_u * 2 / (width - 1.0), b_v * 2 / (height - 1.0)
    bx_ref = np.mean((z - z.mean()) * (j - j.mean())) / np.var(j)
    by_ref = np.mean((z - z.mean()) * (i - i.mean())) / np.var(i)
    check("L1 normalized slopes convert to per-pixel slopes",
          abs(bx - bx_ref) < TOL and abs(by - by_ref) < TOL,
          f"bx={bx:.6e} (ref {bx_ref:.6e}), by={by:.6e} (ref {by_ref:.6e})")

    # ---- L2 ------------------------------------------------------------------
    out = z - bx * (j - width / 2.0) - by * (i - height / 2.0)
    expected = z.mean() + 0.5 * (bx + by)
    check("L2 plane mean shift is +(bx+by)/2",
          abs(out.mean() - expected) < TOL,
          f"mean(out)-mean(z)={out.mean() - z.mean():+.6e}, (bx+by)/2={0.5 * (bx + by):+.6e}")
    centred = z - bx * (j - (width - 1) / 2.0) - by * (i - (height - 1) / 2.0)
    check("L2 control: a true-centre pivot preserves the mean exactly",
          abs(centred.mean() - z.mean()) < TOL,
          f"mean shift {centred.mean() - z.mean():+.3e}")

    # ---- L3 + L4 -------------------------------------------------------------
    x_pix = np.arange(width) - (width - 1) / 2.0
    u_line = 2 * np.arange(width) / (width - 1.0) - 1.0
    avg = z.mean()
    for degree in (0, 1, 2, 5):
        vm = mono.polyvander(x_pix, degree)
        vl = leg.legvander(u_line, degree)
        cm, *_ = np.linalg.lstsq(vm, z.T, rcond=None)  # one multi-RHS solve for all rows
        cl, *_ = np.linalg.lstsq(vl, z.T, rcond=None)
        res_m = z - (vm @ cm).T
        res_l = z - (vl @ cl).T
        d = np.max(np.abs(res_m - res_l))
        check(f"L3 degree={degree} pixel-monomial == normalized-Legendre residuals", d < 1e-10, f"max diff {d:.3e}")
        out_line = res_l + avg
        row_dev = np.max(np.abs(out_line.mean(axis=1) - avg))
        check(f"L4 degree={degree} every row mean == input global mean", row_dev < TOL,
              f"max |rowmean - mean(z)| = {row_dev:.3e}")

    # ---- L5 ------------------------------------------------------------------
    rowmeans = z.mean(axis=1, keepdims=True)
    shift_form = z - (rowmeans - rowmeans.mean())
    poly0_form = (z - rowmeans) + avg
    d = np.max(np.abs(shift_form - poly0_form))
    check("L5 degree-0 row-shift form == degree-0 polynomial form", d < TOL, f"max diff {d:.3e}")

    # ---- L6 ------------------------------------------------------------------
    off = z - z.mean()
    check("L6 offset residual mean is 0", abs(off.mean()) < TOL, f"{off.mean():+.3e}")
    for independent in (True, False):
        order = 3
        terms = ([(px, py) for px in range(order + 1) for py in range(order + 1)] if independent
                 else [(px, py) for px in range(order + 1) for py in range(order + 1 - px)])
        lu, lv = leg.legvander(u.ravel(), order), leg.legvander(v.ravel(), order)
        a = np.stack([lu[:, px] * lv[:, py] for px, py in terms], axis=1)
        c, *_ = np.linalg.lstsq(a, z.ravel(), rcond=None)
        res = z.ravel() - a @ c
        check(f"L6 polynomial (independent={independent}) residual mean is 0", abs(res.mean()) < TOL,
              f"{res.mean():+.3e}")

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
