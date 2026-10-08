"""Basis-equivalence claims behind SubtractPolySurface's deviation row D1.

Re-derives, from scratch, the numeric facts the spec relies on when it replaces
Gwyddion's monomial normal-equation solve with a Legendre ``lstsq`` solve:

B1  On Gwyddion's normalized coordinates (u = 2j/(W-1) - 1, v = 2i/(H-1) - 1),
    the Legendre and monomial term sets span the same space, for both the
    per-axis ("independent", tensor-product) and the total-degree term sets, so
    the least-squares surfaces agree to within a rounding bound derived from
    the monomial design's condition number.
B2  The monomial Gram matrix (what a normal-equation/Cholesky solve factors)
    becomes ill-conditioned as the order grows toward Gwyddion's MAX_DEGREE=11,
    while the Legendre Gram matrix stays well conditioned. This is why the
    deviation is FORCED rather than cosmetic.
B3  Evaluating the fitted Legendre coefficients on the full grid with the
    separable ``legendre.leggrid2d`` equals evaluating the dense design matrix,
    so the full-resolution surface never needs an H*W*n_terms matrix.

Depends only on the stdlib + numpy. Never imports ``phenotypic``.
Exit status is non-zero if any claim fails.
"""

from __future__ import annotations

import sys

import numpy as np
from numpy.polynomial import legendre as leg
from numpy.polynomial import polynomial as mono

EPS = np.finfo(np.float64).eps
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def gwyddion_coords(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Row (v) and column (u) coordinates normalized to [-1, 1] as Gwyddion does."""
    v = 2 * np.arange(height) / (height - 1.0) - 1.0
    u = 2 * np.arange(width) / (width - 1.0) - 1.0
    return v, u


def term_powers(order: int, independent: bool) -> list[tuple[int, int]]:
    """(x power, y power) pairs: tensor product if independent, else total degree."""
    if independent:
        return [(px, py) for px in range(order + 1) for py in range(order + 1)]
    return [(px, py) for px in range(order + 1) for py in range(order + 1 - px)]


def design(vander, u_flat, v_flat, order, terms):
    vu = vander(u_flat, order)
    vv = vander(v_flat, order)
    return np.stack([vu[:, px] * vv[:, py] for px, py in terms], axis=1)


def synthetic_field(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v, u = gwyddion_coords(height, width)
    vv, uu = np.meshgrid(v, u, indexing="ij")
    smooth = 0.3 + 0.1 * uu - 0.07 * vv + 0.05 * uu**2 * vv - 0.04 * np.cos(2.5 * uu * vv)
    return smooth + rng.normal(0.0, 0.01, (height, width))


def main() -> int:
    height, width = 97, 131
    z = synthetic_field(height, width, seed=7)
    v, u = gwyddion_coords(height, width)
    vv, uu = np.meshgrid(v, u, indexing="ij")
    u_flat, v_flat, z_flat = uu.ravel(), vv.ravel(), z.ravel()

    # ---- B1 + B2 -------------------------------------------------------------
    print("order indep  n_terms  max|S_leg - S_mono|   bound      cond(Gram_mono)  cond(Gram_leg)")
    for independent in (True, False):
        for order in (2, 3, 5, 8, 11):
            terms = term_powers(order, independent)
            a_mono = design(mono.polyvander, u_flat, v_flat, order, terms)
            a_leg = design(leg.legvander, u_flat, v_flat, order, terms)
            c_mono, *_ = np.linalg.lstsq(a_mono, z_flat, rcond=None)
            c_leg, *_ = np.linalg.lstsq(a_leg, z_flat, rcond=None)
            diff = np.max(np.abs(a_mono @ c_mono - a_leg @ c_leg))
            cond_a = np.linalg.cond(a_mono)
            # Backward-stable lstsq: forward error on the fitted values is
            # O(cond(A) * eps * |z|); a factor of 10 covers the constant.
            bound = 10.0 * cond_a * EPS * np.max(np.abs(z_flat))
            g_mono = np.linalg.cond(a_mono.T @ a_mono)
            g_leg = np.linalg.cond(a_leg.T @ a_leg)
            print(f"{order:5d} {str(independent):5s} {len(terms):8d}  {diff:18.3e}  {bound:9.3e}  {g_mono:15.3e}  {g_leg:13.3e}")
            check(f"B1 order={order} independent={independent}", diff <= bound,
                  f"surfaces agree to {diff:.3e} (bound {bound:.3e})")
            if order == 11 and independent:
                check("B2 monomial Gram ill-conditioned at Gwyddion MAX_DEGREE", g_mono > 1e10,
                      f"cond(Gram_mono)={g_mono:.3e} > 1e10")
                check("B2 Legendre Gram well-conditioned at the same order", g_leg < 1e6,
                      f"cond(Gram_leg)={g_leg:.3e} < 1e6")

    # ---- B3 ------------------------------------------------------------------
    order = 3
    terms = term_powers(order, independent=True)
    a_leg = design(leg.legvander, u_flat, v_flat, order, terms)
    c_leg, *_ = np.linalg.lstsq(a_leg, z_flat, rcond=None)
    coef = np.zeros((order + 1, order + 1))  # coef[py, px] -> L_py(v) * L_px(u)
    for c, (px, py) in zip(c_leg, terms):
        coef[py, px] = c
    surface_grid = leg.leggrid2d(v, u, coef)
    surface_dense = (a_leg @ c_leg).reshape(height, width)
    diff = np.max(np.abs(surface_grid - surface_dense))
    check("B3 leggrid2d == dense design evaluation", surface_grid.shape == (height, width) and diff < 1e-12,
          f"shape {surface_grid.shape}, max diff {diff:.3e}")

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
