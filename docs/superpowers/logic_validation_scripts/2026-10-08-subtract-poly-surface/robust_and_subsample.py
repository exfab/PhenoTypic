"""Robust-fit and subsampling claims behind SubtractPolySurface.

R1  Plain least squares is biased by bright colonies: on a synthetic plate its
    background error exceeds the noise sigma at every tested colony cover, for
    both a whole-image surface and per-line fits.
R2  The spec's robust loop -- astropy FittingWithOutlierRemoval's structure
    (initial fit on all points; per round: clip residuals with the mask carried
    forward, refit on survivors; stop when the mask stops growing) with a
    single-pass symmetric clip at |r - median| > clip_sigma * mad_std --
    recovers the background to within half the noise sigma at the shipped
    defaults clip_sigma=3.0, max_iter=10, on every one of three seeds:
    a whole-image surface up to 40% colony cover of the plate, and per-line
    fits for every line whose OWN colony fraction is below 20% -- per-line
    fits break down per line, so the plate's cover is the wrong measure.
R3  The shipped max_iter=10 is converged: raising it to 50 moves the
    recovered background by less than a tenth of the noise sigma (lines:
    over the lines R2 covers).
R4  Fitting the surface on a strided subsample (stride chosen so at most
    MAX_FIT_POINTS samples are used) and evaluating it on the full grid differs
    from the full-resolution fit by at most 3 * sigma * sqrt(p / n_sub) RMS,
    the standard error of a p-parameter least-squares surface on n_sub points.
R5  astropy's own default niter=3 is NOT converged at 40% cover (error above
    half a sigma on some seed). This is the evidence for deviation row D6.
R6  The median's 50% breakdown point: at 50% cover the robust surface fails
    (error above one sigma) on at least one seed. A documented limit of the
    method, stated in the docstring -- not a defect to fix.
R7  The same breakdown, per line: some line whose own colony fraction is at
    least 50% fails (error above one sigma). On an arrayed plate a scan line
    through a row of colony centres is such a line, so robust line leveling
    cannot level it. Documented limit, stated in the docstring.
R8  The ~40% surface limit holds only for DISPERSED foreground. A contiguous
    band along one image edge (plate rim, out-of-plate scan border, meniscus)
    defeats the robust surface fit once it covers ~10% of the width: a band of
    5% of the width recovers (< 0.5 sigma) while 10% and 20% fail (> 1 sigma),
    for amplitudes +-0.3 and +1.0, with and without colonies, on every seed.
    The error is scored outside the band, where the plate is. Ruling R16.
R9  The per-line analogue: a contiguous defect at the END of every line
    defeats the robust line fit below the 20% dispersed limit -- 15% and 20%
    of the line fail (> 1 sigma) -- while the same 20% defect centred in the
    line recovers (< 0.5 sigma). The end of a line has leverage on the
    initial lstsq tilt, which inflates the MAD so nothing is clipped. Ruling
    R15.

Colonies are soft-edged domes, so their rims leave low-amplitude halo pixels
inside the clip band -- the realistic case, and harder than flat discs.

Depends only on the stdlib + numpy + scipy. Never imports ``phenotypic``.
Exit status is non-zero if any claim fails.
"""

from __future__ import annotations

import math
import sys

import numpy as np
from numpy.polynomial import legendre as leg
from scipy.stats import median_abs_deviation

FAILURES: list[str] = []
NOISE = 0.01
MAX_FIT_POINTS = 262_144
CLIP_SIGMA = 3.0  # shipped default (astropy SigmaClip default)
MAX_ITER = 10  # shipped default (drift D12; astropy's niter=3 is not converged at 40% cover)
LINE_FRACTION_OK = 0.2  # a line's OWN colony fraction below which per-line robust fits are claimed reliable


def check(name: str, ok: bool, detail: str) -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def tensor_design(u_flat: np.ndarray, v_flat: np.ndarray, order: int) -> np.ndarray:
    lu, lv = leg.legvander(u_flat, order), leg.legvander(v_flat, order)
    return np.stack([lu[:, px] * lv[:, py] for px in range(order + 1) for py in range(order + 1)], axis=1)


def robust_lstsq(a: np.ndarray, z: np.ndarray, clip_sigma: float, max_iter: int) -> np.ndarray:
    """The spec's robust loop for one design matrix (whole surface)."""
    keep = np.ones(z.size, dtype=bool)
    coef, *_ = np.linalg.lstsq(a, z, rcond=None)
    for _ in range(max_iter):
        r = z - a @ coef
        center = np.median(r[keep])
        scale = median_abs_deviation(r[keep], scale="normal")
        if scale == 0.0:
            break
        new_keep = keep & (np.abs(r - center) <= clip_sigma * scale)
        if new_keep.sum() < a.shape[1]:
            break
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
        coef, *_ = np.linalg.lstsq(a[keep], z[keep], rcond=None)
    return coef


def robust_lines(z: np.ndarray, degree: int, clip_sigma: float, max_iter: int) -> np.ndarray:
    """The spec's robust loop applied independently to every row; returns fitted rows."""
    n_lines, n = z.shape
    vander = leg.legvander(2 * np.arange(n) / (n - 1.0) - 1.0, degree)  # (n, k+1)
    coef, *_ = np.linalg.lstsq(vander, z.T, rcond=None)  # (k+1, L): one multi-RHS solve
    coef = coef.T
    keep = np.ones_like(z, dtype=bool)
    active = np.ones(n_lines, dtype=bool)
    for _ in range(max_iter):
        r = z - coef @ vander.T
        rm = np.where(keep, r, np.nan)
        center = np.nanmedian(rm, axis=1, keepdims=True)
        scale = median_abs_deviation(rm, axis=1, scale="normal", nan_policy="omit")[:, None]
        new_keep = keep & (np.abs(r - center) <= clip_sigma * scale)
        grew = new_keep.sum(axis=1) != keep.sum(axis=1)
        enough = new_keep.sum(axis=1) >= degree + 1
        update = active & grew & enough & (scale[:, 0] > 0)
        if not update.any():
            break
        keep[update] = new_keep[update]
        w = keep[update].astype(float)  # batched weighted normal equations for the updated rows
        gram = np.einsum("ln,ni,nj->lij", w, vander, vander)
        rhs = np.einsum("ln,ni->li", w * z[update], vander)
        coef[update] = np.linalg.solve(gram, rhs[..., None])[..., 0]
        active = update
    return coef @ vander.T


def plate(height: int, width: int, cover: float, seed: int, row_artifacts: bool):
    """Synthetic plate. ``row_artifacts`` gives every row its own offset and slope.

    With row artifacts on, the background is exactly degree 1 along each row
    (the 2-D curvature is in ``v`` only), so a degree-1 line fit can represent
    it. Curvature along the row is a job for a surface method; handing it to a
    line fit makes clipping eat real background at the misfit ends -- a test
    of the wrong method, not of the robust loop.
    """
    rng = np.random.default_rng(seed)
    i, j = np.mgrid[0:height, 0:width].astype(float)
    u, v = 2 * j / (width - 1) - 1, 2 * i / (height - 1) - 1
    if row_artifacts:
        background = (0.30 + 0.08 * u - 0.05 * v + 0.04 * v * v
                      + rng.normal(0, 0.03, (height, 1)) + rng.normal(0, 0.02, (height, 1)) * u)
    else:
        background = 0.30 + 0.08 * u - 0.05 * v + 0.06 * u * u + 0.04 * v * v - 0.03 * u * v + 0.02 * u**3 * v
    colonies = np.zeros((height, width))
    radius = 16.0
    while (colonies > 0).mean() < cover:
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        rr = np.hypot(i - cy, j - cx) / radius
        dome = rng.uniform(0.15, 0.45) * np.sqrt(np.clip(1 - rr**2, 0, None))
        colonies = np.maximum(colonies, dome)
    image = background + colonies + rng.normal(0, NOISE, (height, width))
    return image, background, u, v, (colonies > 0).mean(), colonies > 0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


EDGE_AMPLITUDES = (0.3, -0.3, 1.0)


def edge_band_error(frac: float, amplitude: float, cover: float, seed: int) -> float:
    """Robust tensor-order-3 surface error (noise sigmas) outside a left-edge band."""
    height, width, order = 300, 450, 3
    z, truth, u, v, _, _ = plate(height, width, cover, seed=seed, row_artifacts=False)
    band = np.zeros(z.shape, dtype=bool)
    band[:, : int(round(frac * width))] = True
    a = tensor_design(u.ravel(), v.ravel(), order)
    fitted = (a @ robust_lstsq(a, (z + amplitude * band).ravel(), CLIP_SIGMA, MAX_ITER)).reshape(z.shape)
    return rmse(fitted[~band], truth[~band]) / NOISE


def line_defect_error(frac: float, amplitude: float, where: str, seed: int) -> float:
    """Worst per-line robust degree-1 error (noise sigmas) with a contiguous defect in every line."""
    height, width = 300, 450
    zl, truth_l, _, _, _, _ = plate(height, width, 0.0, seed=seed, row_artifacts=True)
    k = int(round(frac * width))
    start = width - k if where == "end" else (width - k) // 2
    defect = np.zeros(zl.shape, dtype=bool)
    defect[:, start:start + k] = True
    fit = robust_lines(zl + amplitude * defect, 1, CLIP_SIGMA, MAX_ITER)
    return float(np.max(np.sqrt(np.mean((fit - truth_l) ** 2, axis=1)))) / NOISE


def check_edge_contiguous_limits() -> None:
    seeds = (1, 2, 3)
    surface = {frac: [edge_band_error(frac, amp, cover, seed * 1000 + 50)
                      for amp in EDGE_AMPLITUDES for cover in (0.0, 0.10) for seed in seeds]
               for frac in (0.05, 0.10, 0.20)}
    check("R8 edge band of 5% of the width: robust surface recovers (max over cases)",
          max(surface[0.05]) < 0.5, f"max {max(surface[0.05]):.3f} sigma over {len(surface[0.05])} cases")
    for frac in (0.10, 0.20):
        check(f"R8 edge band of {frac:.0%} of the width defeats the robust surface (min over cases)",
              min(surface[frac]) > 1.0,
              f"{min(surface[frac]):.2f}-{max(surface[frac]):.2f} sigma over {len(surface[frac])} cases")
    lines = {(frac, where): [line_defect_error(frac, amp, where, seed * 1000 + 77)
                             for amp in EDGE_AMPLITUDES for seed in seeds]
             for frac, where in ((0.15, "end"), (0.20, "end"), (0.20, "centre"))}
    for frac in (0.15, 0.20):
        errs = lines[(frac, "end")]
        check(f"R9 end-of-line defect of {frac:.0%} of the line defeats the robust line fit (min over cases)",
              min(errs) > 1.0, f"{min(errs):.2f}-{max(errs):.2f} sigma, worst line, {len(errs)} cases")
    errs = lines[(0.20, "centre")]
    check("R9 the same 20% defect centred in the line recovers (max over cases)",
          max(errs) < 0.5, f"max {max(errs):.3f} sigma, worst line, {len(errs)} cases")


def main() -> int:
    height, width, order = 300, 450, 3
    seeds = (1, 2, 3)
    worst: dict[tuple[float, str], list[float]] = {}
    print("cover seed  surface: lstsq  robust(3)  robust(10)  robust(50) |  lines: lstsq  robust(10)  robust(50)"
          "   (RMSE / noise sigma)")
    for cover in (0.10, 0.25, 0.40, 0.50):
        for seed in seeds:
            z, truth, u, v, achieved, _ = plate(height, width, cover, seed=seed * 1000 + int(cover * 100),
                                                row_artifacts=False)
            a = tensor_design(u.ravel(), v.ravel(), order)
            s_ls = (a @ np.linalg.lstsq(a, z.ravel(), rcond=None)[0]).reshape(z.shape)
            s_r3 = (a @ robust_lstsq(a, z.ravel(), CLIP_SIGMA, 3)).reshape(z.shape)
            s_rd = (a @ robust_lstsq(a, z.ravel(), CLIP_SIGMA, MAX_ITER)).reshape(z.shape)
            s_r50 = (a @ robust_lstsq(a, z.ravel(), CLIP_SIGMA, 50)).reshape(z.shape)

            zl, truth_l, _, _, _, colony_l = plate(height, width, cover, seed=seed * 1000 + 7 + int(cover * 100),
                                                   row_artifacts=True)
            vander = leg.legvander(2 * np.arange(width) / (width - 1.0) - 1.0, 1)
            l_ls = (vander @ np.linalg.lstsq(vander, zl.T, rcond=None)[0]).T
            l_rd = robust_lines(zl, 1, CLIP_SIGMA, MAX_ITER)
            l_r50 = robust_lines(zl, 1, CLIP_SIGMA, 50)
            # Per-line fits break down per LINE: what matters is each line's own
            # colony fraction, not the plate's. Score lines by their own fraction.
            line_frac = colony_l.mean(axis=1)
            sparse, dense = line_frac < LINE_FRACTION_OK, line_frac >= 0.5

            def line_err(fit: np.ndarray, rows: np.ndarray) -> float:
                if not rows.any():
                    return float("nan")
                return float(np.max(np.sqrt(np.mean((fit[rows] - truth_l[rows]) ** 2, axis=1))) / NOISE)

            e = {k: rmse(x, truth) / NOISE for k, x in (("ls", s_ls), ("r3", s_r3), ("rd", s_rd), ("r50", s_r50))}
            el = {"ls": line_err(l_ls, sparse), "rd": line_err(l_rd, sparse), "r50": line_err(l_r50, sparse),
                  "dense": line_err(l_rd, dense)}
            print(f"{achieved:5.2f} {seed:4d}  {e['ls']:13.3f} {e['r3']:10.3f} {e['rd']:11.3f} {e['r50']:11.3f} |"
                  f" {el['ls']:12.3f} {el['rd']:11.3f} {el['r50']:11.3f}   (lines: worst line with own"
                  f" fraction < {LINE_FRACTION_OK}, n={int(sparse.sum())}; worst line >= 0.5: {el['dense']:.2f})")
            for key, val in (("ls", e["ls"]), ("r3", e["r3"]), ("rd", e["rd"]),
                             ("conv", rmse(s_rd, s_r50) / NOISE),
                             ("l_ls", el["ls"]), ("l_rd", el["rd"]),
                             ("l_conv", rmse(l_rd[sparse], l_r50[sparse]) / NOISE if sparse.any() else float("nan")),
                             ("l_dense", el["dense"])):
                worst.setdefault((cover, key), []).append(val)

    for cover in (0.10, 0.25, 0.40):
        for key, label in (("ls", "surface"), ("l_ls", "lines")):
            check(f"R1 cover={cover:.2f} lstsq {label} biased beyond noise (min over seeds)",
                  min(worst[(cover, key)]) > 1.0, f"min {min(worst[(cover, key)]):.3f} sigma")
        for key, label in (("rd", "surface"), ("l_rd", "lines")):
            check(f"R2 cover={cover:.2f} robust {label} at defaults within 0.5 sigma (max over seeds)",
                  max(worst[(cover, key)]) < 0.5, f"max {max(worst[(cover, key)]):.3f} sigma")
        for key, label in (("conv", "surface"), ("l_conv", "lines")):
            check(f"R3 cover={cover:.2f} max_iter={MAX_ITER} converged ({label} vs 50 rounds)",
                  max(worst[(cover, key)]) < 0.1, f"max {max(worst[(cover, key)]):.3f} sigma")
    check("R5 astropy's niter=3 is NOT converged at 40% cover (evidence for D12)",
          max(worst[(0.40, "r3")]) > 0.5, f"max {max(worst[(0.40, 'r3')]):.3f} sigma")
    check("R6 50% breakdown: robust surface fails on some seed at 50% cover (documented limit)",
          max(worst[(0.50, "rd")]) > 1.0, f"max {max(worst[(0.50, 'rd')]):.3f} sigma")
    dense = [x for c in (0.25, 0.40, 0.50) for x in worst[(c, "l_dense")] if not math.isnan(x)]
    check("R7 per-line breakdown: some line whose own colony fraction >= 0.5 fails (documented limit)",
          bool(dense) and max(dense) > 1.0, f"worst such line {max(dense):.3f} sigma over {len(dense)} plates")

    check_edge_contiguous_limits()

    # ---- R4 ------------------------------------------------------------------
    rng = np.random.default_rng(3)
    big_h, big_w = 1024, 1536
    i, j = np.mgrid[0:big_h, 0:big_w].astype(float)
    u, v = 2 * j / (big_w - 1) - 1, 2 * i / (big_h - 1) - 1
    z = 0.3 + 0.08 * u - 0.05 * v + 0.06 * u * u * v + rng.normal(0, NOISE, (big_h, big_w))
    stride = math.ceil(math.sqrt(big_h * big_w / MAX_FIT_POINTS))
    zs, us, vs = z[::stride, ::stride], u[::stride, ::stride], v[::stride, ::stride]
    p = (order + 1) ** 2
    terms = [(px, py) for px in range(order + 1) for py in range(order + 1)]
    c_sub, *_ = np.linalg.lstsq(tensor_design(us.ravel(), vs.ravel(), order), zs.ravel(), rcond=None)
    c_full, *_ = np.linalg.lstsq(tensor_design(u.ravel(), v.ravel(), order), z.ravel(), rcond=None)

    def grid(c: np.ndarray) -> np.ndarray:
        coef = np.zeros((order + 1, order + 1))
        for c_k, (px, py) in zip(c, terms):
            coef[py, px] = c_k
        return leg.leggrid2d(v[:, 0], u[0, :], coef)

    diff = rmse(grid(c_sub), grid(c_full))
    bound = 3 * NOISE * math.sqrt(p / zs.size)
    check("R4 subsampled surface within 3*sigma*sqrt(p/n_sub) of full fit", diff <= bound,
          f"stride={stride}, n_sub={zs.size}, RMS diff={diff:.3e}, bound={bound:.3e}")

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
