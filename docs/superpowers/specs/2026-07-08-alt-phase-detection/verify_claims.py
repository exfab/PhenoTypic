"""Executable verification of every mathematical claim made in this spec folder.

Run it::

    uv run python docs/superpowers/specs/2026-07-08-alt-phase-detection/verify_claims.py

Exits non-zero if any check fails. Depends only on ``numpy`` and ``scipy``; it does
**not** import ``phenotypic``. Peak memory ~180 MB, runtime ~1.5 s. Every integrand that
is radially symmetric is integrated in 1-D; do not reintroduce 2-D grids for them.

The last four checks (09b, 15, 16, 17) run against Peter Kovesi's synthetic test images,
ported from ImagePhaseCongruency.jl under MIT with the notice retained. They matter
disproportionately: their ground truth was authored by the algorithm's own author, so
unlike the rest of this file it is not this spec marking its own homework. They also
supply the only **positive** control here (check_15) -- everything else establishes what
something *is not*.

Three of these checks contradict a published paper — Fleischmann, Wietzke & Sommer,
"Image Analysis by Conformal Embedding," *J. Math. Imaging Vis.* **40**(3):305-325
(2011), cited as JMIV. Claims that strong must be reproducible on demand, which is why
this file exists. The relevant equations were read directly from the source:

  * Eq. (20)  ``q^i_s(x) ∝ x_i / (|x|^2 + s^2)^((n+1)/2)`` — the regulariser ``s`` is
              the *paper's own*, not this spec's.
  * Eq. (65)  ``dm(v) = delta(|v - c|) dv`` — the integral is over the sphere's
              **surface**, so the one-sided support in ``v_3`` is the paper's.
  * Eq. (88)  the conformal monogenic signal is ``H_i`` applied to the Poisson
              scale-space embedding ``g^{x,s}``.
  * Eq. (89)  Corollary 4 states ``Q^i_s[g^{x,s}] = omega_i * T(0)``, as an equality,
              justified by "``g^{x,s}`` **approximates** the plane wave ``psi_s``".

Eq. (89) fails in three independent ways, in decreasing order of robustness:

  1. **A DC term is omitted** (check_04a). For *any* positive radial weight ``w``,
     ``mu_3 = int v_3 w dsigma > 0`` because ``v_3 = |v|^2 >= 0`` on this sphere. So a
     straight edge (``omega_3 = 0``) with a nonzero mean gives ``Q^3 != 0`` while
     Eq. (89) predicts ``0``. **Regulariser-independent.**
  2. **It assumes an isotropy that does not hold** (check_04b). The exact law is
     ``Q^i = g0*mu_i + s*(M omega)_i`` with ``M = diag(A, A, B)``. Eq. (89) requires
     ``A == B``, true at exactly one ``s`` (``s0* = 0.19269068``) and nowhere else.
     **Conditional on ``s != s0*``.**
  3. **The claimed frequency-independence fails** (check_09, check_09b). Eq. (89) implies
     ``Q^3 / sqrt(Q^1^2 + Q^2^2) = cot(phi_m)``, independent of the signal's radial
     frequency. It is not: on JMIV's own oscillatory circular signal, ``kappa*r`` varies
     3x with wavelength at fixed ``r``, and 10.6x on Kovesi's ``circsine``.

Consequence for the design (``conformal-lift.md`` §5): ``kappa`` is scale-*covariant* —
``kappa*r`` depends only on ``(r/sigma, R/sigma)`` — but it is **not isophote
curvature**. It tracks ``f''/f' + 1/r``. Three radial profiles with identical isophote
curvature give three different answers (check_09). The two coincide only when
``f'' == 0``, i.e. for a cone — which is exactly JMIV's Fig. 13 test signal.

Notation follows ``references.md`` §0 and §4::

    lambda = 1/(1+r^2)          conformal factor of the inverse stereographic map
    u      = S^-1(y)            the lifted point, on the sphere of radius 1/2
    rho^2  = |u|^2 = u_3        the identity that drives everything (check_01)
    J      = lambda^2           surface-measure pullback
    w(u_3) = 1/(u_3 + s0^2)^2   the paper's regularised radial weight (Eq. 20)
    sigma  = pixels per sphere diameter   (correction P3; no source states it)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.ndimage import correlate, gaussian_filter, laplace
from scipy.optimize import brentq

# The unique s0 at which B/A == 1 on the *full* sphere. At any other s0 the second-moment
# tensor is anisotropic, which is what Eq. (89) overlooks (check_04b).
S0_ISOTROPY = 0.19269068


# --------------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------------


def lift_to_sphere(y1: np.ndarray, y2: np.ndarray) -> tuple[np.ndarray, ...]:
    """Inverse stereographic projection onto the sphere of centre (0,0,1/2), radius 1/2.

    JMIV Eq. (31). Returns ``(u1, u2, u3, lam)``.
    """
    r2 = y1 * y1 + y2 * y2
    lam = 1.0 / (1.0 + r2)
    return y1 * lam, y2 * lam, r2 * lam, lam


def conformal_masks(
    mask_radius: int, sigma: float, s0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """The four pulled-back masks, plus the mask constants ``A``, ``B``, ``mu3``.

    ``references.md`` §4.3. Correlation convention (positive ``u_i``).

    Because ``np.meshgrid(..., indexing="ij")`` puts ``Y1`` on **axis 0**, ``Q1`` is odd
    along axis 0 and ``Q2`` along axis 1 (check_02).

    Returns ``(P, Q1, Q2, Q3, A, B, mu3)``.
    """
    ax = np.arange(-mask_radius, mask_radius + 1, dtype=float)
    Y1, Y2 = np.meshgrid(ax, ax, indexing="ij")
    u1, u2, u3, lam = lift_to_sphere(Y1 / sigma, Y2 / sigma)
    w = 1.0 / (u3 + s0 * s0) ** 2
    J = lam**2
    A = float((u1 * u1 * w * J).sum())
    B = float((u3 * u3 * w * J).sum())
    mu3 = float((u3 * w * J).sum())
    return s0 * w * J, u1 * w * J, u2 * w * J, u3 * w * J, A, B, mu3


def moment_tensor_closed_form(s0: float) -> tuple[float, float]:
    """``(A, B)`` in closed form, over the *full* sphere.

    Archimedes' hat-box gives ``int_{S^2} F(v3) dsigma = pi * int_0^1 F(h) dh`` (total
    area ``pi`` for a radius-1/2 sphere), and on this sphere ``v1^2 + v2^2 = v3(1-v3)``.
    ``references.md`` §4.3.1.
    """
    a = s0 * s0
    L = np.log((1.0 + a) / a)
    I1 = L + a / (1.0 + a) - 1.0
    I2 = 1.0 - 2.0 * a * L - a * a / (1.0 + a) + a
    return (np.pi / 2.0) * (I1 - I2), np.pi * I2


def planar_poisson_unit_mass(scale: float, half: int) -> np.ndarray:
    """Planar (n=2) Poisson kernel, unit-mass-normalised **on the lattice**.

    Unit mass makes a difference of two of these exactly zero-mean on the lattice, which
    the continuum identity alone does not deliver. ``references.md`` §4.4.2, Step B.
    """
    ax = np.arange(-half, half + 1, dtype=float)
    Y1, Y2 = np.meshgrid(ax, ax, indexing="ij")
    p = (1.0 / (2.0 * np.pi)) * scale / (Y1**2 + Y2**2 + scale**2) ** 1.5
    return p / p.sum()


def planar_dop(scale: float, step: float, half: int) -> np.ndarray:
    """Difference of Poisson, coarse minus fine, per the CMPCM paper's Eq. (11)."""
    return planar_poisson_unit_mass(step * scale, half) - planar_poisson_unit_mass(scale, half)


def conformal_components(
    img: np.ndarray, mask_radius: int, sigma: float, s0: float
) -> tuple[np.ndarray, ...]:
    """Step A of ``references.md`` §4.4.2, including correction (P1).

    ``Q3`` subtracts the test-point value. ``Q1``/``Q2`` are odd and have zero mass, so
    they are untouched and the common scale factor ``T`` survives.
    """
    P, Q1, Q2, Q3, _, _, _ = conformal_masks(mask_radius, sigma, s0)
    c = correlate(img, P, mode="nearest")
    q1 = correlate(img, Q1, mode="nearest")
    q2 = correlate(img, Q2, mode="nearest")
    q3 = correlate(img, Q3, mode="nearest") - img * Q3.sum()  # (P1)
    return c, q1, q2, q3


def curvature_estimate(
    img: np.ndarray,
    row: int,
    col: int,
    mask_radius: int,
    sigma: float,
    s0: float = S0_ISOTROPY,
    *,
    remove_dc: bool = True,
    apply_gain: bool = True,
) -> float:
    """JMIV Eq. (97)'s ``kappa`` at one point, with (P1) and (P2) individually switchable.

    ``kappa = (2/sigma) * (A/B) * |Q3| / sqrt(Q1^2 + Q2^2)``. The switches let check_08
    show that *both* corrections are necessary. **This quantity is not isophote
    curvature** — see check_09.
    """
    _, Q1, Q2, Q3, A, B, _ = conformal_masks(mask_radius, sigma, s0)
    patch = img[row - mask_radius : row + mask_radius + 1, col - mask_radius : col + mask_radius + 1]
    q1 = float((patch * Q1).sum())
    q2 = float((patch * Q2).sum())
    q3 = float(((patch - img[row, col]) * Q3).sum()) if remove_dc else float((patch * Q3).sum())
    gain = (A / B) if apply_gain else 1.0
    return (2.0 / sigma) * gain * abs(q3) / (np.hypot(q1, q2) + 1e-300)


def radial_image(profile: Callable[[np.ndarray], np.ndarray], n: int) -> tuple[np.ndarray, int]:
    """An image ``f(|x|)`` with concentric circular isophotes. Returns ``(img, centre)``."""
    c = n // 2
    ax = np.arange(n) - c
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    return profile(np.hypot(X, Y).astype(float)), c


def kappa_r_on_profile(profile: Callable[[np.ndarray], np.ndarray], r: int, sigma: float, R: int) -> float:
    """``kappa * r`` for a radial profile, probed at radius ``r``. Should be 1 if kappa is curvature."""
    img, c = radial_image(profile, 2 * (r + R) + 41)
    return curvature_estimate(img, c + r, c, R, sigma) * r


# --------------------------------------------------------------------------------
# Kovesi's synthetic test images
#
# Ported from ``src/syntheticimages.jl`` of ImagePhaseCongruency.jl. The originals
# describe themselves as images that "cause considerable grief for gradient based
# operators", which is exactly why they are the right controls for a congruency
# measure: an independent, adversarial ground truth authored by the algorithm's own
# author rather than by this spec.
#
#   Copyright (c) 2015-2017 Peter Kovesi -- peterkovesi.com
#
#   MIT License:
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of
#   this software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#   following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#   PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
#   OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Faithfulness notes, in the order they bite:
#
#   * Only ODD harmonics are summed (``1:2:(2*nscales-1)``). Even harmonics would break
#     the half-wave symmetry that makes the feature type well defined.
#   * ``ampexponent = -1`` sums ``1/k`` -> a square wave (step features). ``-2`` with
#     ``offset = pi/2`` sums ``cos(k x)/k^2`` -> a triangle wave (line features).
#   * Julia's ``[f(x, y) for x = l:u, y = l:u]`` puts ``x`` on the FIRST axis, so the
#     ports use ``indexing="ij"`` and ``theta = arctan2(Y, X)`` with ``X`` the row
#     coordinate. Getting this backwards transposes every image.
#   * ``circsine``'s ``trim`` option is not ported: in the original it multiplies by
#     ``(r < c) + (r >= c)``, which is identically 1.
# --------------------------------------------------------------------------------


def _centred_axis(sze: int) -> np.ndarray:
    """Kovesi's ``l:u``: ``-sze/2 : sze/2-1`` when even, ``-(sze-1)/2 : (sze-1)/2`` when odd."""
    if sze % 2 == 0:
        return np.arange(-sze // 2, sze // 2, dtype=float)
    return np.arange(-(sze - 1) // 2, (sze - 1) // 2 + 1, dtype=float)


def filter_grid(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadrant-shifted normalised frequency grid: ``(radius, fx, fy)``, DC at ``[0, 0]``."""
    fy = np.fft.ifftshift((np.arange(rows) - rows // 2) / rows)
    fx = np.fft.ifftshift((np.arange(cols) - cols // 2) / cols)
    FX, FY = np.meshgrid(fx, fy)  # 'xy': FX varies across columns
    return np.sqrt(FX**2 + FY**2), FX, FY


def step2line(sze: int = 512, *, nscales: int = 50, ampexponent: float = -1.0,
              ncycles: float = 1.5, phasecycles: float = 0.25) -> np.ndarray:
    """A phase-congruent image whose FEATURE TYPE sweeps step -> line down the rows.

    Every row is the same odd-harmonic series with a growing phase offset. The congruency
    points sit at fixed columns ``x = m*pi`` for every row -- at ``x = m*pi`` each odd
    harmonic ``sin(k x + phi)`` has the same phase ``m*pi + phi``, so they align whatever
    ``phi`` is. Congruency is therefore constant down the image while the feature morphs
    from a step into a line. Gradient magnitude is not (check_15).
    """
    x = np.arange(sze) / (sze - 1) * ncycles * 2 * np.pi
    offsets = phasecycles * 2 * np.pi * np.arange(sze) / sze
    img = np.zeros((sze, sze))
    for scale in range(1, 2 * nscales, 2):  # ODD harmonics only
        img += float(scale) ** ampexponent * np.sin(scale * x[None, :] + offsets[:, None])
    return img


def circsine(sze: int = 512, *, wavelength: float = 40.0, nscales: int = 50,
             ampexponent: float = -1.0, offset: float = 0.0, p: int = 2) -> np.ndarray:
    """Concentric circular waveform. Isophotes are exact circles: curvature ``1/r``, always.

    Only the RADIAL frequency content changes with ``nscales``/``ampexponent``/
    ``wavelength``; the geometry does not. That makes it an oracle for any quantity
    claiming to be isophote curvature (check_09b).
    """
    if p % 2:
        raise ValueError("p should be an even number")
    ax = _centred_axis(sze)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    r = (X**p + Y**p) ** (1.0 / p)
    img = np.zeros_like(r)
    for scale in range(1, 2 * nscales, 2):
        img += float(scale) ** ampexponent * np.sin(scale * r * 2 * np.pi / wavelength + offset)
    return img


def starsine(sze: int = 512, *, ncycles: float = 10.0, nscales: int = 50,
             ampexponent: float = -1.0, offset: float = 0.0) -> np.ndarray:
    """An angular waveform: radial rays at every orientation at once (check_17)."""
    ax = _centred_axis(sze)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    theta = np.arctan2(Y, X)  # Julia: atan(y, x), y on the SECOND axis
    img = np.zeros_like(theta)
    for scale in range(1, 2 * nscales, 2):
        img += float(scale) ** ampexponent * np.sin(scale * ncycles * theta + offset)
    return img


def noiseonf(sze: int, p: float, *, seed: int = 0) -> np.ndarray:
    """``1/f^p`` noise: random phase, amplitude spectrum replaced by ``1/radius^p``.

    The phase spectrum is pure noise, so there is no congruency anywhere -- the negative
    control for the Rayleigh threshold ``T`` (check_16). ``p = 1.5`` is roughly the
    amplitude falloff of natural images.
    """
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft2(rng.normal(size=(sze, sze)))
    magnitude = np.abs(spectrum)
    magnitude[magnitude == 0.0] = 1.0
    radius = filter_grid(sze, sze)[0] * sze + 1.0
    return np.real(np.fft.ifft2((spectrum / magnitude) / radius**p))


def unit_variance(img: np.ndarray) -> np.ndarray:
    """Zero mean, unit standard deviation. ``epsilon = 1e-4`` is absolute, so scale matters."""
    return (img - img.mean()) / img.std()


# --------------------------------------------------------------------------------
# monogenic phase congruency (monogenic-phase-congruency.md §2)
# --------------------------------------------------------------------------------


@dataclass
class Monogenic:
    pc: np.ndarray
    orientation: np.ndarray  # mod pi; 0 == vertical edge
    feature_type: np.ndarray  # 0 == step, +-pi/2 == line
    threshold: float
    n_clamped: int  # how often acos's argument left [-1, 1] (must be 0)


def monogenic_phase_congruency(
    img: np.ndarray,
    *,
    nscale: int = 4,
    min_wavelength: float = 3.0,
    mult: float = 2.1,
    sigma_onf: float = 0.55,
    k: float = 3.0,
    cutoff: float = 0.5,
    g: float = 10.0,
    deviation_gain: float = 1.5,
    eps: float = 1e-4,
    noise_threshold: bool = True,
    swap_axes: bool = False,
    flip_h2_sign: bool = False,
) -> Monogenic:
    """The operator specified in ``monogenic-phase-congruency.md`` §2, transcribed here.

    Log-Gabor bandpass + Riesz transform, summed over scales::

        PC = W * max(1 - deviation_gain*acos(E/(sumAn + eps)), 0) * max(E - T, 0)/(E + eps)

    ``swap_axes`` and ``flip_h2_sign`` inject the two axis-convention bugs that §7 warns
    about, so check_17 can show which tests catch them.
    """
    rows, cols = img.shape
    radius, FX, FY = filter_grid(rows, cols)
    radius[0, 0] = 1.0
    if swap_axes:
        FX, FY = FY, FX
    riesz = (1j * FX - FY) / radius
    riesz[0, 0] = 0.0
    lowpass = 1.0 / (1.0 + (radius / 0.45) ** 30)
    spectrum = np.fft.fft2(img)

    sum_an = np.zeros((rows, cols))
    max_an = np.zeros((rows, cols))
    sum_f = np.zeros((rows, cols))
    sum_h1 = np.zeros((rows, cols))
    sum_h2 = np.zeros((rows, cols))
    tau = 0.0
    for s in range(nscale):
        f0 = 1.0 / (min_wavelength * mult**s)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_gabor = np.exp(-(np.log(radius / f0) ** 2) / (2 * np.log(sigma_onf) ** 2))
        log_gabor[0, 0] = 0.0
        log_gabor *= lowpass

        even = np.real(np.fft.ifft2(spectrum * log_gabor))
        odd = np.fft.ifft2(spectrum * log_gabor * riesz)
        h1, h2 = odd.real, odd.imag
        an = np.sqrt(even * even + h1 * h1 + h2 * h2)

        sum_an += an
        sum_f += even
        sum_h1 += h1
        sum_h2 += h2
        if s == 0:
            # noiseMethod = -1: the Rayleigh parameter from the smallest scale's median.
            tau = float(np.median(sum_an)) / np.sqrt(np.log(4.0))
            max_an = an.copy()
        else:
            max_an = np.maximum(max_an, an)

    width = (sum_an / (max_an + eps) - 1.0) / (nscale - 1)
    weight = 1.0 / (1.0 + np.exp(g * (cutoff - width)))

    total_tau = tau * (1 - (1 / mult) ** nscale) / (1 - 1 / mult)  # geometric sum
    threshold = 0.0
    if noise_threshold:
        threshold = total_tau * np.sqrt(np.pi / 2) + k * total_tau * np.sqrt((4 - np.pi) / 2)

    energy = np.sqrt(sum_f**2 + sum_h1**2 + sum_h2**2)
    arg = energy / (sum_an + eps)
    n_clamped = int(np.count_nonzero((arg > 1.0) | (arg < -1.0)))
    pc = (
        weight
        * np.maximum(1 - deviation_gain * np.arccos(np.clip(arg, -1.0, 1.0)), 0)
        * np.maximum(energy - threshold, 0)
        / (energy + eps)
    )
    # Kovesi writes atan(-sumh2/sumh1); atan2 is equal mod pi and never divides by zero.
    h2_signed = sum_h2 if flip_h2_sign else -sum_h2
    orientation = np.arctan2(h2_signed, sum_h1) % np.pi
    feature_type = np.arctan2(sum_f, np.hypot(sum_h1, sum_h2))
    return Monogenic(pc, orientation, feature_type, threshold, n_clamped)


def angular_distance_mod_pi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest angle between two undirected orientations."""
    return np.abs(((a - b) + np.pi / 2) % np.pi - np.pi / 2)


# --------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    passed: bool
    detail: str


def check_01_sphere_geometry() -> Result:
    """S^-1 lands on the sphere; ``rho^2 == u_3``; the area element pulls back to ``lambda^2``.

    The sphere has radius 1/2, so its area is ``pi``, not ``4 pi``. The familiar
    ``4/(1+r^2)^2`` conformal factor belongs to the *unit* sphere. ``references.md`` §4.3.
    """
    rng = np.random.default_rng(0)
    pts = rng.normal(scale=2.0, size=(200, 2))
    u1, u2, u3, _ = lift_to_sphere(pts[:, 0], pts[:, 1])
    on_sphere = float(np.abs(u1**2 + u2**2 + (u3 - 0.5) ** 2 - 0.25).max())

    r2 = (pts**2).sum(1)
    rho2 = u1**2 + u2**2 + u3**2
    rho_identity = float(np.abs(rho2 - r2 / (1 + r2)).max())
    u3_identity = float(np.abs(rho2 - u3).max())

    # Radially symmetric => 1-D quadrature. int_0^inf 2 pi r/(1+r^2)^2 dr = pi.
    r = np.concatenate([np.linspace(0.0, 10.0, 200_001), np.geomspace(10.0, 1e6, 20_001)[1:]])
    area = float(np.trapezoid(2 * np.pi * r / (1.0 + r * r) ** 2, r))

    ok = on_sphere < 1e-15 and rho_identity < 1e-15 and u3_identity < 1e-15 and abs(area - np.pi) < 1e-6
    return Result(
        "01 sphere geometry: S^-1 on sphere, rho^2 == u3, int J dy == pi",
        ok,
        f"on-sphere {on_sphere:.1e}, rho^2 id {rho_identity:.1e}, "
        f"rho^2==u3 {u3_identity:.1e}, area {area:.6f} (pi = {np.pi:.6f})",
    )


def check_02_mask_symmetries() -> Result:
    """``Q1`` odd on **axis 0**, ``Q2`` odd on **axis 1**, ``Q3`` even and single-signed.

    ``indexing="ij"`` puts ``Y1`` on axis 0. An earlier revision of this test asserted the
    opposite axis for both masks and reported only the mask *sums*, which look fine either
    way -- so it failed while printing nothing useful. Report each property.

    ``Q3``'s evenness is the whole story: on ``S^2``, ``v3 = |v|^2 >= 0``, so the support is
    one-sided and the ``R^3`` Riesz kernel ``h3`` -- odd in ``v3``, and that oddness is what
    makes it a quadrature operator -- degenerates into an even, single-signed operator.
    """
    P, Q1, Q2, Q3, _, _, _ = conformal_masks(5, 4.0, 0.5)
    props = {
        "Q1 odd axis0": bool(np.allclose(Q1, -Q1[::-1, :])),
        "Q2 odd axis1": bool(np.allclose(Q2, -Q2[:, ::-1])),
        "Q1 zero mass": abs(float(Q1.sum())) < 1e-12,
        "Q2 zero mass": abs(float(Q2.sum())) < 1e-12,
        "Q3 even": bool(np.allclose(Q3, Q3[::-1, ::-1])),
        "Q3 single-signed": bool((Q3 >= 0).all()),
        "Q3 nonzero mass": float(Q3.sum()) > 0.1,
        "P even positive": bool(np.allclose(P, P[::-1, ::-1]) and (P > 0).all()),
    }
    bad = [k for k, v in props.items() if not v]
    return Result(
        "02 mask symmetries: Q1 odd axis0, Q2 odd axis1, Q3 even single-signed",
        not bad,
        (f"all 8 properties hold; sum(Q3) = {Q3.sum():+.4f} (DC-sensitive)" if not bad
         else f"FAILED: {', '.join(bad)}"),
    )


def check_03_moment_tensor_is_anisotropic() -> Result:
    """``M = diag(A, A, B)`` with ``A != B``, and ``B/A == 1`` at exactly one ``s0``.

    Two independent things are established, and it matters which is which:

    * the **closed-form antiderivatives** ``I1``, ``I2`` are correct -- checked against a
      1-D radial quadrature, which shares the sphere identity but not the antiderivative;
    * the **mask-truncated** ``A``, ``B`` converge to the closed form as ``R/sigma -> inf``.
      They do *not* agree at small ``R/sigma``, which is why the (P2) gain is a truncation
      correction, not an isotropy correction (check_08).

    Off-diagonals vanish by parity on the mask lattice itself -- no separate grid needed.
    """
    r = np.concatenate([np.linspace(0.0, 20.0, 400_001), np.geomspace(20.0, 1e5, 40_001)[1:]])
    lam = 1.0 / (1.0 + r * r)
    u3 = r * r * lam
    jac = 2 * np.pi * r * lam**2

    rows, worst = [], 0.0
    for s0 in (0.1, S0_ISOTROPY, 0.3, 0.5, 1.0):
        w = 1.0 / (u3 + s0 * s0) ** 2
        A = float(np.trapezoid(0.5 * (u3 - u3 * u3) * w * jac, r))
        B = float(np.trapezoid(u3 * u3 * w * jac, r))
        Ac, Bc = moment_tensor_closed_form(s0)
        worst = max(worst, abs(A - Ac) / Ac, abs(B - Bc) / Bc)
        rows.append((s0, B / A))

    ax = np.arange(-40, 41, dtype=float)
    Y1, Y2 = np.meshgrid(ax, ax, indexing="ij")
    g1, g2, g3, glam = lift_to_sphere(Y1 / 8.0, Y2 / 8.0)
    gw = 1.0 / (g3 + S0_ISOTROPY**2) ** 2
    gJ = glam**2
    off_max = max(
        abs(float((g1 * g2 * gw * gJ).sum())),
        abs(float((g1 * g3 * gw * gJ).sum())),
        abs(float((g2 * g3 * gw * gJ).sum())),
    )

    Ac, Bc = moment_tensor_closed_form(S0_ISOTROPY)
    trunc = []
    for ratio in (1, 2, 4, 8, 16):
        _, _, _, _, A, B, _ = conformal_masks(int(ratio * 16), 16.0, S0_ISOTROPY)
        trunc.append(B / A)
    converges = trunc[0] < 0.6 and trunc[-1] > 0.99 and all(a < b for a, b in zip(trunc, trunc[1:]))

    root = brentq(lambda s: moment_tensor_closed_form(s)[1] / moment_tensor_closed_form(s)[0] - 1.0,
                  0.01, 1.0, xtol=1e-12)
    ratio_small = moment_tensor_closed_form(1e-4)[1] / moment_tensor_closed_form(1e-4)[0]

    ok = (
        worst < 1e-6
        and off_max < 1e-12
        and abs(root - S0_ISOTROPY) < 1e-7
        and ratio_small < 0.2
        and any(abs(ba - 1.0) > 0.3 for _, ba in rows)
        and converges
    )
    return Result(
        "03 M = diag(A,A,B) anisotropic; B/A -> 1 only as the mask grows",
        ok,
        f"closed form vs radial quadrature relerr {worst:.1e}; off-diag {off_max:.1e}; "
        f"B/A=1 at s0*={root:.8f}; full-sphere B/A: "
        + ", ".join(f"{s:.3f}->{ba:.3f}" for s, ba in rows)
        + f"; mask B/A at R/sigma=1,2,4,8,16: {['%.3f' % t for t in trunc]}",
    )


def _linear_response(omega: np.ndarray, g0: float, slope: float, R: int, sigma: float, s0: float):
    """``Q^i`` for the exactly-affine signal ``g(v) = g0 + slope*<v, omega>``, and the mask constants.

    Exactly affine, so the lattice response is exact: ``Q^i = g0*mu_i + slope*(M omega)_i``.
    No small-frequency approximation is involved.
    """
    _, Q1m, Q2m, Q3m, A, B, mu3 = conformal_masks(R, sigma, s0)
    ax = np.arange(-R, R + 1, dtype=float)
    Y1, Y2 = np.meshgrid(ax, ax, indexing="ij")
    u1, u2, u3, _ = lift_to_sphere(Y1 / sigma, Y2 / sigma)
    g = g0 + slope * (u1 * omega[0] + u2 * omega[1] + u3 * omega[2])
    q = np.array([float((g * m).sum()) for m in (Q1m, Q2m, Q3m)])
    return q, A, B, mu3


def check_04a_eq89_omits_a_dc_term() -> Result:
    """**Counter-claim 1, regulariser-independent.** Eq. (89) drops ``g(0) * mu_3``.

    For *any* positive radial weight ``w``, ``mu_3 = int v_3 w dsigma > 0``, because
    ``v_3 = |v|^2 >= 0`` on this sphere. Take a **straight edge**: ``omega_3 = 0``, so
    Eq. (89) predicts ``Q^3 = omega_3 * T = 0`` exactly. With a nonzero mean ``g0`` the
    true response is ``Q^3 = g0 * mu_3 != 0``.

    This does not depend on the choice of ``s0``, on the mask radius, or on any
    normalisation. It is a structural consequence of the one-sided support.
    """
    omega = np.array([0.6, 0.8, 0.0])  # a straight edge: omega_3 == 0
    details = []
    ok = True
    for s0 in (0.1, S0_ISOTROPY, 0.5, 1.0):
        q_dc, _, _, mu3 = _linear_response(omega, 2.3, 1.7, 24, 6.0, s0)
        q_nodc, _, _, _ = _linear_response(omega, 0.0, 1.7, 24, 6.0, s0)
        eq89_predicts_zero = abs(q_nodc[2]) < 1e-9 * abs(q_nodc[0])
        actual_is_the_dc_term = abs(q_dc[2] - 2.3 * mu3) < 1e-12 * abs(2.3 * mu3)
        big = abs(q_dc[2]) > 0.1 * abs(q_dc[0])
        ok &= eq89_predicts_zero and actual_is_the_dc_term and big
        details.append(f"s0={s0:.3f}: Q3(g0=0)={q_nodc[2]:+.1e}, Q3(g0=2.3)={q_dc[2]:.4f}")
    return Result(
        "04a Eq.(89) omits a DC term g(0)*mu_3 -- REGULARISER-INDEPENDENT",
        ok,
        "straight edge (omega_3 = 0): Eq.(89) predicts Q3 = 0. " + "; ".join(details),
    )


def check_04b_eq89_assumes_an_isotropy_that_fails() -> Result:
    """**Counter-claim 2, conditional.** Eq. (89) needs ``A == B``; it holds only at ``s0*``.

    The exact law for ``g(v) = g0 + slope*<v, omega>``::

        Q1 = slope*A*omega_1     Q2 = slope*A*omega_2     Q3 = g0*mu_3 + slope*B*omega_3

    Eq. (89) says ``Q^i = omega_i * T``, i.e. ``(Q3/omega_3)/(Q1/omega_1) == 1``. It equals
    ``B/A``.

    Two of the four things this checks are **identities, not evidence**: that the DC term
    equals ``g0*mu_3`` (linearity) and that the ratio equals ``B/A`` (once the law holds).
    The evidence is ``law_err`` -- the exact diagonal form -- plus ``B/A != 1``.

    The regulariser ``s`` is JMIV's own (Eq. 20 + Eq. 88), so the anisotropy is a property
    of the paper's kernel, not of this spec. But it **degenerates at ``s0*``**, where the
    full-sphere ``A == B``; the check is run at ``s0 = 0.5``, and this conditionality is
    stated in ``references.md`` §4.3.1.
    """
    omega = np.array([0.30, -0.50, 0.40])
    omega /= np.linalg.norm(omega)
    slope, g0, s0 = 1.7, 2.3, 0.5

    q_nodc, A, B, mu3 = _linear_response(omega, 0.0, slope, 24, 6.0, s0)
    q_dc, _, _, _ = _linear_response(omega, g0, slope, 24, 6.0, s0)

    predicted = slope * np.array([A * omega[0], A * omega[1], B * omega[2]])
    law_err = float(np.abs(q_nodc - predicted).max() / np.abs(predicted).max())
    dc_leaks_into_odd = float(np.abs((q_dc - q_nodc)[:2]).max())
    eq89_ratio = (q_nodc[2] / omega[2]) / (q_nodc[0] / omega[0])

    # The degeneracy: at s0* the FULL-SPHERE A == B, so this half of the claim vanishes.
    Ac, Bc = moment_tensor_closed_form(S0_ISOTROPY)
    degenerate = abs(Bc / Ac - 1.0) < 1e-6

    ok = law_err < 1e-12 and dc_leaks_into_odd < 1e-9 and abs(eq89_ratio - 1.0) > 0.2 and degenerate
    return Result(
        "04b Eq.(89) assumes A == B -- CONDITIONAL (fails for s != s0*)",
        ok,
        f"exact diagonal law relerr {law_err:.1e}; DC does not leak into Q1,Q2 "
        f"({dc_leaks_into_odd:.1e}); at s0=0.5 (Q3/w3)/(Q1/w1) = {eq89_ratio:.4f} = B/A, not 1; "
        f"but at s0* the full-sphere B/A = {Bc / Ac:.7f} (claim degenerates there)",
    )


def check_05_jmiv_omega3_sign_is_wrong() -> Result:
    """The lifted isophote's plane normal is ``(2m1, 2m2, -1)``, not ``(2m1, 2m2, +1)``.

    A circle through the origin with centre ``m`` satisfies ``|y|^2 = 2<y, m>``. Lifting,
    ``u = lambda*(y, |y|^2)``, so ``<u, (2m, -1)> = lambda*(2<y,m> - |y|^2) = 0`` exactly.
    ``|kappa|`` is unaffected; ``kappa`` flips sign and ``theta`` rotates by pi.

    This verifies the *geometry*. Whether JMIV writes ``+1`` in this sphere convention is a
    reading of Eqs. (41)/(67)/(91); see ``references.md`` §4.3.
    """
    rng = np.random.default_rng(1)
    worst_correct, best_wrong = 0.0, np.inf
    for _ in range(5):
        m = rng.normal(size=2)
        theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        y = m[None, :] + np.linalg.norm(m) * np.stack([np.cos(theta), np.sin(theta)], 1)
        u1, u2, u3, _ = lift_to_sphere(y[:, 0], y[:, 1])
        u = np.stack([u1, u2, u3], 1)
        worst_correct = max(worst_correct, float(np.abs(u @ np.array([2 * m[0], 2 * m[1], -1.0])).max()))
        best_wrong = min(best_wrong, float(np.abs(u @ np.array([2 * m[0], 2 * m[1], +1.0])).max()))
    ok = worst_correct < 1e-13 and best_wrong > 0.1
    return Result(
        "05 the lifted isophote's normal is (2m1, 2m2, -1), not (+1)",
        ok,
        f"max |<lift, (2m,-1)>| = {worst_correct:.1e} (planar); "
        f"min max |<lift, (2m,+1)>| = {best_wrong:.4f} (not planar)",
    )


def check_06_value_removal_tames_the_divergence() -> Result:
    """The scale-free ``M3`` diverges; (P1) value-removal makes it converge to exactly 1/2.

    ``M3(y) = (1/pi^2)/(r^2(1+r^2))`` is even, positive and ``~1/(pi^2 r^2)`` at the origin,
    so ``int M3 * 2 pi r dr`` diverges logarithmically. With ``g~(0) = 0`` and ``g~``
    Lipschitz the integrand is ``O(1) dr``. For ``f(y) = 1 + r``::

        int_0^1 (f - f(0)) * M3 * 2 pi r dr = (2/pi) * int_0^1 dr/(1+r^2) = 1/2   exactly.

    So ``s0`` regularises the *DC response*, not the operator, and the faithful scale-free
    construction is available. ``references.md`` §4.3.2.
    """
    c = 1.0 / np.pi**2

    def radial(g_of_r: Callable[[np.ndarray], np.ndarray], eps: float, n: int = 400_000) -> float:
        r = np.linspace(eps, 1.0, n)
        return float(np.trapezoid(g_of_r(r) * (c / (r * r * (1 + r * r))) * 2 * np.pi * r, r))

    eps_list = (1e-1, 1e-2, 1e-4, 1e-6)
    raw = [radial(lambda r: 1.0 + r, e) for e in eps_list]
    removed = [radial(lambda r: r, e) for e in eps_list]
    # the divergence rate must match (2/pi) * ln(1/eps)
    predicted_slope = 2.0 / np.pi * np.log(10)
    slopes = [(raw[i + 1] - raw[i]) / np.log10(eps_list[i] / eps_list[i + 1]) for i in range(3)]
    rate_ok = all(abs(sl - predicted_slope) < 0.15 for sl in slopes)
    converges = abs(removed[-1] - 0.5) < 1e-4 and abs(removed[-1] - removed[-2]) < 1e-4
    return Result(
        "06 scale-free M3 diverges as -(2/pi)ln(eps); (P1) converges to exactly 1/2",
        rate_ok and converges,
        f"raw {['%.3f' % v for v in raw]} (slope/decade {['%.3f' % s for s in slopes]} vs "
        f"{predicted_slope:.3f}); value-removed {['%.6f' % v for v in removed]} -> 0.5",
    )


def check_07_s0_degeneracy_bound() -> Result:
    """``s0 >> 1`` collapses the kernel shape onto the Jacobian, because ``rho^2 < 1``.

    The sphere is compact, so once ``s0 >> 1`` the factor ``(rho^2 + s0^2)^2`` is effectively
    constant across the mask. The bound ``s0 <~ 1.5`` is where the shape deviation crosses
    ``1e-3``; assert the crossing rather than quoting an uninterpolated number.
    """
    half = 5
    ax = np.arange(-half, half + 1, dtype=float)
    Y1, Y2 = np.meshgrid(ax, ax, indexing="ij")
    _, _, _, lam = lift_to_sphere(Y1, Y2)
    limit = (Y1 * lam) * lam**2
    limit = limit / np.linalg.norm(limit)

    def deviation(s0: float) -> float:
        _, Q1, _, _, _, _, _ = conformal_masks(half, 1.0, s0)
        q = Q1 / np.linalg.norm(Q1)
        return 1.0 - abs(float((q * limit).sum()))

    devs = {s0: deviation(s0) for s0 in (0.25, 0.5, 1.0, 1.5, 2.0, 4.0)}
    crossing = brentq(lambda s: deviation(s) - 1e-3, 0.5, 4.0, xtol=1e-4)
    ok = devs[0.25] > 1e-2 and devs[2.0] < 1e-3 and 1.0 < crossing < 2.0
    return Result(
        "07 s0 degeneracy: shape -> J alone; the 1e-3 crossing is the s0 <~ 1.5 bound",
        ok,
        ", ".join(f"s0={s}:{d:.1e}" for s, d in devs.items()) + f"; crossing at s0 = {crossing:.3f}",
    )


def check_08_both_corrections_are_necessary() -> Result:
    """(P1) and (P2) are each necessary -- shown by SPREAD over ``(sigma, R)``, not one point.

    An earlier revision asserted this at a single ``(sigma=16, R=16)`` where ``raw = 1.0002``:
    the DC error and the gain error happen to cancel there. Sweeping shows ``raw`` spanning an
    order of magnitude while ``DC+gain`` stays near 1.

    It also removes the old "matched scale" arm. ``kappa*r`` is a function of ``(r/sigma,
    R/sigma)`` alone, so holding both fixed makes agreement a *dimensional necessity*, not
    evidence.

    And the (P2) gain corrects **mask truncation**, not isotropy: at ``s0*`` the full-sphere
    ``A == B`` (check_03), so on an untruncated mask the gain is a no-op.
    """
    grid = ((16, 16), (16, 32), (16, 64), (8, 32), (32, 32), (4, 16), (16, 8))
    cone = lambda rr: rr
    raw, dconly, gonly, both = [], [], [], []
    for sigma, R in grid:
        img, c = radial_image(cone, 2 * (4 + R) + 41)
        probe = lambda dc_, g_: curvature_estimate(
            img, c + 4, c, R, float(sigma), remove_dc=dc_, apply_gain=g_
        ) * 4
        raw.append(probe(False, False))
        dconly.append(probe(True, False))
        gonly.append(probe(False, True))
        both.append(probe(True, True))

    spread = lambda v: max(v) / min(v)
    ok = (
        spread(raw) > 3.0
        and spread(gonly) > 3.0
        and min(dconly) < 0.5
        and spread(both) < 2.0
    )
    return Result(
        "08 both (P1) and (P2) are necessary -- via spread over (sigma, R)",
        ok,
        f"raw spans {min(raw):.3f}-{max(raw):.3f} ({spread(raw):.1f}x); "
        f"DC-only {min(dconly):.3f}-{max(dconly):.3f}; gain-only spread {spread(gonly):.1f}x; "
        f"DC+gain {min(both):.3f}-{max(both):.3f} ({spread(both):.2f}x)",
    )


def check_09_kappa_is_scale_covariant_but_is_not_curvature() -> Result:
    """``kappa`` is scale-covariant, and it is **not isophote curvature**.

    Two separate facts, and an earlier revision of this spec confused them.

    *Scale covariance.* ``kappa*r`` depends only on ``(r/sigma, R/sigma)``. So doubling all
    three leaves it unchanged, and a "spread across ``r`` at fixed ``sigma``" is really a
    sweep over ``r/sigma``. The spec's old "``2.04x`` spread => not scale-free" was measured
    at ``R/sigma ~ 0.5``, a mask too small to contain the isophote -- the regime the spec
    itself declares invalid.

    *Not curvature.* Three radial profiles have **identical** isophote curvature ``1/r``
    (concentric circles), but ``Laplacian/|grad| = f''/f' + 1/r`` differs::

        f = r    -> 1/r    f = r^2 -> 2/r    f = r^3 -> 3/r

    The estimator tracks the latter. It coincides with curvature only when ``f'' == 0``, i.e.
    for a cone -- exactly JMIV's Fig. 13 test signal.

    Even inside JMIV's own signal class (an oscillatory circular signal, where Eq. (89)
    implies the frequency cancels), ``kappa*r`` varies ~3x with wavelength at fixed ``r``.
    """
    sigma, R = 16.0, 48

    covariant = [kappa_r_on_profile(lambda rr: rr, r, sig, Rm)
                 for r, sig, Rm in ((4, 16.0, 48), (8, 32.0, 96), (16, 64.0, 192))]
    covariance_ok = (max(covariant) / min(covariant)) < 1.01

    profiles = {"r": lambda rr: rr, "r^2": lambda rr: rr**2, "r^3": lambda rr: rr**3}
    means = {}
    per_profile_flat = True
    for name, fn in profiles.items():
        vals = [kappa_r_on_profile(fn, r, sigma, R) for r in (4, 8, 16)]
        means[name] = float(np.mean(vals))
        per_profile_flat &= (max(vals) / min(vals)) < 1.10
    g = means["r"]
    not_curvature = (means["r^2"] / g > 1.4) and (means["r^3"] / g > 2.0)

    freq = []
    for lam in (48.0, 200.0, 1600.0):
        r = 16
        n = 2 * (r + R) + 41
        c = n // 2
        ax = np.arange(n) - c
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        img = np.sin(2 * np.pi * (np.hypot(X, Y) - r) / lam)
        freq.append(curvature_estimate(img, c + r, c, R, sigma) * r)
    freq_dependent = (max(freq) / min(freq)) > 2.0

    ok = covariance_ok and per_profile_flat and not_curvature and freq_dependent
    return Result(
        "09 kappa is scale-COVARIANT but is NOT isophote curvature",
        ok,
        f"covariance: {['%.4f' % v for v in covariant]} (identical r/sigma, R/sigma); "
        f"profiles with the SAME curvature 1/r give kappa*r = "
        f"{means['r']:.3f}/{means['r^2']:.3f}/{means['r^3']:.3f} "
        f"(ratios 1.00/{means['r^2'] / g:.2f}/{means['r^3'] / g:.2f}; curvature predicts 1/1/1, "
        f"lap/grad predicts 1/2/3); frequency sweep at fixed r: {['%.3f' % v for v in freq]} "
        f"({max(freq) / min(freq):.2f}x)",
    )


def check_09b_circsine_confirms_kappa_is_not_curvature() -> Result:
    """The same verdict as check_09, on a test image authored by the algorithm's own author.

    ``circsine``'s isophotes are exact circles for every parameter setting, so the isophote
    curvature at radius ``r`` is ``1/r`` by construction and ``kappa*r == 1`` is the only
    admissible answer. check_09 built its own radial profiles; this one uses Kovesi's
    published generator, so the ground truth is not ours to get wrong.

    Two sweeps, both at **fixed geometry** (``r = 40``, ``sigma = 16``, ``R = 48``) -- necessary
    because check_09 established that ``kappa*r`` depends on ``(r/sigma, R/sigma)``, so varying
    ``r`` alongside the frequency would confound the two effects.

    Both sweeps probe ``r = m*wavelength/2``, where ``sin(k*2*pi*r/wavelength) = sin(k*m*pi) = 0``
    for every odd harmonic ``k`` -- a zero crossing of all harmonics at once, i.e. a phase-congruent
    step edge. The feature type is therefore identical across the sweep too; only the spectrum moves.
    """
    sigma, mask_r, r = 16.0, 48, 40

    def kappa_r(**kwargs: object) -> float:
        n = 2 * (r + mask_r) + 41
        img = circsine(n, **kwargs)  # type: ignore[arg-type]
        c = n // 2
        return curvature_estimate(img, c + r, c, mask_r, sigma) * r

    waveform = [
        kappa_r(wavelength=80.0, nscales=1, ampexponent=-1.0),   # pure sine
        kappa_r(wavelength=80.0, nscales=50, ampexponent=-1.0),  # square wave
        kappa_r(wavelength=80.0, nscales=50, ampexponent=-2.0),
        kappa_r(wavelength=80.0, nscales=50, ampexponent=-3.0),
    ]
    # r = m*wavelength/2 keeps the probe on a congruent zero crossing as the wavelength shrinks.
    frequency = [kappa_r(wavelength=2.0 * r / m, nscales=1, ampexponent=-1.0) for m in (1, 2, 3, 5)]

    waveform_spread = max(waveform) / min(waveform)
    frequency_spread = max(frequency) / min(frequency)
    # Not one setting recovers the true value, and both spreads are far outside float noise.
    never_unity = all(abs(v - 1.0) > 0.3 for v in waveform + frequency)
    ok = waveform_spread > 1.4 and frequency_spread > 4.0 and never_unity

    return Result(
        "09b circsine (Kovesi's own generator) confirms kappa is not curvature",
        ok,
        f"identical isophotes (circle r={r}, curvature 1/{r}) and identical feature type; "
        f"radial waveform sweep gives kappa*r = {['%.4f' % v for v in waveform]} "
        f"({waveform_spread:.2f}x); wavelength sweep at fixed r gives {['%.4f' % v for v in frequency]} "
        f"({frequency_spread:.2f}x). Curvature predicts 1.0000 for all eight; none is within 30%",
    )


def check_10_fz_is_an_even_channel() -> Result:
    """``f_z`` is even -- a Laplacian -- not an odd quadrature channel.

    ``Q3~`` is a positive radially symmetric kernel, so ``f * Q3~`` is a smoothing and (P1)
    makes it ``(blur - identity) f``, i.e. a Laplacian. Within a band that is a scalar
    multiple of the even channel ``c``, so the conformal lift adds nothing to the congruency
    output. ``references.md`` §9, ``conformal-lift.md`` §2.

    The magnitude of ``|corr(c, f_z)|`` is configuration-dependent (0.65 at ``s0=0.5``, up to
    0.92 at ``s0*``). The *structure* is what matters: ``f_z`` is orders of magnitude more
    correlated with ``c`` than ``f_x`` is.
    """
    rng = np.random.default_rng(3)
    img = gaussian_filter(rng.normal(size=(256, 256)), 2.0)
    c, q1, _, q3 = conformal_components(img, 8, 8.0, S0_ISOTROPY)
    b = planar_dop(1.0, 1.5, 12)
    c_bp, q1_bp, q3_bp = (correlate(x, b, mode="wrap") for x in (c, q1, q3))

    cc = lambda a, d: abs(float(np.corrcoef(a.ravel(), d.ravel())[0, 1]))
    even_odd = cc(c_bp, q1_bp)
    even_even = cc(c_bp, q3_bp)
    lap = cc(q3, laplace(img))

    ok = even_odd < 0.02 and even_even > 0.6 and (even_even / max(even_odd, 1e-9)) > 50 and lap > 0.75
    return Result(
        "10 f_z is EVEN (a Laplacian), redundant with the even channel c",
        ok,
        f"|corr(c, f_x)| = {even_odd:.4f} (odd, uncorrelated); |corr(c, f_z)| = {even_even:.4f} "
        f"({even_even / max(even_odd, 1e-9):.0f}x larger); |corr(f_z, laplacian)| = {lap:.4f}",
    )


def _congruency(img: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    comps = conformal_components(img, 5, 4.0, 0.25)
    vs = [np.zeros_like(img) for _ in comps]
    a_sum = np.zeros_like(img)
    for s in (1, 2, 3, 4):
        b = planar_dop(float(s), 1.5, 12)
        bp = [correlate(x, b, mode="nearest") for x in comps]
        a_sum += np.sqrt(sum(x * x for x in bp))
        for i, x in enumerate(bp):
            vs[i] += x
    return np.sqrt(sum(v * v for v in vs)), a_sum


def check_11_pipeline_is_dc_free_and_exactly_affine_invariant() -> Result:
    """DC-free composite masks; real edge contrast; **exact** affine invariance.

    An earlier revision asserted ``drift < 1e-9`` on ``E/(A + eps)`` and failed at ``3.4e-4``.
    That drift is entirely the ``eps``: with ``eps = 0`` and the amplitude masked, the
    construction is invariant to ``5.6e-16``. Assert the exact statement, and bound the
    ``eps`` case by ``2*eps/(3*a)`` rather than by a made-up constant.

    The old "flat background = 0" control was vacuous: the flat region has ``a_sum == 0``
    exactly, so ``0/eps = 0`` by construction. Use a *textured but incoherent* control
    (noise), where the amplitude is large and the congruency must still be low.
    """
    from scipy.signal import fftconvolve

    P, Q1, Q2, Q3, _, _, _ = conformal_masks(5, 4.0, 0.25)
    b = planar_dop(1.0, 1.5, 12)
    dc = max(abs(float(fftconvolve(b, m, mode="full").sum())) for m in (P, Q1, Q2, Q3))

    n = 192
    step = np.zeros((n, n))
    step[:, n // 2 :] = 1.0
    rng = np.random.default_rng(7)
    step[: n // 3, : n // 3] += 0.3 * rng.normal(size=(n // 3, n // 3))  # incoherent control

    E0, A0 = _congruency(step, 0.0)
    E1, A1 = _congruency(3.0 * step + 7.0, 0.0)
    # Mask RELATIVE to the peak amplitude: dividing by a vanishing a_sum amplifies float64
    # roundoff without saying anything about the invariance.
    m = A0 > 1e-3 * A0.max()
    drift_exact = float(np.abs(E0[m] / A0[m] - E1[m] / A1[m]).max())

    # The invariance is exact in exact arithmetic; in float64 the convolution chain
    # accumulates ~1e-12. The point is that the eps > 0 drift is ORDERS larger, so the
    # observed 3.4e-4 is the epsilon and not the algebra.
    eps = 1e-5
    drift_eps = float(np.abs(E0 / (A0 + eps) - E1 / (A1 + eps)).max())

    # Discriminate on the CONGRUENCY FUNCTIONAL, not on the raw ratio. Kovesi:
    # "PC = energy/sumAn ... is not very localised." Raw E/A is ~0.95 even for noise,
    # because adjacent DoP scales (t=1.5) overlap heavily and their responses correlate.
    # The sharpening is the functional's job: F = exp(-|E/A - 1| / b^2), b = 0.3.
    b = 0.3
    functional = np.zeros_like(E0)
    functional[m] = np.exp(-np.abs(E0[m] / A0[m] - 1.0) / b**2)
    edge = float(functional[n // 2, n // 2])
    noisy = float(np.median(functional[5 : n // 3 - 5, 5 : n // 3 - 5]))

    ok = (
        dc < 1e-10
        and drift_exact < 1e-10
        and drift_eps / drift_exact > 1e6
        and (edge - noisy) > 0.2
    )
    return Result(
        "11 pipeline: DC-free, EXACTLY affine-invariant, discriminates coherent structure",
        ok,
        f"max composite mask DC {dc:.1e}; |f -> 3f+7| drift: {drift_exact:.1e} at eps=0 "
        f"(float64 chain noise) vs {drift_eps:.1e} at eps=1e-5 -- {drift_eps / drift_exact:.0e}x "
        f"larger, so the drift IS the epsilon; functional F: edge {edge:.4f} vs incoherent noise "
        f"{noisy:.4f} (contrast {edge - noisy:.4f}) -- raw E/A alone gives only 0.038, as Kovesi warns",
    )


def check_12_fusion_numerator_must_match_denominator() -> Result:
    """An L2 numerator over an L1 denominator annihilates coherent multi-channel edges.

    For a perfect edge, ``E_i == A_i`` in every firing channel. ``color-phase-congruency.md``
    §3.1. Every case listed is asserted; an earlier revision computed the third and asserted
    nothing about it.
    """
    b = 0.3
    cases = {
        "one channel": np.array([1.0, 0.0, 0.0]),
        "all three equal": np.array([1.0, 1.0, 1.0]),
        "80/1/18 mix": np.array([0.804, 0.013, 0.183]),
    }
    resp = {}
    for name, e in cases.items():
        ratio = float(np.sqrt((e**2).sum()) / e.sum())
        resp[name] = (ratio, float(np.exp(-abs(ratio - 1) / b**2)))
    ok = (
        resp["one channel"][1] > 0.99
        and resp["all three equal"][1] < 0.02
        and resp["80/1/18 mix"][1] < 0.30
        and resp["one channel"][1] / resp["all three equal"][1] > 50
    )
    return Result(
        "12 L2-over-L1 fusion annihilates coherent multi-channel edges",
        ok,
        "; ".join(f"{k}: ratio {v[0]:.4f} -> response {v[1]:.4f}" for k, v in resp.items())
        + f"  ({resp['one channel'][1] / resp['all three equal'][1]:.0f}x the wrong way)",
    )


def check_13_congruency_functional_properties() -> Result:
    """``exp(-|x-1|/b^2) <= 1`` for every real ``x``; strictly increasing below 1, decreasing above.

    So the ``[0,1]`` bound never needed the triangle inequality. What ``E <= A`` buys is that
    the operating point stays on the **increasing** branch, and this check asserts that both
    branches exist and are strictly monotone -- not merely that the function is symmetric.
    ``references.md`` §4.5.
    """
    b = 0.3
    f = lambda x: np.exp(-np.abs(x - 1.0) / b**2)
    xs = np.linspace(-5, 5, 20001)
    bounded = float(f(xs).max()) <= 1.0 + 1e-12
    below = np.linspace(-2.0, 0.999, 2000)
    above = np.linspace(1.001, 4.0, 2000)
    increasing = bool(np.all(np.diff(f(below)) > 0))
    decreasing = bool(np.all(np.diff(f(above)) < 0))
    symmetric = abs(f(0.9) - f(1.1)) < 1e-12
    return Result(
        "13 exp(-|x-1|/b^2) <= 1 always; increasing below 1, decreasing above",
        bounded and increasing and decreasing and symmetric,
        f"max = {f(xs).max():.6f}; strictly increasing on x<1: {increasing}; decreasing on x>1: "
        f"{decreasing}; f(0.9) = f(1.1) = {f(0.9):.6f} (a super-congruent pixel is penalised)",
    )


def check_14_kernel_norm_rule_does_not_reduce_to_kovesi() -> Result:
    """The kernel-norm rule agrees with ``(1/mult)^s`` only on **consecutive** ratios, ``j >= 3``.

    An earlier revision claimed the rule "reduces to Kovesi's factor on Kovesi's bank" and
    tested it by slicing ``consecutive[2:]`` -- silently dropping ``consecutive[1] = 0.4918``,
    3.3% off, which would have failed the tolerance. That was a cherry-pick.

    What is actually true:

    * the *consecutive* norm ratios converge to ``1/mult`` from ``j = 3``;
    * the *cumulative* ``tau_j/tau_0`` is a persistent ``1.551x`` Kovesi's ``(1/mult)^j``,
      because the two already disagree at the anchor step (``0.7149`` vs ``0.4762``);
    * that anchor disagreement is **not** the lowpass -- with it removed the ratio is still
      ``0.5889``. The finest log-Gabor is not band-limited on the lattice;
    * ``totalTau`` therefore differs by a constant ``~1.25x``, which is exactly the sort of
      per-bank constant Kovesi says ``k`` absorbs.

    The recommendation (use the kernel-norm rule for a *new* bank) survives; the
    justification does not. ``references.md`` §4.4.3.
    """
    from numpy.fft import ifftshift

    n, mult, sigma_onf, min_wl = 512, 2.1, 0.55, 3.0
    fr = np.arange(-n / 2, n / 2) / n
    fx, fy = np.meshgrid(ifftshift(fr), ifftshift(fr))
    radius = np.hypot(fx, fy)
    radius[0, 0] = 1.0
    lowpass = 1.0 / (1.0 + (radius / 0.45) ** 30)

    def norms(apply_lowpass: bool) -> list[float]:
        out = []
        for s in range(6):
            f0 = 1.0 / (min_wl * mult**s)
            with np.errstate(divide="ignore", invalid="ignore"):
                lg = np.exp(-(np.log(radius / f0) ** 2) / (2 * np.log(sigma_onf) ** 2))
            lg[0, 0] = 0.0
            if apply_lowpass:
                lg = lg * lowpass
            out.append(float(np.sqrt((lg**2).sum())))
        return out

    nm = norms(True)
    consecutive = [nm[s] / nm[s - 1] for s in range(1, 6)]
    cumulative = [nm[j] / nm[0] for j in range(6)]
    kovesi = [(1 / mult) ** j for j in range(6)]
    offsets = [cumulative[j] / kovesi[j] for j in range(1, 6)]

    tail_converges = all(abs(r - 1 / mult) < 2e-3 for r in consecutive[2:])
    anchor_outlier = abs(consecutive[0] - 1 / mult) > 0.1
    second_step_off = 0.01 < abs(consecutive[1] - 1 / mult) < 0.05
    persistent_offset = all(abs(o - 1.551) < 0.06 for o in offsets[1:])
    not_the_lowpass = abs(norms(False)[1] / norms(False)[0] - 1 / mult) > 0.05
    total_tau_ratio = (1 + sum(cumulative[1:])) / (1 + sum(kovesi[1:]))

    # planar DoP closed form: ||p_a||^2 = 1/(8 pi a^2); <p_a,p_b> = 1/(2 pi (a+b)^2)
    sq = lambda a: 1.0 / (8 * np.pi * a * a)
    ip = lambda a, b_: 1.0 / (2 * np.pi * (a + b_) ** 2)
    dop_norm = lambda s, t: np.sqrt(sq(t * s) + sq(s) - 2 * ip(t * s, s))
    dop_ratios = [dop_norm(s, 1.5) / dop_norm(1.0, 1.5) for s in (1, 2, 3, 4)]
    closed = all(abs(r - 1.0 / s) < 1e-12 for r, s in zip(dop_ratios, (1, 2, 3, 4)))

    ok = (
        tail_converges and anchor_outlier and second_step_off
        and persistent_offset and not_the_lowpass and closed
        and 1.2 < total_tau_ratio < 1.3
    )
    return Result(
        "14 kernel-norm rule does NOT reduce to Kovesi's (1/mult)^s",
        ok,
        f"consecutive {['%.6f' % r for r in consecutive]} -> 1/mult = {1 / mult:.6f} (j>=3); "
        f"cumulative offset {['%.3f' % o for o in offsets]} (persistent 1.551x); "
        f"anchor 0.7149 is not the lowpass (0.5889 without it); totalTau ratio "
        f"{total_tau_ratio:.3f} (a constant, absorbed by k); planar DoP tau_j/tau_0 == s0/sj",
    )


def check_15_step2line_congruency_survives_the_feature_type_sweep() -> Result:
    """POSITIVE control. Congruency is invariant to feature type; gradient magnitude is not.

    Until now this file had no positive control -- every check said what something *is not*.
    ``step2line`` supplies one, and it is the claim the whole operator rests on: the same
    physical feature, seen as a step at the top of the image and as a line at the bottom,
    must score the same.

    The congruency column is ``x = 2*pi`` (col 170 of 256). ``pc`` is read down that column
    while ``feature_type`` sweeps from a step (``ft = 0``) to a line (``ft = +-pi/2``).

    Gradient magnitude *localises* the feature at every row -- its argmax lands on the same
    column -- but its VALUE collapses ~18x, so any fixed threshold on it misses the line
    rows. That is the failure mode phase congruency exists to fix.
    """
    n, col = 256, 170
    img = unit_variance(step2line(n))
    mono = monogenic_phase_congruency(img)
    gy, gx = np.gradient(img)
    grad = np.hypot(gx, gy)

    rows = np.arange(8, n - 8)  # trim the FFT wrap-around
    pc_col = mono.pc[rows, col]
    grad_col = grad[rows, col]
    ft_deg = np.degrees(mono.feature_type[rows, col])

    ft_span = ft_deg[-1] - ft_deg[0]
    ft_sweeps = 80.0 < ft_span < 100.0 and bool(np.all(np.diff(ft_deg) > 0))  # phasecycles=0.25 -> 90 deg

    pc_collapse = pc_col.max() / pc_col.min()
    grad_collapse = grad_col.max() / grad_col.min()
    endpoints = pc_col[-1] / pc_col[0]  # step row vs line row

    localised = np.mean([abs(int(np.argmax(mono.pc[i, col - 20 : col + 21])) - 20) <= 2 for i in rows])

    ok = (
        ft_sweeps
        and mono.n_clamped == 0  # M1: the acos argument never leaves [-1, 1]
        and 0.8 < endpoints < 1.25
        and pc_collapse < 2.0
        and grad_collapse > 8.0
        and grad_col[-1] / grad_col[0] < 0.15
        and localised == 1.0
    )
    return Result(
        "15 step2line: pc survives the step->line sweep, |grad| does not (POSITIVE control)",
        ok,
        f"feature_type sweeps {ft_deg[0]:+.1f} -> {ft_deg[-1]:+.1f} deg (monotone, expected 90); "
        f"pc {pc_col.min():.4f}-{pc_col.max():.4f} ({pc_collapse:.2f}x, endpoint ratio {endpoints:.3f}) "
        f"vs |grad| {grad_col.min():.4f}-{grad_col.max():.4f} ({grad_collapse:.1f}x, endpoint ratio "
        f"{grad_col[-1] / grad_col[0]:.4f}); pc peaks on the feature column in {localised * 100:.0f}% of "
        f"rows; acos clamped {mono.n_clamped} times",
    )


def check_16_noiseonf_exercises_the_rayleigh_threshold() -> Result:
    """NEGATIVE control. ``T`` is what separates signal from noise -- congruency alone does not.

    ``noiseonf`` has a pure-noise phase spectrum, so it contains no congruent features. Yet with
    the noise threshold disabled its 99.9th-percentile ``pc`` reaches ~0.72-0.76, against ~0.95 for
    a genuinely congruent image: a 1.3x margin, useless as a detector. This restates check_11's
    lesson (``E/A`` is high even for noise) on an image built to be adversarial.

    Switching ``T`` on cuts the noise 1.4-2.6x while touching the signal by ~1%, because ``tau`` is
    estimated from the image's own amplitude median: 1/f noise has a high noise floor and a high
    ``T``, while ``step2line`` is mostly flat and gets a ``T`` 5-60x smaller. The threshold adapts.

    Honest note: ``T`` does not drive 1/f noise to zero (0.29-0.41 remains at the 99.9th percentile).
    Kovesi's ``k`` is a standard-deviation count, not a guarantee.
    """
    n = 256
    signal = unit_variance(step2line(n))
    sig_on = monogenic_phase_congruency(signal)
    sig_off = monogenic_phase_congruency(signal, noise_threshold=False)

    def tail(mono: Monogenic) -> float:
        return float(np.quantile(mono.pc, 0.999))

    rows = []
    ok = True
    for p in (1.0, 1.5, 2.0):
        noise = unit_variance(noiseonf(n, p, seed=1))
        off = monogenic_phase_congruency(noise, noise_threshold=False)
        on = monogenic_phase_congruency(noise)
        rows.append((p, on.threshold, tail(off), tail(on)))
        ok &= tail(off) > 0.5  # congruency alone cannot reject noise
        ok &= tail(on) < 0.5  # the threshold can
        ok &= tail(off) / tail(on) > 1.35  # and it bites
        ok &= on.threshold / sig_on.threshold > 3.0  # T adapts to the image's noise floor

    ok &= tail(sig_on) > 0.9  # the signal is essentially untouched
    ok &= tail(sig_off) / tail(sig_on) < 1.05

    detail = "; ".join(
        f"p={p}: T={t:.4f} ({t / sig_on.threshold:.0f}x signal's), pc.999 {a:.4f} -> {b:.4f}"
        for p, t, a, b in rows
    )
    return Result(
        "16 noiseonf: the Rayleigh threshold T does the work (NEGATIVE control)",
        ok,
        f"{detail}; signal T={sig_on.threshold:.4f}, pc.999 {tail(sig_off):.4f} -> {tail(sig_on):.4f} "
        f"(cut {tail(sig_off) / tail(sig_on):.3f}x). Without T, 1/f noise scores within 1.3x of a "
        f"congruent image",
    )


def check_17_starsine_pins_the_orientation_convention() -> Result:
    """``starsine`` catches both axis bugs of §7. Axis-aligned edges catch neither reliably.

    The star's intensity depends only on the polar angle, so its edge normals sweep every
    orientation. The ground truth is the generator's own ``theta`` field -- no separate
    derivation to get wrong.

    Two injectable bugs, and what each test sees:

    ``swap_axes`` (fx/fy transposed)
        On a straight edge ``pc`` is identical to ``1.5e-17`` -- a ``pc`` test is blind. An
        orientation test on a vertical/horizontal pair *does* catch it (0 deg <-> 90 deg).

    ``flip_h2_sign`` (``atan2(+h2, h1)`` instead of ``-h2``)
        Reflects every orientation about the x-axis. On axis-aligned edges 0 and 90 deg are
        their own mirror images mod pi, so a vertical/horizontal pair is **blind to it**. Only a
        pattern with off-axis orientations catches it. The ``-h2`` sign encodes a y-UP
        convention; ``starsine``'s recovered orientation equals ``+theta``, not ``-theta``.
    """
    n = 128
    vertical = np.zeros((n, n))
    vertical[:, n // 2 :] = 1.0  # intensity varies across COLUMNS
    horizontal = np.zeros((n, n))
    horizontal[n // 2 :, :] = 1.0

    def orientation_at_peak(img: np.ndarray, **kw: bool) -> float:
        mono = monogenic_phase_congruency(img, **kw)
        return float(np.degrees(mono.orientation[np.unravel_index(np.argmax(mono.pc), mono.pc.shape)]))

    v_deg, h_deg = orientation_at_peak(vertical), orientation_at_peak(horizontal)
    convention_ok = abs(v_deg) < 1e-6 and abs(h_deg - 90.0) < 1e-6

    # The straight-edge pair is blind to the sign flip (0 and 90 are self-mirrored mod pi)...
    v_flip = orientation_at_peak(vertical, flip_h2_sign=True) % 180.0
    h_flip = orientation_at_peak(horizontal, flip_h2_sign=True) % 180.0
    flip_invisible = min(v_flip, 180.0 - v_flip) < 1e-6 and abs(h_flip - 90.0) < 1e-6

    # ...and pc is blind to the swap.
    pc_plain = monogenic_phase_congruency(vertical).pc
    pc_swapped = monogenic_phase_congruency(vertical, swap_axes=True).pc
    pc_blind = float(np.abs(pc_plain - pc_swapped).max())

    star = unit_variance(starsine(256, ncycles=8))
    ax = _centred_axis(256)
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    theta, r = np.arctan2(Y, X), np.hypot(X, Y)

    mono = monogenic_phase_congruency(star)
    mask = (mono.pc > 0.4) & (r > 30) & (r < 256 // 2 - 10)
    err = angular_distance_mod_pi(mono.orientation[mask], theta[mask] % np.pi)
    accurate = float(np.degrees(np.median(err))) < 2.0 and float(np.degrees(np.quantile(err, 0.9))) < 8.0

    flipped = monogenic_phase_congruency(star, flip_h2_sign=True)
    swapped = monogenic_phase_congruency(star, swap_axes=True)
    d_flip = float(np.degrees(np.median(angular_distance_mod_pi(flipped.orientation[mask], mono.orientation[mask]))))
    d_swap = float(np.degrees(np.median(angular_distance_mod_pi(swapped.orientation[mask], mono.orientation[mask]))))

    ok = convention_ok and flip_invisible and pc_blind < 1e-12 and accurate and d_flip > 30.0 and d_swap > 30.0
    return Result(
        "17 starsine pins the orientation convention and catches both axis bugs",
        ok,
        f"vertical edge -> {v_deg:.2f} deg, horizontal -> {h_deg:.2f} deg; on those edges pc is blind to "
        f"an fx/fy swap (max|dpc| = {pc_blind:.1e}) and orientation is blind to an h2 sign flip. "
        f"starsine ({int(mask.sum())} feature px): orientation matches the generator's theta to "
        f"{float(np.degrees(np.median(err))):.2f} deg median / "
        f"{float(np.degrees(np.quantile(err, 0.9))):.2f} deg at the 90th pct; sign flip shifts it "
        f"{d_flip:.1f} deg, axis swap {d_swap:.1f} deg",
    )


# --------------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------------

CHECKS: tuple[Callable[[], Result], ...] = (
    check_01_sphere_geometry,
    check_02_mask_symmetries,
    check_03_moment_tensor_is_anisotropic,
    check_04a_eq89_omits_a_dc_term,
    check_04b_eq89_assumes_an_isotropy_that_fails,
    check_05_jmiv_omega3_sign_is_wrong,
    check_06_value_removal_tames_the_divergence,
    check_07_s0_degeneracy_bound,
    check_08_both_corrections_are_necessary,
    check_09_kappa_is_scale_covariant_but_is_not_curvature,
    check_09b_circsine_confirms_kappa_is_not_curvature,
    check_10_fz_is_an_even_channel,
    check_11_pipeline_is_dc_free_and_exactly_affine_invariant,
    check_12_fusion_numerator_must_match_denominator,
    check_13_congruency_functional_properties,
    check_14_kernel_norm_rule_does_not_reduce_to_kovesi,
    check_15_step2line_congruency_survives_the_feature_type_sweep,
    check_16_noiseonf_exercises_the_rayleigh_threshold,
    check_17_starsine_pins_the_orientation_convention,
)


def run_all_checks() -> int:
    """Run every check, print a report, and return a process exit code."""
    print("Verifying the mathematical claims of 2026-07-08-alt-phase-detection\n")
    failures = 0
    for check in CHECKS:
        result = check()
        failures += not result.passed
        print(f"[{'PASS' if result.passed else 'FAIL'}] {result.name}")
        print(f"       {result.detail}\n")
    total = len(CHECKS)
    print(f"{total - failures}/{total} checks passed.")
    if failures:
        print("\nA failure here means the spec is wrong, or the test is. Find out which before fixing either.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run_all_checks())
