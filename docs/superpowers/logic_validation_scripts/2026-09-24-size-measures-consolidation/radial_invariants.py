#!/usr/bin/env python3
"""Re-derive the size-measures-consolidation spec's load-bearing numeric claims.

Spec: ``docs/superpowers/specs/2026-09-24-size-measures-consolidation/design.md``.

Depends only on the stdlib, numpy and scipy. Never imports ``phenotypic``: the point
is to check the *spec*, not the implementation of it. Exits non-zero if any claim
has stopped being true.

The radial signature here is evaluated **analytically** at the ``K`` bin centres,
not traced from a pixel contour: scikit-image's ``find_contours`` is outside this
directory's dependency contract. The shipped implementation (ported from branch
``shape-radial-measures``, ``_trace_radial_signature``) bins a subpixel contour and
keeps the outermost crossing per bin, so its values on rasterised shapes differ
from these by sub-pixel amounts. The unit tests pin the rasterised values; this
script pins the geometry the spec's prose quotes.

Run:  uv run --no-project --with numpy --with scipy python radial_invariants.py
"""

import sys

import numpy as np
from scipy import ndimage as ndi
from scipy import stats

K = 360  # angular_bins default
TRIM = 0.2  # trim_proportion default
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILURES.append(name)


def bin_centres(k: int = K) -> np.ndarray:
    """Angles of the k bin centres over [-pi, pi)."""
    return (np.arange(k) + 0.5) * 2.0 * np.pi / k - np.pi


def five_radii(signature: np.ndarray, inscribed: float) -> dict[str, float]:
    return {
        "inscribed": inscribed,
        "median": float(np.median(signature)),
        "mean": float(signature.mean()),
        "robust_mean": float(stats.trim_mean(signature, TRIM)),
        "max": float(signature.max()),
    }


def check_01_edt_statistics_on_a_disk_are_not_radii() -> None:
    """Old Shape_Mean/Median/MaxRadius on a disk: R/3, R(1-1/sqrt2), R.

    Tolerance: rasterising a disk moves its effective boundary by at most ~1 px,
    so each ratio to R may move by ~1/R. At R=100 that is 0.01, which is still
    far tighter than the gap between the claimed ratios (0.333 vs 0.293 vs 1).
    """
    r = 100
    y, x = np.ogrid[-150:151, -150:151]
    mask = (x * x + y * y) <= r * r
    d = ndi.distance_transform_edt(mask)[mask]
    tol = 1.0 / r
    for label, got, want in [
        ("mean/R", d.mean() / r, 1.0 / 3.0),
        ("median/R", float(np.median(d)) / r, 1.0 - 1.0 / np.sqrt(2.0)),
        ("max/R", d.max() / r, 1.0),
    ]:
        check(f"01 disk EDT {label}", abs(got - want) <= tol,
              f"got {got:.4f}, want {want:.4f} +/- {tol}")


def check_02_whole_objmap_edt_merges_touching_colonies() -> None:
    """Two labels sharing a full edge: whole-map EDT 20/21, per-object EDT 10/11.

    Integer geometry, so the EDT maxima are exact integers; compared exactly.
    """
    objmap = np.zeros((43, 43), dtype=int)
    objmap[1:42, 1:21] = 1  # 41 x 20
    objmap[1:42, 21:42] = 2  # 41 x 21
    whole = ndi.distance_transform_edt(objmap)  # binarises its input
    per_object = []
    merged = []
    for lab in (1, 2):
        sl = ndi.find_objects((objmap == lab).astype(int))[0]
        crop = np.pad(objmap[sl] == lab, 1)
        per_object.append(float(ndi.distance_transform_edt(crop).max()))
        merged.append(float(whole[objmap == lab].max()))
    check("02 whole-objmap EDT inflates", merged == [20.0, 21.0], f"got {merged}")
    check("02 per-object EDT is correct", per_object == [10.0, 11.0], f"got {per_object}")


def check_03_disk_all_five_radii_equal_r() -> None:
    r = 37.0
    sig = np.full(K, r)
    radii = five_radii(sig, inscribed=r)
    check("03 disk: all five radii == R",
          all(abs(v - r) < 1e-12 for v in radii.values()), f"{radii}")


def rectangle_signature(half_len: float, half_wid: float) -> np.ndarray:
    th = bin_centres()
    c, s = np.abs(np.cos(th)), np.abs(np.sin(th))
    with np.errstate(divide="ignore"):
        return np.minimum(half_len / c, half_wid / s)


def check_04_elongated_colony_values() -> None:
    """100 x 20 rectangle: 10.0 / 14.1 / 21.0 / 16.2 / 50.9 as printed in the spec.

    The spec prints one decimal place, so each printed value is within 0.05 of the
    true value by rounding alone; the tolerance is exactly that.
    """
    sig = rectangle_signature(50.0, 10.0)
    radii = five_radii(sig, inscribed=10.0)  # nearest edge from the centre: half-width
    printed = {"inscribed": 10.0, "median": 14.1, "mean": 21.0,
               "robust_mean": 16.2, "max": 50.9}
    for key, want in printed.items():
        got = radii[key]
        check(f"04 rectangle {key}", abs(got - want) <= 0.05,
              f"got {got:.3f}, spec prints {want}")
    # The nearest bin centre sits half a bin (pi/K) off the perpendicular, so the
    # sampled minimum overshoots the true nearest-edge distance by exactly
    # h/cos(pi/K) - h. That is why InscribedRadius comes from the EDT, not the signature.
    upper = 10.0 / np.cos(np.pi / K)
    check("04 signature min approximates the inscribed radius",
          10.0 <= sig.min() <= upper + 1e-12,
          f"signature min {sig.min():.6f} in [10, {upper:.6f}]")
    check("04 trim discounts elongation (robust < mean)",
          radii["robust_mean"] < radii["mean"] - 4.0,
          f"robust {radii['robust_mean']:.2f} vs mean {radii['mean']:.2f}")


def runner_signature(r: float, half_wid: float, reach: float) -> np.ndarray:
    """Disk of radius r plus a runner {0 <= x <= reach, |y| <= half_wid}, centre at origin.

    Outermost crossing per ray: the disk edge, or the runner's far end / side,
    whichever is further out along that ray.
    """
    th = bin_centres()
    c, s = np.cos(th), np.abs(np.sin(th))
    runner = np.zeros_like(th)
    fwd = c > 0
    tan = s[fwd] / c[fwd]
    with np.errstate(divide="ignore"):
        x_exit = np.minimum(reach, np.where(tan > 0, half_wid / tan, reach))
    runner[fwd] = x_exit / c[fwd]
    return np.maximum(r, runner)


def check_05_angular_sampling_keeps_runner_inside_breakdown() -> None:
    """Colony r=40 with a half-width-3 runner to x=90 (the branch plan's worked case).

    Boundary-pixel sampling: 29.9% of samples beyond r=45, which exceeds the 20% trim's
    breakdown point -> trimmed mean 42.03. Angle sampling (analytic, bin centres):
    2.2% -> trimmed mean 40.00. The branch plan printed 2.5% / 40.03 from its pixel
    contour; the spec quotes this script's analytic 2.2% and cites the branch figure.
    """
    r, hw, reach = 40, 3, 90
    y, x = np.mgrid[-100:101, -100:101]
    mask = (x * x + y * y <= r * r) | ((np.abs(y) <= hw) & (x >= 0) & (x <= reach))
    boundary = mask & ~ndi.binary_erosion(mask)
    d = np.hypot(x[boundary], y[boundary])
    frac_px = float((d > 45).mean())
    trim_px = float(stats.trim_mean(d, TRIM))
    check("05 boundary-pixel contamination exceeds breakdown",
          abs(frac_px - 0.299) <= 0.0005 and frac_px > TRIM, f"{frac_px:.4f}")
    check("05 boundary-pixel trimmed mean is dragged", abs(trim_px - 42.03) <= 0.005,
          f"{trim_px:.4f}")

    sig = runner_signature(r, hw, reach)
    frac_ang = float((sig > 45).mean())
    radii = five_radii(sig, inscribed=float(r))
    check("05 angle contamination inside breakdown",
          abs(frac_ang - 0.022) <= 0.0005 and frac_ang < TRIM, f"{frac_ang:.4f}")
    check("05 robust mean ignores the runner", abs(radii["robust_mean"] - r) <= 0.01,
          f"{radii['robust_mean']:.4f}")
    check("05 plain mean is pulled up by the runner", radii["mean"] > r + 0.5,
          f"{radii['mean']:.4f}")
    check("05 max radius reaches the runner tip", abs(radii["max"] - np.hypot(reach, hw)) < 0.1,
          f"{radii['max']:.3f} vs tip {np.hypot(reach, hw):.3f}")


def check_06_wide_runner_separates_mean_from_robust_mean() -> None:
    """Implementation plan's unit-test fixture: r=40, runner half-width 8 to x=100.

    The unit test compares a rasterised colony against these values at a 0.6 px
    tolerance (half-pixel contour offset), so it needs a mean-vs-robust gap well
    above 2 x 0.6 = 1.2 px to catch a swap of the two estimators. Analytic gap:
    42.35 - 40.00 = 2.35 px. The runner still covers only 5.6% of directions,
    inside the 20% breakdown point, so the robust mean stays on the body radius.
    """
    sig = runner_signature(40, 8, 100)
    radii = five_radii(sig, inscribed=40.0)
    frac = float((sig > 45).mean())
    check("06 wide runner inside breakdown", frac < TRIM and abs(frac - 0.0556) <= 0.0005,
          f"{frac:.4f}")
    check("06 robust mean on the body", abs(radii["robust_mean"] - 40.0) <= 0.01,
          f"{radii['robust_mean']:.4f}")
    check("06 mean pulled up by 2.35 px", abs(radii["mean"] - 42.348) <= 0.005,
          f"{radii['mean']:.4f}")
    check("06 gap exceeds twice the unit-test tolerance",
          radii["mean"] - radii["robust_mean"] > 2 * 0.6,
          f"gap {radii['mean'] - radii['robust_mean']:.3f}")
    check("06 median on the body", abs(radii["median"] - 40.0) <= 1e-9,
          f"{radii['median']:.4f}")


def main_checks() -> int:
    check_01_edt_statistics_on_a_disk_are_not_radii()
    check_02_whole_objmap_edt_merges_touching_colonies()
    check_03_disk_all_five_radii_equal_r()
    check_04_elongated_colony_values()
    check_05_angular_sampling_keeps_runner_inside_breakdown()
    check_06_wide_runner_separates_mean_from_robust_mean()
    print(f"\n{len(FAILURES)} failure(s)" + (f": {FAILURES}" if FAILURES else ""))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main_checks())
