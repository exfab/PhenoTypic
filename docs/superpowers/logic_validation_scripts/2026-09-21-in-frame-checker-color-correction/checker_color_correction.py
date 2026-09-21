#!/usr/bin/env python3
"""Independent witness for the numeric claims in the in-frame-checker
colour-correction spec (``docs/superpowers/specs/
2026-09-21-in-frame-checker-color-correction/README.md``).

Depends only on the stdlib + numpy/scipy and never imports ``phenotypic`` or
``colour``: CIEDE2000, the Weiszfeld geometric median and the Finlayson-2015
root-polynomial expansion are all re-derived here, so a reader can check the
spec's load-bearing numbers without trusting the code the spec describes.

Claims checked
--------------
C1  The candidate-restricted ΔE2000 medoid (score the K pixels nearest the Lab
    geometric median against *every* pixel) returns the same pixel as the
    exhaustive medoid, for unimodal tile-like clouds, at O(K·n) instead of
    O(n²) cost and with no random subsampling.
C2  The seeded-subsample medoid that ``phenotypic.util.medoid_ciede2000`` uses
    by default is not seed-stable at the cap sizes affordable here, which is
    why the spec replaces it rather than raising ``medoid_max_pixels``.
C3  The root-polynomial expansion is positively homogeneous of degree 1
    (Φ(k·RGB) = k·Φ(RGB)); a fit on the frame's own checker therefore absorbs
    an exposure gain that a transferred profile cannot.
C4  Degree-3 (13-term) root-polynomial least squares is rank-deficient below 13
    patches and badly conditioned at 24, whereas degree-2 (6-term) stays
    well-conditioned — the basis of the spec's degree-demotion ladder.

Exit code is 0 only if every claim holds.
"""

from __future__ import annotations

import sys

import numpy as np

# ---------------------------------------------------------------------------
# CIEDE2000 (Sharma, Wu & Dalal 2005), vectorised, no external colour library
# ---------------------------------------------------------------------------


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """ΔE00 between broadcastable ``(..., 3)`` CIE L*a*b* arrays (kL=kC=kH=1)."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar**7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0**7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    h1p = np.where((np.abs(a1p) + np.abs(b1)) == 0.0, 0.0, h1p)
    h2p = np.where((np.abs(a2p) + np.abs(b2)) == 0.0, 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    both = (C1p * C2p) != 0.0
    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where(both, dhp, 0.0)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(0.5 * dhp))

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    hsum = h1p + h2p
    hdiff = np.abs(h1p - h2p)
    hp_bar = np.where(
        both,
        np.where(
            hdiff <= 180.0,
            0.5 * hsum,
            np.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0)),
        ),
        hsum,
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )

    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar**7
    R_C = 2.0 * np.sqrt(Cp_bar7 / (Cp_bar7 + 25.0**7))
    S_L = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(
        20.0 + (Lp_bar - 50.0) ** 2
    )
    S_C = 1.0 + 0.045 * Cp_bar
    S_H = 1.0 + 0.015 * Cp_bar * T
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    return np.sqrt(
        (dLp / S_L) ** 2
        + (dCp / S_C) ** 2
        + (dHp / S_H) ** 2
        + R_T * (dCp / S_C) * (dHp / S_H)
    )


#: Sharma, Wu & Dalal (2005) Table 1 — a subset spanning the hue-rotation,
#: chroma and neutral branches. (Lab1, Lab2, expected ΔE00).
SHARMA_CASES = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


def check_c0_ciede2000() -> bool:
    """C0 — the local ΔE00 matches the published Sharma et al. test vectors."""
    worst = 0.0
    for lab1, lab2, expected in SHARMA_CASES:
        got = float(delta_e_2000(np.array(lab1), np.array(lab2)))
        worst = max(worst, abs(got - expected))
    ok = worst < 1e-4
    print(f"C0 CIEDE2000 vs Sharma et al. table: max |Δ| = {worst:.2e}  -> {_v(ok)}")
    return ok


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def weiszfeld(points: np.ndarray, tol: float = 1e-6, max_iter: int = 200):
    """Euclidean geometric median; plain Weiszfeld with a distance floor."""
    x = points.mean(axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(points - x, axis=1)
        w = 1.0 / np.maximum(d, 1e-12)
        nxt = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(nxt - x) < tol:
            return nxt
        x = nxt
    return x


def exhaustive_medoid(lab: np.ndarray, chunk: int = 128):
    """Real pixel minimising total ΔE2000 to every other pixel. O(n²)."""
    n = lab.shape[0]
    row_sums = np.empty(n)
    for s in range(0, n, chunk):
        blk = lab[s : s + chunk]
        row_sums[s : s + chunk] = delta_e_2000(blk[:, None, :], lab[None, :, :]).sum(1)
    i = int(row_sums.argmin())
    return i, lab[i], float(row_sums[i])


def candidate_medoid(lab: np.ndarray, k: int = 256, chunk: int = 64):
    """Spec's estimator: score only the *k* pixels nearest the Lab geometric
    median, each against **all** pixels. Deterministic, O(k·n)."""
    g = weiszfeld(lab)
    order = np.argsort(np.linalg.norm(lab - g, axis=1))
    cand_idx = order[: min(k, lab.shape[0])]
    cand = lab[cand_idx]
    row_sums = np.empty(cand.shape[0])
    for s in range(0, cand.shape[0], chunk):
        blk = cand[s : s + chunk]
        row_sums[s : s + chunk] = delta_e_2000(blk[:, None, :], lab[None, :, :]).sum(1)
    w = int(row_sums.argmin())
    return int(cand_idx[w]), cand[w], float(row_sums[w]), w


def subsample_medoid(lab: np.ndarray, max_pixels: int, seed: int, chunk: int = 128):
    """Shipped ``medoid_ciede2000`` behaviour: medoid of a seeded subsample."""
    rng = np.random.default_rng(seed)
    n = lab.shape[0]
    sample = lab[rng.choice(n, size=min(max_pixels, n), replace=False)]
    m = sample.shape[0]
    row_sums = np.empty(m)
    for s in range(0, m, chunk):
        blk = sample[s : s + chunk]
        row_sums[s : s + chunk] = delta_e_2000(blk[:, None, :], sample[None, :, :]).sum(1)
    return sample[int(row_sums.argmin())]


def tile_cloud(seed: int, n: int) -> np.ndarray:
    """A checker-tile-like Lab pixel cloud: unimodal, sensor-noise dominated."""
    rng = np.random.default_rng(seed)
    centre = np.array(
        [rng.uniform(20.0, 85.0), rng.uniform(-45.0, 60.0), rng.uniform(-55.0, 65.0)]
    )
    sd = np.array([rng.uniform(1.5, 4.0), rng.uniform(1.0, 3.0), rng.uniform(1.0, 3.0)])
    return centre + rng.normal(0.0, 1.0, (n, 3)) * sd


# ---------------------------------------------------------------------------
# Root-polynomial expansion (Finlayson, Mackiewicz & Hurlbert 2015)
# ---------------------------------------------------------------------------


def root_polynomial(rgb: np.ndarray, degree: int) -> np.ndarray:
    """Root-polynomial feature expansion: 3 terms (d1), 6 (d2), 13 (d3)."""
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    terms = [R, G, B]
    if degree >= 2:
        terms += [np.sqrt(R * G), np.sqrt(G * B), np.sqrt(R * B)]
    if degree >= 3:
        cr = lambda x: np.cbrt(x)  # noqa: E731
        terms += [
            cr(R * G * B),
            cr(R * R * G),
            cr(R * R * B),
            cr(G * G * R),
            cr(G * G * B),
            cr(B * B * R),
            cr(B * B * G),
        ]
    return np.column_stack(terms)


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------


def _v(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def check_c1_candidate_medoid(n_trials: int = 8, k: int = 256) -> bool:
    """C1 — candidate-restricted medoid == exhaustive medoid, deterministically."""
    mismatches, worst_de, worst_rank = 0, 0.0, 0.0
    for t in range(n_trials):
        lab = tile_cloud(seed=1000 + t, n=int(np.random.default_rng(t).integers(3000, 5200)))
        i_ex, med_ex, _ = exhaustive_medoid(lab)
        i_ca, med_ca, _, rank = candidate_medoid(lab, k=k)
        de = float(delta_e_2000(med_ex, med_ca))
        worst_de = max(worst_de, de)
        worst_rank = max(worst_rank, rank / k)
        if i_ex != i_ca:
            mismatches += 1
        # determinism: no RNG anywhere in the candidate path
        assert candidate_medoid(lab, k=k)[0] == i_ca
    ok = mismatches == 0 and worst_de < 1e-12 and worst_rank < 0.8
    print(
        f"C1 candidate medoid (K={k}) vs exhaustive over {n_trials} tiles: "
        f"{mismatches} pixel mismatches, max ΔE00 {worst_de:.2e}, "
        f"worst winner rank {worst_rank:.1%} of K  -> {_v(ok)}"
    )
    return ok


def check_c2_subsample_instability(n_tiles: int = 4, caps=(1000, 2000)) -> bool:
    """C2 — the seeded-subsample medoid moves with the seed by ≳0.1 ΔE00."""
    worst = {}
    for cap in caps:
        spreads = []
        for t in range(n_tiles):
            lab = tile_cloud(seed=2000 + t, n=4200)
            centres = [subsample_medoid(lab, cap, seed=s) for s in range(5)]
            pair = [
                float(delta_e_2000(centres[i], centres[j]))
                for i in range(len(centres))
                for j in range(i + 1, len(centres))
            ]
            spreads.append(max(pair))
        worst[cap] = float(np.median(spreads))
    ok = all(v > 0.1 for v in worst.values())
    detail = ", ".join(f"cap={c}: median max-pair {v:.3f} ΔE00" for c, v in worst.items())
    print(f"C2 seeded-subsample medoid is seed-dependent ({detail})  -> {_v(ok)}")
    return ok


def check_c3_homogeneity(degrees=(1, 2, 3)) -> bool:
    """C3 — Φ(k·RGB) = k·Φ(RGB) for every supported degree."""
    rng = np.random.default_rng(7)
    rgb = rng.uniform(0.01, 1.0, (500, 3))
    worst = 0.0
    for d in degrees:
        for gain in (0.25, 0.5, 1.6, 4.0):
            lhs = root_polynomial(rgb * gain, d)
            rhs = gain * root_polynomial(rgb, d)
            worst = max(worst, float(np.abs(lhs - rhs).max()))
    ok = worst < 1e-12
    print(f"C3 root-polynomial homogeneity, max |Φ(kx) − kΦ(x)| = {worst:.2e}  -> {_v(ok)}")
    return ok


def check_c4_conditioning() -> bool:
    """C4 — degree-3 is rank-deficient below 13 patches and ill-conditioned at 24."""
    rng = np.random.default_rng(11)
    # Stand-in for a 24-patch chart: linear RGB spread over the unit cube.
    patches = np.clip(rng.uniform(0.02, 0.95, (24, 3)), 1e-6, None)
    rows = []
    for n in (24, 20, 18, 16, 13, 12, 6):
        sub = patches[:n]
        for d in (2, 3):
            phi = root_polynomial(sub, d)
            n_terms = phi.shape[1]
            rank = int(np.linalg.matrix_rank(phi))
            cond = float(np.linalg.cond(phi))
            rows.append((n, d, n_terms, rank, cond, rank >= n_terms))
    print("C4 design-matrix rank / conditioning")
    print("    patches  degree  terms  rank  cond(Φ)      determined")
    for n, d, k, r, c, det in rows:
        print(f"    {n:7d}  {d:6d}  {k:5d}  {r:4d}  {c:10.3e}  {det}")
    d3 = {n: (det, c) for n, d, _, _, c, det in rows if d == 3}
    d2 = {n: (det, c) for n, d, _, _, c, det in rows if d == 2}
    ok = (
        d3[12][0] is False  # degree 3 underdetermined at 12 patches
        and d3[13][0] is True  # exactly determined at 13
        and d2[6][0] is True  # degree 2 determined down to 6
        and d3[24][1] > 10.0 * d2[24][1]  # and far worse conditioned at 24
    )
    print(
        f"    degree-3 undetermined at 12 / determined at 13; degree-2 determined at 6; "
        f"cond ratio at 24 patches = {d3[24][1] / d2[24][1]:.1f}x  -> {_v(ok)}"
    )
    return ok


def main() -> int:
    results = [
        check_c0_ciede2000(),
        check_c1_candidate_medoid(),
        check_c2_subsample_instability(),
        check_c3_homogeneity(),
        check_c4_conditioning(),
    ]
    print()
    if all(results):
        print("All claims hold.")
        return 0
    print(f"{results.count(False)} claim(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
