"""Re-derive the claims the nearest-object search in MeasureNeighborDist rests on.

Independent witness for docs/superpowers/specs/2026-09-24-measure-neighbor-nearest/
design.md. Depends only on the stdlib + numpy/scipy; never imports phenotypic.

Claims checked (spec section "Correctness argument"):

C1  Boundary sufficiency: for two disjoint pixel sets A and B, the minimum
    pixel-centre Euclidean distance over all (a, b) pairs equals the minimum over
    4-connected boundary pixels only (pixels with at least one 4-neighbour
    outside the set). Holds for non-convex shapes and shapes with holes.
C2  Bounding-box lower bound: hypot(gap_r, gap_c), with gaps taken from the
    *inclusive* pixel bounding boxes, never exceeds the true mask distance.
C3  Branch-and-bound exactness: visiting candidates in ascending (lower bound,
    label) order and stopping once the next lower bound strictly exceeds the
    best distance yields the same (label, distance) as brute force over every
    other object, including the smaller-label tie-break.
C4  Directional dominance: the minimum over a Voronoi-restricted subset of
    another object's pixels (what the existing directional columns report) is
    never smaller than the unrestricted pair minimum, so
    NearestDistance <= every non-NaN directional distance holds exactly.

Exit status is non-zero on any failure.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

RNG = np.random.default_rng(20260924)
N_TRIALS = 300


def _random_blob_map(shape: tuple[int, int], n_objects: int) -> np.ndarray:
    """Label map of random, possibly non-convex, possibly holed blobs."""
    objmap = np.zeros(shape, dtype=np.int32)
    next_label = 1
    for _ in range(n_objects * 4):
        if next_label > n_objects:
            break
        rr, cc = RNG.integers(0, shape[0]), RNG.integers(0, shape[1])
        blob = np.zeros(shape, dtype=bool)
        for _ in range(RNG.integers(1, 4)):  # union of discs -> non-convex
            r0 = int(np.clip(rr + RNG.integers(-4, 5), 0, shape[0] - 1))
            c0 = int(np.clip(cc + RNG.integers(-4, 5), 0, shape[1] - 1))
            rad = int(RNG.integers(1, 6))
            yy, xx = np.ogrid[:shape[0], :shape[1]]
            blob |= (yy - r0) ** 2 + (xx - c0) ** 2 <= rad ** 2
        if RNG.random() < 0.3:  # punch a hole
            yy, xx = np.ogrid[:shape[0], :shape[1]]
            blob &= ~((yy - rr) ** 2 + (xx - cc) ** 2 <= 1)
        # keep objects disjoint and non-touching-by-overwrite
        if (objmap[ndi.binary_dilation(blob)] != 0).any() or not blob.any():
            continue
        objmap[blob] = next_label
        next_label += 1
    return objmap


def _coords(objmap: np.ndarray, label: int) -> np.ndarray:
    return np.argwhere(objmap == label)


def _boundary_coords(objmap: np.ndarray, label: int) -> np.ndarray:
    mask = objmap == label
    eroded = ndi.binary_erosion(mask)  # default structure: 4-connected cross
    return np.argwhere(mask & ~eroded)


def _pair_min(a: np.ndarray, b: np.ndarray) -> float:
    return float(cKDTree(b).query(a, k=1)[0].min())


def _inclusive_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (int(coords[:, 0].min()), int(coords[:, 0].max()),
            int(coords[:, 1].min()), int(coords[:, 1].max()))


def _bbox_gap(a: tuple, b: tuple) -> float:
    gap_r = max(0, b[0] - a[1], a[0] - b[1])
    gap_c = max(0, b[2] - a[3], a[2] - b[3])
    return float(np.hypot(gap_r, gap_c))


def _branch_and_bound(label: int, labels: list[int], bboxes: dict,
                      boundaries: dict) -> tuple[int | None, float]:
    others = [lab for lab in labels if lab != label]
    order = sorted(others, key=lambda o: (_bbox_gap(bboxes[label], bboxes[o]), o))
    best_label, best = None, np.inf
    for other in order:
        if _bbox_gap(bboxes[label], bboxes[other]) > best:
            break
        d = _pair_min(boundaries[label], boundaries[other])
        if d < best or (d == best and best_label is not None and other < best_label):
            best_label, best = other, d
    return best_label, best


def _brute_force(label: int, labels: list[int], full: dict) -> tuple[int | None, float]:
    best_label, best = None, np.inf
    for other in sorted(lab for lab in labels if lab != label):
        d = _pair_min(full[label], full[other])
        if d < best:
            best_label, best = other, d
    return best_label, best


def check_all() -> list[str]:
    failures: list[str] = []
    counts = dict(c1=0, c2=0, c3=0, c4=0, ties=0)
    for trial in range(N_TRIALS):
        objmap = _random_blob_map((48, 48), int(RNG.integers(2, 9)))
        labels = [int(v) for v in np.unique(objmap) if v != 0]
        if len(labels) < 2:
            continue
        full = {lab: _coords(objmap, lab) for lab in labels}
        bnd = {lab: _boundary_coords(objmap, lab) for lab in labels}
        bboxes = {lab: _inclusive_bbox(full[lab]) for lab in labels}

        for a in labels:
            for b in labels:
                if a >= b:
                    continue
                d_full = _pair_min(full[a], full[b])
                d_bnd = _pair_min(bnd[a], bnd[b])
                counts["c1"] += 1
                if d_full != d_bnd:
                    failures.append(f"C1 trial {trial} ({a},{b}): full {d_full} != boundary {d_bnd}")
                counts["c2"] += 1
                if _bbox_gap(bboxes[a], bboxes[b]) > d_full:
                    failures.append(f"C2 trial {trial} ({a},{b}): bbox gap exceeds mask distance")

        for a in labels:
            bb = _branch_and_bound(a, labels, bboxes, bnd)
            bf = _brute_force(a, labels, full)
            counts["c3"] += 1
            if bb != bf:
                failures.append(f"C3 trial {trial} label {a}: B&B {bb} != brute {bf}")
            dists = sorted(_pair_min(full[a], full[o]) for o in labels if o != a)
            if len(dists) > 1 and dists[0] == dists[1]:
                counts["ties"] += 1

        # C4: Voronoi-restricted min over a random sub-selection is >= pair min.
        a, b = labels[0], labels[1]
        subset = full[b][RNG.random(len(full[b])) < 0.5]
        if len(subset):
            counts["c4"] += 1
            if _pair_min(full[a], subset) < _pair_min(full[a], full[b]):
                failures.append(f"C4 trial {trial}: subset min below pair min")

    # A check that cannot fail proves nothing: C3 must actually see ties.
    if counts["ties"] == 0:
        failures.append("C3 tie-break never exercised; raise N_TRIALS or density")
    print("checked:", counts)
    return failures


if __name__ == "__main__":
    problems = check_all()
    for p in problems:
        print("FAIL", p)
    print("OK" if not problems else f"{len(problems)} failure(s)")
    sys.exit(1 if problems else 0)
