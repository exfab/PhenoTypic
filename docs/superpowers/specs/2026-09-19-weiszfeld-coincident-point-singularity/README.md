# Weiszfeld coincident-point singularity — spec

**Date:** 2026-09-19
**Found on:** `fix/geo-median-convergence` (PR #230), by a guard written for that branch
**Status:** **Specified, not implemented.** Deferred to its own PR.
**Subject:** `weiszfeld_median` in `src/phenotypic/util/_geometric_median.py`
**Executable witness:** [`weiszfeld_singularity.py`](../../logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py)
— 7 checks, `uv run python <path>`, exits non-zero on failure, does not import `phenotypic`.

## Objective

Make `weiszfeld_median` return the geometric median even when a data point
coincides with the running estimate. Today it returns that data point.

## Non-goals

- Implementing the Cohen et al. (2016) `method='cohen'` branch. It still raises.
- Changing the solver's public signature, defaults, or return contract.
- Re-tuning `GEOMEDIAN_TOL` / `GEOMEDIAN_MAX_ITER` in `ColorCheckerProfile`;
  those are correct and are pinned by a test.

## The defect

`weiszfeld_median` floors the per-point distance before inverting it:

```python
distances = np.linalg.norm(points - x, axis=1)
distances = np.maximum(distances, 1e-10)   # <-- here
weights = 1.0 / distances
x = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
```

A point lying on `x` gets weight `1/1e-10 = 1e10`. Measured on a
3,901-pixel swatch, that single point holds **99.9995%** of the total weight
(`1.000e+10` against `4.674e+04` for the other 3,900 combined). The weighted
update is therefore that point, to within rounding; the next iteration finds
`change == 0` and returns with `converged: True` after **one** iteration.

This is the classical Weiszfeld singularity. Weiszfeld's iteration is
undefined when the estimate hits a data point, and clamping the distance does
not define it — it just picks the wrong answer quietly, with a success flag.

### Why it bites in practice

The first estimate is the **centroid** (`x = np.mean(points, axis=0)`). So the
trigger is "a data point at the centroid", and the returned value is the
**mean** — which is exactly the non-robust statistic the geometric median was
chosen to avoid. On a skewed patch the two are far apart: in the witness the
gap is **21.9 8-bit code values**, and the returned point's objective
(the sum of distances the median minimises) is **48.6% worse** than the true
median's — 540.27 against 363.55.

Nothing exotic is needed to hit it: **one** coincident pixel out of 3,901 is
enough. A colour-checker patch core is tens of thousands of 8-bit pixels
sharing a lattice of spacing `1/255`, so a pixel at the quantised centroid is
an ordinary event, not a pathological one.

Absent a coincident point the shipped rule is already correct (error
`9.4e-05` code values against the reference), so this is a singularity defect,
not a general convergence defect. That is why it survived review.

## Blast radius

| Caller | Path | Exposure |
|---|---|---|
| `ColorCheckerProfile._fit_from_rois` | `robust_color_center`, sRGB `[0,1]` | **Highest.** 8-bit quantised pixels on a coarse lattice; a centroid-coincident pixel is plausible. |
| `MeasureColor` `ColorLab_*GeoMedian` | `robust_color_center`, Lab + HSV cone | Lower. Float Lab pixels rarely land within `1e-10` of the estimate, but nothing prevents it. |

Both reach the solver through `phenotypic.util.robust_color_center`, so one
fix covers both. Note this means the fix **changes measurement output** on any
input that currently triggers the singularity, which is why it is its own PR.

## Proposed fix — Vardi & Zhang (2000) modified Weiszfeld

Split the estimate's own coincident mass out of the reweighting and take a
damped step toward the reweighted centroid of the rest:

Let `y` be the current estimate, `η` the number of points equal to `y`, and
`R` the remaining points.

```
T(y) = Σ_{a∈R} a/‖a−y‖  /  Σ_{a∈R} 1/‖a−y‖
r    = ‖ Σ_{a∈R} (a−y)/‖a−y‖ ‖
γ    = min(1, η / r)
y⁺   = (1 − γ)·T(y) + γ·y
```

With `η = 0` this reduces exactly to the current update, so the non-degenerate
path is unchanged. With `η > 0` the step is damped rather than captured, and
`r == 0` is the optimality certificate (`y` is the median; stop).

The witness implements this and confirms the result is a local minimum of the
objective in all 6 axis probe directions at `1e-7`.

### Compare the distance floor

Keep a floor only where it guards division for points that are *near* but not
*on* the estimate. The floor must not be the mechanism that handles
coincidence — that is what failed. Test coincidence explicitly (`d == 0`).

## Acceptance criteria

1. `tests/unit/correction/test_color_checker_geometric_median.py::test_patch_center_is_not_short_circuited_by_a_coincident_pixel`
   passes with its `xfail` marker **removed**. It is written and currently
   `xfail(strict=True)`, so it flips to a failure the moment the bug is fixed —
   the marker cannot be forgotten.
2. `weiszfeld_median` reports `converged: True` only when the returned point is
   a genuine fixed point, not when it has been pinned to a data point.
3. A degenerate input where *every* point is identical still returns that point.
4. `tests/unit/util/test_robust_color_stats.py` passes unchanged — in
   particular `test_robust_center_symmetric_cloud`, whose exact solution is a
   point no input occupies, and `test_robust_center_identical_points`.
5. The `MeasureColor` surface is re-run: `ColorLab_*GeoMedian` columns must be
   unchanged on inputs that never triggered the singularity.
6. The witness script still exits zero.

## References

Vardi, Y., & Zhang, C.-H. (2000). The multivariate L1-median and associated
data depth. *PNAS*, 97(4), 1423–1426. https://doi.org/10.1073/pnas.97.4.1423

Weiszfeld, E. (1937). Sur le point pour lequel la somme des distances de n
points donnés est minimum. *Tôhoku Mathematical Journal*, 43, 355–386.
