# Weiszfeld coincident-point singularity — spec

**Date:** 2026-09-19
**Found on:** `fix/geo-median-convergence` (PR #230), by a guard written for that branch
**Status:** **Implemented** on `fix/geo-median-convergence`. Plan:
[`../../plans/2026-09-19-weiszfeld-coincident-point-singularity/plan.md`](../../plans/2026-09-19-weiszfeld-coincident-point-singularity/plan.md).
Plan review:
[`../../reports/2026-09-19-weiszfeld-coincident-point-singularity/plan-review.md`](../../reports/2026-09-19-weiszfeld-coincident-point-singularity/plan-review.md).
**Subject:** `weiszfeld_median` in `src/phenotypic/util/_geometric_median.py`
**Executable witness:** [`weiszfeld_singularity.py`](../../logic_validation_scripts/2026-09-19-weiszfeld-coincident-point-singularity/weiszfeld_singularity.py)
— 9 checks, `uv run python <path>`, exits non-zero on failure, does not import `phenotypic`.

## Objective

Make `weiszfeld_median` return the geometric median even when a data point
coincides with the running estimate. Before this change it returned that
data point.

## Non-goals

- Implementing the Cohen et al. (2016) `method='cohen'` branch. It still raises.
- Changing the solver's public signature, defaults, or return contract.
- Re-tuning `GEOMEDIAN_TOL` / `GEOMEDIAN_MAX_ITER` in `ColorCheckerProfile`;
  those are correct and are pinned by a test.

## The defect

Before this change, `weiszfeld_median` floored the per-point distance before
inverting it:

```python
distances = np.linalg.norm(points - x, axis=1)
distances = np.maximum(distances, 1e-10)   # <-- here
weights = 1.0 / distances
x = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
```

A point lying on `x` got weight `1/1e-10 = 1e10`. Measured on a
3,901-pixel swatch, that single point held **99.9995%** of the total weight
(`1.000e+10` against `4.674e+04` for the other 3,900 combined). The weighted
update was therefore that point, to within rounding; the next iteration found
`change == 0` and returned with `converged: True` after **one** iteration.

This is the classical Weiszfeld singularity. Weiszfeld's iteration is
undefined when the estimate hits a data point, and clamping the distance does
not define it — it just picks the wrong answer quietly, with a success flag.

### Why it bites in practice

The first estimate is the **centroid** (`x = np.mean(points, axis=0)`). So the
trigger is "a data point at the centroid", and the value returned was the
**mean** — which is exactly the non-robust statistic the geometric median was
chosen to avoid. On a skewed patch the two are far apart: in the witness the
gap is **21.9 8-bit code values**, and the returned point's objective
(the sum of distances the median minimises) is **48.6% worse** than the true
median's — 540.27 against 363.55.

Nothing exotic is needed to hit it: **one** coincident pixel out of 3,901 is
enough. A colour-checker patch core is tens of thousands of 8-bit pixels
sharing a lattice of spacing `1/255`, so a pixel at the quantised centroid is
an ordinary event, not a pathological one.

Absent a coincident point the old rule was already correct (error
`9.4e-05` code values against the reference), so this is a singularity defect,
not a general convergence defect. That is part of why it survived review, but
not all of it.

### The escape is driven by `eps`, not `max_iter`

The floor bound only while the estimate sat *exactly* on the pixel. After the
first step it no longer did, the floor stopped binding, and the classical
iteration would have recovered on its own — except that the post-capture step
was smaller than `eps`, so `change < eps` fired and the loop reported
`converged: True`. The stopping rule mistook post-capture stillness for
convergence.

Measured on the planted swatch at every configuration a caller actually uses,
in 8-bit code values against the converged median:

| Caller | `eps` | `max_iter` | old: iters / error | new: iters / error |
|---|---|---|---|---|
| `ColorCheckerProfile` | `1e-06` | 200 | 1 / **21.896** | 12 / 0.000 |
| `MeasureColor` | `1e-04` | 50 | 1 / **21.896** | 7 / 0.014 |
| `geometric_median` default | `1e-06` | 1000 | 1 / **21.896** | 12 / 0.000 |
| (tight probe) | `1e-09` | 5000 | 21 / 0.000 | 19 / 0.000 |

**A larger budget would not have masked the bug.** At `eps = 1e-06` the old
rule failed after one iteration even with a thousand available; the loop never
spent its budget. Only tightening `eps` to `1e-09` made the post-capture step
register as movement and let the iteration continue. Any numeric claim about
this defect is under-specified unless it names its `eps` **and** its
`max_iter`.

## Blast radius

**Likelihood and severity are separate questions here, and only likelihood
differs between the two callers.**

| Caller | Path | Likelihood of a coincident point | Severity if one occurs |
|---|---|---|---|
| `ColorCheckerProfile._fit_from_rois` | `robust_color_center`, sRGB `[0,1]` | **Highest.** 8-bit quantised pixels on a coarse lattice; a centroid-coincident pixel is plausible. | **21.896** code values of error, the full defect. |
| `MeasureColor` `ColorLab_*GeoMedian` | `robust_color_center`, Lab + HSV cone | Lower. Float Lab pixels rarely land within the coincidence radius (`1e-12 · max(‖x‖, 1)`, ~5.5e-11 at a typical L\*a\*b\* magnitude), but nothing prevents it. | **Identical: 21.896** code values. Rarer, not milder. |

Both reach the solver through `phenotypic.util.robust_color_center`, so one
fix covers both. Note this means the fix **changes measurement output** on any
input that triggered the singularity, which is why it is its own PR.

That change is not purely a correction of the captured answer. `MeasureColor`
runs at `eps = 1e-04` with `max_iter = 50`, a loose budget, so its post-fix
error on the planted swatch is `0.014` code values rather than `0.000` — a
real, small movement in `ColorLab_*GeoMedian` output on any input that
triggered the singularity, on top of the 21.896 the fix removes.

## The fix — Vardi & Zhang (2000) modified Weiszfeld

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

With `η = 0` the step reduces to the classical update, bit-identically —
except where a distance falls in the band the removed floor used to cover, for
which see *The distance floor is removed outright* below. With `η > 0` the step
is damped rather than captured, and **`γ = 1` is the optimality certificate**:
`0 ∈ ∂f(y)` iff `r ≤ η` iff `γ = 1`, which gives `y⁺ = y` and stops the loop on
its own. `r == 0` is the special case where the reweighted pull vanishes
outright.

The witness implements this and confirms the result is a local minimum of the
objective in all 6 axis probe directions at `1e-7`.

### The distance floor is removed outright

The floor is gone, not relocated. Once coincident points are split out of the
reweighting, every remaining distance is strictly greater than the coincidence
radius by construction, so `1/d` is finite for every input and there is nothing
left to guard. Re-adding a floor would reintroduce a weaker form of the
original defect.

**Test coincidence with a scale-relative radius, not `d == 0`.** An exact
comparison is not sufficient: a point `1.11e-16` from the estimate is not
equal to it, so an exact test lets it through to `1/d` with ~9e15 of weight
and it captures the solve exactly as the `1e-10` floor did — measured at
**21.896** code values of error, the same magnitude as the original defect.
The hole is not monotone in the offset (`1e-15` and `1e-13` are both fine),
which makes it a lurking hazard rather than a boundary one can reason about.
That measurement is against a **mean-initialised** solver, which is what
ships; from a median start the singularity is unreachable and the comparison
says nothing.

The shipped radius is `1e-12 · max(‖x‖, 1)`. **Read the clamp first: for the
callers here it is a flat absolute constant, not a relative one.** Below unit
norm the `max` clamps, so the radius is exactly `1e-12` at `[0,0,0]`,
`[0.5,0,0]` and the swatch base `[0.35,0.37,0.40]` alike. `ColorCheckerProfile`
works entirely in sRGB `[0,1]` — the caller this spec rates highest-likelihood —
so across that whole range the rule is not scale-relative at all. The relative
half only begins above unit norm: `1.732051e-12` at `[1,1,1]`, and ~`5.5e-11`
for L\*a\*b\* near (50, 10, 20).

Where it *is* relative, the reason is floating-point, not statistical:
`1e-12·‖x‖` is a few thousand ULPs at `x`, so the test asks whether a point is
indistinguishable from the iterate at double precision — which is exactly the
question `1/d` blows up on. That is why the relative half scales with `‖x‖`
rather than with the data's spread. The radius is harmless at every scale here:
8-bit pixels are `1/255` apart, and L\*a\*b\* nearest-neighbour spacing is far
larger than `5.5e-11`. The `max(‖x‖, 1)` clamp does give the solver a
**1e-12 absolute** resolution floor near the origin, which is now part of the
public contract of `geometric_median`.

One consequence worth stating: removing the floor means the new rule and the
old one differ not only at coincidence but anywhere a distance falls in
`(1e-12·max(‖x‖,1), 1e-10]` — floored before, unfloored now. No colorimetric
caller produces such a distance, but it is the honest scope of "unchanged".

Claims 8 and 9 of the witness re-derive both halves.

## Acceptance criteria

1. `tests/unit/correction/test_color_checker_geometric_median.py::test_patch_center_is_not_short_circuited_by_a_coincident_pixel`
   passes with its `xfail` marker **removed** — **discharged by observation,
   not by argument.** The test was written `xfail(strict=True)` precisely so
   that it would flip to a failure the moment the bug was fixed, and it did:
   immediately before the decorator was deleted the run reported
   `[XPASS(strict)]`, quoting the marker's own reason text back — *"strict=True,
   so this flips to a failure the moment the fix lands and the marker cannot be
   left behind."* The marker was removed in `2560479b`; the file now passes
   4/4, with no xfail and no xpass. That `[XPASS(strict)]` state cannot be
   reproduced once the decorator is gone, so it was captured deliberately and
   is quoted verbatim in
   [`../../reports/2026-09-19-weiszfeld-coincident-point-singularity/cluster-a-report.md`](../../reports/2026-09-19-weiszfeld-coincident-point-singularity/cluster-a-report.md).
2. `weiszfeld_median` is no longer able to report `converged: True` while
   *pinned to* a data point: a coincident point is excluded from the
   reweighting, so it cannot dominate the update. Where the answer genuinely
   is a data point, `γ = 1` is an exact optimality certificate —
   `0 ∈ ∂f(x)` iff `r ≤ η` iff `γ = 1`.

   The stopping rule's guarantee in general is `r ≤ η + eps·W`, where
   `W = Σ_{a∈R} 1/‖a−x‖`, **not** `r ≤ η`. Damping shortens the step by `η/W`,
   so the loop can stop at `γ < 1`; measured on the planted swatch,
   `η/W = 1.87e-06` against `eps = 1e-06`, so this is reachable rather than
   theoretical. That slack is `eps`-scaled and is the same slack the classical
   rule has carried at `η = 0` since 1937 — this change neither introduces nor
   removes it. Pinned by
   `tests/unit/util/test_geometric_median.py::test_converged_carries_the_subgradient_certificate`,
   which asserts the true bound against the objective's own subgradient rather
   than trusting the solver's stopping test.
3. A degenerate input where *every* point is identical still returns that
   point. Pinned by
   `tests/unit/util/test_geometric_median.py::test_every_point_identical_returns_that_point`,
   which pins the **contract**, not the branch its name points at: the explicit
   all-coincident guard is redundant with the `r == 0` return. **Measured, on a
   solver transcription carrying no `errstate` suppression** — on five copies of
   `[7, 7, 7]`, with the guard and with it deleted:

   | | returned | `iterations` | `converged` | warnings |
   |---|---|---|---|---|
   | guard present (as shipped) | `[7. 7. 7.]` | 1 | `True` | none |
   | guard deleted | `[7. 7. 7.]` | 1 | `True` | `RuntimeWarning` |

   An empty reweighting gives `r = 0`, so the `r == 0` return fires instead and
   produces the same answer by a different route. **The `0/0` `RuntimeWarning`
   is the single observable difference**, and nothing in the suite would fail on
   it either — `pyproject.toml`'s `filterwarnings` has no `error` entry. The
   guard is kept for that warning, for legibility, and to avoid a `nan`
   intermediate — so this criterion is not what would catch its removal. A
   mutation run confirms the same boundary independently: with the suppression
   removed, the `no_far_guard` mutant dies (`1 failed, 31 passed, 3 warnings`),
   where under `errstate(invalid="ignore")` it had survived.
4. `tests/unit/util/test_robust_color_stats.py` passes unchanged — in
   particular `test_robust_center_symmetric_cloud`, whose exact solution is a
   point no input occupies, and `test_robust_center_identical_points`.
5. `ColorLab_*GeoMedian` columns are unchanged on inputs that never triggered
   the singularity. **Argued, not tested — a carried risk, not a closed
   check.** There is no numeric assertion on any `ColorLab_*GeoMedian` value
   anywhere in the repo and no golden corpus:
   `tests/unit/measure/test_measure_color.py` checks header presence, hex
   format, ΔE non-negativity, opt-in XYZ/xy, a serialization round-trip and a
   numeric reduction — none of which pins a geometric-median value. The claim
   rests on two things instead: the `η == 0` branch being arithmetically
   identical to the old update outside the `(1e-12·max(‖x‖,1), 1e-10]` band
   (*The distance floor is removed outright*; witness claim 9 leg A shows the
   two updates bit-identical on a clean cloud), and the affected surface
   running green (101 passed). It does **not** rest on a before/after numeric
   comparison, because none was run. Note also criterion 2's counterpart
   observation: on inputs that *did* trigger the singularity the columns move,
   and at `MeasureColor`'s own `eps = 1e-04` / `max_iter = 50` they land
   `0.014` code values off the converged median rather than on it.
6. The witness script still exits zero — 9 checks, 9 passing.

## References

Vardi, Y., & Zhang, C.-H. (2000). The multivariate L1-median and associated
data depth. *PNAS*, 97(4), 1423–1426. https://doi.org/10.1073/pnas.97.4.1423

Weiszfeld, E. (1937). Sur le point pour lequel la somme des distances de n
points donnés est minimum. *Tôhoku Mathematical Journal*, 43, 355–386.
