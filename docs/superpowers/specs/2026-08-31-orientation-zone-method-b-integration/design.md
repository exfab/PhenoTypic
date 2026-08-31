# Method B branch-orientation zone integration

**Date:** 2026-08-31
**Status:** Draft, selected for integration prototyping
**Scope:** The branch-orientation measurement zones produced by
`MeasureOrientationZones`. General morphological zones produced by
`MeasureSymZones` are out of scope.

## Summary

Method B partitions a radially sampled, multivariate branch-evidence profile into
three contiguous regions with exactly two change points:

1. **Branch-Orientation Unresolved**
2. **Dense resolved growth**
3. **Sparse resolved growth**

The user selected Method B as the integration direction after reviewing crop-level
overlays made with the final detector choices. Method B is a fixed mathematical
procedure; only its declared parameters may vary between species × medium strata.
Every crop in the same stratum and scene conditions must use the same parameter set.

Method B will initially be an opt-in strategy in `MeasureOrientationZones`. The
existing colony-ness segmentation remains the default and remains the only strategy
used by `MeasureSymZones`. This protects existing results while Method B is validated
on additional species, media, scenes, and phenotypes.

## Evidence and interpretation

The executed comparison artifact is:

`scratch/orientation_zone_species_media_parameter_regime.ipynb`

Its durable score tables are generated under:

`scratch/orientation_zone_final_detector_parameter_outputs/`

The hand annotations are qualitative human judgments, not physical ground truth.
Consequently, radius drift measures agreement with the supplied grading, not an
absolute biological segmentation accuracy.

Held-out-selection results favored Method B for the two *Neurospora crassa* cohorts:

| Stratum | Method B median normalized all-boundary drift | Original |
|---|---:|---:|
| *N. crassa* × menadione | 0.056 | 0.086 |
| *N. crassa* × xylan | 0.072 | 0.089 |

Method B did not win the supplied Ganoderma radius-drift evaluation: its median was
0.297, versus 0.144 for the original method and 0.071 for Option C, whose overall
boundary availability was 0.917. The crop overlays nevertheless led the user to
prefer Method B's capture. This is a real objective mismatch, not a rounding issue.
The integration must therefore remain opt-in until the acceptance criterion is made
explicit: qualitative enclosure of the usable orientation region, agreement with
hand radii, or downstream orientation-measurement stability.

The existing production PELT calculation is not Method B. PELT proposes a penalized,
unknown number of density-profile changes and is currently used only to estimate a
core candidate. Method B requires exactly two changes in a multivariate profile. In
the final-detector comparison, raw production PELT placed the menadione boundary near
the colony exterior and returned no positive candidate for the four xylan crops. It
remains a diagnostic comparator, not an implementation shortcut.

## 1. Input contract

Method B consumes the selected detector's object mask plus the enhancement signal
used in the completed evaluation:

| Stratum | Object mask | Radial feature signal |
|---|---|---|
| *N. crassa* × menadione | TwoK | TwoK `detect_mat` |
| *N. crassa* × xylan | monogenic-phase selected mask | TwoK `detect_mat` |
| Ganoderma × glucose/yeast extract | SAM2 selected instance | `max(stretched_gray, oriented_PCT)` |

This distinction is binding. In particular, substituting the monogenic-phase map for
the TwoK feature signal on xylan would implement an unevaluated variant.

The evaluation used the existing zone pipeline to establish the center before
building Method B's radial features. For xylan, this means the center came from the
legacy zone mask while occupancy and crossings used the monogenic selected mask. A
production-faithful rerun must settle this mismatch before Method B becomes a default:

- **Compatibility choice:** preserve the evaluated center contract.
- **Generalized choice:** estimate the center from the final selected mask, then rerun
  the complete evaluation because every radial feature and boundary can move.

The outer radius is the 95th percentile of selected-mask pixel distances from the
chosen center. It is Method B's sparse endpoint and caps the other two radii.

## 2. Radial feature profile

Let the Sholl-style ring width be `w`, evaluated at `w = 8 px`. Ring centers are

\[
r_i = \frac{w}{2} + iw.
\]

Each ring provides seven continuous features:

1. robust-scaled enhancement mean;
2. robust-scaled enhancement variance;
3. selected-mask occupancy;
4. mean structure-tensor coherence over reliable selected pixels;
5. axial resultant of local fiber tilt relative to the radial spoke;
6. mean enhancement-gradient magnitude over selected pixels;
7. literal skeleton-crossing axial resultant.

It also provides a Boolean orientation-support indicator. A ring is supported when all
three conditions hold:

\[
N_{cross} \ge N_{min}, \qquad
R_{cross} \ge R_{min}, \qquad
C_{ring} \ge C_{min}.
\]

Interior unsupported gaps no longer than `maximum_gap` are bridged. Edge gaps are
never bridged. All selected fitted regimes currently use `maximum_gap = 0`.

## 3. Feature normalization

Continuous features are normalized independently within each crop and feature
column. Non-finite values are replaced with the finite column median, or zero when
the entire column is non-finite. For values `x`, the normalized feature is

\[
z = \frac{x - \operatorname{median}(x)}
         {\max(1.4826\operatorname{MAD}(x),\operatorname{std}(x),\epsilon)}.
\]

The eighth feature is the Boolean support indicator converted to `{0, 1}` and
multiplied by `support_weight`. It is not standardized after weighting.

This crop-wise normalization makes the change-point objective insensitive to a
feature's additive offset and positive scalar units, subject to floating-point
rounding and the explicit support thresholds.

## 4. Exact two-change-point objective

For `n` rings, `m = minimum_segment`, and feature matrix `X`, Method B searches

\[
m \le b_1 \le n - 2m, \qquad
b_1 + m \le b_2 \le n - m.
\]

Every segment therefore contains at least `m` rings. The objective is

\[
J(b_1,b_2) =
\operatorname{SSE}(X_{0:b_1}) +
\operatorname{SSE}(X_{b_1:b_2}) +
\operatorname{SSE}(X_{b_2:n}),
\]

where each segment's sum of squared errors is measured around its own feature-wise
mean. Prefix sums and prefix sums of squares evaluate each candidate in constant time
after an `O(nd)` setup, making the complete search `O(n²d)`. With at most approximately
100 rings, the exact NumPy search is preferable to introducing solver-dependent
behavior.

The first change point is accepted only when

\[
\operatorname{mean}(support_{b_1:n}) -
\operatorname{mean}(support_{0:b_1}) \ge support\_margin.
\]

Candidates are compared as `(cost, b1, b2)`. Exact cost ties therefore select the
smallest first boundary and then the smallest second boundary. This deterministic
tie-breaking is part of the contract.

The ring-index boundaries become physical radii at the inner edge of the selected
ring:

\[
r_{unresolved} = r_{b_1} - \frac{w}{2}, \qquad
r_{dense} = r_{b_2} - \frac{w}{2}.
\]

Both radii are clipped to preserve

\[
0 \le r_{unresolved} \le r_{dense} \le r_{sparse}.
\]

## 5. No-candidate behavior

The evaluated notebook uses this fallback when no valid pair `(b1, b2)` survives:

1. Fit one change point to `1 - support` with the same minimum segment length.
2. Use that boundary for both unresolved and dense.
3. Keep the selected-mask 95th-percentile radius as sparse.
4. If even the one-change fit is impossible, unresolved and dense are missing.

Production integration should additionally expose a failure policy:

- `original`: use the original colony-ness boundaries and record the fallback.
- `missing`: emit missing orientation-zone measurements.

The recommended initial default is `original`, because it preserves batch throughput
and backward behavior. The chosen method and fallback reason must be recorded in
opt-in diagnostics so a fallback cannot be mistaken for Method B output.

## 6. Fitted evaluation regimes

These are the all-crop fitted values used in the completed comparison. They are
starting presets for validation, not universal biological constants.

| Stratum | min crossings | min resultant | min ring coherence | min segment | support weight | support margin | max gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ganoderma × glucose/yeast extract | 3 | 0.30 | 0.15 | 8 | 0.5 | 0.0 | 0 |
| *N. crassa* × menadione | 3 | 0.15 | 0.15 | 4 | 4.0 | 0.0 | 0 |
| *N. crassa* × xylan | 3 | 0.30 | 0.15 | 4 | 4.0 | 0.0 | 0 |

Parameters may vary only between species × medium strata. Core library code must not
hard-code species names. Separate pipeline configurations are sufficient when strata
run separately. A mixed-stratum run would require an explicit metadata-to-parameter
router, which is a separate design decision.

## 7. Recommended code integration

Add a private, pure helper module, provisionally
`phenotypic.measure._orientation_zone_segmentation`, containing:

```python
@dataclass(frozen=True)
class OrientationChangePointParams:
    minimum_segment: int
    min_crossings: int
    min_resultant: float
    min_ring_coherence: float
    support_weight: float
    outer_support_margin: float
    maximum_gap: int


@dataclass(frozen=True)
class OrientationZoneBoundaries:
    unresolved_radius: float
    dense_radius: float
    sparse_radius: float
    supported_fraction: float
    objective: float
    method_used: str
    fallback_reason: str | None
```

The public operation receives an opt-in strategy field:

```python
zone_method: Literal[
    "colony_ness",
    "orientation_change_points",
] = "colony_ness"
```

Its Method B parameters are annotated Pydantic fields. `radial_ring_width` is reused
for boundary sampling so segmentation and the downstream literal-crossing measurement
use the same ring grid.

The per-object flow becomes:

```text
final detector
  -> base center and tile geometry
  -> orientation field and literal crossings, computed once
  -> Method B radial feature matrix
  -> exact two-change-point boundaries
  -> Dense/Sparse orientation aggregation outside Unresolved
```

`MeasureOrientationZones._iter_object_fields` currently computes the original zone
segmentation before resolving the tile and orientation field. Refactor it to:

1. obtain the base center and compatibility geometry;
2. resolve the analysis tile;
3. compute the orientation field once;
4. invoke Method B only when selected;
5. apply the returned radii to a copy of the segmentation record;
6. pass that copy through existing aggregation and inspect paths.

Do not mutate the shared segmentation result in place. Do not change
`compute_zone_segmentation` or `MeasureSymZones` in the first implementation.

## 8. Solver decision

Retain the current explicit NumPy search for the first implementation.

The project already depends on `ruptures`. Its `Dynp` solver is an exact alternative
when the number of changes is fixed, but Method B also has a first-boundary support
constraint and explicit lexicographic tie-breaking. Replacing the evaluated search
with `Dynp` would require a behavioral-equivalence proof and provides no meaningful
performance benefit at the evaluated ring counts.

PELT is rejected as the Method B solver because it solves a different problem: the
number of changes is selected through a penalty instead of fixed at two.

## 9. Diagnostics and outputs

When orientation diagnostics are enabled, record at least:

- unresolved, dense, and sparse radii;
- requested and actually used segmentation method;
- whether a fallback occurred and its reason;
- supported-ring fraction;
- selected objective value;
- ring count and minimum segment length.

The inspect figure must show Method B boundaries and distinguish them from original
fallback boundaries. The unresolved disk is excluded from branch-orientation
aggregation. This expresses measurement resolvability, not biological absence of
orientation.

## 10. Verification requirements

1. **Independent numeric validation.** Run
   `docs/superpowers/logic_validation_scripts/2026-08-31-orientation-zone-method-b-integration/method_b_change_points.py`.
2. **Default compatibility.** With `zone_method="colony_ness"`, all existing
   `MeasureOrientationZones` outputs and serialized defaults remain unchanged.
3. **Notebook parity.** Extracted per-crop ring profiles must reproduce the notebook's
   Method B boundaries exactly, including ties and fallback behavior.
4. **Single orientation computation.** A test must prove Method B does not trigger a
   second structure-tensor or skeletonization pass.
5. **Serialization.** JSON round trips preserve every new parameter and reject invalid
   closed-set values.
6. **Minimum length.** Fewer than `3 * minimum_segment` rings must take the documented
   fallback path without an index error.
7. **Monotonic radii.** Every result satisfies
   `0 <= unresolved <= dense <= sparse` or explicitly reports missing boundaries.
8. **Signal provenance.** Detector-specific tests pin the mask and feature signal used
   for each evaluated stratum, especially the xylan mask/signal split.
9. **Species × medium consistency.** Every crop in one configured stratum receives the
   same parameter values.
10. **Regression evaluation.** Rerun all 16 supplied crops after moving the helper into
    production code. Treat any boundary drift as an implementation discrepancy until
    explained.

## 11. Deferred decisions

1. Whether one CLI run must route different Method B parameters by image metadata.
2. Whether the xylan compatibility center or a final-mask center becomes canonical.
3. Whether visual enclosure, hand-radius drift, or downstream measurement stability is
   the primary acceptance objective for Ganoderma.
4. When, if ever, Method B becomes the public default.
5. Whether another measurer needs orientation-specific boundaries strongly enough to
   justify a persisted zone-segmentation operation.
