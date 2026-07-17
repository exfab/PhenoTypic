# Robust outward-orientation aggregates for `MeasureOrientationZones`

**Status:** Approved and in implementation
**Date:** 2026-07-16
**Scope:** Aggregate the already-developed literal skeleton ring-crossing profile
into interpretable Overall, Dense, and Sparse zone measurements.

## Decision summary

Do not compress outward orientation into one scalar. A single number cannot
distinguish a colony that steadily turns from one that reaches the same angle by
one abrupt step or by oscillating back and forth. The proposed phenotype is a
four-part description:

1. **Sustained peak rotation**: robust unsigned magnitude in degrees.
2. **Robust net rotation**: signed inner-to-outer change in degrees.
3. **Robust rotation rate**: signed spatial slope in degrees per pixel.
4. **Trend consistency**: strength of monotonic outward turning on `[0, 1]`.

Retain the ordinary raw peak as an explicitly named diagnostic. Do not make a
95th percentile the primary robust maximum. With only 12 supported rings, the
usual interpolated 95th percentile lies between the largest two values and is
therefore still effectively a near-maximum statistic.

`RotationRateGradient`, in degrees per pixel squared, is useful for exploring
whether turning becomes stronger or weaker toward the edge. It remains a cached
diagnostic in the first implementation because splitting a short or gappy radial
profile into two slopes makes it substantially less stable than the four primary
measurements.

## Phenotype and non-goals

The target is collective outward branch turning relative to straight radial
growth from the inoculum center, without detecting or tracking individual
branches. It should be insensitive to branch count once a ring has enough
crossings to support an orientation estimate.

This change does not:

- estimate parent-to-daughter bifurcation angles;
- infer branch identity or continuity between rings;
- use branch density as a phenotype;
- hide coherence or support failures by interpolating across gaps;
- run coherence-enhancing diffusion implicitly inside the measurer; or
- call a spatial derivative an acceleration. Radius is distance, not time.

The proposed rate is `dC/dr` and the exploratory rate gradient is `d²C/dr²`.

## Input profile

The aggregate layer consumes the literal skeleton ring-crossing profile already
developed under `phenotypic.sdk_.orientation_fields`:

1. Skeletonize the complete object mask.
2. Exclude the inoculum core only after skeletonization.
3. At radii separated by `radial_ring_width`, collect literal skeleton crossings
   within the crossing band.
4. Give each crossing one vote in doubled-angle axial space.
5. Accept a ring only when its crossing count and resultant pass the existing
   support thresholds.
6. Accumulate signed axial changes only across consecutive supported rings.
   Gaps and exact 90-degree ambiguity start a new run at zero.

The radial domain begins at `core_end_radius` and extends to the first complete
ring boundary beyond the farthest detected object pixel. It is not trimmed to
the symmetric colony radius or to the sparse-zone end. This prevents the old
symmetry assumption from discarding informative outer growth.

The profile value at ring `i` is cumulative signed rotation `C_i` in degrees at
radius `r_i` in pixels. Positive and negative signs preserve opposite turning
directions. All public angular values are degrees.

## Candidate aggregates

| Candidate | Strength | Failure mode | Decision |
|---|---|---|---|
| Raw maximum `max(abs(C))` | Direct and intuitive | One bad ring can dominate | Keep as diagnostic |
| 95th percentile of `abs(C)` | Familiar robust-summary label | Near-maximal for 12 to 20 rings; ignores radial order | Do not use as primary |
| 90th percentile of `abs(C)` | Less sensitive than P95 | Still ignores persistence and radial order | Useful exploratory comparator only |
| Median of largest three values | Rejects one isolated peak | Count-dependent and can select nonadjacent rings | Superseded by sustained peak |
| Maximum rolling median | Requires a high angle to persist across neighboring rings | Depends on ring width and window | Primary magnitude metric |
| Radial median of `abs(C)` | Robust typical cumulative magnitude | Deliberately misses a localized strong-turn region | Validation comparator |
| Span-normalized area under `abs(C)` | Captures how broadly a rotation persists | An early step remains large at all later radii even without continued turning | Future persistence descriptor |
| Endpoint median difference | Robust signed inner-to-outer displacement | Can miss a high intermediate excursion | Primary signed net metric |
| Mean adjacent change per ring | Direct answer to average ring-to-ring change | One bad step affects two changes; units depend on ring spacing | Prefer robust degrees-per-pixel slope |
| Ordinary least-squares slope | Familiar rate estimate | Sensitive to a single deviant ring | Reject |
| Median pairwise slope | Robust signed rate with a simple exact definition | Requires a sufficiently long contiguous run | Primary rate metric |
| Absolute Kendall rank association | Captures coherent monotonic progression without assuming linearity | Does not measure magnitude | Primary consistency metric |
| Total absolute turning | Captures oscillation | Grows with profile length and noise | Plot diagnostic only |
| Two-half rate gradient | Tests whether rate changes outward | Unstable on short/gappy profiles | Experimental cached diagnostic |
| Robust LOESS or Savitzky-Golay derivative | Smooth local derivative profile | Adds bandwidth/window choices; weakly supported by 12 to 20 rings | Future option |

## Exact proposed metrics

Robust net rotation, rate, and consistency operate on one selected contiguous
run within the requested zone. The dominant run is the run with the greatest
radial span. Ties resolve by more supported rings, then by the smaller starting
radius. Those three metrics are missing when the dominant run has fewer than six
rings. Sustained peak instead searches every eligible three-ring window in every
contiguous run, because its purpose is to retain a localized but persistent high
turn. It is missing only when the zone contains no complete eligible window.

### 1. Sustained peak rotation

For the default three-ring window:

\[
P_\mathrm{sustained} =
\max_j \operatorname{median}(|C_{j-1}|, |C_j|, |C_{j+1}|).
\]

Every ring in a window must be consecutive, supported, and inside the requested
zone. Windows are evaluated across all contiguous runs, but never across a run
break. Thus, one isolated orientation error cannot set the phenotype. At the
default 8-pixel ring spacing, three ring centers span 16 pixels from the first to
the last sample. This is an unsigned magnitude.

### 2. Robust net rotation

For a dominant run containing `n` rings, define

\[
k = \max(2, \lceil 0.2n \rceil).
\]

Then

\[
N_\mathrm{robust} =
\operatorname{median}(C_{n-k+1:n}) -
\operatorname{median}(C_{1:k}).
\]

This signed metric reports how much rotation was generated from the inner to the
outer part of the run while reducing sensitivity to either endpoint.

### 3. Robust rotation rate

Use the median of all pairwise slopes in the dominant run:

\[
R_\mathrm{robust} =
\operatorname{median}_{i<j}
\left(\frac{C_j-C_i}{r_j-r_i}\right).
\]

Units are degrees per pixel. This is a fully specified estimator inspired by
Sen's rank-based slope estimator, not a direct port of external code. A positive
value means cumulative rotation becomes more positive outward; a negative value
means it becomes more negative outward.

### 4. Trend consistency

Compute Kendall's `tau-b` rank association between radius and cumulative
rotation, including the standard tie correction, and report its absolute value:

\[
K_\mathrm{consistent} = |\tau(r, C)|.
\]

Because radii are strictly increasing, let `P`, `Q`, and `T_C` be the numbers of
concordant, discordant, and cumulative-rotation-tied pairs. Then

\[
\tau_b = \frac{P-Q}{\sqrt{(P+Q+T_C)(P+Q)}}.
\]

The range of its absolute value is `[0, 1]`. Values near one indicate a
consistently one-directional outward trend. Values near zero indicate reversals,
oscillation, plateaus, or no ordered trend. Direction is deliberately omitted
here because `R_robust` already carries the sign. If every `C` value is tied,
consistency is defined as `0`, not missing.

### Diagnostic raw peak

\[
P_\mathrm{raw} = \max_i |C_i|.
\]

The output name and documentation must contain `RawPeak` and explicitly say that
it is outlier-sensitive. It is not a substitute for sustained peak.

### Experimental rotation-rate gradient

For runs of at least eight rings, split the dominant run into inner and outer
halves with at least four rings each. Compute a median-pairwise slope in each half:

\[
G_R = \frac{R_\mathrm{outer}-R_\mathrm{inner}}
{\operatorname{median}(r_\mathrm{outer})-
 \operatorname{median}(r_\mathrm{inner})}.
\]

Units are degrees per pixel squared. Positive means the signed rotation rate
becomes more positive toward the colony edge; negative means it becomes more
negative. This must be called a spatial rate gradient, not acceleration.

## Zone semantics

Calculate the literal profile once from the inoculum boundary through the full
detected object length. Aggregate it separately over Overall, Dense, and Sparse
radial bounds.

- **Overall:** all radii outside the inoculum core through the full object extent.
- **Dense:** ring centers inside the established dense-zone radial bounds.
- **Sparse:** ring centers inside the established sparse-zone radial bounds,
  continuing to the full detected extent when the current segmentation defines
  no earlier meaningful outer cutoff.

No window may straddle a zone boundary. Runs are reselected within each zone and
never bridged across unsupported rings.

An important interpretation distinction follows from cumulative input values:

- Sustained peak in Sparse means the cumulative rotation *reached while in the
  sparse zone*. It can include rotation accumulated in Dense.
- Robust net and robust rate in Sparse are baseline-invariant differences and
  describe rotation *generated within the sparse zone*.

This distinction belongs in each `MeasurementInfo.desc`.

## Proposed public measurement set

Add four phenotype columns for each of Overall, Dense, and Sparse:

- `OutwardRotationSustainedPeak-Mask-{Zone}` in degrees
- `OutwardRotationNet-Mask-{Zone}` in degrees
- `OutwardRotationRate-Mask-{Zone}` in degrees per pixel
- `OutwardRotationConsistency-Mask-{Zone}` on `[0, 1]`

The 12 columns above are the default output and live in
`ORIENTATION_ZONE_PRIMARY`.

When `include_diagnostics=True`, also add the following for each zone in
`ORIENTATION_ZONE_DIAGNOSTIC`:

- `OutwardRotationRawPeak-Mask-{Zone}` in degrees
- `OutwardRotationP90-Mask-{Zone}` in degrees
- `OutwardRotationP95-Mask-{Zone}` in degrees
- `OutwardRotationMedianMagnitude-Mask-{Zone}` in degrees
- `OutwardRotationAbsoluteArea-Mask-{Zone}` in degrees after span normalization
- `OutwardRotationTotalVariation-Mask-{Zone}` in degrees
- `OutwardRotationRateGradient-Mask-{Zone}` in degrees per pixel squared
- `OutwardRotationRingSupport-Mask-{Zone}` on `[0, 1]`
- `OutwardRotationRunSpanSupport-Mask-{Zone}` on `[0, 1]`
- `OutwardRotationMedianResultant-Mask-{Zone}` on `[0, 1]`

The same opt-in diagnostic enum retains the existing absolute-orientation,
radial-relative, sector-support, fixed-lag, and Dense-to-Sparse calculations as
legacy validation references. Crossing count remains internal because it is
coupled to branch density.

The two enum classes remain in the same `_orientation_zones.py` schema module.
Every `MeasurementInfo.desc` states units and interpretation; `bio_desc` remains
empty. The default output has 12 primary columns. The opt-in flag adds 69
diagnostic/reference columns at the current schema revision.

## Operation parameters

Reuse `radial_ring_width` for the literal rings. Add only:

- `outward_peak_window_rings: int = 3`, constrained to odd integers at least 3.
- `outward_min_run_rings: int = 6`, constrained to integers at least 3.
- `include_diagnostics: bool = False`. When false, emit only
  `ORIENTATION_ZONE_PRIMARY`; when true, also emit
  `ORIENTATION_ZONE_DIAGNOSTIC`.

Keep crossing-band width and ring eligibility thresholds in the orientation-field
helper's existing, documented configuration unless validation demonstrates that
users need to tune them. Do not add a CED parameter. CED remains an explicit
preprocessing operation so measurements remain reproducible and inspectable.

The existing `long_range_lag` parameter and sector-based long-range calculations
remain available as legacy/reference behavior during this change. Deprecation or
removal requires a separate compatibility decision.

### First-validation assumptions

| Setting | Proposed value |
|---|---:|
| Ring-center spacing, `radial_ring_width` | 8 px |
| Crossing half-width | 1.5 px |
| Minimum local coherence | 0.15 |
| Minimum within-crossing resultant | 0.15 |
| Minimum crossings per ring | 3 |
| Minimum ring resultant | 0.15 |
| Sustained-peak window | 3 rings |
| Minimum dominant run | 6 rings |
| Minimum run for experimental rate gradient | 8 rings, at least 4 per half |

These are starting assumptions from the developed literal-crossing prototype,
not biologically validated universal constants.

## Branch-count invariance

The metric is branch-count invariant only conditionally:

- Every accepted crossing receives one equal vote when estimating ring
  orientation.
- Every accepted ring receives one equal vote in the aggregate metrics.
- Neither crossing count nor foreground area weights the phenotype.

However, a ring with too few crossings is unsupported. Sparse and dense colonies
with the same orientation trajectory should therefore match once both pass the
same support rules, but colonies below the eligibility threshold can differ by
missingness. This is a support limitation, not evidence of a density phenotype.

## Diagnostic figures

Each diagnostic remains a separate plotting helper, following the current SDK
structure:

1. **Crossing overlay:** object image, inoculum exclusion, sampled rings, literal
   crossings, axial orientation marks with small direction arrows, and unsupported
   rings visibly distinguished.
2. **Cumulative profile:** `C(r)` in degrees, with run breaks, zone boundaries,
   sustained-peak window, endpoint median windows, and the robust slope line.
3. **Rate profile:** pairwise-slope summary and optional robust-smoothed local rate;
   the exploratory inner/outer slopes and rate gradient are annotated here only.
4. **Metric summary:** raw peak versus sustained peak, net rotation, rate,
   consistency, and support. Raw peak uses a distinct diagnostic style.

No object-label text should be drawn over the image.

## Exploratory check on the two reference colonies

These values were re-derived from the saved ring profiles generated earlier in
this analysis. They are a smoke check, not biological validation. `P90` and `P95`
are included only to show why a percentile alone is insufficient.

| Colony | Input | Rings in selected run | Raw peak (deg) | P90 (deg) | P95 (deg) | Sustained peak (deg) | Robust net (deg) | Robust rate (deg/px) | Median resultant |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R3C4 | Original | 10 | 59.75 | 44.32 | 50.72 | 47.71 | 43.99 | 0.831 | 0.397 |
| R3C4 | CED | 10 | 61.05 | 53.53 | 60.73 | 60.65 | 44.32 | 0.755 | 0.397 |
| R4C6 | Original | 12 | 82.69 | 79.94 | 81.31 | 80.18 | 75.32 | 0.870 | 0.563 |
| R4C6 | CED | 12 | 80.37 | 78.62 | 79.57 | 78.91 | 77.73 | 0.838 | 0.574 |

R3C4 demonstrates the important distinction: CED changes the largest sustained
excursion substantially while the robust net change remains similar. That may be
a real profile-shape difference or a preprocessing effect; two colonies cannot
resolve it. R4C6 is more stable across preprocessing. These observations justify
reporting magnitude, net change, rate, and consistency separately.

The table above preserves the earlier, radially truncated development profiles
for provenance. It must not be used as a golden expectation for the production
full-length implementation. In particular, the full R3C4 crop contains a second
supported run from approximately radius 142 through 238 pixels. That run spans
more radius than the earlier 46 through 118 pixel run, so it is the dominant run
under the specified selection rule. The production full-length results are a
47.72 degree sustained peak, a 59.75 degree raw peak, a -23.37 degree robust net
rotation, and a -0.426 degree-per-pixel robust rate. This is not an axial sign
flip: the sustained peak describes the strong intermediate excursion, while net
and rate describe the longer outer run's subsequent negative trend.

## Real-image regression fixture

`tests/fixtures/orientation_zones/r3c4_twok_literal_crossing.npz` is a 512 by
512 crop of the actual notebook `detect_mat` used during development, together
with the isolated TwoK object map for colony R3C4. The crop retains about 50
pixels of real background beyond the detected bounding box and records the crop
origin, source label, colony position, and SHA-256 hashes of both full cached
arrays. The fixture therefore exercises orientation-field estimation,
skeletonization, literal ring crossings, run construction, zone aggregation,
public degree conversion, and diagnostic gating on the real source layer.

## Literature rationale

The literature supports the structure of this proposal, but it does not validate
this colony-level metric directly.

- Bastien et al. model plant-organ orientation as a function of arc length and
  relate local curvature to the spatial derivative of orientation. This supports
  separating angle, spatial rotation rate, and temporal acceleration concepts:
  https://doi.org/10.1073/pnas.1214301109
- Sen defines a rank-based regression slope using pairwise slopes, motivating the
  robust rate estimator specified independently above:
  https://doi.org/10.1080/01621459.1968.10480934
- Hart et al. compare retinal-vessel tortuosity measures based on curvature,
  inflection, and high-curvature fractions and conclude that shape cannot always
  be represented well by one descriptor. This supports a metric family rather
  than one tortuosity-like score:
  https://www.siue.edu/~sumbaug/RetinalProjectPapers/Measurement%20and%20classification%20of%20retinal%20vascular%20tortuosity.pdf
- Grisan et al. partition vessels into constant-sign curvature segments before
  combining their contributions. This supports reporting trend consistency or
  reversals separately from turning magnitude:
  https://doi.org/10.1109/TMI.2007.904657
- Savitzky and Golay provide local-polynomial smoothing and differentiation, and
  Cleveland provides robust local regression. Both are plausible future local
  rate estimators, but current profiles of roughly 12 to 20 supported rings do not
  justify an additional smoothing bandwidth without broader validation:
  https://doi.org/10.1021/ac60214a047 and
  https://doi.org/10.1080/01621459.1979.10481038
- Rittaud et al. report helical and oscillatory individual-hypha growth in
  *Candida albicans*. This recent 2026 result establishes biological relevance of
  distinguishing coherent turning from oscillation, but it tracks individual
  hyphae and therefore does not directly validate a colony-level field metric:
  https://doi.org/10.1073/pnas.2526262123

## Validation plan

The implementation must include unit tests for exact formulas and integration
tests against frozen ring profiles. A standalone logic validator accompanies this
spec and imports only NumPy.

Required controls:

- linear positive and negative ramps recover their exact signed rate and
  consistency of one;
- adding a large isolated spike increases raw peak without allowing a one-ring
  event to set the sustained peak;
- gaps cannot be bridged by sustained windows, endpoint windows, or slopes;
- dominant-run selection follows radial span, then deterministic tie-breaks;
- translating every `C_i` by a constant leaves net, rate, and consistency
  unchanged;
- doubling pixel spacing halves degrees-per-pixel rate;
- reversing the sign of `C` reverses net and rate but not sustained peak or
  consistency;
- insufficient support returns missing phenotype values and finite support
  diagnostics;
- changing crossing count while holding ring orientations and eligibility fixed
  leaves every phenotype aggregate unchanged.

The standalone validator was run with Python 3.12.11 and NumPy 2.3.5.

Mutation checks must demonstrate test failure when:

- maximum replaces the rolling median;
- unsupported gaps are silently removed before aggregation;
- ordinary least squares replaces the median pairwise slope;
- signed rate is converted to absolute rate; or
- ring values are weighted by crossing count.

## Acceptance criteria

The following decisions are approved:

1. The four primary descriptors are accepted as a family rather than collapsed
   into a single score.
2. Three-ring sustained peak is accepted as the robust maximum replacement.
3. Rate gradient remains diagnostic-only until tested on a larger colony set.
4. Overall, Dense, and Sparse all receive the primary metrics.
5. Validation comparators, support values, and existing sectorized metrics are
   opt-in through `include_diagnostics=True`.
6. Primary and diagnostic columns use separate MeasurementInfo classes in the
   same schema module.
