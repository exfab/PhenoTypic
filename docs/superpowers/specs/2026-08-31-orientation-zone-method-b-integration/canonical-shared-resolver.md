# Canonical Method B shared zone resolver

Status: accepted for implementation, 2026-09-01.

## 1. Decision

PhenoTypic has one canonical branch-orientation zone algorithm. Both
`MeasureOrientationZones` and `MeasureSymZones` call the same resolver and use
the same public Method B parameters. There is no `zone_method` selector and no
second Method B implementation.

`legacy_mode=False` is the default for newly constructed operations. Setting
`legacy_mode=True` restores the pre-existing colony-ness zone partition. It
does not change the independent PELT core, symmetry, or expansion calculations.

The zone names remain `CoreZone`, `DenseZone`, and `SparseZone`:

- `CoreZone` contains the inoculum and any contiguous inner region where branch
  orientation lacks sufficient crossing support, coherence, or continuity.
- `DenseZone` lies between the first and second Method B boundaries.
- `SparseZone` lies between the second boundary and the configured radial
  percentile of the target mask.

This is an operational image-measurement definition. It does not assert that
biological orientation is absent inside CoreZone.

## 2. Public contract

Both public measurers declare these fields through one private base class:

```python
legacy_mode: bool = False
outer_zone_percentile: float = 100.0
sigma_d: float = 1.5
sigma_i: float = 4.0
radial_ring_width: float = 8.0
zone_minimum_segment: int = 4
zone_min_crossings: int = 3
zone_min_resultant: float = 0.15
zone_min_ring_coherence: float = 0.15
zone_support_weight: float = 4.0
zone_outer_support_margin: float = 0.0
zone_maximum_gap: int = 0
```

The default numeric values are configuration defaults, not biological
constants. Parameter regimes may differ by species x medium, but every crop in
one species x medium stratum must use the same values. Production code never
branches on species, medium, scene, or crop identity.

Canonical mode requires `method="distance"` and always consumes the final
`Image.objmap` target mask and `Image.detect_mat`. The historical
`intensity_source`, `tau_core`, `tau_dense`, and `tau_sparse` fields affect only
legacy zone boundaries. `n_annuli`, `pelt_penalty`, and symmetry parameters
remain active in both modes because they produce the independent morphological
measurements.

`outer_zone_percentile` is finite in `(0, 100]`. P100 is the exact maximum
finite target-mask distance from the distance-transform center. Lower values
use NumPy's linear percentile. Pixel selectors use
`np.nextafter(radius, +inf)` as their exclusive upper bound so boundary pixels
are retained. The same exact radius drives measurement and visualization.

## 3. Shared object flow

For every non-tiny object, the shared resolver performs:

```text
final target object mask
  -> distance-transform center
  -> preserved PELT/symmetry/expansion measurements
  -> one detect_mat crop
  -> one structure-tensor field
  -> one reliable object skeleton
  -> one center-origin literal-crossing transform
  -> raw zoning profile + filtered measurement profile
  -> exact Method B partition
  -> shared ZoneResolution
```

`MeasureSymZones` records the resolved radii and areas. It discards the
ephemeral orientation arrays. `MeasureOrientationZones` reuses those arrays for
literal-crossing and field measurements, then keeps only compact plot records.
No full image, tensor field, skeleton, or distance map is retained in an
operation cache.

The existing `compute_zone_segmentation()` remains the legacy primitive and the
source of `CoreRadius`, `SymmetricRadius`, `MeanExpansion`, and `MaxExpansion`.
Canonical resolution replaces only `CoreEndRadius`, `DenseEndRadius`,
`SparseEndRadius`, and their concentric areas.

## 4. Method B profile

For ring width `w` and selected outer radius `r_outer`:

\[
n=\max(1,\lceil r_{outer}/w\rceil),\qquad
r_i=(i+1/2)w.
\]

The terminal ring is clipped to the exact outer radius. Signal scaling uses
finite target-mask pixels through P100, independent of the requested outer
percentile. The P2 and P98 limits scale the signal to `[0, 1]`; non-finite input
locations remain unavailable to statistics and receive a finite median fill
only for derivatives and the tensor.

Each ring contains seven continuous features:

1. mean scaled signal;
2. scaled-signal variance;
3. target-mask occupancy;
4. mean reliable-pixel coherence;
5. radial-tilt axial resultant;
6. mean scaled-signal edge energy; and
7. literal-crossing ring resultant.

Each continuous feature is median-imputed and standardized within the object by
the larger of scaled MAD, standard deviation, and floating-point epsilon. A
Boolean support feature is appended with `zone_support_weight` and is not
standardized.

A ring is supported when its literal crossing count, ring resultant, and mean
coherence pass their configured thresholds. Interior unsupported gaps no longer
than `zone_maximum_gap` are bridged. Leading and trailing gaps are never
bridged.

## 5. Change-point objective and determinism

With feature matrix `X`, ring count `n`, and minimum segment length `m`, exact
Method B searches:

\[
m\le b_1\le n-2m,\qquad b_1+m\le b_2\le n-m
\]

and minimizes the sum of within-segment squared errors across
`X[0:b1]`, `X[b1:b2]`, and `X[b2:n]`. The first boundary must not reduce the
outer support fraction by more than the configured margin. The middle and outer
segments must each contain a supported ring.

Candidates are ordered by `(cost, b1, b2)`. Exact ties therefore select the
earliest first boundary and then the earliest second boundary. Boundaries map to
the inner edge of the selected rings and are clipped to
`0 <= CoreEnd <= DenseEnd <= Outer`.

If no exact candidate exists, one change is fitted to `1 - support`. It is
accepted only when supported rings exist after the boundary. This collapsed
solution sets `CoreEndRadius == DenseEndRadius`; `DenseArea` is zero and
dense-only orientation outputs are missing.

If the collapsed fit also fails, all public zone radii, areas, and
zone-dependent orientation outputs are missing. The valid selected outer radius
may still appear in diagnostics. Canonical failure never invokes colony-ness.

Stable diagnostic method codes are:

| Code | Meaning |
|---:|---|
| 0 | explicit legacy colony-ness |
| 1 | exact two-change Method B |
| 2 | collapsed one-change Method B |
| 4 | canonical failure with missing zone measurements |

## 6. Orientation measurement geometry

Successful exact and collapsed canonical fits use one geometry for every
orientation consumer:

```text
Overall: [CoreEndRadius, SparseEndRadius)
Dense:   [CoreEndRadius, DenseEndRadius)
Sparse:  [DenseEndRadius, SparseEndRadius)
```

This geometry applies to field aggregation, radial-relative measurements,
literal crossings, long-range rotation, bend, signed turning, cumulative and
matched overlays, quiver clipping, and displayed circles. Legacy mode retains
its historical family-specific domains.

`SymmetricRadius` remains visible as an independent mask-symmetry measurement;
it is not the canonical Method B outer boundary.

## 7. Serialization compatibility

Direct construction with no arguments selects canonical mode. Newly serialized
operations always include `"legacy_mode": false` unless the user selected
legacy mode.

Serialized `MeasureOrientationZones` and `MeasureSymZones` payloads that lack
the field predate this redesign. All operation deserialization paths run a
class-specific migration hook that inserts `legacy_mode=True` for those payloads:

- standalone `BaseOperation.from_json()`;
- pipeline measurement deserialization;
- legacy nested operation markers; and
- `OperationField` reconstruction.

This distinguishes historical reproducibility from the new-construction
default without maintaining two canonical surfaces.

## 8. Evaluation and interpretation

The supplied hand masks are subjective qualitative grades. Neurospora label 3
is normalized to label 4. Ganoderma CoreZone reference is the union of labels 1
and 3. Reference radii are cumulative radial summaries, not pixel-perfect
biological ground truth.

Evaluation must use the fixed final detector outputs already selected for each
species x medium stratum. It reports per-boundary absolute and normalized radial
drift, one-ring accuracy, availability, retained-mask fraction, and P95/P100
sensitivity. Crop-specific tuning and supervised classification are prohibited.

Because Ganoderma has eight crops from three scenes and each Neurospora cohort
has four crops from one scene, Ganoderma scene-held-out results are stronger
evidence. Neurospora leave-one-crop-out results remain sensitivity analyses.

### Executed 16-crop diagnostic

The executed scratch notebook is
`scratch/orientation_zone_method_b_p95_p100_evaluation.ipynb`. Its committed
manifest, boundary rows, aggregate scores, montage, and 16 per-crop overlays
are under
`docs/superpowers/artifacts/2026-09-01-zone-segmentation-improvement/`.
The notebook uses the established detector feature caches and the following
fixed strata: Ganoderma × glucose/yeast extract, Neurospora × menadione, and
Neurospora × xylan.

Across 48 crop-boundary comparisons, the descriptive results are:

| Method | Availability | One-ring agreement | Median absolute drift |
|---|---:|---:|---:|
| Legacy control | 100% | 31.25% | 15.49 px |
| Method B P95 | 100% | 29.17% | 13.19 px |
| Method B P100 | 100% | 14.58% | 29.13 px |

P95 retained 95.01% of detector-mask pixels on average. P100 intentionally
uses the full mask extent and therefore does not optimize agreement with a
human-drawn sparse-zone circle. In this small evaluation it had lower circular
agreement, particularly for Neurospora xylan. These values are observations
about agreement with one qualitative annotation set, not estimates of
biological accuracy. They do not override the accepted P100 default; they make
the extent-policy trade-off visible and preserve P95 as a configurable
sensitivity setting.

## 9. Acceptance tests

- Exact, collapsed, missing, gap-bridging, percentile, and tie-breaking unit
  tests pin the pure solver.
- Both measurers must return identical canonical radii for the same object and
  parameters.
- Canonical failure must not expose legacy zone boundaries.
- `legacy_mode=True` must preserve the historical measurement goldens.
- Old standalone, pipeline, legacy-nested, and `OperationField` payloads must
  migrate to legacy mode; new payloads must serialize the canonical default.
- Inspection figures must show the exact configured outer percentile and apply
  the same inner exclusion used by measurement.
- The crop notebook must execute top-to-bottom and save traceable manifests,
  compact score summaries, and per-crop overlays.
