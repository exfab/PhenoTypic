# Canonical Method B shared zone resolver

Status: implemented with corrective revalidation, 2026-09-01.

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
center_detector: OperationField | None = ImagePipeline(
    ops=[
        SetDetectMode(mode="gray"),
        InoculumDetector(
            min_diameter=20.0,
            max_diameter=140.0,
            thresh_method="otsu",
            enable_gmm=True,
            gmm_n_components=2,
            gmm_separation_threshold=0.9,
            validate_obj_count=True,
        ),
    ]
)
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

`center_detector` is an `ObjectDetector`, `ImagePipeline`, or `None`. Its
default is the deterministic grayscale-reset `InoculumDetector` pipeline shown
above, selected by the 16-crop scene-derived center campaign. Canonical
measurement applies it once to an image copy. Each positive center component
is assigned to the final colony
with which it has the greatest pixel overlap; ties use the lowest numeric
label. If several components map to one colony, greatest overlap wins with the
same label tie-break. The centroid of the winning component's pixels inside
the final colony is the authoritative center for morphology, Method B,
orientation measurements, and figures. A configured detector that has no
overlapping center for an object produces canonical failure code 4 with
`failure_reason="center_not_found"`; it never silently falls back to the final
mask. `legacy_mode=True` ignores this field.

Explicit `center_detector=None` preserves the V1 fallback behavior: canonical mode uses
the final-mask distance-transform center selected by `method="distance"`.
This fallback exists for compatibility and does not claim that the final
branch mask is the best inoculum-center support mask.

The default numeric values are configuration defaults, not biological
constants. Parameter regimes may differ by species x medium, but every crop in
one species x medium stratum must use the same values. Production code never
branches on species, medium, scene, or crop identity.

Canonical mode requires `method="distance"` for the fallback estimator and
always consumes the final `Image.objmap` target mask and `Image.detect_mat`.
When configured, `center_detector` supplies only the radial origin; it does not
replace the target mask or Method B feature signal. The historical
`intensity_source`, `tau_core`, `tau_dense`, and `tau_sparse` fields affect only
legacy zone boundaries. `n_annuli`, `pelt_penalty`, and symmetry parameters
remain active in both modes because they produce the independent morphological
measurements.

`outer_zone_percentile` is finite in `(0, 100]`. P100 is the exact maximum
finite target-mask distance from the selected radial origin. Lower values use
NumPy's linear percentile. Pixel selectors use
`np.nextafter(radius, +inf)` as their exclusive upper bound so boundary pixels
are retained. The same exact radius drives measurement and visualization.

## 3. Shared object flow

For every non-tiny object, the shared resolver performs:

```text
final target object mask
  -> configured compact center detector, or final-mask EDT fallback
  -> shared mask-only PELT/symmetry/expansion primitive
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

One private morphology primitive is the source of `CoreRadius`,
`SymmetricRadius`, `MeanExpansion`, and `MaxExpansion`. In canonical mode it
uses the detector-selected center when configured, otherwise the historical
distance-transform fallback. Canonical mode passes that geometry directly to
Method B and never calculates colony-ness profiles or thresholds. The existing
`compute_zone_segmentation()` extends the morphology result only for explicit
legacy mode.

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
| 4 | canonical failure with missing zone measurements, including a requested center that was not found |

## 6. Orientation measurement geometry

Successful exact and collapsed canonical fits use one geometry for every
orientation consumer:

```text
Overall: [CoreEndRadius, nextafter(SparseEndRadius, +inf))
Dense:   [CoreEndRadius, DenseEndRadius)
Sparse:  [DenseEndRadius, nextafter(SparseEndRadius, +inf))
```

Overall uses `nextafter(SparseEndRadius, +inf)` as its exclusive upper bound as
well. Thus the internal Core/Dense and Dense/Sparse boundaries remain half-open,
while a pixel exactly on the global P95/P100 outer circle is included. Field,
literal-crossing, long-range, quiver, and inspection-figure selectors apply the
same rule.

This geometry applies to field aggregation, radial-relative measurements,
literal crossings, long-range rotation, bend, signed turning, cumulative and
matched overlays, quiver clipping, and displayed circles. Legacy mode retains
its historical family-specific domains.

`SymmetricRadius` remains visible as an independent mask-symmetry measurement;
it is not the canonical Method B outer boundary.

## 7. Serialization compatibility

Direct construction with no arguments selects canonical mode and the default
grayscale-reset inoculum-center pipeline. Newly serialized operations always
include `"legacy_mode": false` unless the user selected legacy mode, and
include the complete default center pipeline unless the user explicitly chose
`"center_detector": null`. A configured detector round-trips through the
shared `OperationField` class-tagged representation.

Serialized `MeasureOrientationZones` and `MeasureSymZones` payloads that lack
`legacy_mode` predate the Method B redesign. The class-specific migration hook
inserts `legacy_mode=True`. Payloads that lack `center_detector` predate the
new center default, so the same hook inserts `center_detector=None` and retains
their original final-mask EDT behavior. This applies through every operation
deserialization path:

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

### Corrected executed 16-crop diagnostic

The earlier scores from
`scratch/orientation_zone_method_b_p95_p100_evaluation.ipynb` are superseded.
That historical notebook used the pre-correction tensor input, reported finite
boundaries as zone availability, reused cached centers, and did not carry enough
provenance to prove the detector mask, signal, and center for every crop.

The replacement is the fully executed
`scratch/orientation_zone_method_b_corrected_evaluation.ipynb`. Its committed
manifest, boundary rows, orientation-availability rows, selected regimes,
cross-validation rows, montage, and 16 per-crop overlays are under
`docs/superpowers/artifacts/2026-09-01-zone-segmentation-improvement/`. Every
center is recomputed as the deterministic EDT argmax of the final detector mask.
Both production measurers are run for every reported method, and summary values
are asserted against the boundary-level rows.

The exact detector/signal provenance is:

- Neurospora menadione: current TwoK mask and TwoK `detect_mat`.
- Neurospora xylan: monogenic-branch TwoK final mask and the TwoK `detect_mat`
  used by the cache producer. The EDT center is recomputed from the final
  monogenic mask rather than the legacy TwoK mask.
- Ganoderma: SAM2 mask and feature signal from
  `max(stretched_gray, oriented_PCT)` without background subtraction. The SAM2
  proposal selection used human label 1, so these rows isolate zone segmentation
  only and cannot support end-to-end detector-performance claims.

The predeclared P100 grid selected one regime per species × medium by failures,
collapsed fits, median normalized CoreZone drift, median normalized all-boundary
drift, then deterministic parameter JSON. The selected regime was applied
unchanged at P95. Across 48 crop-boundary comparisons per method:

| Method | Boundary availability | Zone-geometry availability | One-ring agreement | Median absolute drift | Collapsed crops |
|---|---:|---:|---:|---:|---:|
| Legacy control | 100% | 87.50% | 22.92% | 21.97 px | n/a |
| Corrected fixed P95 | 100% | 93.75% | 31.25% | 25.09 px | 3 |
| Corrected fixed P100 | 100% | 93.75% | 14.58% | 29.03 px | 3 |
| Retuned P95 | 100% | 100% | 33.33% | 14.11 px | 0 |
| Retuned P100 | 100% | 100% | 18.75% | 26.79 px | 0 |

P95 retained 95.01% of final detector-mask pixels on average. The corrected
notebook reports finite primary orientation metrics separately for Overall,
Dense, and Sparse; a collapsed DenseZone is unavailable rather than counted as
100% available. Ganoderma leave-one-scene-out validation produced no missing
fits and two collapsed held-out fits. Neurospora leave-one-crop-out values are
reported only as single-scene sensitivity analyses.

These results describe agreement with one qualitative annotation set, not
biological accuracy. P100 remains the full-mask default; P95 is an explicit
extent sensitivity setting.

## 9. Acceptance tests

- Exact, collapsed, missing, gap-bridging, percentile, and tie-breaking unit
  tests pin the pure solver.
- Positive affine scale-invariance tests cover very small, very large, and
  offset signals after the finite-filled P2/P98 tensor correction.
- Exact outer-circle pixels are included while internal boundaries remain
  half-open.
- Both measurers must return identical canonical radii for the same object and
  parameters.
- Canonical tiny objects emit missing morphology, zone, and orientation values
  with method code 4; legacy mode retains its historical zero-valued zones.
- Canonical failure must not expose legacy zone boundaries.
- Canonical mode must not execute the legacy colony-ness extension.
- `legacy_mode=True` must preserve the historical measurement goldens.
- The pre-simplification migration golden at
  `tests/unit/measure/_golden/orientation_zones_pre_simplification.json` freezes
  complete `MeasureSymZones` and diagnostic `MeasureOrientationZones` tables,
  selected centers, solver states, configured-detector serialization, and old
  payload migration for exact, collapsed, missing, tiny, detector-center, and
  legacy cases. Regenerate it only through the opt-in capture test in
  `test_orientation_zone_migration_golden.py` before intentional behavior
  changes. Its initial SHA-256 is
  `7f67797ea72d2e9c54b1352d06db55c5f3a6eff260819141e89e899f2b1dd073`.
- Old standalone, pipeline, legacy-nested, and `OperationField` payloads must
  migrate to legacy mode; new payloads must serialize the canonical default.
- Inspection figures must show the exact configured outer percentile and apply
  the same inner exclusion used by measurement.
- The crop notebook must execute top-to-bottom and save traceable manifests,
  compact score summaries, and per-crop overlays.
