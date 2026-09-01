# Method B branch-orientation zone integration

**Date:** 2026-08-31, revised 2026-09-01

**Status:** Proposed; implementation-ready after evidence Gate 0

**Target branch:** `codex/zone-segmentation-improvement`, stacked on
`ome-zarr-merged`

**Scope:** Orientation-specific radial zones used by
`MeasureOrientationZones`. The morphological zones produced by
`MeasureSymZones` are explicitly out of scope.

## 1. Decision summary

The implementation will add an opt-in `orientation_change_points` zoning method
to `MeasureOrientationZones`. This is the previously evaluated Method B: an exact
two-change-point partition of a center-origin, multivariate Sholl-ring profile.
It divides the object into three contiguous measurement regions:

1. **Branch-Orientation Unresolved**
2. **Dense candidate orientation zone**
3. **Sparse candidate orientation zone**

The unresolved region is operational, not biological. It is any region where
branch orientation cannot be estimated with sufficient crossing support,
coherence, or continuity. It may contain the inoculum, indistinguishable
core-dense growth, overlapping hyphae, or locally blurred or saturated growth.
No claim is made that biological orientation is absent.

The public extent parameter is:

```python
outer_zone_percentile: float = 100.0
```

It is valid in `(0, 100]`. `100.0` means the exact furthest selected-mask pixel
and is the user-selected default. A value of `95.0` reproduces the outer-radius
policy used by the historical evaluation, subject to the intentional production
changes recorded in Section 14. The active percentile controls the Method B
profile window, the sparse outer boundary, all Method B orientation selectors,
and the displayed outer circle. It is ignored by the legacy `colony_ness` method.

`zone_method="colony_ness"` remains the operation default. Therefore adding the
new fields does not change existing numerical results unless Method B is selected.
Old JSON configurations that omit the fields continue to load. Newly serialized
operations will contain the new defaulted fields, so byte-identical JSON output is
not promised.

## 2. User decisions and constraints

The following decisions are binding:

- Method B is the selected integration direction after crop-overlay review.
- The unresolved category includes the inoculum core and any other region whose
  branch orientation is not resolvable.
- Ganoderma does not require biological core and dense growth to be separated.
  A combined unresolved region is acceptable if usable orientation measurements
  can still be collected outside it.
- Parameters may vary only by species x medium. Images acquired under the same
  scene conditions in the same stratum must receive the same parameter values.
- The library must implement one mathematical method. It must not contain
  species-name branches or crop-specific parameter selection.
- Hand labels are subjective qualitative references. Circular boundaries should
  approximately enclose the judged zones; pixel-perfect mask agreement is not the
  target.
- Supervised zone classification is out of scope.
- Detection quality assurance remains in the detector evaluation workflow, not
  in the orientation-zone notebook or operation.
- `outer_zone_percentile` is configurable and must be visible in the inspection
  figure. Full detected extent, `100.0`, is the default.

## 3. Scope and non-goals

### 3.1 In scope

- A private, pure Method B segmentation module.
- Public Pydantic configuration on `MeasureOrientationZones`.
- One center-origin orientation and literal-crossing analysis per object.
- Exclusion of the unresolved region from every Method B orientation metric.
- Percentile-controlled Method B outer extent.
- Numeric diagnostics and compact categorical provenance.
- Inspection overlays that show the exact boundaries used for measurement.
- Unit, regression, serialization, figure, and all-crop evaluation tests.

### 3.2 Out of scope

- Changes to `compute_zone_segmentation` or `MeasureSymZones`.
- Automatic detector choice by species or medium.
- A metadata router for mixed-stratum runs.
- Training a supervised zone classifier.
- Replacing the upstream detector or SAM2 instance-selection strategy.
- Making Method B the default zoning method.
- Persisting a new shared morphological-zone object for other measurers.

## 4. Current production architecture

The existing shared morphology path is in
`src/phenotypic/measure/_zone_segmentation.py`:

- `ZoneSegmentationParams` begins at line 81.
- `compute_zone_segmentation` begins at line 145.
- the current density PELT core call is at lines 247-250;
- colony-ness is calculated at lines 376-383;
- threshold radii are selected at lines 387-392; and
- the mutable `ZoneSegmentation` result is returned at lines 404-435.

`MeasureSymZones` delegates directly to this helper at
`src/phenotypic/measure/_measure_symzones.py:239-278`. Changing the shared helper
would therefore change a different public measurement operation and violate this
specification.

`MeasureOrientationZones` currently:

1. computes the legacy zone segmentation;
2. resolves a grid section or expanded tile;
3. computes the structure-tensor field; and
4. separately skeletonizes the mask when primary literal-crossing metrics are
   filled.

The main integration seam is
`src/phenotypic/measure/_measure_orientation_zones.py:1176-1198`.
The existing literal-crossing construction is at lines 1275-1366. Current ring
centers start outside the already estimated core. Method B instead requires
center-origin rings before its unresolved boundary is known.

The current metric families do not all use the same radial domain:

- primary literal-crossing metrics start at `core_end_radius` and continue to the
  furthest detected mask radius;
- legacy field diagnostics define Overall from zero to `symmetric_radius`; and
- long-range metrics use `min(sparse_end_radius, symmetric_radius)`.

The Method B path will use one internally consistent domain. The default legacy
path will retain its current behavior exactly.

## 5. Evidence base and epistemic limits

### 5.1 Evaluation data

The historical evaluation contains 16 supplied crops:

- 8 Ganoderma crops from 3 scenes;
- 4 *N. crassa* x menadione crops from 1 scene; and
- 4 *N. crassa* x xylan crops from 1 scene.

Ganoderma validation held out complete scenes. Each Neurospora cohort contains
only one scene, so its leave-one-crop-out result is a sensitivity analysis, not
independent scene validation.

The final fitted score table contains 288 rows: 16 crops x 6 methods x 3
boundaries. It has no duplicate crop-method-boundary keys and no missing fitted
boundaries. The cross-validation table contains 240 rows for five candidate
methods, with two missing boundaries.

### 5.2 Human-reference transformation

Source label masks are not modified. References are derived as follows:

- Neurospora label `3` is remapped to label `4` for scoring.
- Ganoderma unresolved is the union of labels `1` and `3`.
- Dense is the cumulative unresolved region plus label `2`.
- Sparse is the cumulative dense region plus label `4`.
- Each reference boundary is the 95th percentile of the cumulative label-mask
  distances from the algorithm-estimated center.

These radii quantify agreement with a qualitative human grading. They are not
physical ground truth, and drift from them is not automatically evidence of a
biologically inferior boundary.

### 5.3 Historical held-out results

All values in this table use the historical 8 px ring width and Method B outer
extent of 95 percent.

| Stratum | Method | Median normalized drift | Mean normalized drift | One-ring accuracy | Availability |
|---|---|---:|---:|---:|---:|
| Ganoderma x glucose/yeast extract | Method B | 0.297405 | 0.309005 | 0.125000 | 1.000000 |
| Ganoderma x glucose/yeast extract | Original | 0.144328 | 0.248102 | 0.208333 | 1.000000 |
| *N. crassa* x menadione | Method B | 0.056222 | 0.052854 | 0.750000 | 1.000000 |
| *N. crassa* x menadione | Original | 0.085574 | 0.090481 | 0.500000 | 1.000000 |
| *N. crassa* x xylan | Method B | 0.072268 | 0.098196 | 0.333333 | 1.000000 |
| *N. crassa* x xylan | Original | 0.088517 | 0.105318 | 0.333333 | 1.000000 |

One-ring accuracy means
`abs(predicted_radius - reference_radius) <= 8 px`.

Normalized drift and availability are:

\[
drift_{normalized}=
\frac{|r_{predicted}-r_{reference}|}{r_{reference,sparse}},
\qquad
availability=\frac{N_{finite\ predicted\ radii}}{N_{expected\ radii}}.
\]

Ganoderma Option C had a lower held-out median drift of `0.071228`, mean drift
of `0.084648`, one-ring accuracy of `0.166667`, and availability of `0.916667`.
The user's visual preference for Method B therefore conflicts with the supplied
radius-drift objective. Method B remains opt-in. No claim that it is numerically
superior for Ganoderma is permitted from the current evidence.

### 5.4 Evidence provenance

The executed notebook and its full crop-level outputs remain under ignored
`scratch/` and are not reproducible from a clean clone. On the originating host,
the exact paths are:

```text
Data root:
/Users/alex/Library/CloudStorage/GoogleDrive-anguy344@ucr.edu/My Drive/Active Projects/rbeck/BranchZoneMask/Images/HyphaeAnalysis

Notebook:
/Users/alex/Projects/PhenoTypic/scratch/orientation_zone_species_media_parameter_regime.ipynb

Output directory:
/Users/alex/Projects/PhenoTypic/scratch/orientation_zone_final_detector_parameter_outputs

Feature cache:
/Users/alex/Projects/PhenoTypic/scratch/orientation_zone_final_detector_feature_cache
```

Their verified SHA-256 identifiers are:

| Artifact | SHA-256 |
|---|---|
| `scratch/orientation_zone_species_media_parameter_regime.ipynb` | `64e5550f8ea36952e025d01a29897ce5855777f6ee2d534ea46e257c4de5c1c6` |
| `selected_parameters.csv` | `dc883682eacb60611e6048a8b1c6c0498a4a5dc955cd47cd2f46cc349d237892` |
| `species_media_parameter_regime_scores.csv` | `e1ea982b8509f7e6c8395f89a2ec5f1fd8a0dbbfeb670e7f5ae68025d409b122` |
| `cross_validation_scores.csv` | `d24392ba57aed6d30f4bc4f520ff233ebdf01257ba8b9fb5b0e357336fb75279` |
| `original_vs_b_vs_pelt_crop_boundaries.csv` | `73a815ccfb8c2a95c970e0be38c9524f16a9dff24d6a984ea082def1984b6ee8` |

Before a production accuracy claim is merged, the implementation session must
commit a compact, auditable crop manifest and the new P100 evaluation summary.
Raw microscopy crops and model weights need not be committed.

The manifest must contain at least these columns:

```text
dataset,group,scene,crop,image_path,label_path,image_sha256,label_sha256,
detector_mask_source,feature_signal_source,center_source
```

Verify the existing artifacts without rewriting them:

```bash
shasum -a 256 \
  /Users/alex/Projects/PhenoTypic/scratch/orientation_zone_species_media_parameter_regime.ipynb \
  /Users/alex/Projects/PhenoTypic/scratch/orientation_zone_final_detector_parameter_outputs/selected_parameters.csv \
  /Users/alex/Projects/PhenoTypic/scratch/orientation_zone_final_detector_parameter_outputs/species_media_parameter_regime_scores.csv \
  /Users/alex/Projects/PhenoTypic/scratch/orientation_zone_final_detector_parameter_outputs/cross_validation_scores.csv \
  /Users/alex/Projects/PhenoTypic/scratch/orientation_zone_final_detector_parameter_outputs/original_vs_b_vs_pelt_crop_boundaries.csv
```

To rerun the notebook on the same host, first copy it to a new evaluation
notebook rather than overwriting the historical artifact, then use:

```bash
uv run jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output orientation_zone_method_b_p95_p100_evaluation.executed.ipynb \
  /Users/alex/Projects/PhenoTypic/scratch/orientation_zone_method_b_p95_p100_evaluation.ipynb
```

Write new tabular outputs beneath
`/Users/alex/Projects/PhenoTypic/scratch/orientation_zone_method_b_p95_p100_evaluation_outputs/`
and commit only the manifest, compact score summary, and inspection figures.

The durable detector decision artifacts are under
`docs/superpowers/artifacts/2026-08-28-hyphae-detection/`. The earlier Method B
specification and independent numeric validator were introduced by commits
`e835cedeb`, `b3077fa82`, and `d8cc5b981` on this branch lineage.

## 6. Detector and signal contract

Method B consumes the final `Image.objmap` object mask and the configured
orientation signal. It does not choose or run a species-specific detector.

The historical evaluation used:

| Stratum | Evaluated object mask | Evaluated radial feature signal |
|---|---|---|
| *N. crassa* x menadione | current TwoK | current TwoK `detect_mat` |
| *N. crassa* x xylan | monogenic-phase selected mask | TwoK `detect_mat` |
| Ganoderma x glucose/yeast extract | selected SAM2 instance | `max(stretched_gray, oriented_PCT)` without background subtraction |

Production configurations must arrange for the selected signal to be present in
`detect_mat` and must set `intensity_source="detect_mat"`. The model-level
validator must reject Method B with another intensity source. The operation
cannot infer species, medium, or detector provenance.

The evaluated Ganoderma cache selected a SAM2 instance using overlap with human
label `1`. That rule is not deployable without annotation. Annotation-free SAM2
instance selection is an upstream detector prerequisite for end-to-end deployment,
but it does not block implementation or testing of the zone algorithm against the
already selected masks.

## 7. Center and outer-radius contract

### 7.1 Center

The production Method B center is the center returned by the existing base
segmentation for the final `Image.objmap` object. Method B requires
`method="distance"`: the center is the row-major first maximum of the Euclidean
distance transform. The current intensity center is gray-intensity weighted and
has not been evaluated for this use. A model-level validator must reject
`zone_method="orientation_change_points"` with any other center method.

This resolves a historical mismatch. The xylan notebook used the monogenic mask
for occupancy and crossings but a legacy TwoK zone mask for the center. The
current production `Image` exposes one final object map, so the mixed-mask center
is not representable without a second-mask API. This specification chooses the
final-mask center and requires a full rerun.

### 7.2 Configured outer radius

Let `D` be the finite distances of selected-mask pixels from the center. The
configured Method B outer radius is:

\[
r_{outer}(p) =
\begin{cases}
\max(D), & p=100, \\
\operatorname{percentile}_{linear}(D,p), & 0<p<100.
\end{cases}
\]

The explicit `p=100` branch guarantees exact full extent. The displayed radius
is never rounded to a ring edge. Pixel selectors use
`np.nextafter(r_outer, +inf)` as their exclusive upper bound so a pixel exactly
on the full-extent boundary is included.

`outer_zone_percentile` applies only when
`zone_method="orientation_change_points"`. Changing the percentile changes the
profile window and can therefore move the unresolved and dense change points. It
does more than truncate the sparse zone.

## 8. Center-origin ring profile

For ring width `w = radial_ring_width`, define:

\[
n = \max(1, \lceil r_{outer}/w \rceil), \qquad
r_i = (i + 1/2)w, \quad i=0,\ldots,n-1.
\]

The historical evaluation used `w = 8 px`. Each raster annulus contains pixels
whose distances satisfy `abs(distance - r_i) <= w/2`. The production selector is
also clipped to the exact configured outer radius. Therefore the last annulus may
be partial when the selected percentile is not a ring edge.

The literal skeleton-ring transform uses a radial half-width of `1.5 px`. Method
B's raw zoning profile must request `minimum_points=1` and
`minimum_resultant=0.0`, because `zone_min_crossings` permits one or two and the
zoning support thresholds must receive unfiltered evidence. The existing
downstream measurement profile retains `minimum_points=3`, the reliable-pixel
coherence floor of `0.15`, and the individual-crossing resultant floor of
`0.15`. Both profiles reuse one prepared transform and skeleton.

`sigma_d`, `sigma_i`, and `radial_ring_width` remain the existing operation
parameters. The historical evaluation fixed them at `1.5`, `4.0`, and `8.0`
across all strata. If they are tuned later, the species x medium consistency rule
also applies to them.

## 9. Two-stage normalization and ring features

### 9.1 Signal scaling

Before ring statistics, the selected tile signal is robustly scaled. Compute
`P2` and `P98` over finite target-object mask pixels through the full selected-mask
extent, P100, independent of `outer_zone_percentile`. Adjacent objects,
background, grid boundaries, and fallback tile size do not enter this scaling
population. Then scale the full tile with those fixed limits so edge derivatives
remain available; ring features still sample only the target object. This choice
makes the input scaling invariant to the requested outer percentile:

\[
s = \operatorname{clip}\left(
\frac{I-P_2(I)}{P_{98}(I)-P_2(I)}, 0, 1
\right).
\]

Keep a separate finite-source validity mask. Non-finite source positions remain
unavailable: they do not enter ring intensity, edge-energy, coherence, or
radial-tilt statistics and are handled later by feature-column imputation. For
derivatives and the structure tensor only, replace non-finite source positions
with the median finite target-object value after the corresponding scaling. This
finite fill is computational support, not measured signal. If no finite
target-object pixels exist or the percentile range is nonpositive, `s` is zero
and the signal-dependent features are unavailable.

Edge energy is

\[
E = \sqrt{(\partial_x s)^2 + (\partial_y s)^2}.
\]

### 9.2 Continuous features

Each ring supplies seven continuous values:

1. mean scaled signal over selected-mask ring pixels;
2. scaled-signal variance over selected-mask ring pixels;
3. selected-mask occupancy, object pixels divided by geometric ring pixels;
4. mean structure-tensor coherence over selected pixels whose coherence is at
   least `0.15`;
5. radial-tilt resultant
   `abs(mean(exp(2j * radial_relative_tilt)))` over those reliable pixels;
6. mean edge energy over selected-mask ring pixels; and
7. the literal-crossing ring-level axial resultant.

The fifth feature measures concentration of radial-relative tilt. It does not
measure preference for radial growth. Perfectly radial and perfectly tangential
fibers both yield a resultant of one. Production code and tests should call this
feature `radial_tilt_resultant`, not `radial_alignment`.

### 9.3 Feature-column standardization

Continuous features are normalized independently within each object and column.
Non-finite values are replaced with the finite column median, or zero if the
entire column is non-finite. For imputed values `x`:

\[
z = \frac{x - \operatorname{median}(x)}
         {\max(1.4826\operatorname{MAD}(x),\operatorname{std}(x),\epsilon)}.
\]

This normalization is separate from the earlier image-signal scaling.

## 10. Orientation support

A ring has raw support when all three conditions hold:

\[
N_{cross} \ge N_{min}, \qquad
R_{ring} \ge R_{min}, \qquad
C_{ring} \ge C_{min}.
\]

Here `N_cross` is the literal crossing count, `R_ring` is the ring-level crossing
resultant, and `C_ring` is the reliable-pixel ring coherence mean.

The configurable `zone_min_resultant` is a ring-support threshold. It does not
change the fixed `0.15` individual-crossing filter. Similarly,
`zone_min_ring_coherence` is applied after reliable pixels have already been
filtered at `0.15`.

Interior unsupported runs no longer than `zone_maximum_gap` are bridged. Leading
and trailing gaps are never bridged. The Boolean support feature is converted to
zero or one, multiplied by `zone_support_weight`, and appended to the seven
standardized features without further standardization.

## 11. Exact two-change-point objective

Let `X` be the eight-column feature matrix, `n` the ring count, and `m` the
minimum segment length. Method B searches:

\[
m \le b_1 \le n-2m, \qquad
b_1+m \le b_2 \le n-m.
\]

Every segment contains at least `m` rings. The objective is:

\[
J(b_1,b_2) =
\operatorname{SSE}(X_{0:b_1}) +
\operatorname{SSE}(X_{b_1:b_2}) +
\operatorname{SSE}(X_{b_2:n}),
\]

where SSE is summed across features around each segment's feature-wise mean.
Prefix sums and prefix squared sums make each candidate constant time after an
`O(nd)` setup. Complete search is `O(n^2 d)`. Evaluated crops contain sufficiently
few rings that this exact NumPy search is preferable to solver-dependent behavior.

The first boundary must also satisfy:

\[
\operatorname{mean}(support_{b_1:n}) -
\operatorname{mean}(support_{0:b_1}) \ge
zone\_outer\_support\_margin.
\]

A margin of zero still rejects a split whose outer support fraction is lower
than its inner support fraction.

Every accepted candidate must also contain at least one supported ring in both
the middle segment `support[b1:b2]` and the outer segment `support[b2:n]`. This
prevents an all-unresolved profile from being labeled as two resolved zones.
These are candidate measurement segments, not proof that the whole segment is
resolved. Before any orientation family emits a value, that family must pass its
existing family-specific evidence requirements within the selected zone. For
literal outward metrics, this includes the existing
`outward_min_run_rings` contiguous-run requirement. Field, bend, turning,
matched, cumulative, and quiver outputs retain their existing finite-pixel,
coherence, and sample-count requirements, but evaluate them only inside the
Method B geometry. A failed family-level gate emits `NaN`; it does not relabel
the boundary or borrow evidence from another zone.

Candidates are ordered as `(cost, b1, b2)`. Exact ties therefore choose the
earliest first boundary and then the earliest second boundary. This ordering is
part of the reproducibility contract.

The change points map to inner ring edges:

\[
r_{unresolved}=r_{b_1}-w/2, \qquad
r_{dense}=r_{b_2}-w/2.
\]

Final radii are clipped to preserve:

\[
0 \le r_{unresolved} \le r_{dense} \le r_{outer}.
\]

## 12. No-candidate and failure behavior

The evaluated degraded path is retained:

1. If no valid two-change solution exists, fit one change to `1 - support` with
   the same minimum segment length.
2. Accept it only if at least one supported ring occurs after the boundary.
3. Set unresolved and dense to that one boundary.
4. Keep the configured outer radius as sparse.
5. Mark the used method as collapsed one-change Method B.

A collapsed dense interval emits missing Dense measurements, but Overall and
Sparse may still be measured outside the unresolved region.

If the one-change fit is also impossible, apply:

```python
zone_failure_policy: Literal["colony_ness", "missing"] = "colony_ness"
```

- `colony_ness` uses the original unresolved and dense boundary estimates,
  clipped as `unresolved=clip(unresolved, 0, outer)` and
  `dense=clip(dense, unresolved, outer)`, but
  retains the configured Method B outer extent and Method B measurement-domain
  semantics. It is a boundary fallback, not a claim that Method B succeeded.
- `missing` emits missing values for all orientation-zone measurements on that
  object.

If Method B fails, the failure policy is `colony_ness`, and the legacy
segmentation has `zones_computed=False`, the result is missing, with used-method
code 4. A successful exact or collapsed Method B result is valid regardless of
the legacy `zones_computed` value.

The fallback must never be silent. Stable provenance tokens are:

```text
method_requested:
  colony_ness
  orientation_change_points

method_used:
  colony_ness
  orientation_change_points_exact
  orientation_change_points_collapsed
  colony_ness_boundary_fallback
  missing

fallback_reason:
  none
  insufficient_rings_for_two_change
  no_valid_two_change_candidate
  no_resolved_support_after_collapsed_boundary
  insufficient_rings_for_one_change
  invalid_outer_extent
  legacy_zones_unavailable_after_method_b_failure
  missing_requested_by_policy
```

Failure reasons are appended to an ordered `failure_trace` as each stage fails:
outer extent, two-change search, one-change search, and terminal policy fallback.
The public `fallback_reason` is the last token in that trace, so the terminal
cause wins deterministically; the compact cache retains the complete trace so
earlier causes are not lost. When `zone_failure_policy="missing"` terminates an
otherwise valid-extent attempt, append the stable token
`missing_requested_by_policy`. Exact success has an empty trace and public
reason `none`.

Exact Method B reports its finite three-segment objective. Collapsed Method B
reports the finite one-change objective. Boundary fallback and missing output
report `NaN` for the objective.

## 13. Public operation configuration

Add the following fields to `MeasureOrientationZones`, before plot-only fields:

```python
zone_method: Literal[
    "colony_ness",
    "orientation_change_points",
] = "colony_ness"
outer_zone_percentile: float = 100.0
zone_failure_policy: Literal["colony_ness", "missing"] = "colony_ness"
zone_minimum_segment: int = 4
zone_min_crossings: int = 3
zone_min_resultant: float = 0.15
zone_min_ring_coherence: float = 0.15
zone_support_weight: float = 4.0
zone_outer_support_margin: float = 0.0
zone_maximum_gap: int = 0
```

The class docstring `Args:` section must define these fields in the same order so
JSON schema descriptions and GUI forms remain correct.

Validation is exact:

- `outer_zone_percentile` is a finite non-Boolean float in `(0, 100]`;
- `zone_minimum_segment` and `zone_min_crossings` are non-Boolean integers at
  least one;
- `zone_min_resultant` and `zone_min_ring_coherence` are finite in `[0.15, 1]`;
- `zone_support_weight` is finite and nonnegative;
- `zone_outer_support_margin` is finite in `[0, 1]`;
- `zone_maximum_gap` is a non-Boolean integer at least zero; and
- invalid `Literal` values are rejected by Pydantic.

A model validator enforces `method="distance"` and
`intensity_source="detect_mat"` whenever
`zone_method="orientation_change_points"`.

The default numeric Method B values are configuration defaults, not universal
biological constants. The operation does not select fitted presets automatically.

## 14. Historical P95 presets and intentional production deviations

The fitted historical P95 regimes were:

| Stratum | crossings | resultant | coherence | min segment | support weight | support margin | max gap | outer percentile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ganoderma x glucose/yeast extract | 3 | 0.30 | 0.15 | 8 | 0.5 | 0.0 | 0 | 95 |
| *N. crassa* x menadione | 3 | 0.15 | 0.15 | 4 | 4.0 | 0.0 | 0 | 95 |
| *N. crassa* x xylan | 3 | 0.30 | 0.15 | 4 | 4.0 | 0.0 | 0 | 95 |

These are starting configurations, not validated constants for new scenes.

The production implementation intentionally differs from the historical notebook
in six ways:

1. it uses the final selected mask for the center, rather than the mixed legacy
   xylan center;
2. it clips the terminal feature annulus to the exact selected percentile, while
   the notebook allowed the final complete annulus to extend slightly beyond P95;
3. it defaults `outer_zone_percentile` to P100 rather than P95.
4. it computes signal-scaling percentiles from finite target-object pixels at
   P100, while the historical notebook used the full crop.
5. it builds an unfiltered zoning ring profile with `minimum_points=1` and
   `minimum_resultant=0.0`, while the historical notebook used three points and
   `0.15`; and
6. it requires at least one supported ring in each candidate middle and outer
   segment, which narrows the historical feasible candidate set.

Consequently, this is a production revision of Method B. Historical crop
boundaries and presets are evidence for the algorithmic direction, not golden
outputs or validated parameters for the revised formulation. The direct solver
retains the historical SSE objective, but the production candidate set and input
features differ. All parameters must be refit and all 16 crops re-evaluated.

## 15. Private module architecture

Add `src/phenotypic/measure/_orientation_zone_segmentation.py` containing pure
array logic and frozen dataclasses. It must not import `Image`, detector classes,
species metadata, plotting code, or `MeasureSymZones`.

The provisional data contracts are:

```python
@dataclass(frozen=True)
class OrientationChangePointParams:
    ring_width: float
    outer_zone_percentile: float
    minimum_segment: int
    min_crossings: int
    min_resultant: float
    min_ring_coherence: float
    support_weight: float
    outer_support_margin: float
    maximum_gap: int


@dataclass(frozen=True)
class OrientationRadialProfile:
    radii: NDArray[np.float64]
    continuous_features: NDArray[np.float64]
    crossing_count: NDArray[np.int64]
    crossing_resultant: NDArray[np.float64]
    ring_coherence: NDArray[np.float64]
    raw_support: NDArray[np.bool_]
    bridged_support: NDArray[np.bool_]


@dataclass(frozen=True)
class OrientationZoneResult:
    unresolved_radius: float
    dense_radius: float
    outer_radius: float
    full_extent_radius: float
    requested_percentile: float
    retained_mask_fraction: float
    supported_fraction: float
    objective: float
    ring_count: int
    method_used: str
    fallback_reason: str
    failure_trace: tuple[str, ...]
```

Use small testable functions for:

- selected-mask outer percentile;
- center-origin ring construction;
- signal scaling and edge energy;
- ring feature reduction;
- support construction and interior-gap bridging;
- feature imputation and standardization;
- prefix-SSE calculation;
- exact two-change search;
- collapsed one-change fallback; and
- monotonic radius clipping.

## 16. Orientation measurement geometry

Do not mutate the shared `ZoneSegmentation` record and do not overload its
`symmetric_radius` with orientation-only meaning. Introduce a private immutable
`OrientationMeasurementGeometry` adapter used only inside
`MeasureOrientationZones`.

For successful or collapsed Method B, every orientation family uses:

```text
Overall: [unresolved, outer)
Dense:   [unresolved, dense)
Sparse:  [dense, outer)
```

This excludes the unresolved disk from all Method B branch-orientation values.
At P100, `outer` is the exact full selected-mask extent. At lower percentiles,
all metric selectors stop at the displayed percentile circle.

For `zone_method="colony_ness"`, the adapter reproduces the current family-specific
bounds exactly, including the current legacy Overall and long-range behavior.
This separation is required for default numerical compatibility.

## 17. Single-compute object flow

The Method B per-object flow is:

```text
final objmap and configured signal
  -> base center and tile geometry
  -> one structure-tensor orientation field
  -> one reliable object skeleton
  -> center-origin literal crossings and ring features
  -> exact Method B boundaries
  -> immutable orientation measurement geometry
  -> existing orientation aggregators using the same crossing evidence
```

Refactor the existing literal crossing implementation so skeleton preparation and
ring sampling are separable. The public existing wrapper must retain its behavior.
Method B samples the center-origin grid once and reuses that profile for its exact
solution and downstream literal metrics. Colony-ness boundary fallback, method
code 3, also uses this center-origin ring phase and the Method B geometry. It does
not resample the legacy core-origin grid. Do not run `skeletonize` a second time.

The operation must compute `orientation_field` once per object during one
`measure()` call. A later independent `inspect()` call may recompute the object
analysis because full-resolution arrays are intentionally not cached.

The geometry adapter must be threaded through every radial consumer, not only
the primary metrics. This includes field aggregation, long-range coherence,
bend overlays, signed-turning metrics, cumulative and matched overlays, quiver
clipping, and boundary circles. Each consumer must apply both its geometry's
inner exclusion and outer limit. The current quiver path has no inner exclusion;
Method B must add one without changing legacy behavior.

## 18. Diagnostics and public schema

When `include_diagnostics=True`, append the following exact public schema entries.
The left column is the enum name; the next two columns are the unprefixed label
and emitted header.

| Enum | Label | Header |
|---|---|---|
| `ZONE_SEGMENTATION_METHOD_CODE` | `ZoneSegmentationMethodCode` | `OrientZones_ZoneSegmentationMethodCode` |
| `UNRESOLVED_RADIUS` | `UnresolvedRadius` | `OrientZones_UnresolvedRadius` |
| `DENSE_RADIUS` | `DenseRadius` | `OrientZones_DenseRadius` |
| `OUTER_RADIUS` | `OuterRadius` | `OrientZones_OuterRadius` |
| `FULL_EXTENT_RADIUS` | `FullExtentRadius` | `OrientZones_FullExtentRadius` |
| `OUTER_ZONE_PERCENTILE` | `OuterZonePercentile` | `OrientZones_OuterZonePercentile` |
| `OUTER_ZONE_RETAINED_MASK_FRACTION` | `OuterZoneRetainedMaskFraction` | `OrientZones_OuterZoneRetainedMaskFraction` |
| `ZONE_SUPPORTED_RING_FRACTION` | `ZoneSupportedRingFraction` | `OrientZones_ZoneSupportedRingFraction` |
| `ZONE_CHANGE_POINT_OBJECTIVE` | `ZoneChangePointObjective` | `OrientZones_ZoneChangePointObjective` |
| `ZONE_CHANGE_POINT_RING_COUNT` | `ZoneChangePointRingCount` | `OrientZones_ZoneChangePointRingCount` |
| `ZONE_CHANGE_POINT_MINIMUM_SEGMENT` | `ZoneChangePointMinimumSegment` | `OrientZones_ZoneChangePointMinimumSegment` |

The stable segmentation-method code is:

| Code | Meaning |
|---:|---|
| 0 | requested and used legacy colony-ness |
| 1 | exact two-change Method B |
| 2 | collapsed one-change Method B |
| 3 | colony-ness boundary fallback after Method B failure |
| 4 | Method B failure with missing measurements |

Diagnostic finiteness is part of the contract. `F` means finite, `N` means
`NaN`, and `F/N` is split by the stated reason. Supported-ring fraction always
uses the gap-bridged support profile.

| Code and state | U radius | D radius | outer/full radius | percentile/retained | bridged support/ring count/min segment | objective |
|---|---:|---:|---:|---:|---:|---:|
| 0 legacy | N | N | N | N | N | N |
| 1 exact | F | F | F | F | F | F, three-segment SSE |
| 2 collapsed | F, equals D | F, equals U | F | F | F | F, one-change SSE |
| 3 boundary fallback | F, clipped | F, clipped | F | F | F | N |
| 4 missing after valid Method B extent/profile | N | N | F | F | F | N |
| 4 `invalid_outer_extent` | N | N | N | N | N except configured minimum segment is F | N |

The compact figure cache also stores `method_requested`, `method_used`, and the
categorical `fallback_reason`. Full-resolution masks, orientation fields,
skeletons, and segmentation objects must not be retained in the operation cache.

In legacy mode, the method code is zero and every other new Method-B diagnostic
is `NaN`. Compatibility for `include_diagnostics=True` means all pre-existing
columns retain identical values and order, with the new columns appended. Since
`outer_zone_percentile` is ignored in legacy mode, changing it cannot alter a
pre-existing value or replace the required new `NaN` values.

New `MeasurementInfo` entries must author only `label` and technical `desc`.
`bio_desc` remains empty and `image` remains unset.

Every affected schema entry that hardcodes legacy radial domains such as
`0..symmetric`, `core_end..dense`, or "full detected object extent" must be
revised. Its technical description must state the conditional Method B domain
from Section 16 and the unchanged legacy domain. Default colony-ness output
meaning remains unchanged.

## 19. Visualization contract

The inspection figure is a measurement audit, not decoration. It must:

- draw the exact unresolved, dense, and configured outer radii used by selectors;
- label the outer circle `Sparse measurement limit (full extent, P100)` when
  `outer_zone_percentile=100`;
- otherwise label it `Sparse measurement limit (P<value>)`;
- draw a thin dotted `Detected full extent (excluded outer tail)` reference circle
  when the configured percentile is below 100;
- show the requested percentile, selected radius, full radius, retained fraction,
  used method, and fallback state in hover or summary text;
- distinguish colony-ness fallback boundaries from exact Method B boundaries; and
- preserve the same rings in static `for_save=True` output.

The decision figure and its reproducible source are:

- `docs/superpowers/artifacts/2026-09-01-zone-segmentation-improvement/extent-policy-comparison.png`
- `docs/superpowers/artifacts/2026-09-01-zone-segmentation-improvement/make_extent_policy_comparison.py`

The figure uses an illustrative xylan extreme case chosen to make the extent
difference visible; it is not presented as representative. It recomputes the
final-mask distance-transform center and independently refits Method B for P95
and P100. It displays crossing counts and demonstrates that the two settings can
differ in the included distal mask tail, crossing evidence, and fitted inner
boundaries. The dotted magenta circle in the P95 panel is the detected P100 full
extent, not a hand reference.

The pinned external figure inputs are:

| Input | SHA-256 |
|---|---|
| TIFF crop | `fae2af401d518b1d2a321089e2bacb205f8363d44e525464eb9a4a5d1853520d` |
| label PNG | `fbacf8b5cabdc63baa853462775d2d78a69af7ef54f2864ae71491bba5f72d83` |
| feature-cache NPZ | `d360542a39f82c967413861ec0d371e9fc62a514e26aec43bacbd476f47b534b` |

The source verifies these hashes before rendering. Reproduce it from this
worktree with:

```bash
uv run python \
  docs/superpowers/artifacts/2026-09-01-zone-segmentation-improvement/make_extent_policy_comparison.py
```

## 20. PELT and solver decision

The existing PELT implementation in
`src/phenotypic/measure/_zone_segmentation.py:477-505` is not Method B. It uses a
penalty to select an unknown number of density-profile changes, then uses only the
first change as a core candidate. Method B fixes exactly two changes in a
multivariate profile and adds an outward-support constraint and deterministic tie
ordering.

In the historical comparison, PELT returned no positive candidate for all four
xylan crops. Menadione PELT candidates were approximately `83.38-121.05 px`, while
the hand unresolved radii were approximately `31.30-38.85 px`. PELT remains a
diagnostic comparator, not an implementation shortcut.

The project dependency on `ruptures` does not justify replacing the explicit
NumPy search. Any future solver substitution requires a behavioral-equivalence
proof covering the support constraint, exact ties, every output boundary, and
fallback behavior.

## 21. Testing requirements

### 21.1 Pure helper tests

Create `tests/unit/measure/test_orientation_zone_segmentation.py` covering:

- P100 equals the exact maximum finite selected-mask distance;
- lower percentiles use NumPy linear percentile behavior;
- Boolean percentiles are rejected;
- invalid, empty, and all-nonfinite distance inputs;
- deterministic center-origin ring centers and partial final rings;
- exact clipping of feature and measurement selectors at the selected radius;
- exact-boundary inclusion using `np.nextafter` and retained-mask fraction;
- robust signal scaling and constant-signal behavior;
- feature median imputation and all-missing columns;
- positive-scale and additive-offset invariance;
- radial-tilt resultant semantics for radial and tangential fields;
- support thresholds at, below, and above each boundary;
- bridging interior gaps but not edge gaps;
- prefix-SSE equality with direct brute-force SSE;
- exact two-change recovery on synthetic profiles;
- deterministic lexicographic ties;
- support-margin rejection;
- rejection when the middle or outer candidate segment has no supported ring;
- fewer than `3 * minimum_segment` rings;
- collapsed one-change behavior;
- fewer than `2 * minimum_segment` rings; and
- monotonic clipped radii.

### 21.2 Operation integration tests

Extend `tests/unit/measure/test_measure_orientation_zones.py` to prove:

- old JSON without new fields loads;
- every new field round-trips through JSON;
- invalid closed sets and numeric values are rejected;
- Method B rejects center methods other than `distance`;
- `outer_zone_percentile` rejects Boolean values;
- `zone_method="colony_ness"` produces the same DataFrame and figures as before;
- changing `outer_zone_percentile` has no effect in colony-ness mode;
- successful Method B computes one orientation field and one skeleton per object;
- successful Method B remains valid when legacy `zones_computed=False`;
- P100 includes the furthest selected-mask distance;
- a lower percentile clips every Method B metric family consistently;
- Method B Overall excludes unresolved pixels;
- collapsed dense produces missing Dense values without losing valid Sparse values;
- colony-ness fallback is marked and does not masquerade as Method B;
- missing failure policy emits missing orientation metrics;
- diagnostic codes and radii match the actual path;
- no full-resolution array appears in the compact cache; and
- inspect and static figures draw the exact selector radii and dynamic labels.

Add focused inner-and-outer clipping assertions for field aggregation,
long-range metrics, bend overlays, signed turning, cumulative and matched
overlays, quiver traces, and boundary circles.

### 21.3 Shared-regression tests

- `MeasureSymZones` regression fixtures remain unchanged.
- Existing orientation primary columns remain unchanged in default mode.
- With diagnostics enabled, every pre-existing column retains its value and
  position, and the new diagnostics are appended.
- Existing schema ordering still leaves `quiver_block` after all measurement
  fields.
- README measurement-schema generation includes the new technical diagnostics.

### 21.4 Evaluation tests

Rerun all 16 supplied crops with one fixed final-mask distance-transform center
per crop. Use that same center for every algorithm and human-reference radius in
the new comparison. Report displacement from both the human label-1 centroid and
the historical algorithm center, and include a center-sensitivity analysis. The
new drift values are not directly comparable to the historical table because
the reference radii move with the center.

The evaluation has two distinct percentile questions:

1. Paired sensitivity: run the historical P95 parameter regimes at P95 and P100
   without retuning. This isolates the effect of changing extent.
2. Achievable P100 performance: tune P100-specific parameters within the same
   species x medium held-out folds, applying one parameter set to every image in
   a stratum and scene condition.
3. Control: run the original colony-ness segmentation with the same fixed center.

P100 is sensitive to a remote selected-mask pixel or thin distal filament. The
retained-mask fraction and full-versus-selected radius must accompany every P95
and P100 score so that this sensitivity is visible.

Produce crop overlays with hand-reference colors and a compact score table. Do
not reuse historical P95 accuracy numbers for P100. Report scene-held-out
Ganoderma results separately from single-scene Neurospora sensitivity results.

## 22. Independent numeric validation

The independent script is:

`docs/superpowers/logic_validation_scripts/2026-08-31-orientation-zone-method-b-integration/method_b_change_points.py`

It must continue to import only the standard library and NumPy, never
`phenotypic`. It independently validates:

- exact two-change recovery;
- prefix-SSE equality with direct brute-force SSE;
- deterministic ties;
- the outward-support constraint;
- required support in the candidate middle and outer segments;
- interior-gap bridging;
- P2/P98 scaling from finite full-object pixels independent of the active extent,
  plus imputation and robust standardization;
- P100 exact full extent, P95 linear percentile behavior, Boolean rejection,
  exact-boundary inclusion, and retained-mask fraction;
- center-origin ring construction and partial terminal annulus clipping; and
- the collapsed one-change objective and outer-support requirement.

## 23. Implementation sequence

1. Gate 0: commit the manifest from Section 5, verify historical hashes, and
   create a separate reproducible P95/P100 evaluation notebook.
2. Add the pure private Method B module and helper tests.
3. Refactor literal skeleton preparation so Method B can reuse one skeleton.
4. Add and validate the public operation fields.
5. Add the immutable orientation measurement geometry adapter.
6. Integrate the opt-in Method B branch without changing the legacy branch.
7. Add diagnostics and update technical schema descriptions.
8. Update inspect and static figure traces.
9. Run targeted helper, operation, schema, and figure tests.
10. Rerun the 16-crop fixed-parameter sensitivity comparison and P100-specific
    held-out tuning described in Section 21.4.
11. Review Ganoderma overlays and downstream orientation stability before any
    proposal to change the public `zone_method` default.

## 24. Acceptance gates

The implementation is ready to merge only when:

1. the independent numeric validator passes;
2. default colony-ness measurements and `MeasureSymZones` remain unchanged;
3. P100 and lower-percentile selector radii equal their displayed circles;
4. Method B performs one orientation-field and one skeletonization pass per object;
5. all new parameters serialize and validate correctly;
6. all radial metric and overlay consumers apply the geometry's inner and outer
   bounds;
7. all 16 crops have new P95 and P100 overlays and score summaries;
8. the P100 results are explicitly identified as new evidence, not inherited P95
   performance;
9. a human review confirms that the selected boundaries enclose the intended
   usable orientation regions; and
10. Ganoderma remains opt-in unless a later evaluation establishes an acceptance
   objective and supports changing the default.

No numeric biological acceptance threshold is invented here. The current labels
are qualitative and the existing Ganoderma visual and radius-drift objectives
disagree.

## 25. Deferred work and known dependencies

- A metadata-to-parameter router if mixed species x medium strata must run in one
  pipeline invocation.
- Annotation-free SAM2 instance selection for Ganoderma.
- Independent scenes for both Neurospora media.
- A formal Ganoderma acceptance objective combining visual enclosure, hand-radius
  agreement, and downstream orientation-measurement stability.
- A decision on whether Method B should ever become the default zoning method.
- Persisted orientation-specific geometry for other measurement operations.

## 26. Pickup checklist for another session

Start in:

```text
/Users/alex/Projects/PhenoTypic/.worktrees/zone-segmentation-improvement
```

Read, in order:

1. this specification;
2. the independent numeric validator;
3. `src/phenotypic/measure/_zone_segmentation.py`;
4. `src/phenotypic/measure/_measure_orientation_zones.py`;
5. `src/phenotypic/sdk_/orientation_fields/_literal_crossings.py`;
6. `src/phenotypic/sdk_/orientation_fields/_aggregates.py`;
7. `src/phenotypic/schema/_orientation_zones.py`;
8. `tests/unit/measure/test_measure_orientation_zones.py`; and
9. `tests/fixtures/orientation_zones/README.md`.

Before implementing, verify the ignored notebook and CSV hashes in Section 5 if
those files are present. Do not silently regenerate or overwrite them. Use the
public parameter name `outer_zone_percentile` everywhere. P100 is the configured
default, but P95 is the setting associated with the historical evaluation.

Run the independent validator with:

```bash
uv run python \
  docs/superpowers/logic_validation_scripts/2026-08-31-orientation-zone-method-b-integration/method_b_change_points.py
```

The first implementation change should be the private pure helper and its direct
tests. Do not begin by editing the shared `compute_zone_segmentation` function.
