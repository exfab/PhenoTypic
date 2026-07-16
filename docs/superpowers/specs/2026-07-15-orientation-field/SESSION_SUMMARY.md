# Orientation-field analysis session summary

- **Project:** PhenoTypic
- **Purpose:** Develop an interpretable, center-aware phenotype for colony-wide
  hyphal rotation without individually detecting branches.
- **Branch:** `branch-reconnection`
- **Status:** Experimental. The initial orientation-field implementation and analysis
  bundle are committed, but the later skeleton-crossing, CED, and parameter-sweep work
  remains uncommitted. No final public measurement has been selected.
- **Date:** 2026-07-15
- **Primary continuation point:** Literal skeleton-ring crossings with an equal-crossing
  population consensus. CED remains an optional preprocessing experiment, not a settled
  default.

## 1. Original request

> Please review the `inspect()` method for
> [_measure_symmetric_zones.py](src/phenotypic/measure/_measure_symmetric_zones.py) and
> [_measure_orientation_zones.py](src/phenotypic/measure/_measure_orientation_zones.py) .
> Please hide the object labels from them, and review the figure for orientation zones.
> I'm confused how to read it, and what does the figure say about the orientation field
> and whats being calculated? Is there a better figure we can make to visualize the
> calculation that helps users understand what's happening?

The task expanded into designing and testing an outward-rotation phenotype on two real
colonies from the referenced Neurospora workflow.

## 2. Decision and steering log

This table reconstructs the important human steering decisions. It is not a verbatim
transcript.

| Order | Decision or correction | Why it mattered |
|---:|---|---|
| 1 | Remove object labels from both inspection figures and replace the orientation view with calculation-oriented overlays. | Object IDs obscured the image and did not explain what the orientation field measured. |
| 2 | Test every proposal on the real notebook image and emphasize two colonies with visibly different branch organization. | Synthetic examples alone could hide failures caused by dotted intensity, segmentation, tangency, and dense inoculum structure. |
| 3 | Define the phenotype relative to the inoculum center rather than the image coordinate system. | A straight branch at 0 degrees and a straight branch at 90 degrees must both read as straight radial growth rather than apparent rotation. |
| 4 | Exclude the inferred inoculum core from orientation evidence while retaining the center as the geometric anchor. | The core is dense and would dominate a dense-zone average even though the phenotype concerns outward branches. |
| 5 | Report angles in degrees, retain raw peak amplitude as a separately marked diagnostic, and avoid making branch density the phenotype. | Degrees are immediately interpretable; the raw peak is useful for review but should not be confused with a normalized colony statistic. |
| 6 | Use signed diverging color rather than mean absolute direction. The final shared diagnostic palette was Spectral over the full cumulative range. | Signed color distinguishes coherent clockwise and counterclockwise turning and avoids hiding cancellation. |
| 7 | Evaluate longer spatial scales, including sigma 16, 32, and 48, and test a 45-degree spatial arc gate. | Large Cartesian kernels can mix unrelated orientation families; an angular gate was proposed to limit that mixing. |
| 8 | Explore Sholl-inspired outward accumulation and matched-ring continuation. | Concentric sampling is naturally centered on the inoculum and can summarize how orientation changes with radius. |
| 9 | Compare strict, gap-bridged, restart, and gap-plus-restart continuity rules, including tangential alternatives. | Missing lattice points could represent absent evidence, an overly strict rule, or true ambiguity. These cases should not be silently conflated. |
| 10 | Preserve full colony length for ring mean/median compounding instead of stopping at a symmetric-growth boundary. | The remaining outer area contained substantial orientation information relevant to the phenotype. |
| 11 | Skeletonize the object and then move to literal skeleton-ring crossings. | Crossings sample the branch centerlines at known radii without requiring explicit biological branch tracing. |
| 12 | Compare reciprocal one-to-one, independent many-to-one, and global one-to-one inheritance. | Reciprocal matching is conservative, many-to-one can represent bifurcation-like expansion, and global assignment tests whether local choices are misleading. |
| 13 | Prefer the branch-tracking-free ring population trend over parent-child matching as the current conceptual direction. | If a rotational phenotype affects many branches, the distribution of literal crossings can reveal the shared trend without committing to individual branch identity. |
| 14 | Apply coherence-enhancing diffusion and then sweep its parameters. | Dotted branch intensity appeared to rotate local axes spuriously, but a single CED setting could not establish that smoothing was properly tuned. |

## 3. What was built and changed

### Committed implementation

Commit `c050b468b` (`feat(measure): preserve orientation field analysis`) contains the
first production implementation and the initial analysis bundle. It:

- removes Plotly object-label annotations from both symmetric-zone and orientation-zone
  inspection figures;
- adds radial-relative tilt, outward turning, long-range Sholl-style comparison, and
  nematic bend calculations;
- adds cumulative and matched-ring diagnostic views;
- updates the orientation-zone schema and unit tests;
- saves the notebook-derived composite, object maps, scripts, validation programs, and
  real-image figures in this spec folder.

Commit `7375e0ec5` (`docs(measure): checkpoint full-length orientation analysis`) is the
later committed checkpoint. It preserves full-length equal-sector mean/median
compounding, axial-median change diagnostics, colormap comparisons, the R3C4 ring-gate
diagnostic, and independent full-length validation scripts.

### Uncommitted experiments after the checkpoint

The current worktree additionally contains:

- a skeleton-masked ring-median comparison;
- a point-level literal skeleton-ring crossing collector;
- three previous-ring matching policies with auditable per-point state;
- CED-controlled literal-crossing comparisons;
- a branch-tracking-free equal-crossing ring population profile;
- a 26-setting CED parameter sweep across both real colonies;
- independent validators for point matching and population consensus.

These later files are deliberately diagnostic. They do not add a public measurement
column or replace the committed reference calculations.

## 4. External inputs and dependencies

- **Workflow reference:**
  `/Users/alex/Projects/Neurospora/notebooks/LightDetectFungi_Workflow.ipynb`
- **Original TIFF recorded by the scripts:**
  `/Volumes/T9/exfab/UCR-010-I-D_Neurospora/data/denoised_media_subsets_FrameIdx10-12/xylan/d000273_300_001_2025-12-12_02-00-49_rgb.tiff`
- **Cached analysis layers:** `cache/composite.npy`, `cache/objmap.npy`, and
  `cache/twok_branch_reconnection_objmap.npy` allow the expensive image preparation and
  TwoK detection stages to be reused.
- **Real-image cases:** detector label 24 (`R3C4`) and detector label 36 (`R4C6`).
- **CED implementation:** the existing `phenotypic.enhance.StructureSmoothing`
  operation, whose docstring references Weickert's coherence-enhancing diffusion
  formulation. No new runtime package was added for this experiment.
- **Conceptual Sholl references:** Sholl's original concentric-circle analysis and
  automated traced-neurite extensions are discussed in [design.md](design.md). The
  fungal calculation uses rings as sampling surfaces rather than claiming classical
  Sholl branch counts.

## 5. Data and calculation transformations

### Radial-relative orientation

For every selected structure pixel, the fiber axis is compared to the radial spoke from
the inferred inoculum center. Axial orientation is period 180 degrees, so an unoriented
line and the same line reversed are equivalent. A straight outward branch therefore has
near-zero radial-relative tilt regardless of its absolute image angle.

The center still defines radius and polar position, but pixels inside the inferred
inoculum radius are excluded from measurement and visualization.

### Ring and sector prototypes

The committed Sholl-style reference samples 8-pixel annuli and fixed polar sectors.
Long-range comparisons and ring compounding were tested with both radial-relative tilt
and absolute fiber-axis change. Fixed sectors are simple, but a curved branch can move
between sectors as it grows.

### Literal skeleton-ring crossings

The object mask is skeletonized. At each ring, connected skeleton pixels within the
rasterized circle band form one literal crossing record. Each record stores its image
location, absolute fiber axis, radial-relative tilt, coherence, doubled-angle
resultant, and pixel support. This is an observed orientation sample, not an inferred
biological branch ID.

The point matcher permits inheritance only from the immediately previous ring and
requires annular skeleton connectivity, a geometric reach gate, a seam-safe axial gate,
history-aware period-180 lifting, and explicit ambiguity rejection. The complete
contract is in [design.md](design.md#16-point-level-previous-ring-orientation-inheritance).

### Branch-tracking-free population consensus

Every literal crossing receives one vote within its ring. The ring angle is the
doubled-angle axial mean, and the axial resultant reports whether those crossing
orientations agree. At least three crossings and a resultant of at least 0.15 are
required. Signed changes accumulate only within contiguous supported runs. Missing
rings and exact 90-degree transitions break the run rather than receiving an invented
sign or continuation.

Conditional on passing the minimum-crossing support guard, the ring consensus is
invariant to uniform replication of an unchanged crossing-angle distribution. Support
itself is count-sensitive. The calculation is not invariant to uneven skeleton
fragmentation or to adding branches with genuinely different orientations.

## 6. Current quantitative evidence

### Point-level matching remains sparse

The latest [point-matching summary](artifacts/twok_point_matched_orientation_summary.csv)
contains 602 literal crossings for R3C4 and 200 for R4C6.

| Colony | Policy | Supported points | Accepted edges | Raw cumulative peak |
|---|---|---:|---:|---:|
| R3C4 | Reciprocal one-to-one | 53 | 22 | 25.34 degrees |
| R3C4 | Independent many-to-one | 123 | 92 | 45.08 degrees |
| R3C4 | Global one-to-one | 57 | 26 | 19.58 degrees |
| R4C6 | Reciprocal one-to-one | 45 | 17 | 18.18 degrees |
| R4C6 | Independent many-to-one | 62 | 34 | 30.50 degrees |
| R4C6 | Global one-to-one | 45 | 17 | 18.18 degrees |

Reciprocal matching remains the cleanest correspondence policy, but its support is too
sparse to serve as the primary colony phenotype. Independent many-to-one preserves more
paths, but several outer crossings can inherit the same parent, so it can duplicate
evidence and become sensitive to skeleton fragmentation.

### Equal-crossing population trend

The latest [CED comparison summary](artifacts/twok_ced_literal_crossing_summary.csv)
uses gap-safe contiguous runs. Under the original orientation field, R3C4 has 16
supported rings and a 59.75-degree population raw peak; R4C6 has 12 supported rings and
an 82.69-degree population raw peak. These are diagnostic extrema, not selected effect
sizes.

The first CED condition (`sigma=1.5`, `rho=3.0`, 30 iterations, `C=90`) raises median
crossing coherence from 0.471 to 0.552 for R3C4 and from 0.541 to 0.687 for R4C6. Its
population raw peaks are 61.05 and 80.37 degrees, respectively. The R4C6 full peak is
sensitive to low-resultant inner rings; later-radius evidence is more defensible than
the single maximum.

### CED parameter sweep

The [sweep table](artifacts/twok_ced_literal_crossing_parameter_sweep.csv) and
[aggregate table](artifacts/twok_ced_literal_crossing_parameter_sweep_aggregate.csv)
compare 26 CED configurations. Selection proxies reward higher crossing coherence and
lower non-junction skeleton-angle roughness while auditing crossing-count deviation and
source-image distortion.

No tested setting reduced the mean 90th-percentile non-junction angle roughness across
both colonies. Every `MeanRoughnessReduction` value in the aggregate table is negative.
Therefore, greater CED coherence did not demonstrate correction of the dotted-angle
failure mode.

`CED24` was retained only as a conservative visualization candidate because it limits
mean roughness degradation to 0.40% and worst crossing-count deviation to 1.50% while
raising mean coherence by 13.69%. Its parameters are `sigma=2.5`, `rho=5.0`, 30
iterations, `C=95`, `dt=0.1`, and `alpha=0.001`.

| Colony | Condition | Crossings | Median coherence | P90 branch-interior angle difference | Population raw peak |
|---|---|---:|---:|---:|---:|
| R3C4 | Original | 602 | 0.471 | 5.644 degrees | 59.75 degrees |
| R3C4 | CED24 | 600 | 0.516 | 5.646 degrees | 60.68 degrees |
| R4C6 | Original | 200 | 0.541 | 4.694 degrees | 82.69 degrees |
| R4C6 | CED24 | 203 | 0.637 | 4.730 degrees | 91.49 degrees |

The crossing-count changes from 602 to 600 for R3C4 and from 200 to 203 for R4C6
are intentional condition differences, not conflicting reports of the same quantity.
The object geometry and ring locations are fixed, but CED changes which skeleton-ring
samples pass the orientation-coherence gate.

CED24 is not a final tuned setting. In particular, the R4C6 raw population peak is
parameter-sensitive even though the underlying literal-crossing pattern remains
visually similar.

## 7. Methodological caveats

- **Axial versus directional angles:** Fiber axes are unoriented. All differences must
  be period-180 and seam-safe. Exact 90-degree steps have no defensible sign.
- **Raw peak:** A maximum is useful for inspection but is sensitive to weakly supported
  rings, preprocessing, and one local excursion. It must remain visually distinct from
  any eventual normalized phenotype.
- **Support is separate:** Unsupported cells are missing evidence, not zero rotation.
  Every future metric must report support or resultant separately.
- **Branch-density invariance is conditional:** Uniformly duplicating the same
  orientation distribution leaves an equal-crossing axial mean unchanged. Adding a new
  orientation family legitimately changes it.
- **Skeleton crossings are samples, not branches:** A branch tangent to a ring, a
  skeleton junction, or raster fragmentation can create more than one crossing.
- **CED has no ground-truth objective here:** Higher coherence alone can make an
  orientation estimate look more certain without making it more accurate or smoother.
- **R3C4 is heterogeneous:** It contains low-resultant intervals and gaps. CED does not
  resolve those ambiguities.
- **R4C6 has a coherent later-radius pattern, but its full raw peak is anchored by weak
  inner rings:** Avoid interpreting the largest cumulative value as a validated
  biological effect size.

## 8. Validation and review

The analysis includes independent NumPy/SciPy validators that do not import
`phenotypic`, plus unit tests for production utilities and measurement behavior.

Key saved validators cover:

- radial-relative orientation and controls;
- nematic bend invariants;
- matched-ring recurrence and tangential continuation;
- full-length ring boundaries;
- cumulative axial-median change;
- point-level period-180 lifting, ambiguity, connectivity, and no-restart behavior;
- equal-crossing consensus, uniform-replication invariance, gap handling, and exact
  90-degree ambiguity.

During the working session, scoped lint, the independent validators, generated-CSV
reconciliation, targeted CED unit tests, and independent code review were run. Before
promoting any later prototype, rerun them from the dirty worktree and add synthetic
branch-count, width, ring-phase, and known-rotation controls.

## 9. Corrections and course changes

- The orientation display changed from opaque zone/resultant graphics to explicit
  local axes, radial rings, calculation overlays, and signed color.
- Angles changed from radians to degrees in user-facing diagnostics.
- The inoculum was initially visible in some views; later diagnostics exclude it while
  preserving its center.
- HSV was rejected for signed cumulative rotation because both period endpoints look
  similarly red. RdBu and trimmed cyclic alternatives were reviewed; Spectral became
  the shared signed diagnostic map.
- Yellow overlays that obscured direction color were removed in direction-only views.
- Ring compounding was initially truncated by symmetric/sparse bounds; the dedicated
  full-length prototype extends to the first complete ring boundary beyond the farthest
  object pixel.
- A point-matching explanation originally claimed that an old R3C4 ring-median flip
  failed the 20-degree endpoint gate. That was corrected: strict inherited paths ended
  before the transition, so the point matcher never evaluated it.
- CED initially re-inferred the core geometry. The corrected comparison mechanically
  reuses the original object mask, center, distance map, inoculum exclusion, and rings.
- Population unwrapping initially skipped unsupported rings. It now restarts within
  each contiguous supported run and rejects exact 90-degree signed continuation.
- **Documentation drift to fix:** [design.md](design.md#172-branch-tracking-free-population-calculation)
  still states the pre-correction R3C4 population peaks. The current values are in
  `artifacts/twok_ced_literal_crossing_summary.csv` and are the values used here.

## 10. What did not work or remains inconclusive

- **Large global bend kernels:** Increasing sigma supplied longer spatial context but
  mixed orientation families and caused low-resultant blank regions.
- **Fixed polar sectors:** Simple and density-normalized, but a curved branch can leave
  its original sector, so the same sector at two radii may sample different structures.
- **Strict matched-ring tracking:** Scientifically conservative but too sparse on the
  real colonies.
- **Gap and restart variants:** Useful diagnostics, but gap bridging adds correspondence
  assumptions and restarts are segment-relative rather than inoculum-relative.
- **Tangential graph and streamline variants:** They recover additional support but
  introduce stronger path-following assumptions and have not been validated against
  known branch trajectories.
- **Whole-ring mean/median compounding:** Captures colony-wide changes but can jump at
  the axial seam or become unstable when a ring has multiple competing modes.
- **Skeleton masking alone:** Changes support and peak values but does not prove that
  the resulting angle is biologically more accurate.
- **CED tuning by coherence:** The parameter sweep found no setting that improved the
  chosen branch-interior roughness proxy across both colonies. CED remains promising
  visually, but its benefit is unproven.

## 11. Current recommendation

Continue with **literal skeleton-ring crossings plus an equal-crossing axial population
profile** as the primary research direction.

This approach matches the biological goal most directly:

1. the inoculum center defines the radial frame;
2. the core is excluded;
3. individual branches do not need to be detected or followed;
4. every crossing contributes one orientation observation at a known radius;
5. a shared rotational phenotype should appear as a coherent shift in the crossing
   distribution across rings;
6. ring resultant and support expose ambiguity rather than converting it to zero.

Keep the parent-child point matcher as a secondary audit tool. Keep both the original
orientation source and CED24 visualizations, but do not make CED24 the production
default. Raw peak amplitude should remain displayed and exported separately.

## 12. Next steps

1. Synchronize Section 17 of `design.md` with the gap-safe CSV values and the CED sweep.
2. Define the population-profile output contract before selecting a single colony
   statistic: signed ring angle, contiguous-run change, ring resultant, crossing count,
   and supported-ring fraction.
3. Build analytic phantoms with known radial, spiral, and tangential fields. Include
   exact 90-degree ambiguity and missing-ring controls.
4. Test branch-count invariance by uniformly replicating identical branches, then add
   differently oriented branches to document the intended sensitivity.
5. Test skeleton width, fragmentation, ring spacing, ring phase, and inoculum-center
   perturbation.
6. Replace or supplement the CED tuning proxy with a dotted-line phantom whose true
   centerline orientation is known. Only then choose CED parameters.
7. Review the R4C6 weak inner rings separately from its later-radius trend and decide
   whether a minimum resultant should define the starting anchor.
8. Review more colonies before defining any production aggregation or threshold.
9. Rerun lint, validators, CSV reconciliation, and focused tests; then create a new
   checkpoint containing only orientation-analysis files, leaving unrelated detector
   work untouched.

## 13. Lessons learned

- Anchor orientation to biological geometry, not the image axes.
- Treat axial angles as period-180 quantities everywhere, including visualization,
  interpolation, median selection, unwrapping, and ambiguity tests.
- Never interpret missing support as zero phenotype.
- Keep raw amplitude, reliability, and the normalized phenotype as separate outputs.
- More coherent is not automatically more accurate. Tune preprocessing against a known
  error mode, not against the desired biological result.
- A population calculation can avoid fragile branch correspondence when the phenotype
  is expected to affect many branches coherently.
- Preserve conservative reference implementations while experimenting so changes can
  be attributed to one methodological choice at a time.
- Real-image overlays are essential, but analytic phantoms are required to establish
  correctness.

## 14. Artifact index

### Core documentation and checkpoints

- [README.md](README.md): bundle inventory and reproduction notes.
- [design.md](design.md): detailed algorithms, assumptions, and historical decisions.
- Commit `c050b468b`: initial production implementation and saved analysis bundle.
- Commit `7375e0ec5`: full-length orientation-analysis checkpoint.

### Current point-crossing and CED figures

- [R3C4 point matching](artifacts/twok_R3C4_point_matched_orientation_2x2.png)
- [R4C6 point matching](artifacts/twok_R4C6_point_matched_orientation_2x2.png)
- [R3C4 CED24 crossing comparison](artifacts/twok_R3C4_ced_point_crossing_comparison_CED24_2x2.png)
- [R4C6 CED24 crossing comparison](artifacts/twok_R4C6_ced_point_crossing_comparison_CED24_2x2.png)
- [R3C4 CED24 population trend](artifacts/twok_R3C4_ced_literal_crossing_trend_CED24_2x2.png)
- [R4C6 CED24 population trend](artifacts/twok_R4C6_ced_literal_crossing_trend_CED24_2x2.png)
- [CED parameter sweep](artifacts/twok_ced_literal_crossing_parameter_sweep.png)

### Current data tables

- [Literal crossings](artifacts/twok_point_ring_crossings.csv)
- [Point inheritance states](artifacts/twok_point_matched_orientation_states.csv)
- [Point-matching summary](artifacts/twok_point_matched_orientation_summary.csv)
- [CED ring profiles](artifacts/twok_ced_literal_crossing_ring_profiles.csv)
- [CED comparison summary](artifacts/twok_ced_literal_crossing_summary.csv)
- [CED sweep detail](artifacts/twok_ced_literal_crossing_parameter_sweep.csv)
- [CED sweep aggregate](artifacts/twok_ced_literal_crossing_parameter_sweep_aggregate.csv)

### Reproduction scripts

- [Point crossing and matching](scripts/render_point_matched_ring_orientation.py)
- [CED comparison and population trend](scripts/render_ced_point_crossing_comparison.py)
- [CED parameter sweep](scripts/sweep_ced_literal_crossings.py)
- [Skeleton-masked comparison](scripts/render_skeletonized_ring_median_axial_change.py)
- [Full-length ring median](scripts/render_ring_median_axial_change.py)
- [Ring compounding](scripts/render_ring_compounded_rotation.py)
- [Tangential-method comparison](scripts/render_tangential_method_comparison.py)

### Independent logic validators

- [Point-ring matching](../../logic_validation_scripts/2026-07-15-orientation-field/point_ring_orientation_matching.py)
- [CED crossing consensus](../../logic_validation_scripts/2026-07-15-orientation-field/ced_literal_crossing_consensus.py)
- [Cumulative axial median](../../logic_validation_scripts/2026-07-15-orientation-field/cumulative_axial_median_change.py)
- [Full-length ring extent](../../logic_validation_scripts/2026-07-15-orientation-field/full_length_ring_extent.py)

The complete historical figure inventory is maintained in [README.md](README.md).
