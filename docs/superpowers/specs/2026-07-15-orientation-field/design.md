# Design: angular-arc bend and Sholl-guided branch rotation

- **Date:** 2026-07-15
- **Status:** Angular-arc gating remains proposed. A private matched-ring
  cumulative-rotation diagnostic has been selected as the first Sholl-style
  prototype; no public measurement schema has been added.
- **Scope:** Two diagnostic prototypes for `MeasureOrientationZones`. Existing
  orientation-zone calculations remain unchanged as test references.
- **Primary user goal:** Measure how the orientation field departs from straight,
  outward-radiating growth without detecting individual branches and without making
  branch density the target phenotype.

## 1. Decision summary

### 1.1 Decision update: matched rings before continuous tracks

After review, the first Sholl-style implementation is a **matched-ring cumulative
rotation** intermediate between the existing fixed-sector calculation and the continuous
director-track design below. Each ring cell may connect to a reliable nearby sector on
the immediately following ring. Seam-safe fiber-axis changes are accumulated along that
discrete path. The existing fixed-sector radial-relative accumulation remains unchanged
as the primary reference, and a fixed-sector fiber-axis control isolates the effect of
matching from the effect of changing the accumulated angle.

The continuous outward director tracker in Section 5 remains a deferred comparison. It
should be implemented only if the discrete 8-pixel ring and 10-degree sector grid proves
too coarse or boundary-sensitive on the real colonies.

### 1.2 Diagnostic continuity-rule comparison

The matched-ring prototype supports four private diagnostic conditions so missing polar
lattice cells can be attributed to continuity policy rather than conflated with absent
orientation evidence:

1. **Strict baseline:** adjacent supported rings are required and terminated seeds do
   not restart.
2. **Gap bridging:** rings with no reliable nearby candidate may be skipped. Their cells
   remain blank, the next match uses the complete radial interval, and geometric or
   axial-ambiguity failures still terminate the segment.
3. **Segment restarts:** gap bridging remains disabled. After termination, the same
   geometric seed sector starts a new segment at its next reliable ring and resets its
   cumulative rotation to zero.
4. **Gap bridging plus restarts:** bridging is attempted first; if it cannot produce a
   defensible continuation, the later segment may restart at zero.

Restarted values are segment-relative and must never be presented as cumulative rotation
from the inoculum. Gap bridging does not interpolate missing orientations, so skipped
ring cells remain unsupported in the lattice. These modes remain diagnostic-only and do
not alter the public measurement schema.

Prototype both methods, but treat them as answering different questions:

1. **Angular-arc-gated bend** asks: “Is there a coherent curved orientation field
   within this local angular region at a chosen spatial scale?” It is the smaller,
   lower-risk modification to the current large-sigma bend overlay.
2. **Sholl-guided outward director tracks** ask: “If an outward-growing trajectory
   follows the measured orientation field from one radial band to the next, how far
   does its angular position rotate?” This is the better match to the desired
   phenotype, but it is also more assumption-dependent.

The recommended evaluation order is angular-arc gating first, then Sholl-guided tracks
on the same two colonies and the same cached TwoK detector output. Neither method should
replace the existing fixed-sector Sholl metric until analytic phantoms and real-image
review establish what each failure state means.

**Confidence:** This recommendation is a defensible design judgment, not an established
optimal method. It is based on the geometry of the current implementation and on general
reasoning about unoriented line fields. No empirical comparison of these two proposed
methods has yet been run.

## 2. Assumptions and terminology

### 2.1 Meaning of a 45-degree arc

“45-degree arc” means a **spatial polar wedge** around the inferred inoculum center. It
does not mean retaining fibers whose absolute orientation is 45 degrees. For example,
one reporting arc may cover polar positions `[0°, 45°)`, while retaining radial,
tangential, or oblique fibers inside that wedge.

This distinction is load-bearing. Filtering by fiber orientation would reintroduce the
0° versus 90° global-axis problem that radial-relative measurements are intended to
avoid.

### 2.2 Center and active region

Both methods use the inferred inoculum center to define polar radius `r` and azimuth
`alpha`. The active structure selector is:

```text
detected object mask
AND r >= core_end_radius
AND r < min(sparse_end_radius, symmetric_radius)
```

Thus the center anchors the geometry but the inferred inoculum core contributes no
orientation evidence and is blank in every view.

### 2.3 What “branch-density invariant” can mean here

Neither method can be absolutely invariant to every change in branch number without
detecting and matching individual branches. Adding a new branch can add a previously
unsupported angular region or introduce a second orientation into an existing region.
That is real new orientation evidence and may change the estimate.

The achievable contract is narrower and testable:

- duplicating same-orientation evidence inside an already supported angular unit does
  not change that unit's normalized orientation estimate;
- each supported 45-degree reporting arc receives equal final weight, independent of
  its pixel count;
- the fraction of supported arcs or tracks is reported separately as a density-sensitive
  quality diagnostic;
- the primary orientation phenotype is never multiplied by branch count, mask area, or
  number of detected pixels.

This matches the current equal-sector aggregation principle in
`src/phenotypic/measure/_measure_orientation_zones.py:242-338`.

## 3. Current baseline and the gaps it leaves

### 3.1 Global large-sigma bend

The current bend field coherence-weights the doubled-angle fiber director and applies a
Cartesian Gaussian to the full active selector. It blanks pixels when the scale-local
resultant is below `0.15`
(`src/phenotypic/util/_nematic_bend.py:61-110` and
`src/phenotypic/measure/_measure_orientation_zones.py:1978-1992`). At sigma 32 or 48,
different orientation families can enter the same averaging window and cancel even when
each family is locally coherent.

### 3.2 Existing Sholl-style reference

The repository already divides the core-excluded colony into equal-width annular bands
and 36 fixed 10-degree sectors
(`src/phenotypic/measure/_measure_orientation_zones.py:415-493`). Its long-range metric
compares the **same sector index** at two radii
(`src/phenotypic/measure/_measure_orientation_zones.py:644-700`). This is a useful,
simple reference, but a curved branch can move into a neighboring angular sector as it
grows. The fixed-sector comparison can then compare different structures, lose support,
or miss the trajectory.

Classical Sholl analysis counts structure crossings of concentric circles rather than
tracking orientation trajectories. That foundational use is described in Sholl's 1953
paper, [“Dendritic organization in the neurons of the visual and motor cortices of the
cat”](https://pmc.ncbi.nlm.nih.gov/articles/PMC1244622/). Automated extensions commonly
operate on traced neurite morphology and connectivity, which conflicts with this
project's requirement to avoid individual branch detection
([Kutzing et al., 2010](https://pubmed.ncbi.nlm.nih.gov/21113115/)). Therefore, the
second prototype uses Sholl rings as sampling surfaces but follows a continuous director
field rather than counting or matching segmented branches.

## 4. Prototype A: overlapping angular-arc-gated bend

### 4.1 Question answered

Within a limited spatial arc, how strongly does the local unoriented fiber field bend at
sigma 16, 32, or 48, without allowing evidence from the rest of the colony to enter the
Gaussian average?

### 4.2 Data flow

```text
[orientation phi, coherence C, center, active mask]
                         |
                         v
             [overlapping 45-degree arc weights]
                         |
                         v
       [masked doubled-angle Gaussian Q field per arc]
                         |
                         v
             [arc-local bend and resultant]
                         |
              +----------+----------+
              v                     v
       [full-field overlay]   [equal-arc summaries]
```

The prototype is synchronous, array-local, and uses only NumPy and SciPy already present
in the project. No new dependency is proposed.

### 4.3 Angular windows

Use overlapping raised-cosine windows rather than eight disjoint hard wedges. Let the
full arc width be `A = 45°`, with arc centers every `A/2 = 22.5°`. For polar azimuth
`alpha` and arc center `alpha_j`, define the wrapped angular difference `d_j` in
`[-180°, 180°)` and:

```text
a_j(alpha) = 0.5 * (1 + cos(2*pi*d_j/A))  when |d_j| <= A/2
             0                             otherwise
```

This gives 16 overlapping computational arcs. The overlap avoids a sharp orientation
discontinuity at an arbitrary wedge boundary. A hard eight-wedge implementation remains
the “boring baseline” because it is easier to reason about, but it should appear only as
a debugging toggle, not the recommended overlay.

### 4.4 Arc-gated Q field

For fiber axis `theta = phi + pi/2`, active selector `M`, coherence `C`, and Gaussian
`G_sigma`, compute each arc independently:

```text
w_j       = M * C * a_j
Q_j       = G_sigma * (w_j * exp(i*2*theta))
W_j       = G_sigma * w_j
q_j       = Q_j / W_j
R_j       = |q_j|
theta_j   = 0.5 * arg(Q_j)
B_j       = |n_j dot grad(theta_j)|
```

`B_j` is the same director-field bend definition as the current prototype, in degrees
per pixel after conversion. The doubled angle preserves the equivalence of `theta` and
`theta + 180°`.

At each active pixel, combine only the one or two overlapping arc results whose angular
windows include that pixel. Blend the **scalar bends**, weighted by `a_j * R_j`, after
each arc's phase derivative has been computed. Do not average the complex Q fields from
different arcs again, because doing so would recreate the cancellation the gating is
intended to prevent.

### 4.5 Reliability and blank pixels

A pixel is shown only when all of the following are true:

- it is in the core-excluded active selector;
- local structure-tensor coherence is at least `0.15`;
- at least one angular window has finite normalized support;
- the contributing arc's scale resultant is at least `0.15`.

Blank pixels continue to mean “unsupported or directionally ambiguous,” not zero bend.
The diagnostic must report separate fractions for:

- retained reliable active structure;
- scale-resultant cancellation;
- insufficient arc support;
- local-coherence exclusion.

### 4.6 Equal-arc summary and density handling

Use eight non-overlapping 45-degree **reporting arcs** even though the computation uses
16 overlapping windows. Within each supported reporting arc, calculate a normalized
`C * R` weighted mean bend. Average the eight supported arc means equally. Report:

- candidate `ArcBendMean` in degrees per pixel;
- `ArcBendSupport`, the fraction of eight reporting arcs that are reliable;
- `ArcBendRawPeak` as a distinctly labeled diagnostic, never as the primary phenotype;
- `ArcPhaseSensitivity`, the absolute difference between summaries from reporting grids
  offset by 0 degrees and 22.5 degrees.

These remain prototype names. No `MeasurementInfo` entries or public DataFrame columns
are added before real-image review.

### 4.7 Visualization

Produce one plate-crop figure per colony with:

1. actual detected-array layer in grayscale;
2. three bend panels for sigma 16, 32, and 48;
3. inoculum core blank;
4. optional faint white 45-degree reporting boundaries, disabled by default;
5. a sequential color map for unsigned bend, with each panel's numeric range printed;
6. retained, cancelled, and unsupported percentages printed above each panel;
7. raw peak printed separately from the robust/equal-arc summary;
8. no object labels and no yellow overlay.

The diagnostic supports two views:

- **single-arc view:** the user supplies an arc center and sees only the corresponding
  45-degree spatial wedge, its effective Gaussian support, bend, and resultant;
- **stitched view:** the complete colony is assembled from the overlapping arc-local
  results so it can be compared directly with the current global-sigma overlay.

The single-arc view answers exactly what evidence entered one calculation. The stitched
view is the one used for coverage and equal-arc summaries.

Because bend is unsigned, `Spectral` is not appropriate for this panel. `Spectral` is
reserved for the signed rotation view in Prototype B. Dynamic per-panel color limits may
be used for spatial inspection, but the caption must say that brightness is not directly
comparable across sigma. A second fixed-limit export should be available for amplitude
comparison.

### 4.8 Strengths, limitations, and failure modes

**Strengths**

- Small conceptual change to the existing Q-field calculation.
- Directly tests whether azimuthal mixing caused the missing sigma 32/48 regions.
- Preserves the axial 180-degree equivalence.
- Easy to compare with the current global-sigma reference.

**Limitations and failures**

- Multiple incompatible orientations inside the same 45-degree arc still cancel.
- A fixed angular width spans a larger physical arc length at larger radius.
- Large sigma still mixes inner and outer radii inside an arc, so it is not a trajectory
  tracker.
- Results can depend on reporting-grid phase; `ArcPhaseSensitivity` makes that visible.
- Adding a branch with a new orientation to an occupied arc can legitimately change the
  field estimate. This is not strict branch-count invariance.

The decision is reversible because this begins as a diagnostic figure and private helper
without a schema commitment.

## 5. Prototype B: Sholl-guided outward director tracks

### 5.1 Question answered

Starting at a uniformly spaced angular seed outside the inoculum, how does a hypothetical
outward-growing trajectory move in azimuth when it follows the measured local fiber
director from one Sholl band to the next?

This tracks the **orientation field's integral curves**, not detected branches. A recent
open-access methods paper describes streamlines as curves tangent to a fiber-orientation
field and discusses their value and failure modes for fiber visualization
([Roney et al., 2024](https://doi.org/10.1016/j.cmpb.2024.108202)). Applying that idea to
this fungal orientation field is an adaptation, not an established fungal phenotype
method.

### 5.2 Data flow

```text
[orientation phi, coherence C, center, active mask]
                         |
                         v
         [overlapping ring-by-angle orientation lattice]
                         |
                         v
          [uniform angular seeds at core boundary]
                         |
                         v
    [predict/correct next ring crossing from local tilt]
                         |
              +----------+----------+
              v                     v
       [outward tracks]       [zone rotations]
              |                     |
              +----------+----------+
                         v
          [Spectral overlay + support diagnostics]
```

### 5.3 Polar orientation lattice

Use complete annular bands beginning at `core_end_radius`, initially with the existing
`radial_ring_width = 8 px`. Sample signed radial-relative tilt in overlapping angular
windows centered every 10 degrees. A 20-degree full angular window is the starting
prototype value so neighboring cells overlap instead of creating hard 10-degree seams.

For every ring-angle cell, store:

- coherence-weighted doubled-angle mean radial-relative tilt `delta` in `[-90°, 90°]`;
- axial resultant `R`;
- detected-structure support;
- mean local coherence.

Cells with fewer than three reliable pixels, coherence below `0.15`, or resultant below
`0.15` are unsupported. These starting thresholds match the current reference and are
not claimed to be optimal.

### 5.4 Outward track equation

At polar position `(r, alpha)`, let `delta` be the signed angle between the fiber axis and
the outward radial unit vector. Choose the sign of the axial director so its radial
component is nonnegative. The tangent components are then proportional to:

```text
radial component     = cos(delta)
azimuthal component  = sin(delta)
```

The corresponding outward path satisfies:

```text
d alpha / d r = tan(delta) / r
```

For one ring interval `[r_k, r_(k+1)]`, a constant-tilt predictor is:

```text
delta_alpha = tan(delta_k) * log(r_(k+1) / r_k)
alpha_pred  = alpha_k + delta_alpha
```

Use a midpoint or Heun predictor-corrector: sample the lattice again at the predicted
position, average the two slopes, and update `alpha_(k+1)`. This is preferable to simply
jumping to the nearest fixed sector.

Terminate a track rather than hallucinating through ambiguity when:

- the current or predicted cell is unsupported;
- `|delta|` exceeds an initial 75-degree limit, where outward radial progress becomes
  poorly conditioned;
- the predicted angular step exceeds a configurable safety limit;
- the trajectory leaves the active radial range;
- an exact 90-degree axial ambiguity is encountered.

The strict default does not bridge missing rings. The private diagnostic comparison may
scan across any number of consecutive rings that contain no reliable nearby candidate,
using the complete radial interval in the predictor. A gap containing a geometric or
exact-90-degree axial failure is not bridgeable. Skipped cells remain unsupported, and
interactive overlays render the connection as a labeled dashed bridge rather than an
observed path segment.

### 5.5 Seeds and aggregation

Seed 36 trajectories uniformly every 10 degrees at the first supported ring outside the
inoculum. Seeds are geometric and fixed, not created per branch or per detected pixel.

For each track and radial zone, define net path rotation as the continuously unwrapped
azimuth change between the supported zone entry and exit:

```text
PathRotation(track, zone) = alpha_exit - alpha_entry
```

Positive means clockwise in image coordinates, matching the existing signed-turning
convention. A zone-level track is eligible only if it has continuous support from its
entry to exit; partial tracks contribute to the support diagnostic but not the net zone
rotation.

To reduce branch-density weighting:

1. group the 36 seeds into eight 45-degree reporting arcs;
2. summarize eligible tracks within each arc once;
3. give each supported arc equal weight in the colony/zone summary.

Prototype outputs are:

- `PathRotationMeanAbsolute`, equal-arc mean absolute net rotation in degrees;
- `PathRotationMeanSigned`, equal-arc mean signed net rotation in degrees;
- `PathRotationSupport`, fraction of eight reporting arcs with at least one complete
  track;
- `PathPersistence`, median fraction of eligible radial bands reached by seeded tracks;
- `PathRotationRawMax`, distinctly labeled diagnostic only.

The exact within-arc statistic, mean versus median, remains a review decision. Export both
in the prototype table, but do not add either to the public schema until the two real
colonies show which one corresponds to the visible phenotype.

### 5.6 Interpretation against straight branches

- A straight branch radiating from the inoculum has `delta = 0`, so its angular position
  does not change and its path rotation is zero regardless of whether its global image
  axis is 0 degrees, 90 degrees, or anything else.
- Opposite directions on the same straight axis are equivalent because the field is
  axial and the outward sign is selected locally.
- A constant nonzero radial-relative tilt follows a spiral-like path and accumulates
  azimuthal rotation. It should not be treated as a straight radial branch.
- A local parent-to-daughter bifurcation cannot be recovered from a single averaged
  director cell. That remains explicitly deferred.

### 5.7 Visualization

The primary prototype figure contains four coordinated panels:

1. **Actual-array overlay:** grayscale detected-array layer, core blank, faint Sholl
   rings, and uniformly seeded outward paths. Small arrowheads indicate outward tracking.
2. **Signed cumulative rotation overlay:** the active structure colored by the track
   value assigned to its ring-angle cell, using the `Spectral` diverging map fixed to the
   full `[-180°, 180°]` range. Unsupported cells remain transparent.
3. **Polar track map:** radius on x, seed angle on y, signed cumulative path rotation as
   color. This exposes support gaps and coherent clockwise/counterclockwise bands.
4. **Support profile:** fraction of tracks remaining versus radius, with dense/sparse
   boundaries marked.

No object labels or yellow orientation layer are shown. Hover or a compact table reports
zone mean absolute rotation, signed rotation, support, persistence, and raw maximum.

### 5.8 Strengths, limitations, and failure modes

**Strengths**

- Directly expresses rotation away from the seed's original radial spoke in degrees.
- A global 0-degree versus 90-degree axis cannot create a false rotation.
- Can follow a trajectory into an adjacent angular sector.
- Uniform seeds and equal-arc aggregation reduce direct branch-count weighting.

**Limitations and failures**

- It follows an averaged orientation field, not a biological branch identity.
- Crossings and two incompatible families lower the resultant and terminate tracks.
- A path can switch between nearby parallel branches without that switch being visible.
- Sparse masks reduce track completion; support must always accompany the phenotype.
- Near-tangential directions are numerically unstable for outward radial integration.
- The result depends on ring width, angular sampling width, coherence threshold, and
  termination rules.
- A smooth orientation field can produce visually plausible tracks even when the
  detector has connected structures that are not biologically continuous. Overlay review
  on the TwoK output is mandatory.

The algorithm is reversible while private. Public metric names would be sticky, so schema
addition is deliberately deferred.

## 6. Comparison

| Criterion | Global bend reference | Arc-gated bend | Fixed-sector Sholl reference | Sholl-guided tracks |
|---|---|---|---|---|
| Prevents distant azimuth mixing | No | Yes, within arc width | Yes | Yes, within sampling window |
| Follows motion into adjacent sector | No | No | No | Yes |
| Output unit | degrees/pixel | degrees/pixel | degrees across fixed lag | degrees of path rotation |
| Straight radial branch gives zero | Bend only if field constant | Bend only if field constant | Yes for radial-relative tilt change | Yes by construction |
| Branch-count sensitivity | Pixel-weighted locally | Equal reporting arcs | Equal reliable cells | Uniform seeds plus equal arcs |
| Main failure | Q cancellation | within-arc cancellation | sector mismatch | track ambiguity/termination |
| Implementation cost | Existing | Low to moderate | Existing | Moderate to high |
| Technical debt if kept private | None | Low | None | Moderate |

The deciding axis is **trajectory correspondence**, not sigma alone. Arc gating can show
whether the bend field survives local averaging, but only outward director tracks attempt
to maintain correspondence as the field moves around the colony.

## 7. Prototype parameters

These are explicit starting values for review, not production defaults:

| Parameter | Arc-gated bend | Sholl-guided tracks |
|---|---:|---:|
| Spatial scales | 16, 32, 48 px | n/a |
| Computational arc width | 45 degrees | 20 degrees |
| Computational arc spacing | 22.5 degrees | 10 degrees |
| Reporting arcs | 8 x 45 degrees | 8 x 45 degrees |
| Radial ring width | n/a | 8 px |
| Minimum coherence | 0.15 | 0.15 |
| Minimum axial resultant | 0.15 | 0.15 |
| Minimum pixels per sample | 3 | 3 |
| Maximum absolute outward tilt | n/a | 75 degrees |
| Missing-ring bridge | n/a | 0 rings |
| Signed display range | n/a | -180 to 180 degrees |

The prototypes should accept these as diagnostic method arguments rather than adding
Pydantic operation fields. If a method is approved for measurement, its parameters must
then follow the repository's operation-parameter and serialization conventions.

### 7.1 Diagnostic API sketch

The names are deliberately diagnostic and do not commit the public measurement schema:

```python
def angular_arc_bend_overlay(
    self,
    image=None,
    *,
    scales: tuple[float, ...] = (16.0, 32.0, 48.0),
    arc_width_degrees: float = 45.0,
    arc_center_degrees: float | None = None,
    overlap_fraction: float = 0.5,
    fixed_color_limit: float | None = None,
): ...

def sholl_path_rotation_overlay(
    self,
    image=None,
    *,
    ring_width: float = 8.0,
    seed_spacing_degrees: float = 10.0,
    sample_arc_width_degrees: float = 20.0,
    max_abs_tilt_degrees: float = 75.0,
    max_gap_rings: int = 0,
    signed_limit_degrees: float = 180.0,
): ...
```

`arc_center_degrees=None` selects the stitched full-colony view. A finite value selects
the single spatial arc centered at that image-coordinate azimuth. Validation and
normalization belong in private helpers until a prototype is approved.

## 8. Validation plan before public metrics

### 8.1 Independent analytic phantoms

The implementation phase must first add independent NumPy/SciPy logic-validation scripts
under:

```text
docs/superpowers/logic_validation_scripts/
  2026-07-15-angular-arc-sholl-branch-rotation/
```

They must not import `phenotypic`. Required cases:

1. straight radial spokes at different global axes return zero path rotation;
2. duplicating same-angle evidence changes support but not the primary equal-arc value;
3. clockwise and counterclockwise fields return equal magnitude and opposite sign;
4. two orthogonal families placed in different 45-degree arcs cancel in the global
   sigma field but remain supported in the arc-gated field;
5. the same two families placed inside one arc still cancel, demonstrating the method's
   intended limit;
6. rotating a phantom relative to the reporting-grid origin quantifies phase sensitivity;
7. a known analytic spiral is recovered within a predeclared numerical tolerance;
8. support gaps terminate tracks and never silently bridge them;
9. core pixels cannot affect either result.

The scripts and exact tolerances are part of implementation, not assumed proven by this
unexecuted specification.

### 8.2 Unit tests

Add focused tests for:

- axial seam safety at plus/minus 90 degrees;
- arc-window partition and overlap normalization;
- finite support and all-NaN behavior;
- deterministic results and rotation of the reporting-grid offset;
- exact zone-entry and zone-exit handling;
- no track propagation through unsupported cells;
- equal-arc aggregation and separate support;
- figure traces contain no object-label text;
- signed `Spectral` overlay uses fixed `[-180, 180]` limits;
- existing orientation-zone tests and columns remain unchanged.

### 8.3 Real-image review

Use the cached notebook image and branch-reconnection TwoK object map already used for
the sigma study. Render the same two colonies:

- detector label 24, grid section 23, corresponding to R3C4;
- detector label 36, grid section 35, corresponding to R4C6.

For Prototype A, compare global and arc-gated sigma 16/32/48 fields, retained coverage,
resultant cancellation, equal-arc value, phase sensitivity, and raw peak.

For Prototype B, compare the fixed-sector Sholl reference and tracked paths on the same
base layer. Review whether paths follow visible branch corridors, whether obvious turns
produce coherent signed rotation, and where support terminates.

### 8.4 Decision gates

No method is promoted to a public measurement unless:

- phantom invariants pass;
- the inoculum remains excluded;
- the primary value is stable when same-orientation evidence is replicated;
- support changes are reported separately;
- figure interpretation agrees with the field actually being calculated;
- parameter changes are explainable rather than merely visually attractive;
- both real colonies are reviewed at full resolution;
- the existing calculations remain available as regression references.

## 9. Implementation sequence after approval

1. Add independent logic-validation scripts and analytic tests.
2. Add the private matched-ring cumulative profile without changing the schema.
3. Compare the existing fixed-sector radial-relative calculation, a fixed-sector
   fiber-axis control, and nearby-sector matched fiber-axis accumulation.
4. Implement Prototype A diagnostic overlay and compare sigma 16/32/48.
5. Defer continuous Prototype B tracks unless matched rings fail the decision gates.
6. Run selected diagnostics on R3C4 and R4C6 using the cached TwoK object map.
7. Review figures and select statistics/parameters with the user.
8. Only then decide whether to add operation fields and `MeasurementInfo` entries.

## 10. Recommendation and what would change it

Proceed with both diagnostic prototypes, retaining the current global bend and
fixed-sector Sholl calculations unchanged. Use arc-gated bend as the low-risk diagnostic
for the sigma-mixing problem. Use Sholl-guided outward tracks as the leading candidate
for the final “rotation away from a straight radial branch” phenotype.

This recommendation would change if arc-gated sigma 32 alone consistently matches the
visible turns while tracked paths frequently terminate or switch corridors. In that case,
the simpler arc-gated field should remain the phenotype and Sholl-guided paths should be
kept only as a visualization. Conversely, if the tracked path rotation is stable across
ring widths and sparse/dense versions while the bend magnitude remains strongly
sigma-dependent, path rotation should become the primary candidate metric.

## 11. Tangential-continuation comparison

### 11.1 Compared diagnostic methods

The real-image follow-up compares four private diagnostics while leaving the existing
matched-ring implementation unchanged as the reference:

1. the strict radial matched-ring profile;
2. a bounded tangential rescue graph;
3. Cartesian integral curves of a smoothed axial director field;
4. local director bend at sigma 32 px with near-tangential pixels marked separately.

The rescue graph may move at most two 10-degree cells around the same ring. Same-ring
edges are accepted only when the measured fiber axes at both endpoints are within 35
degrees of the chord bearing between sector centers, including the half-sector bearing
offset. A tangential route is considered only after no direct
outward route is defensible, so it cannot replace an existing strict continuation. It
does not bridge missing rings. The accumulated value includes seam-safe fiber-axis
changes along both the lateral and outward edges.

The Cartesian comparison smooths coherence-weighted doubled-angle components at sigma
8 px and bilinearly samples those components, never the wrapped axial angle. It launches
up to 36 geometric seed locations from detected pixels in the first reliable ring-sector
cell and requires every accepted point to remain within 4 px of the active mask. When the seed
director is within the numerical tangent tolerance, both director signs are displayed as
dashed hypotheses. Such a pair is a visualization of ambiguity, not a signed outward-turn
measurement. Integral curves are terminated at the core, outer boundary, low resultant,
image boundary, excessive mask distance, or a spatial revisit.

The local panel uses `fiber_bend_field` at sigma 32 px. Bend color is clipped at its 95th
percentile for legibility, while the raw maximum remains stated in the title. Cyan marks
pixels whose absolute radial-relative tilt is at least 75 degrees. This panel is local,
unsigned, and not directly comparable numerically with the three cumulative panels.

### 11.2 Real-image results

The July 15 branch-reconnection TwoK output gave the following diagnostic values. These
are prototype observations, not public phenotype columns:

| Colony | Method | Support | Absolute p95 | Raw peak |
|---|---|---:|---:|---:|
| R3C4 | strict matched rings | 28.5% lattice cells | 93.7 degrees | 155.8 degrees |
| R3C4 | tangential rescue | 29.1% lattice cells | 98.6 degrees | 164.3 degrees |
| R3C4 | Cartesian streamlines | 34/36 supported seed locations | 135.4 degrees per-seed endpoint | 269.4 degrees raw path-sample peak |
| R3C4 | local bend + tangent occupancy | 2,894/16,682 bend-valid pixels, 17.3% | 0.916 degrees/px | 1.676 degrees/px |
| R4C6 | strict matched rings | 42.6% lattice cells | 49.8 degrees | 70.5 degrees |
| R4C6 | tangential rescue | 43.1% lattice cells | 54.5 degrees | 83.3 degrees |
| R4C6 | Cartesian streamlines | 32/36 supported seed locations | 73.2 degrees per-seed endpoint | 76.9 degrees raw path-sample peak |
| R4C6 | local bend + tangent occupancy | 440/3,957 bend-valid pixels, 11.1% | 0.775 degrees/px | 1.938 degrees/px |

The bounded rescue added little support on either colony. That is evidence that most
strict failures in these two fields are not repaired by one or two reliable,
bearing-aligned same-ring cells. An initial unconstrained Cartesian render crossed
visually empty gaps; the reported version therefore applies the 4 px active-mask-distance
limit. Even with that limit, the Cartesian view shows long, strongly curved field lines
in R3C4. Its per-seed endpoint statistic is not numerically comparable with the A/B
ring-cell distribution, so no cross-method magnitude conclusion is made from the p95
values. Apparent streamline continuity is not sufficient evidence of biological branch identity
because a smoothed director line can still switch between nearby detected structures.
The Cartesian view is useful for displaying field geometry but should not become the
primary phenotype without phantom validation and branch-switch sensitivity tests.

### 11.3 Current recommendation

Keep strict matched rings as the quantitative reference. Retain bounded tangential
rescue as a narrowly scoped sensitivity analysis because it preserves direct paths and
changes these two colonies only modestly. Retain Cartesian streamlines and the local
bend/tangent panel as explanatory overlays. Do not combine their numbers into one score:
they measure different quantities and have different units.

Before promoting tangential rescue, implement a two-best-path ambiguity margin and test
sector-grid rotation sensitivity. Before promoting Cartesian streamlines, show stability
across active-mask distance, smoothing scale, and integration step size.

## 12. Colony-wide ring-compounded tilt

### 12.1 Question answered

This diagnostic asks: if every Sholl ring is represented once by its colony-wide typical
radial-relative tilt, how much azimuthal rotation would those ring tilts predict as they
are compounded outward?

The calculation uses the inoculum-centered 8 px rings and 36 fixed 10-degree sectors
already used by the matched-ring reference. The inoculum core is excluded. Within each
ring, every reliable sector contributes once, regardless of its pixel count or resultant
magnitude after eligibility. Two ring summaries are compared:

- the equal-sector doubled-angle mean;
- the equal-sector sample axial median, defined as the observed sector tilt minimizing
  total absolute axial distance, with mean proximity as the deterministic tie breaker.

At least three reliable sectors and an equal-sector ring resultant of at least 0.15 are
required. Below that resultant, the colony-wide axial direction is treated as ambiguous,
so neither mean nor median is compounded. Sector support and ring resultant are plotted
separately from continuous cumulative-ring support.

### 12.2 Geometric compounding

Raw orientation angles are not added directly because that would make the result depend
arbitrarily on the number of sampled rings. For ring summary tilt `delta_k`, the method
uses the constant-tilt polar step:

```text
delta_alpha_k = tan(delta_k) * log(r_(k+1) / r_k)
A_(k+1) = A_k + delta_alpha_k
```

Thus a straight radial field has zero accumulation. A constant nonzero tilt produces the
analytic logarithmic-spiral rotation `tan(delta) * log(r/r_start)`, independent of how
finely the radial interval is subdivided. A missing ring or absolute tilt above 75
degrees terminates continuous accumulation rather than extrapolating through tangency.

This is a colony-wide field model, not a branch tracker. Opposing clockwise and
counterclockwise sector populations may cancel in the mean, while the median selects the
majority tendency. Agreement between mean and median is therefore an important robustness
check.

### 12.3 Real-image prototype results

| Colony | Ring aggregator | Continuous ring support | Absolute p95 | Raw peak |
|---|---|---:|---:|---:|
| R3C4 | equal-sector axial mean | 10/21 rings, 47.6% | 40.7 degrees | 46.7 degrees |
| R3C4 | equal-sector axial median | 10/21 rings, 47.6% | 37.2 degrees | 42.6 degrees |
| R4C6 | equal-sector axial mean | 13/14 rings, 92.9% | 18.0 degrees | 18.1 degrees |
| R4C6 | equal-sector axial median | 13/14 rings, 92.9% | 17.2 degrees | 17.3 degrees |

The eligible R3C4 ring summaries compound in the positive image-coordinate turning
direction until a near-tangential ring terminates the profile. R4C6 accumulates about 18
degrees in the negative direction, mainly in its inner measured rings, then flattens and
partially recovers in the outer rings. Mean and median agree closely for both colonies,
which suggests that the choice between these two ring aggregators is not driving the
result. It does not establish branch coherence: the plotted ring resultants show broad or
mixed sector-tilt distributions, particularly for several R3C4 rings.

### 12.4 Interpretation and limitations

This version is closer to the requested overall rotation phenotype than summing changes
of a global absolute fiber axis. It is invariant to global image rotation because the
input is radial-relative tilt. Equal-sector aggregation reduces direct sensitivity to
branch density, but eligibility remains threshold-sensitive and multiple branches within
one sector still share one coherence-weighted orientation cell.

The main limitation is loss of spatial correspondence. A positive sector on one ring and
a different positive sector on the next can contribute to one smooth colony-wide curve
without representing the same branch. For that reason this method should be interpreted
as colony-wide rotational tendency, not tracked branch rotation or a bifurcation angle.

## 13. Full-length cumulative axial-median change

### 13.1 Question answered

This diagnostic asks how the colony-wide equal-sector axial-median radial-relative
orientation changes from one ring to the next. It is an orientation-state calculation,
not a polar path predictor. It was added because R3C4 retained rich, locally coherent
orientation evidence after the tangent-based method reached an 87.4-degree ring median
and terminated at its 75-degree radial-integration guard.

The calculation uses the full detected-object radius rather than the symmetric-growth or
sparse-zone radius. The exclusive outer bound is the first complete 8 px ring boundary
beyond the farthest detected object pixel. The inoculum core remains excluded. Ring
medians retain the same equal-sector eligibility, three-sector minimum, and 0.15 axial
ring-resultant threshold as Section 12.

### 13.2 Calculation

For reliable adjacent ring medians `m_k`, accumulate only their seam-safe axial change:

```text
d_k = 0.5 * atan2(sin(2 * (m_k - m_(k-1))),
                  cos(2 * (m_k - m_(k-1))))
C_k = C_(k-1) + d_k
```

The first supported ring is zero. A missing ring consensus or an exactly 90-degree axial
change terminates continuous accumulation. There is no absolute-tilt cutoff and no
`tan(tilt)` term, so near-tangential ring medians remain finite. Raw orientation angles
are not summed. Consequently, subdividing an unchanged orientation interval does not
inflate the result, global image rotation cancels, and crossings of the axial
plus/minus-90-degree seam remain continuous. The unwrapped cumulative value may exceed
90 degrees; its sign assumes the true change between each adjacent ring is less than 90
degrees.

### 13.3 Real-image prototype results

| Colony | Continuous ring support | Absolute p95 | Raw peak |
|---|---:|---:|---:|
| R3C4 | 17/21 rings, 81.0% | 51.4 degrees | 60.5 degrees |
| R4C6 | 13/14 rings, 92.9% | 90.9 degrees | 92.8 degrees |

For R3C4, the profile continues through the 87.4- and 89.5-degree near-tangential ring
medians and remains defined through ring 16. It stops at ring 17 because the ring
resultant falls below 0.15, not because of tangency or symmetric-radius trimming.

For R4C6, the large positive cumulative change reflects a ring median that moves from
-51.3 degrees in the first measured ring to approximately +34 degrees in the outer
supported rings. The tangent-based radial-path reference remains near 17 degrees because
it answers a different geometric question. Neither result should be substituted for the
other without naming the definition.

### 13.4 Interpretation and limitations

This is the preferred colony-wide complement for capturing tangential orientation
changes. It retains equal-sector weighting, so adding more same-orientation pixels to an
already reliable sector does not directly increase the value. Support remains a separate
quantity and can still change with branch density near eligibility thresholds.

The result is not a tracked biological branch trajectory. An abrupt change can occur
when different sector populations dominate adjacent rings, even if no individual branch
turns by that amount. Use the sector-level tangential or Cartesian diagnostics when
spatial correspondence is required. Keep the tangent-based radial-path calculation as a
reference rather than combining its values with this orientation-state metric.

## 14. Skeleton-masked sampling diagnostic

### 14.1 Controlled comparison

This diagnostic tests whether branch width is materially affecting the full-length
cumulative axial-median result. The detected object mask is morphologically skeletonized
to a one-pixel centreline. The skeleton and detected-mask variants then use the same
image-derived local orientation field, coherence field, inoculum centre and exclusion,
8 px rings, 36 sectors, sector eligibility rules, ring-consensus threshold, and
cumulative axial-change calculation from Section 13. Only the pixels allowed to
contribute to a ring-sector mean change.

The skeleton-derived cumulative values are painted back over the reliable detected mask
in the comparison figure so the spatial result remains legible. A separate panel shows
the actual one-pixel measurement skeleton, widened by one pixel for display only.

### 14.2 Two-colony results

| Colony | Sampling | Reliable pixels | Continuous support | Median sector support | Median ring resultant | Absolute p95 | Raw peak |
|---|---|---:|---:|---:|---:|---:|---:|
| R3C4 | detected mask | 22,750 | 17/21 rings | 0.722 | 0.281 | 51.4 degrees | 60.5 degrees |
| R3C4 | skeleton | 7,559 | 19/21 rings | 0.667 | 0.324 | 40.1 degrees | 43.9 degrees |
| R4C6 | detected mask | 6,951 | 13/14 rings | 0.528 | 0.549 | 90.9 degrees | 92.8 degrees |
| R4C6 | skeleton | 2,265 | 13/14 rings | 0.472 | 0.566 | 72.2 degrees | 73.0 degrees |

On rings supported by both variants, the mean absolute cumulative-profile difference is
13.0 degrees for R3C4 and 17.1 degrees for R4C6. These are sample-specific results from
the cached branch-reconnection detector output; the complete values are exported in
`artifacts/twok_skeletonized_axial_change_summary.csv` and
`artifacts/twok_skeletonized_axial_change_profiles.csv`.

### 14.3 Interpretation

The result is mixed, not a demonstrated accuracy improvement. Skeletonization increases
R3C4's continuous ring support and modestly raises median ring resultant for both
colonies, but it lowers median sector coverage and substantially reduces the reported
cumulative magnitude. Without branch-trajectory ground truth, the lower peaks cannot be
classified as more accurate rather than attenuated.

Skeleton sampling is useful as a branch-width sensitivity diagnostic because a thick
branch no longer contributes across its full cross-section. It does not make the metric
invariant to branch number. Additional branches can still change the within-sector axial
mean, the sectors that pass support thresholds, and which sector population determines
the ring median. Skeleton topology is also detector- and morphology-dependent: connected
dense regions can create loops and short spurs that do not correspond one-to-one with
biological branches.

Keep the detected-mask result as the current primary prototype and show the skeleton
result alongside it as a sensitivity analysis. Promotion of skeleton sampling would
require manually annotated or synthetic branch trajectories with known radial-relative
turning, including controlled changes in branch width and count.

## 15. Continuity-aware axial lifting at tangency

### 15.1 Status and question

This section is a specification only. It does not change the current reference
calculation.

An axial orientation is periodic over 180 degrees. At exact tangency, +90 and -90
degrees therefore describe the same line. The current seam-safe axial difference already
handles a simple representation flip such as +89 to -89 degrees as a +2-degree change:

```text
d_axial(a, b) = 0.5 * atan2(sin(2 * (a - b)), cos(2 * (a - b)))
```

This is equivalent to the conservative one-dimensional period-pi behavior provided by
`numpy.unwrap(..., period=pi)`: it selects the period-complementary representation that
minimizes the adjacent jump. NumPy's implementation is suitable as an independent
reference, but it does not use a turning trend or choose among multiple ring modes.
[NumPy documents the period-aware rule here](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html).

R3C4 demonstrates a different ambiguity. Its skeleton ring median changes from 88.69 to
10.98 degrees. The same axial observation permits a -77.71-degree increment or a
+102.29-degree increment. Neither is determined by the two unoriented ring medians.
Choosing the latter because earlier rings turned positively would be a dynamical prior,
not additional image evidence.

### 15.2 Goals and non-goals

The proposed prototype will:

- preserve continuity through harmless +90/-90 representation changes;
- use the previous accepted change only when the current ring supports more than one
  nearly equivalent axial state;
- expose when continuity, rather than the best independent ring estimate, selected a
  state;
- return an explicit unsupported or ambiguous state when no defensible continuation
  exists;
- retain equal-sector weighting and separate support from orientation magnitude.

It will not infer axial polarity from a single ring, prove clockwise versus
counterclockwise biological growth, track a particular branch, bridge missing rings, or
estimate bifurcation angles. Those tasks require spatial correspondence.

### 15.3 Considered designs

#### Design A: principal period-pi unwrap

```text
wrapped ring median -> principal axial difference -> cumulative change
```

For each reliable ring median `m_k`:

```text
u_k = u_(k-1) + d_axial(m_k, u_(k-1))
```

This is the boring, reliable reference and matches the current calculation. It is
rotation invariant and seam safe, but assumes the true adjacent change has magnitude
below 90 degrees. It cannot distinguish a -77.71-degree change from its +102.29-degree
axial lift and cannot prevent a switch between different sector populations.

#### Design B: predictive lift of one ring median

```text
wrapped median -> candidate 180-degree lifts -> previous-step predictor -> chosen lift
```

Let `u_(k-1)` be the previous unwrapped state and
`v_(k-1) = u_(k-1) - u_(k-2)` its previous change. Predict:

```text
p_k = u_(k-1) + clip(v_(k-1), -60 degrees, +60 degrees)
U_k = {m_k + 180 degrees * n : n is an integer}
u_k = argmin over U_k of |u - p_k|
```

This can select a non-principal lift when a sustained trend crosses the axial seam. It
is simple and online, but it can turn an earlier error into a persistent trajectory and
can impose a direction unsupported by the current ring.

#### Design C: history-aware selection among admissible ring modes

```text
ring-sector tilts -> data-admissible axial modes -> continuity gate -> unwrapped state
                                      |                    |
                                      +-> support/QC ------+
```

Do not collapse a ring to one median before applying continuity. For every reliable
sector tilt `theta_(k,s)`, calculate its equal-sector axial data loss as a candidate
ring state:

```text
L_(k,j) = mean_s |d_axial(theta_(k,s), theta_(k,j))|
```

Retain distinct candidates whose loss is within 5 degrees of the minimum and which have
at least three reliable sectors within 15 degrees. Each retained axial candidate is
expanded to its nearby 180-degree lifts and compared with the predictor `p_k` defined in
Design B. Continuity may choose only among this data-admissible set; it may not rescue a
poorly supported orientation.

### 15.4 Recommended prototype

Use Design C, with Design A retained unchanged as the reference. Do not use a generic
two-dimensional phase-unwrapping routine. For example,
`skimage.restoration.unwrap_phase` is intended for spatial phase arrays and is not a
substitute for equal-sector ring-state selection.
[The scikit-image phase-unwrapping contract is documented here](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.unwrap_phase).

The initial deterministic rules are:

1. Require the existing three-sector minimum, sector-resultant eligibility, and ring
   resultant of at least 0.15 before constructing candidates.
2. Give every eligible angular sector one vote. Raw pixel count and detected branch
   count are not explicit optimization weights.
3. Use the independently best axial median for the first supported ring.
4. Use the principal axial step for the second ring because no previous velocity exists.
5. From the third ring onward, predict from the previous accepted change and choose the
   nearest data-admissible lifted candidate.
6. Activate history-based selection only when the previous or independently selected
   current state is within 15 degrees of tangency. Outside that gate, retain the
   independently selected median.
7. Require an accepted step no larger than 60 degrees and a prediction residual no
   larger than 30 degrees. These are prototype guards to be sensitivity-tested, not
   established biological limits.
8. If the best and second-best admissible continuations differ in prediction residual by
   less than 10 degrees, mark the ring `history_ambiguous`.
9. If no candidate passes the step and residual guards, mark it
   `history_discontinuity`. Do not emit zero and do not restart on later rings.
10. Flag every ring where history selects a state other than the independently best
    candidate as `continuity_tiebreak`.

The reported continuous state is:

```text
C_k = u_k - u_start
```

The prototype must export the wrapped independent median, chosen unwrapped state,
principal axial increment, chosen increment, candidate data loss, prediction residual,
continuation margin, status flag, sector support, and ring resultant. The overlay should
show the current reference and continuity-aware profile together, with ambiguous or
discontinuous rings left blank rather than colored as zero.

Under these conservative rules, the R3C4 88.69-to-10.98-degree transition is expected to
remain unsupported unless ring 11 contains a near-tangential alternative whose
equal-sector data loss is within the admissible 5-degree window. Previous state alone is
not sufficient evidence to force the +102.29-degree lift.

### 15.5 Required controls

1. +89 to -89 degrees unwraps to a +2-degree change; -89 to +89 unwraps to -2 degrees.
2. Global image rotation leaves radial-relative results unchanged.
3. Straight radial, horizontal, and vertical fields do not accumulate false rotation.
4. A synthetic smooth crossing through tangency follows the known unwrapped direction.
5. Reversing the synthetic turning direction reverses the reported sign.
6. Exact equidistance between two lifts returns `history_ambiguous`.
7. A strongly supported true reversal outside the tangency gate is preserved rather than
   smoothed away.
8. A mode outside the 5-degree data-loss window cannot be selected by history.
9. Duplicating pixels or branch width within already eligible sectors does not change the
   selected state.
10. Adding branches that occupy new sectors may change support or the ring state, and is
    reported as such rather than claimed to be branch-count invariant.
11. Missing or rejected rings terminate the profile without zero-filling or restart.
12. The existing principal axial-change output remains numerically unchanged.

Sensitivity figures must sweep the candidate-loss slack, maximum accepted step, and
prediction-residual guard. Promotion requires synthetic trajectories with known turns
and manually reviewed real paths through the tangential R3C4 region.

## 16. Point-level previous-ring orientation inheritance

### 16.1 Collection contract

This diagnostic replaces colony-wide ring medians and 10-degree sector cells with
literal crossings between the reliable one-pixel object skeleton and each full-length
Sholl circle. It is an orientation-collection prototype, not a colony phenotype.

For every 8 px ring centre, skeleton pixels within 1.5 px of the mathematical circle are
grouped by 8-connectivity. Each connected crossing stores its coherence-weighted
coordinate, absolute fiber-axis orientation, radial-relative tilt, coherence, axial
resultant, and contributing skeleton-pixel count. The inoculum exclusion and coherence
threshold remain active. Crossings are never discarded merely because they cannot be
matched; all raw records are exported.

An outer crossing may inherit only from a crossing on the immediately preceding ring.
Both crossings must:

- be connected through the reliable skeleton inside the intervening annular corridor;
- lie within `ring_width / cos(75 degrees)`, which is 30.91 px for 8 px rings;
- differ by no more than 20 degrees under the seam-safe axial difference;
- admit a 180-degree lift whose step is at most 60 degrees and whose residual from the
  previous accepted step is at most 30 degrees.

The common candidate cost is:

```text
cost = (distance / maximum_distance)^2
     + (absolute_axial_difference / 20 degrees)^2
```

A local choice is rejected when the second-best cost lies within 0.05 of the best cost.
Only finite parent accumulation may propagate. There is no gap bridge, late seed, or
restart. A missing or rejected intermediate crossing therefore terminates that path.

For an accepted parent `i` and child `o`, the raw state retains both:

```text
signed_o   = signed_i   + chosen_unwrapped_step(i, o)
absolute_o = absolute_i + abs(chosen_unwrapped_step(i, o))
```

No colony-level aggregation is defined yet.

### 16.2 Matching policies

All policies operate on the identical hard-gated candidate graph:

- **Reciprocal one-to-one:** accept an edge only when each endpoint is the other's unique
  decisive best candidate. This is the conservative primary comparison.
- **Independent many-to-one:** every outer crossing accepts its own decisive best parent;
  several children may inherit from the same parent. This preserves branch splits but
  can also duplicate one inner orientation into nearby parallel children.
- **Global one-to-one:** solve one minimum-cost assignment per adjacent ring pair using a
  private unmatched option for every outer point. A valid but poor edge is not forced,
  and assignments with a nearly equal alternative total cost are rejected.

The current sector-based matcher remains unchanged as a reference. It is not used to
construct these point records.

### 16.3 Real-image collection results

| Colony | Policy | Raw crossings | Supported points | Accepted edges | Raw cumulative peak |
|---|---|---:|---:|---:|---:|
| R3C4 | reciprocal one-to-one | 602 | 53 | 22 | 25.3 degrees |
| R3C4 | independent many-to-one | 602 | 123 | 92 | 45.1 degrees |
| R3C4 | global one-to-one | 602 | 57 | 26 | 19.6 degrees |
| R4C6 | reciprocal one-to-one | 200 | 45 | 17 | 18.2 degrees |
| R4C6 | independent many-to-one | 200 | 62 | 34 | 30.5 degrees |
| R4C6 | global one-to-one | 200 | 45 | 17 | 18.2 degrees |

The point collector retains crossings through the full detected length: R3C4 has raw
crossings through ring 20 and R4C6 through ring 12. Strict accepted inheritance reaches
ring 6 for reciprocal and global matching and ring 9 for many-to-one matching in R3C4;
it reaches ring 5 for one-to-one and many-to-one matching in R4C6. Later raw crossings
remain in the export with explicit unsupported states rather than being reset to zero.

The point prototype does not reproduce or directly evaluate the former R3C4
88.69-to-10.98-degree colony-median transition. Its strict inherited point paths end by
ring 6 for one-to-one matching and ring 9 for many-to-one matching, before the former
ring-10-to-ring-11 transition. The 20-degree orientation-sharing gate applies to the
axes at candidate point endpoints, not to colony-wide ring medians. Continuing through
that region would require endpoint-level evidence on every intervening ring.

The annular skeleton-connectivity requirement is essential. Angle and Euclidean distance
alone produced visually plausible but unsupported chords between unrelated branches in
the first diagnostic run. Connectivity removed those edges but reduced propagation
support substantially. The present figures should therefore be read as an audit of what
the strict collection rules accept, not as a completed outward-turning measurement.

### 16.4 Preserved diagnostics and next checks

The raw crossing CSV stores local orientation evidence independently of matching policy.
The state CSV stores parent ID, status, distance, axial change, prediction residual,
unwrapped orientation, signed step, signed accumulation, absolute accumulation, and
normalized cost for every policy and every crossing.

Before selecting a policy or calculating one colony statistic, compare:

1. axial gates of 10, 20, and 30 degrees;
2. spatial gates of 1.5 and 2 ring widths against the 75-degree geometric reach;
3. crossing half-widths of 1.0, 1.5, and 2.0 px;
4. strict corridor connectivity against a one-ring skeleton geodesic limit;
5. reciprocal, many-to-one, and global policy stability on known synthetic branches;
6. width and uniformly replicated branch-count controls.

Any final aggregation must report correspondence support separately and must not convert
unmatched crossings into zero rotation.

## 17. Coherence-enhancing diffusion and equal-crossing population trend

### 17.1 Controlled preprocessing comparison

The literal skeleton-ring collector was rerun after applying `StructureSmoothing` to the
orientation source with:

```text
num_iter=30, sigma=1.5 px, rho=3.0 px, dt=0.1, alpha=0.001, C=90
```

These are prototype parameters, not established biological thresholds. The TwoK object
map, inferred inoculum center, distance map, 8 px ring positions, inoculum exclusion, and
point-matching guards are taken from the original condition and mechanically reused for
CED. A mismatch in the object mask, center, or distance map raises an error. CED therefore
changes the intensity-derived orientation and coherence fields, not the detected colony
geometry.

| Colony | Condition | Crossings | Median coherence | Many-to-one support | Many-to-one raw peak |
|---|---|---:|---:|---:|---:|
| R3C4 | original | 602 | 0.471 | 123 | 45.1 degrees |
| R3C4 | CED | 610 | 0.552 | 143 | 38.1 degrees |
| R4C6 | original | 200 | 0.541 | 62 | 30.5 degrees |
| R4C6 | CED | 202 | 0.687 | 60 | 25.8 degrees |

CED increased median local coherence in both colonies while leaving the number of
literal crossings nearly unchanged. The inherited many-to-one raw peak decreased in
both colonies. These observations show that the preprocessing materially changes the
orientation evidence; they do not independently establish that every changed angle is
closer to biological ground truth.

### 17.2 Branch-tracking-free population calculation

For each ring, every literal crossing contributes one radial-relative axial tilt. The
ring population consensus is the equal-crossing doubled-angle mean:

```text
ring_angle = 0.5 * atan2(mean(sin(2 * tilt)), mean(cos(2 * tilt)))
ring_resultant = hypot(mean(cos(2 * tilt)), mean(sin(2 * tilt)))
```

A ring requires at least three crossings and a resultant of at least 0.15. Consecutive
supported ring angles accumulate seam-safe period-180-degree changes within each
contiguous run. A missing ring starts a new zero-relative run, and an exact 90-degree
inter-ring step is directionally ambiguous, remains unsupported, and also breaks the
run. This calculation does not match or directly follow individual branches.
Conditional on both rings passing the minimum-crossing guard, uniformly replicating the
same crossing-orientation distribution leaves its consensus unchanged. Support itself is
count-sensitive, and uneven skeleton fragmentation can still change the empirical
distribution.

R4C6 shows a similar outward population profile under both conditions: the raw peak is
82.7 degrees in the original field and 80.4 degrees after CED, with 12 of 14 rings
supported. That full peak is sensitive to the first two rings, whose resultants are low
(0.24 and 0.26 originally; 0.16 and 0.17 after CED). From the first clearly coherent
ring in this example (ring 2, resultant about 0.77) to the outermost supported ring, the
observed shift is about 43 degrees in both conditions. The outermost supported-ring
resultant changes from 0.775 to 0.873, while several other outer resultants are similar
or slightly lower after CED. This persistence is evidence that the later-radius trend is
not caused solely by dotted intensity, but it is not yet a validated biological effect
size. R3C4 remains heterogeneous and has low-resultant intervals. After changing the
calculation to restart after unsupported gaps rather than carrying an unidentified lift
across them, its largest within-run population change is 59.8 degrees originally and
61.0 degrees after the first CED setting. CED does not resolve the low-resultant
intervals or make the separate supported runs comparable.

The population raw peak remains a diagnostic, not a selected phenotype. A production
metric should retain the signed radial profile and ring resultant, and should report the
fraction of supported rings separately.

### 17.3 CED parameter sweep

Twenty-six CED configurations were evaluated on both colonies. The coarse grid varied:

- `sigma`: 0.75, 1.5, and 2.5 px;
- `rho`: 1x and 2x `sigma`;
- `num_iter`: 15 and 30;
- `C`: 80 and 95;

The library default and the first diagnostic setting were added as explicit controls;
`dt=0.1` and `alpha=0.001` remained fixed. Parameter selection did not use the observed
rotation amplitude. It compared median crossing coherence, the 90th percentile of
seam-safe neighboring orientation changes along non-junction skeleton interiors,
crossing-count preservation, reliable-skeleton preservation, and normalized source
RMSE.

No tested CED configuration reduced the mean branch-interior P90 angular roughness
across the two colonies. Changes ranged from a 0.4% increase to a 10.3% increase.
Consequently, increased structure-tensor coherence cannot be interpreted as evidence
that CED corrected the dotted-intensity angle error. Strong low-`C` settings provided
the largest coherence gains, but also changed local angles and source intensity more.

For a conservative visualization, CED24 was selected by two prespecified guards:

```text
mean P90 roughness degradation <= 1%
worst crossing-count deviation <= 2%
```

Among configurations passing both guards, CED24 had the largest mean coherence gain.
Its parameters are `sigma=2.5`, `rho=5.0`, `num_iter=30`, `C=95`, `dt=0.1`, and
`alpha=0.001`. Across the colonies it raises median crossing coherence by 13.7%, changes
P90 branch-interior roughness by -0.4% under the reduction convention (a 0.4% worsening),
has at most 1.5% crossing-count deviation, and has mean normalized source RMSE 0.073.
This is a sweep-selected conservative display setting, not a validated optimum.

### 17.4 Outward-normalized arrow overlay

The local structure-tensor orientation is axial: `theta` and `theta + 180 degrees` are
the same observation, so the image does not measure a head-to-tail direction. For the
literal-crossing overlay, each axis is represented by the equivalent arrow whose dot
product with the center-to-crossing radial vector is nonnegative. The arrowhead therefore
points toward increasing radius by construction. It helps show whether the local axis
leans clockwise or counterclockwise as it leaves the inoculum, but it must not be read as
measured growth polarity or material flow. Color continues to encode signed
radial-relative tilt.

The focused outward-orientation diagnostic contains only the literal-crossing arrows,
the equal-crossing ring consensus, its resultant, and contiguous-run consensus change.
It does not infer branch correspondence. Each crossing contributes one vote within its
ring. Conditional on eligibility, uniformly replicating the same orientation distribution
leaves the consensus unchanged. The minimum-crossing support guard remains count-sensitive,
and uneven skeleton fragmentation can still alter the sampled distribution.

### 17.5 Public SDK helper boundary

The approved literal-crossing calculation is available from
`phenotypic.sdk_.orientation_fields`. `literal_skeleton_ring_crossings` performs the
skeleton-ring transform, and `literal_crossing_ring_profile` calculates the
equal-crossing ring consensus and contiguous outward change. Both functions accept
explicit arrays and geometry; neither performs CED, detects an object, or infers an
inoculum center.

Each diagnostic is a separate, composable function:

- `plot_literal_crossing_map` draws the source-array overlay, accepted skeleton,
  sampled rings, and outward-normalized local arrows;
- `plot_literal_crossing_population` draws every local crossing tilt and the black
  equal-crossing ring consensus;
- `plot_literal_crossing_outward_profile` draws contiguous-run consensus change and
  ring resultant on separate axes.

Plot functions accept a caller-owned Matplotlib `Axes` and return their principal
artist. They express values in degrees for interpretation while transform and profile
arrays remain in radians for calculation. The real-colony CED diagnostic now calls
these public helpers rather than maintaining a second implementation.

The public local-tilt plots default to the cyclic `twilight_shifted` colormap so the
equivalent -90-degree and +90-degree axial seam has matching endpoint colors. The saved
session diagnostics retain the explicitly requested `Spectral` map, which is
non-cyclic and therefore has a visible color discontinuity at that seam. In either map,
exact tangency has no defensible arrow polarity; outward arrowheads are a display
convention rather than a measured direction.
