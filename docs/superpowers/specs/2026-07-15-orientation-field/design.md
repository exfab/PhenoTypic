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
| R3C4 | equal-sector axial mean | 66.7% | 40.7 degrees | 46.7 degrees |
| R3C4 | equal-sector axial median | 66.7% | 37.2 degrees | 42.6 degrees |
| R4C6 | equal-sector axial mean | 100.0% | 18.0 degrees | 18.1 degrees |
| R4C6 | equal-sector axial median | 100.0% | 17.3 degrees | 17.3 degrees |

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
