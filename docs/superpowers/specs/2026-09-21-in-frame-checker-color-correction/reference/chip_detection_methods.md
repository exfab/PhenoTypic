# Automatic colour-chip detection for the SnP plate rig

Four plate frames (`d000220_300_021`, `_022`, `_038`, `d000320_300_003`), two
imaging sessions (Sept 2025, Feb 2026). `d000320_300_003` was rendered from CR3
with `rawtherapee-cli` and `rawtherapee_profiles/InvertLinearLMMSE.pp3`; the
other three used the pre-existing `.tiff` renders.

## 1. What the detector is up against

Each frame carries one 2×6 ColorChecker half-card at each edge. The card is
mounted rotated, so the chart's six columns run down the image and its four
rows are split two per card:

| band | inner column (toward plate) | outer column (toward frame edge) |
|---|---|---|
| left  | chart row 1 (`A1…F1`) | chart row 2 (`A2…F2`) |
| right | chart row 4, neutrals (`A4…F4`) | chart row 3 (`A3…F3`) |

This mapping was not assumed — it was recovered independently by matching
measured tile colour to the reference chart (§4, M4), which reproduces it on
every band up to at most one swapped pair of near-identical patches.

Three properties make naive segmentation fail:

1. **Both outer columns are clipped by the frame border.** There is no visible
   tile centre to lock onto, and the old pipeline's `center_and_pad_checker`
   reflect-padding exists to invent one. That padding is the source of the
   smeared reconstruction — it fabricates the majority of every rebuilt tile.
2. **The transparent plate wall casts refracted ghost copies of the neutral
   column** into the plate region of the right band, at x ≈ 100–150. A blind
   segmenter could take them for the real neutrals at x ≈ 165–207.
3. **The plate lies inboard of both cards and obstructs neither.** Measured
   from the per-column vertical variance, the patch-bearing zones are left
   x 0–76 and 143–203 and right x 152–207 and 282–340, while the plate
   structure occupies left x > 203 and right x < 152. Horizontal L\* profiles
   across the inner-column patches are flat with clean edges and no veil
   step. An earlier draft of this note described the inner columns as
   "wall-hazed"; that was wrong — see §4.

Tiles are ≈195 × 65 px on a 255.5 px row pitch, identical on both sides: the
row phases fitted independently on the left and right bands agree to 0.3 px.

## 2. How much does the card actually move?

Measured against the `_021` reference band by two independent estimators
(cross-correlation and OpenCV ECC), which agree to ≈1 px (`rig_displacement.csv`):

* within the Sept-2025 session: **≤ 4.4 px**
* Sept-2025 → Feb-2026: **≤ 9.7 px**
* rotation: **≤ 0.084°** — negligible over a 1650 px card

So the rig is stable, and detection is posed as constrained refinement of a
stored prior (`rig_prior.json`: per side, two columns with an x-window, a row
start, pitch 255.5 px and a duty fraction) rather than a global search.

## 3. The detectors

All share the prior and return a refined lattice; they differ in the refinement.

**M0 frozen** — the prior, unrefined. The null model, and the honest baseline:
at the rig's real stability it is not obviously wrong.

**M1 translation** — subpixel cross-correlation of the band against a stored
reference band, then a rigid shift of the prior. Note: phase-whitened
correlation (`normalization="phase"`) recovers synthetic shifts exactly but
collapses the horizontal estimate to zero on real cross-session pairs, where
the illumination differs; plain cross-correlation is used instead.

**M1 euclidean** — the same idea with rotation, via OpenCV ECC
(`MOTION_EUCLIDEAN`). Returns the correlation coefficient as a confidence.

**M2 profile** — reference-free. Re-snaps each column's extent from the
per-column Lab standard deviation inside a window around the prior, then
re-fits the 6-period row lattice as a square wave of the prior's pitch.

**M2 rigid** — reference-free, and the best of them. Measures the horizontal
shift from the *unclipped* inner column only, then applies it rigidly to the
whole lattice keeping the prior's tile widths, with the row phase taken from
whichever column fits at higher contrast. This is what lifts the reference-free
horizontal range from −10…+6 px to ±30 px.

**M3 blob** — segments patch cores natively (median filter → local Lab variance
→ threshold at a quantile → close → fill → erode → label) and fits the lattice
to the surviving centroids by translation RANSAC seeded at the prior.
Thresholds must be quantiles, not absolute: the Lab noise level differs 2.5×
between sessions (band median local σ 4.3 in Sept vs 1.9 in Feb). Recovers
6–9 of 12 tiles per side; the rest come from the fitted lattice. A single
global x-offset does not fit — the clipped outer column's visible centroid
sits inboard of its nominal one — so offsets are estimated per column.

**M4 identity** — not a locator but a verification gate. Chromaticity (r, g)
plus luminance relative to the side's brightest tile is invariant to a global
gain, which is what separates these frames from the reference chart; tiles are
matched to the side's twelve expected patches with the Hungarian algorithm.
Both sides of the comparison are in linear RGB — the feature is only
gain-invariant within one encoding, so the reference Lab values are converted
to sRGB and then linearised before matching.

**M5 off-the-shelf** — `colour-checker-detection` 0.2.3,
`detect_colour_checkers_segmentation`.

## 4. Results

| method | capture range Δy (px) | capture range Δx (px) | needs a reference image | median s/band | impurity |
|---|---|---|---|---|---|
| M0 frozen | 0 | 0 | no | 0.000 | 1.46 % |
| M1 translation | −60 … +60 | −6 … +3 | yes | 0.011 | 1.25 % |
| **M1 euclidean** | **−60 … +60** | **−40 … +40** | yes | 0.061 | 1.14 % |
| M2 profile | −30 … +30 | −10 … +6 | no | 0.262 | 1.26 % |
| **M2 rigid** | **−45 … +45** | **−30 … +30** | **no** | 0.357 | 1.13 % |
| M3 blob | −45 … +45 | −10 … +6 | no | 1.470 | 1.38 % |
| M23 cascade | −45 … +45 | −6 … +6 | no | 1.825 | 0.96 % |

The five original methods give mean ΔE₂₀₀₀ after a 24-patch fit of 5.33–5.38,
i.e. they differ by less than 0.1 — detection is not the accuracy bottleneck.

*Capture range* = the largest displacement, applied synthetically to the band,
whose worst-case recovery error over all eight bands stays under 3 px
(`capture_range.csv`). ±60 px is the limit of the sweep, not of the method.
*Impurity* = fraction of core-box pixels more than 6 ΔE₂₀₀₀ from their tile
median, measured on a median-filtered band so the number reflects spatial
contamination rather than sensor noise. Every method scored 10–12 of 12 on the
M4 identity check across the eight real bands (mean 11.5, identical for all
five): where it is not 12, a single adjacent pair of genuinely similar patches
is swapped — purplish blue `B2` against blue flower `E1` on the left card, and
the two mid-grey neutrals `D4`/`E4` on the right. The check therefore detects a
missing, flipped or wrong card, not individual tile errors.

**M5 found zero checkers in all 28 configurations tried** — each band alone,
the two half-cards naively concatenated (what the current pipeline builds),
that composite rotated 90°, and rotated-plus-stretched to a chart-like aspect,
each under both the Classic and Nano presets. The library looks for one
contiguous 4×6 chart; this rig does not contain one.

### What separates the methods

* **Rows are easy, columns are not.** On real frames all four refinements place
  rows within 1.3 px of each other but columns differ by up to 14 px
  (`method_agreement.csv`) — about a fifth of a tile width. The disagreement is
  genuine: the outer columns are clipped, so they have no visible centre to
  agree on, and each method resolves that differently.
* **Horizontal placement is the discriminating axis.** ECC holds it to ≤0.7 px
  at 40 px displacement; `M2_rigid` reaches ±30 px without any reference
  image; plain cross-correlation drifts 3–5 px, and `M2_profile` and `M3_blob`
  give out at −10…+6 px. Vertically everything is comfortable.
* **The reference-free limit was a modelling choice, not a search-window
  one.** `M2_profile` re-snaps each column independently, which makes it lag
  on the clipped outer column: as the card moves, that column's *visible*
  extent grows or shrinks, so its measured centre travels about half as far as
  the card. The unclipped inner column tracks displacement almost exactly
  (−24.5 px recovered against −25 applied, +26.0 against +25). `M2_rigid`
  measures the shift there and applies it rigidly, which is what buys the
  ±30 px. Chaining M3 into M2_rigid (`M23_cascade`) was worse than `M2_rigid`
  alone — the coarse stage injects error — so it is not recommended.
* **At the rig's real stability the choice barely matters.** With a 40 % core
  inset, even the frozen prior matches the refinements tile for tile on the
  identity check (mean 11.5 of 12) and reaches 1.5 % impurity. The
  refinements earn their place as insurance against a rig that gets bumped, not
  as a fix for present error.

### Retracted: "clearance from the plate"

An earlier version of this note reported minimum clearance against a fixed
boundary at band x = 203 (left) and 152 (right), and concluded that
`M2_profile` and `M3_blob` pushed boxes across it. **That is withdrawn.**
Those numbers came from the patch-bearing zone of the *September-2025*
reference frame and were then applied unchanged to frames whose cards have
moved. There is no fixed boundary: the card translates a few pixels between
frames, and its patch edge translates with it.

Re-derived per frame from each frame's own variance profile, the left inner
column's patch edge sits at 208–219 px on the subset (median 216), a median
+13 px from the frozen 203, tracking the +9 px card displacement. Against each
frame's own edge, **no method crosses on the plate side** — the side the
frozen line was supposed to protect. (`M3_blob` does cross on the *opposite*,
card-border side; see §8.) On the Feb-2026
calibration frame — the only one of the four far enough from the prior to
produce negatives — every method is fully contained, `M2_profile` by +7.2 px
and `M3_blob` by +3.6 px, where the frozen line had scored them −2.8 and −6.4.

The effect of the error was to penalise methods for correctly following the
card while the frozen prior, which does not move, scored a clean sheet. What
survives is only a margin ordering, not a pass/fail (see §8).

### Retracted: the "wall haze" attribution

An earlier version of this note attributed the colour residual to the plate
wall hazing the inner columns. **That attribution is withdrawn.** The two
groups I compared — inner columns (chart rows 1, 4) against outer columns
(rows 2, 3) — differ in two ways at once: column position *and* chart gamut,
since rows 1 and 4 are the six neutrals plus the six muted naturals while
rows 2 and 3 are the twelve saturated patches.

The control separates them. The calibration capture
`FlatField_Tv60_Av56_ISO800_EM_ColorChip.tif` is the full 24-patch chart,
centred on the rig, with no plate anywhere in the frame. Grouping *its*
patches the same way gives a muted/saturated gain ratio of
**[0.786, 1.222, 1.315], divergence 0.238**, against the plate frames'
0.227–0.243 with ratios 0.711–0.777 / 1.111–1.184 / 1.298–1.331. The
divergence and the blue ratio fall inside the plate-frame range; the red and
green ratios sit *just outside* it, 0.786 against a maximum of 0.777 and
1.222 against 1.184 — that is, the no-plate control is marginally **more**
extreme than any frame with a plate in it, which is the opposite of what a
plate-induced haze would produce. The effect is present, at full strength,
with no plate in the picture, so it is a property of the chart grouping and
of fitting a root-polynomial across a split gamut, not of the optics
(`chart_group_gain_control.csv`).

Two consequences: the `wall_haze_ratio` function has been removed from the
module rather than renamed, because its only demonstrated content is an
artifact of my own grouping; and the earlier proposal to **fit on rows 2 and
3 only is withdrawn** — it rested on the haze claim, and dropping the six
neutrals from a colour-correction fit would be actively harmful.

What survives is the weaker, encoding-independent statement below: the
residual concentrates on the muted half of the chart, which is what fitting
13 root-polynomial terms across a split gamut does, with `A4` (white) at
22–25 ΔE as the most sensitive single patch.

### The detection method is not what limits colour accuracy

Fitting a 24-patch degree-3 RPCC profile per frame gives mean ΔE₂₀₀₀ of 4.8–5.9
and the five detectors differ by <0.1 (`downstream_rpcc.csv`). The residual is
concentrated in the muted half of the chart (`rpcc_per_patch_residual.csv`):
`A4` (white) reaches 22–25 ΔE while the clear columns sit near 0–4.

A degree-2 cross-prediction makes this unambiguous (`rpcc_wall_crossfit.csv`;
6 terms, 12 patches per fit, so the fit is not degenerate):

| fit on | predicts saturated tiles | predicts muted tiles |
|---|---|---|
| clear columns (rows 2, 3) | 3.0 – 5.3 | 12.7 – 18.2 |
| muted rows 1, 4 | 26.3 – 45.8 | 5.6 – 7.2 |

A root-polynomial model fitted on half the chart extrapolates poorly to the
other half, and much worse in the direction that trains on neutrals plus muted
naturals and predicts saturated colour. That is a gamut-coverage result about
the model, and — per the control above — it appears with or without a plate in
the frame. It says nothing about the plate and is not a reason to drop any
patches from the fit.

## 5. QC gate for unattended batches

`qc_record()` accepts a band only if all four signals agree: displacement from
the prior < 30 px, ECC correlation > 0.90, spread between the four methods'
estimates < 8 px, and ≥ 10 of 12 colour identity. All eight real bands pass.
Injected faults each trip at least two signals — card flipped (0/12 identity,
ECC 0.78, 15.9 px spread), card blanked (1/12, ECC 0.30, 51.7 px spread), and
60 px and 120 px displacements (displacement plus spread) — while a 2.5×
exposure change correctly passes (`qc_gate.csv`).

## 6. Recommendation for scaling

**If a reference band is acceptable:** run **M1 euclidean** as primary with
**M2 rigid** as the reference-free cross-check, gated by `qc_record`. ECC is
0.06 s per band, has the widest capture range on both axes, and reports a
usable confidence; the cross-check cannot fail the way a stale reference band
makes ECC fail confidently, so a disagreement between the two is informative
rather than silent.

**If you would rather not maintain a reference band at all:** run **M2 rigid**
alone. At ±45 px vertically and ±30 px horizontally it covers three to four
times the largest displacement this rig has produced (9.7 px), costs 0.36 s per
band, matches ECC on mask impurity (1.13 % against 1.14 %) and on the identity
check, and depends on nothing but the stored lattice. The QC gate still works —
drop the method-spread signal to the M2-rigid-versus-M3 pair, or keep all three
reference-free methods and compare those. Its one structural weakness is that
it takes the horizontal shift from the inner column alone, so a defect confined
to that column would go unnoticed; the identity check is the backstop.

`M3_blob` costs 4× M2 rigid for a worse horizontal range, and `M23_cascade` is
worse than `M2_rigid` alone. Neither belongs in production.

The rig change that would matter more than any detector choice is moving the
cards far enough inboard that the outer columns are not clipped — that is what
creates the horizontal ambiguity all of these methods are working around.

## 7. Caveats

* Capture range was probed by displacing the band with spline interpolation and
  edge replication (`scipy.ndimage.shift(mode="nearest")`). Displacements beyond
  ~20 px therefore replicate an edge strip, which falls on the already-clipped
  outer column; the x-axis numbers are slightly pessimistic for that reason.
* The prior is derived from `d000220_300_021`. A rig teardown and remount would
  need it rebuilt (`M2_profile` output on a new reference frame is enough).
* Only four frames exist, all from one rig, so the real-displacement statistics
  are an existence proof of stability, not a distribution.
* The RPCC condition numbers reported here are of the root-polynomial design
  matrix as constructed in this analysis and are not comparable to the
  condition numbers printed by the `ColorCheckerProfile` fitter.


## 8. Validation on the 20-frame representative subset

`rhodotorula-subset` holds 20 plate frames from 8 experiment runs (Cd, Cu, Fe
x2, Pb, Zn, Salt, pH), February–August 2026. None was used to build anything
here: the prior and the reference band both come from `d000220_300_021`,
September 2025. The subset frames are also 4012x6016, eight pixels smaller
than the frames the prior was built on.

**The prior transfers.** Every band of every frame is found. The subset sits a
consistent **+9 px in x and about −5 px in y** from the 2025 prior — well
inside every method's capture range except the frozen prior's. Colour identity
averages 11.85 of 12 under ECC across the 40 bands.

**Box placement, measured against each frame's own patch extent.** No method
reaches the plate-side patch edge on any band. On the card-border side
`M3_blob` does cross, by up to 1.4 px, leaving 35 % of its boxes not wholly
inside the patch; every other method stays inside on both sides. Margins on
the left inner column
(120 boxes per method), inboard / outboard, and the share of boxes lying
wholly inside the patch:

| method | inboard | outboard | fully inside |
|---|---|---|---|
| M0 frozen | +2.6 | +20.6 | 100 % |
| M1 translation | +11.6 | +11.6 | 100 % |
| M1 euclidean | +11.6 | +11.6 | 100 % |
| M2 profile | +10.0 | +2.2 | 100 % |
| M2 rigid | +13.6 | +7.6 | 100 % |
| **M3 blob** | **−1.4** | **+0.6** | **65 %** |

The two registration methods are the best centred; `M3_blob` is the only one
that actually loses containment. The frozen prior's large outboard margin is
not a virtue — it comes with only 2.6 px on the inboard side, because it does
not follow the card's +9 px shift.

**`M3_blob`'s horizontal estimate collapses on half the frames.** Its left-band
dx reads ≈ −2 px against a +9 px consensus on 9 of 20 frames, while its row
placement stays correct to 0.7 px. The cause is countable: on those frames the
segmenter finds **zero** blobs in the unclipped inner column (all 5–6 sit in
the clipped outer column, whose centroid barely moves with the card), against
at least one inner-column blob on every frame where it succeeds. Its
horizontal accuracy therefore rests on a single low-contrast blob.

**The eight-pixel crop separates the two families.** On the right band ECC
reports ~+7.7 px where `M2_rigid` reports ~+3.5. ECC registers the whole band
pattern, which includes the plate — fixed in absolute coordinates, so it moves
in band coordinates when the frame is cropped differently — while `M2_rigid`
tracks the card, which sits a fixed distance from the frame edge. For placing
tile boxes the card is the right target, so the reference-free method is
arguably the more correct of the two here.

**The criterion that matters: does the box include covered pixels?** There is
no fixed boundary to test against — the card shifts a few pixels per frame — so
this is measured per frame and per tile. A pixel counts as covered if its
median-filtered colour departs by more than 6 ΔE₂₀₀₀ from the median of the
**interior** of that same tile (central 40 % in x, middle 50 % in y).

*The reference must be the tile interior.* An earlier version of this analysis
took it from the outer 30 % of the detected span, which on the left band is the
dark card border (L\* ≈ 25 against a patch body of L\* ≈ 49). That 6 L\* offset
put the entire patch body just over the threshold and reported 52 % of a clean
tile as covered, with a ranking inversion that made the frozen prior look best.
Both were artefacts. With the interior reference the same tile reads flat at
0.5–0.8 ΔE across its body and the real departure begins at band x ≈ 215.

**On the left band, no box touches the plate-side veil at all** — plate-side
contamination is 0.000 for every method on all 720 box-tile combinations. The
only contamination is on the opposite side, the dark card border, and only for
the methods that fail to follow the card's +9 px shift
(`veil_left_by_side.csv`):

| method | card-border side, mean / max | plate side, max |
|---|---|---|
| M2 rigid | 0.0000 / 0.0000 | 0.000 |
| M1 euclidean | 0.0001 / 0.0055 | 0.000 |
| M1 translation | 0.0001 / 0.0055 | 0.000 |
| M2 profile | 0.0006 / 0.0070 | 0.000 |
| M0 frozen | 0.0553 / 0.3475 | 0.000 |
| M3 blob | 0.0851 / 0.6104 | 0.000 |

On the right band's neutral column, over the 19 unobstructed frames, the same
measure gives M2 rigid 0.6 % mean (9.5 % max) as the best and M3 blob 2.4 %
(49 %) as the worst, with the three registration methods between at 1.2–1.3 %
(`veil_right_neutral.csv`).

**One real obstruction event.** In `CadmiumArrayRun/d000375_300_020`, a bright
object sweeps diagonally across the right card's neutral column. No method
resists it — the patches are covered, so every box measures the occluder
(impurity 16.7–28.3 %). What differs is whether the method *says* so:

| method | confidence on the occluded band | over the 39 clean bands | informative? |
|---|---|---|---|
| M1 translation | 0.280 | 0.494 – 0.697 | yes, well below the floor |
| M1 euclidean | 0.824 | 0.969 – 0.989 | yes, well below the floor |
| M2 profile | −7.5 (and identity 7/12) | 5.2 – 19.7 | yes, sign flips |
| M2 rigid | 35.1 | 25.4 – 36.0 | **no — inside the clean range** |
| M3 blob | 0.417 | 0.417 – 0.750 | **no — equals the clean minimum** |

`M2_rigid` holds its position correctly but its confidence on the occluded
band is indistinguishable from a clean one — it sits near the top of the clean
range, not below it. So the primary detector's own confidence is not a usable
occlusion signal in reference-free mode; the band is caught instead by the
other three gate signals, which all fire on it (panel spread 22.2 px against a
12 px limit, identity 8/12, impurity 22.9 %).

**Which is why the gate now carries tile impurity.** On the occluded band it
reads 18.9 % (ECC) and 22.9 % (`M2_rigid`) against ≤1.13 % on all 39 other
bands — a 17–20x margin, needing no reference image. A 5 % threshold separates
them under every recommended method (`M3_blob` reaches 6.1 % on a clean band,
one more reason not to use it).

**Re-tuned gate.** The old gate flagged 16 of 40 clean bands, because it polled
`M2_profile` and `M3_blob`, whose column estimates are erratic. Restricted to
the production panel, the spread is ≤5.8 px on clean bands and 10.4 px on the
occluded one. With that change plus the impurity signal, **39 of 40 bands pass
and the occluded band is caught by three independent signals in both modes**
(`subset_qc_tuned.csv`):

* reference mode — 10.4 px spread, ECC 0.82, impurity 18.9 %
* reference-free mode — 22.2 px spread, identity 8/12, impurity 22.9 %


## 9. Colour-agnostic detection, and identity as a separate stage

The pipeline is three stages and they must not borrow from each other:

1. **Localisation — colour-agnostic.** `M1_*` register on luminance and edge
   structure, `M2_*` on variance profiles, `M3_blob` on border-fill
   segmentation. None reads patch colour, so a rotated or swapped card is
   still found. (`M1_*` do depend on a reference *band*, so a rotated card
   would break the template; the reference-free `M2_rigid` does not.)
2. **Identity — solved, not assumed.** `assign_identity` enumerates the eight
   discrete ways a 2x6 half-card can sit (which chart rows, which column
   order, and the 180-degree flip that reverses A-F) and scores each one
   directly: under a hypothesis, lattice position k must be `ids[k]`. An
   earlier version solved a free Hungarian assignment instead; that lets
   labels wander onto whichever reference fits best and it mislabelled the
   occluded frame outright. Direct scoring fixed it.
3. **Recovery** of occluded tiles may use the identity from stage 2, but the
   occluded tiles themselves must not vote on it (`clean=` mask).

**Reference values come from colour-science, chromatically adapted.**
`reference_linear_rgb()` reads `ColorChecker24 - After November 2014` and
Bradford-adapts from the chart's D50 illuminant to D65. An earlier version
pushed the D50 Lab values through a D65-assuming `lab2rgb`, which displaces
the saturated patches by up to **6.4 Lab units** (A3 blue 6.38, B2 purplish
blue 5.60, F3 cyan 4.83) while moving the neutrals by at most 0.27 — the
signature of a missing adaptation, and exactly the error that makes
near-neighbour patches swap.

**Validation, 40 bands of the subset:** 12/12 correct on every band upright,
and 12/12 on every band with the tile order reversed to simulate a card
rotated 180 degrees. The hypothesis choice is unanimous — `rows(2,1)` for
every left band, `rows(4,3)` for every right band, and the corresponding
`_flip` hypotheses when rotated — and the margin to the second-best
hypothesis never falls below 0.365 (`identity_validation.csv`). For contrast,
restricting the vote to the six clean tiles of an occluded band drops the
margin to 0.044, a near-tie: that is the regime where the margin should be
read as "orientation undetermined" rather than trusted.
