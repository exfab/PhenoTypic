# In-frame checker colour correction — spec

**Status:** proposed
**Module:** `phenotypic.correction`
**New public operation:** `CalibrateColorRpcc`
**Validation script:** `docs/superpowers/logic_validation_scripts/2026-09-21-in-frame-checker-color-correction/checker_color_correction.py`

---

## Objective

Give a plate image a colour correction fitted from **the colour chart in its own
frame**, in one operation, without the user hand-cropping tiles or hand-building
a profile.

The rig mounts two 2×6 half-cards at the left and right edges of every plate
photograph. Today, getting a correction out of them takes a notebook: crop the
edge strips by hard-coded pixel widths, mirror-pad the crops to rebuild a
synthetic 24-patch card, segment it, hand-mask the fabricated pixels, fit a
profile, apply it. That route fabricates up to 91 % of some tiles' pixels, breaks
when the plate drifts a few pixels, and produces a profile with no record of
whether the card was actually where it was assumed to be.

The replacement reads the tiles where they really are, measures each one with a
robust centre, fits the root-polynomial matrix the package already implements,
and refuses to correct a frame whose card it could not trust.

### Why per-frame, and not a stored profile

Fitting from the frame's own card is not a convenience — it is most of the
accuracy. Measured mean ΔE2000 over the 24 in-frame tiles:

| profile source | mean ΔE2000 |
|---|---|
| transferred chip profile | 15.93 |
| per-image fit, in-sample | 1.95 |
| per-image fit, held out on another frame | 3.03 |
| … within one imaging session | 2.09 |
| … across sessions (Sept → Feb) | 3.98 |

The mechanism is in claim **C3** of the validation script: the root-polynomial
expansion is positively homogeneous of degree 1, so Φ(k·RGB) = k·Φ(RGB). A fit
on a frame that happens to be 1.5× dark absorbs that gain as a matrix scaled by
1/k. A profile fitted elsewhere is exposure-invariant by construction and
*preserves* the deficit. Per-plate beat the transferred profile on all 24
patches, an 8.2× overall reduction.

## Non-goals

- Re-validating root-polynomial correction as a method. It is established; LOOCV
  and held-out-patch panels were considered and dropped in an earlier session.
- Replacing `ColorCheckerProfile` or `ColorCorrector`. Both stay; this operation
  drives them.
- Detecting a free-standing, fully-visible chart photographed on its own — that
  is what `ColorCheckerProfile.fit()` with border-fill segmentation already does,
  and it does it better. This operation is for cards that are *part* of the scene.
- Fixing the auto-white-balance confound in RAW development (see Deferred).

## Background: what the prior sessions established

Numbers below are the measured values the design leans on. They came from the
*Rhodotorula* screening frames (6016×4012, RawTherapee `InvertLinearLMMSE.pp3`,
four plate frames across three sessions plus one calibration chip) and are
context for those frames, not universal constants.

**Detection** (`SnP-ColorCorrection/chipdetect/`). Seven lattice-refinement
methods were benchmarked on synthetic displacements and on real cross-session
pairs. Capture range and cost per band:

| method | Δy range | Δx range | needs reference band | s/band |
|---|---|---|---|---|
| frozen prior | 0 | 0 | no | 0.000 |
| translation (phase correlation) | ±60 | −6…+3 | yes | 0.011 |
| euclidean (ECC) | ±60 | ±40 | yes | 0.058 |
| profile snap | ±30 | −10…+6 | no | 0.261 |
| **rigid snap** | ±45 | ±30 | no | 0.357 |
| blob + RANSAC | ±45 | −10…+6 | no | 1.419 |

Real displacement is small: ≤4.4 px within a session, ≤9.7 px across five
months, rotation ≤0.084°. The choice of detector moves the 24-patch fit residual
by **<0.1 ΔE2000** — detection is not the accuracy bottleneck, it is the
*reliability* bottleneck. Two findings drive the design:

- Per-column snapping lags on a column clipped by the frame border (the column's
  visible extent changes with the shift, not just its centre), which is why the
  rigid variant — shift measured on the unclipped column, applied to the whole
  lattice — has 3× the horizontal range of the per-column one.
- The `colour-checker-detection` library found 0 charts in 28 configurations: it
  needs one contiguous 4×6 chart and these are two 2×6 half-cards. Not usable.

**Measurement.** `MeasureColor` emits two centres per object from the same
pixels: a Weiszfeld geometric median in Lab, and a ΔE2000 medoid (a real pixel).
On 120 checker tiles the two sit a median of 0.052 ΔE2000 apart and their
accuracy against reference is statistically indistinguishable (−0.004 ΔE2000
paired, Wilcoxon p = 0.81). The medoid is preferred here for a different reason:
being an actual pixel it cannot land in an empty region of a bimodal cloud, and
it is the estimator `MeasureColor` reports, so a patch colour fitted here and a
colony colour measured later are the same kind of quantity.

**Patch count.** 768 leave-one-out fits and 18 240 subset fits established what
a missing tile costs and how that cost depends on the expansion degree. The two
results the spec acts on: **cyan** is the only single patch whose removal costs
more than 0.5 ΔE2000 (1.96 → 3.08 at degree 3, against < 0.44 for every other
patch), and it happens to sit on the frame-clipped outer column that is hardest
to detect; and degree 3, while more accurate on a full card, is far more
sensitive to which patches are missing than degree 2 (IQR 1.65 vs 0.36 ΔE2000
across 30 removal orders at 18 patches) because its design matrix is
ill-conditioned even at 24 patches (9.5×10⁴ vs 54).

Full tables — per-patch cost, the error-versus-patch-count curves for both
degrees, the miss budget, and the between-frame measurement that fixed the
degree — are in [`degree-and-patch-count.md`](degree-and-patch-count.md).

## Algorithm

Six stages. Stage inputs and outputs are given so each can be tested alone.

### A — ROI extraction (mandatory input)

`rois` is a required list of one `CheckerRoi` per card region, and it carries
**only where to look** — a rectangle. Nothing about what is inside it is
declared: not how many tiles, not which chart patches they are, not how the card
is oriented. All of that is recovered from the image (stages B and D).

```
CheckerRoi(row=(1170, 2840), col=(0, 340))          # left edge band
CheckerRoi(row=(1170, 2840), col=(5856, 6016))      # right edge band
```

#### Bounding-box shorthand

Typing two `CheckerRoi` constructions to say "these two rectangles" is friction
in a notebook, so `rois` also accepts plain coordinate lists in
`[row_min, col_min, row_max, col_max]` order and converts them internally:

```python
CalibrateColorRpcc(rois=[[1170, 0, 2840, 340],
                         [1170, 5856, 2840, 6016]])

# equivalent to, and normalised into:
CalibrateColorRpcc(rois=[CheckerRoi(row=(1170, 2840), col=(0, 340)),
                         CheckerRoi(row=(1170, 2840), col=(5856, 6016))])
```

That ordering is deliberate: it is `skimage.measure.regionprops`' `bbox`
convention, so a region found by any upstream segmentation drops straight in
without re-ordering — `rois=[r.bbox for r in regionprops(label_img)]` works. It
is *not* the `(row_slice, col_slice)` pair `ColorCheckerProfile.rois` takes, and
it is not `(x, y, w, h)`; both of those are easy to supply by accident and
produce a silently wrong rectangle.

Implementation: a `CheckerRoi.from_bbox(seq)` classmethod plus a
`field_validator(mode="before")` on `rois` that coerces any non-`CheckerRoi`
entry. The list may mix forms freely. Coercion rejects, with a message naming
the expected order:

- a sequence that is not exactly four items;
- non-integer coordinates;
- `row_min >= row_max` or `col_min >= col_max` — which is what a
  `(row, col, height, width)` or `(x, y, w, h)` mistake usually produces, so the
  error should say so rather than just "invalid bounds".

The shorthand is **input-only**. `model_dump` and the saved `.json.pht-op`
always emit the canonical `CheckerRoi` form, so a stored configuration is
unambiguous about which convention it used and re-loads to the same rectangles.

Rationale for requiring the rectangle rather than "detect the card anywhere":
whole-frame segmentation on the calibration frame found 36 chips — the
free-standing card *and* a rig-mounted half-card — and the strict chip-count
gate refused the fit. Where to look is the one thing no image-content heuristic
recovers reliably, and the user has it. Rationale for requiring *nothing else*:
a declared layout is a claim about the image that the image can contradict, and
the failure is silent — a card mounted upside down after a rig service still
produces 12 plausible tiles, each fitted against the wrong reference colour.
Deriving identity from the pixels means a wrong card cannot be quietly accepted;
it shows up as a low match margin.

Validation at construction is therefore only geometric: ROIs must be in-bounds
and non-overlapping.

Two optional fields exist purely as assertions, never as inputs to detection:
`expect_tiles: int | None` and `label: str | None` (a name carried into
diagnostics, e.g. `"left band"`). If `expect_tiles` is set and detection finds a
different number, that is a QC failure rather than a silent adjustment.

### B — Lattice fit or refinement

Work in CIE L\*a\*b\* of the ROI, obtained through `Image.color.Lab` on a
sub-image so encoding is handled once (see *Encoding guard* below).

Two entry points, selected by whether a prior is supplied:

- **`lattice_prior=None`** — fit the lattice from the ROI itself, including how
  many tiles there are. Smooth a column-wise and a row-wise cross-channel
  standard-deviation signal; patches are plateaus in it and the gutters between
  them are the minima. **Count the qualifying plateaus along each axis** to get
  the tile grid, then refine phase, pitch and duty by a 1-D search over a
  periodic square wave at that count. Cost ~0.26 s/ROI. Use this to *build* a
  prior on a reference frame, and for one-off images.

  Plateau counting is also what rejects the refracted ghost copies of the
  neutral column seen at the right band of this rig: the ghosts are too blurred
  to produce a qualifying plateau (the real neutrals peak at 26.2 and 52.4 in
  the column signal; the ghost zone produces nothing that qualifies), so they
  are never counted as tiles.
- **`lattice_prior=<CheckerLattice>`** — refine a stored lattice against this
  frame with `refine`:
  - `"rigid"` *(default)* — measure the displacement on the unclipped inner
    column only and translate the whole lattice by it. ±45 px vertical, ±30 px
    horizontal, no reference image needed, 0.357 s/ROI.
  - `"frozen"` — trust the prior unchanged. For rigs that never move; still QC'd.

  **`"ecc"` is not offered by the operation** (decided 2026-09-21, after code
  review). ECC registers against a stored *reference band* — the Lab pixels of
  each ROI on the frame the prior was fitted from, ~2150×340×3 floats per band —
  which would have to be serialised into every pipeline JSON and per-image
  provenance journal (~40 MB per band as JSON). Its only gains over `"rigid"`
  are range (±60 / ±40 vs ±45 / ±30) and a correlation score, and measured rig
  motion (≤9.7 px, ≤0.084°) sits well inside rigid's range. `refine_ecc` stays
  in `_checker_detect` as a building block; re-adding it to the operation means
  first settling how a reference band is stored.

Output: a `CheckerLattice` per ROI (per-column `x0/x1/start/pitch/duty`, plus
`dy`, `dx`, `rot`, `confidence`).

### C — Tile boxes and core trimming

Each tile box is trimmed to its central fraction before any pixel is read:
`core_trim=0.4` keeps the middle 60 % of each axis (36 % of the area). This is
what keeps the measurement off the gutters when the lattice is a few pixels out,
and it is where the detector's residual error is absorbed.

Note this is a *box* trim, deliberately distinct from
`ColorCheckerProfile.core_fraction` (0.5), which is a distance-transform core of
a segmented blob. Different geometry, different parameter, different name.

### D — Patch identity

Identity is **derived here**, not supplied. Nothing upstream knows which chart
patch sits in which detected cell; this stage decides it, and the fit consumes
its answer directly.

**Score whole placements, never a free per-tile assignment.** Enumerate the
discrete ways the detected tile block can sit on the chart grid — which
contiguous sub-block of the 4×6 chart it occupies, in which orientation — and
score each *whole* hypothesis by the mean distance between the observed tiles
and that hypothesis's reference patches, in gain-invariant features
(chromaticity plus relative luminance, so an exposure difference does not
choose the answer). The winner is the placement with the lowest score; each
tile's identity follows from it.

This distinction is the crux. A free Hungarian assignment lets each label wander
onto whichever reference fits it best, so an occluded or contaminated tile is
silently relabelled and quietly fitted against the wrong reference colour. A
placement hypothesis is decided by all twelve tiles at once, so one bad tile
cannot move it. Two genuinely similar patches (purplish blue vs blue flower; the
two mid-greys) swap freely under per-tile matching and not at all under
placement scoring.

**Measured discrimination.** On the four *Rhodotorula* frames, both bands each
(8 cards), scoring against the fully general hypothesis set — every ordered pair
of chart rows × both column directions, 24 hypotheses, nothing rig-specific —
the correct placement won **12/12 tiles on all 8 cards**, with a margin to the
runner-up of **0.294–0.334**. Narrowing the hypothesis set to the rig's actual
geometry raises the margin to ~0.84, but the general set is already decisive, so
the operation does not need to be told the rig layout.

**Margin is the gate, and it has to be tighter than it was.** When identity was
only a cross-check against a declared layout, a wrong match cost a QC counter.
Now it corrupts two rows of the least-squares input, so the threshold is
calibrated against silent failure. Injecting bright occlusions into 1–6 of the
12 tiles (2 240 trials):

| margin threshold | clean cards rejected | wrong identities caught | silent mislabels |
|---|---|---|---|
| 0.10 (the old value) | 0 % | 64.8 % | 11.5 % |
| 0.15 | 0 % | 82.3 % | 5.8 % |
| **0.20** | **0 %** | **92.4 %** | **2.5 %** |
| 0.25 | 0 % | 97.8 % | 0.7 % |

The gate is **refuse below 0.20, warn below 0.25**, against a clean-card
baseline of 0.294 minimum / 0.315 median. 0.25 catches more but leaves only 18 %
headroom under the observed clean minimum on four frames, which is too tight to
ship. The residual silent rate at 0.20 is concentrated at heavy occlusion — 0.0 %
with one tile occluded, 0.3 % with two, rising to 5.9 % at six — and the
impurity and robust-shift gates fire independently in exactly that regime
(17× separation on the one real obstruction event in the test set).

**Corroborating signal.** Run a free Hungarian assignment as well and report how
many tiles it agrees with the winning placement on. Disagreement on more than
two of twelve is a warning: the placement is still used, but something about
those tiles does not look like the patch the geometry says it is.

**There is no fallback.** The prototype's documented recovery from an
undetermined orientation was to fall back to the expected layout. There is no
expected layout here by design, so a card whose placement is undetermined is
refused. That is the cost of not declaring the layout, and it is the right
trade: a refusal is visible and a mislabelled fit is not.

### E — Patch colour: deterministic ΔE2000 medoid

For each tile, take the core pixels in Lab and compute a **candidate-restricted
ΔE2000 medoid**:

1. Weiszfeld geometric median of all core pixels (Lab) — cheap, O(n) per
   iteration, ~3 ms for 4 500 px.
2. Rank pixels by Euclidean distance to it; take the nearest
   `medoid_candidates=256`.
3. Score each candidate by its total ΔE2000 to **every** core pixel; the winner
   is the patch colour.
4. Record the winner's rank within the candidate set, and the ΔE2000 spread from
   it (median / P95) as a per-tile consistency measure.

The patch colour handed to the fitter is that pixel's own sRGB value, so the
existing fit path is unchanged.

**Why not call `medoid_ciede2000` directly.** The shipped estimator picks the
medoid out of a seeded subsample of `medoid_max_pixels=1000`. Measured on real
tiles, re-drawing the seed moves the answer by a median of 0.119 ΔE2000 (worst
0.806) — larger than the difference between the medoid and the geometric median
it is being chosen over. Raising the cap does not fix it affordably: the
selection is O(n²), and measured cost on a 4 500-pixel tile is 0.09 s at cap 1000,
1.71 s at cap 4000 — **41 s per 24-tile frame** — while seed spread only falls
from ~0.32 to ~0.25 ΔE2000 (claim **C2**). The candidate-restricted form is
O(K·n), costs ~0.14 s/tile (3.4 s/frame) at full-tile resolution, uses no RNG at
all, and returned **the same pixel as the exhaustive medoid in 8/8 synthetic
tiles with the winner at rank ≤ 0.4 % of K** (claim **C1**).

Failsafe: if the winning candidate's rank exceeds 80 % of K, the cloud is not
unimodal around its geometric median — widen K once and re-score, then warn.

`MeasureColor` keeps its own estimator; this is the correction path only. If the
two are ever reconciled, the candidate-restricted form is the one to adopt, since
it is a strict improvement on the same estimand.

### F — Reference adaptation, fit, and apply

Reference handling is already correct in `_load_reference_data` and must be
reused rather than reimplemented: it takes the chart's xyY under **the chart's
own illuminant**, applies **Bradford chromatic adaptation** to
`target_illuminant`, and only then produces reference Lab and reference linear
sRGB. Both sides of the ΔE2000 comparison and both sides of the least-squares fit
therefore live in the working illuminant. Never compare measured values against
unadapted chart values.

The operation must *record* that this happened — `diagnostics["illuminant"] =
{"checker": ..., "target": ..., "bradford_adapted": bool}` — so a reader of a
saved profile can see it rather than infer it.

Fit: build `measured_srgb: dict[patch_name -> (3,) sRGB]` from stage E, construct
a `ColorCheckerProfile` with the resolved degree, and hand it to the existing
`_fit_from_measured` path (promoted to a public
`ColorCheckerProfile.fit_from_patch_colors(mapping)`). One solver, one
diagnostics schema, one report — no second implementation.

Apply: delegate to `ColorCorrector(profile=...)`, which decodes sRGB gamma,
applies the root-polynomial matrix in linear light, re-encodes, and recomputes
`gray` and `detect_mat`.

**Pipeline placement.** `CalibrateColorRpcc` → `DenoiseBlockMatch`. Correction
improves accuracy while mildly amplifying per-pixel noise (within-tile ΔE2000
spread rises 3.32 → 3.63 median on the calibration chip; 0.99 → 1.47 on the
per-plate fits), so the denoiser belongs after it, not before.

## Failsafes

Every check below either downgrades the fit or refuses it. The design principle
is that an unattended batch must never silently emit a correction fitted to a
card that was occluded, mis-identified, or clipped.

### Degree is fixed

`degree` is an explicit integer, default `3`. It is set once and applies to every
image in a run. The operation never adapts it to what a frame happened to
detect — not upward, not downward.

The reason is that degree 2 and degree 3 differ in a reproducible direction, not
just in magnitude, so correcting two frames with different degrees adds a bias to
the contrast between them that does not average out over replicates. Choosing the
degree for a batch, and what it costs, is documented in
[`degree-and-patch-count.md`](degree-and-patch-count.md) — the short version is
that batch-wide degree 2 costs +0.02 ΔE00 of between-frame comparability against
batch-wide degree 3, so if any frame in the batch will be short of patches, run
the whole batch at degree 2 rather than mixing.

One hard error remains, because it is arithmetic rather than policy: a degree
whose expansion has more terms than the fit has accepted patches (degree 3 with
< 13, degree 4 with < 22) raises `ValueError`. Such a fit has no unique solution,
and its minimum-norm answer reports a near-zero in-sample residual that is an
artifact of exact interpolation.

Everything else about a short card is a **warning**, not a change of model.

### Missing-tile warnings

Emitted per frame after identity assignment and outlier rejection, and recorded
in `diagnostics["patch_census"]` so a batch log can be swept afterwards.

| condition | level | message carries |
|---|---|---|
| any chart patch not found in any ROI | `UserWarning` | the missing patch names, the count, and which ROI the placement expected them in |
| **cyan missing** | `UserWarning`, separately | that it is the only single patch whose loss exceeds 0.5 ΔE00 (≈ +1.1 at degree 3), and that it sits on the frame-clipped outer column |
| accepted patches < `min_patches` (default 20) | `UserWarning` + QC flag `insufficient_patches` | the count, and the measured error at that count for the configured degree |
| accepted patches < 13 | `UserWarning`, escalated | that this is below the practical accuracy floor for either degree, whatever the rank says |
| patch count differs between images in one run | `UserWarning`, once per run | the per-image counts, and that unequal counts alone roughly double the apparent between-frame difference (1.21 vs 0.50 ΔE00 median) even at fixed degree |
| patches rejected as ΔE outliers > 4 | `UserWarning` | that `outlier_sigma` is likely rejecting a systematic problem rather than outliers |

The `insufficient_patches` flag participates in `on_qc_fail` like any other QC
signal, so a batch runner can be configured to skip those frames; by default they
are corrected and flagged, because at fixed degree a 20-patch frame is usable and
the user should decide.

A convenience for choosing the batch degree, run once before the batch — it
surveys patch counts only and recommends nothing the user cannot override:

```python
census = CalibrateColorRpcc.patch_census(images, rois=[left, right])
# -> per-image detected/accepted counts, the union of missing patches,
#    whether cyan was found in every image, and the worst count in the set
```

### Numeric gates

| gate | limit | action |
|---|---|---|
| condition number of Φ | > 1×10⁷ | refuse — the near-degenerate subset fits that produced spurious 0.00 ΔE00 residuals sat at 1.2×10⁹–7×10¹⁰ |
| condition number of Φ | > 1×10⁶ | warn |

Patch-census conditions are in *Missing-tile warnings* above; they warn and never
alter the model.

### Geometry and card-integrity gates (ported from `qc_record`)

Run per ROI, on the tiles actually measured. Limits are the calibrated values
from the 40-band subset.

| signal | limit | what it catches | action |
|---|---|---|---|
| displacement from prior | > 30 px | rig moved or prior stale | refuse |
| spread between anchor columns (see note below) | > 12 px | columns of one rigid card disagree about where it moved | refuse |
| fewer than two columns wholly inside the ROI | < 2 | the anchor-column check above could not run | warn |
| placement margin | < 0.20 | card missing, flipped, wrong, or too occluded to place | refuse |
| placement margin | < 0.25 | placement weaker than any clean card observed (baseline 0.294–0.334) | warn |
| free-Hungarian disagreement with the winning placement | > 2 of 12 tiles | those tiles do not look like the patch the geometry implies | warn |
| detected tile count ≠ `expect_tiles` (when set) | — | lattice found the wrong number of tiles | refuse |
| mean tile impurity | > 0.05 | something covering the card | refuse |
| worst tile impurity | > 0.05 | localised contamination | warn |
| robust colour shift from contamination | > 1.5 ΔE | contamination actually moved the value | refuse |
| clipped-pixel fraction | > 0.20 | channel pinned at the sensor limit — unrecoverable | refuse |

**Anchor-column spread replaces the method spread (2026-09-21).** This gate
originally compared two refinement *methods* on the same ROI: 8 px when one of
them used a reference band, 12 px when both were reference-free. That check
cannot be built as shipped. ECC, the only reference-based method, is not
offered by the operation (see §Detection), and no second reference-free method
was ported. What runs instead is a consistency check inside the rigid method:
the shift is measured once per column, each time anchoring on that column, and
the spread between the estimates must stay within 12 px, because the columns of
one rigid card must agree about where it moved. Only columns lying wholly
inside the ROI vote. A column clipped by the frame border tracks the shift at
about half rate (§Detection), and letting it vote refused in-range moves: a
25 px shift read as 12.5 px of disagreement. On a two-column half-card whose
outer column is clipped, only one column votes and the check cannot run. The
gate then records `anchor_columns_voting` and warns rather than passing
silently. Restoring the check on clipped cards would mean measuring a clipped
column's shift from its unclipped inner edge, a method this spec has not
evaluated.

The impurity signals answer *is something there*; the robust shift answers *did
it move the answer*, and only the second is grounds to reject. Fault injection
(flipped card, blanked card, 60 px and 120 px displacement) trips ≥ 2 signals
each; a 2.5× exposure change correctly passes, because the gate is keyed to
geometry and card integrity, not photometry.

`on_qc_fail: Literal["raise", "warn", "skip"] = "raise"`. `"skip"` returns the
image uncorrected with the QC record attached — the right setting for a batch
runner that must not die on one bad frame.

### Encoding guard

The prototype fed *linear* RawTherapee RGB into `skimage.rgb2lab`, which expects
sRGB encoding; its Lab values and every threshold expressed in them are an
internal contrast measure, not calibrated CIE quantities. This must not be
ported. All Lab in the new operation comes from `Image.color.Lab`, which knows
the image's `gamma`. The operation asserts that `image.gamma` is set and logs it
in diagnostics.

### Capture-metadata guards

`ColorCorrector` already warns when a corrected image's EXIF shows a different
camera body or lens than the profile. Add, at fit time:

- Record the as-shot white-balance multipliers in `CaptureMetadata`.
- When correcting an image whose WB multipliers differ from the profile's by more
  than 5 %, warn. Measured cost of the mismatch that motivated this: a profile
  fitted on an auto-WB render and applied to a differently-WB'd render of the
  *same* exposure gave 5.69 ΔE2000 against 1.10 in-sample (7.10 uncorrected),
  with the error concentrated in the neutrals.
- When a stored profile is reused rather than fitted per frame, warn once with
  the expected penalty (≈ 2.1 ΔE2000 within a session, ≈ 4.0 across sessions).

## API

```python
class CheckerRoi(BaseModel):
    row: tuple[int, int]                       # where to look — nothing else
    col: tuple[int, int]
    label: str | None = None                   # carried into diagnostics
    expect_tiles: int | None = None            # assertion only; never steers detection
    anchor_col: int | None = None              # unclipped column for rigid refinement

    @classmethod
    def from_bbox(cls, seq, **kw) -> CheckerRoi: ...   # [row_min, col_min, row_max, col_max]

class CheckerLattice(BaseModel):               # serialisable; one per ROI
    columns: list[ColumnLattice]               # x0, x1, start, pitch, duty
    nrows: int
    dy: float = 0.0
    dx: float = 0.0
    rot: float = 0.0

class CalibrateColorRpcc(ImageCorrector):
    # -- required ----------------------------------------------------------
    rois: list[CheckerRoi]                     # accepts [row_min, col_min, row_max, col_max]
                                               # lists too; coerced, stored canonical

    # -- chart & colour space ---------------------------------------------
    checker_type: str = "ColorChecker24 - After November 2014"
    target_illuminant: str = "D65"
    degree: int = 3                        # fixed for the whole run; never adapted
    min_patches: int = 20                  # below this, warn and flag — the degree does not move

    # -- detection ---------------------------------------------------------
    lattice_prior: list[CheckerLattice] | None = None
    refine: Literal["rigid", "frozen"] = "rigid"   # no "ecc": see §Detection

    # -- measurement -------------------------------------------------------
    core_trim: float = 0.4
    medoid_candidates: int = 256

    # -- fit ---------------------------------------------------------------
    outlier_sigma: float = 2.0
    max_condition_number: float = 1e7

    # -- identity ----------------------------------------------------------
    min_identity_margin: float = 0.20          # refuse below; warn below 0.25

    # -- safety ------------------------------------------------------------
    qc_limits: QcLimits = QcLimits()
    on_qc_fail: Literal["raise", "warn", "skip"] = "raise"
    profile: ColorCheckerProfile | None = None   # reuse instead of fitting; warns

    # -- post-fit state ----------------------------------------------------
    fitted_profile: ColorCheckerProfile | None = None
    qc: dict[str, Any] = {}

    @classmethod
    def patch_census(cls, images, *, rois, **kw) -> PatchCensus: ...
```

`CalibrateColorRpcc` subclasses `ImageCorrector` because it rewrites `rgb`, `gray`
and `detect_mat` together — the contract that base class exists to enforce.

Serialisation: `CheckerRoi` uses integer bounds, not `slice` objects.
`ColorCheckerProfile.rois` is currently `Field(exclude=True)` and does not
round-trip; here the ROIs are the operation's defining input and **must**
serialise, so a saved `.json.pht-op` reproduces the run.

### Tuning

Annotate for the `tune` module, matching existing convention:
`core_trim` → `TuneSpec(0.2, 0.6)`; `outlier_sigma` → `TuneSpec(1.5, 4.0)`.

`degree` is **not** a tuning target, despite `ColorCheckerProfile.degree`
carrying a `TuneSpec`. Letting a tuner vary it per image is the per-frame
demotion this spec prohibits, arrived at from the other direction; and tuning it
against in-sample residual would select degree 3 every time while degrading
between-frame comparability. `medoid_candidates` is likewise a correctness/cost
knob, not a tuning target.

## Integration

### Files

```
src/phenotypic/correction/_color_correction/
    _checker_roi.py          CheckerRoi, ColumnLattice, CheckerLattice
    _checker_detect.py       plateau counting, lattice fit, rigid / ecc / frozen refinement
    _checker_identity.py     placement enumeration, gain-invariant scoring, margin
    _checker_measure.py      core trimming, candidate-restricted medoid, impurity
    _checker_qc.py           QcLimits, qc_record, gate evaluation
    _color_correction_op.py  CalibrateColorRpcc
    _color_checker_profile.py   (edit) promote fit_from_patch_colors to public
    __init__.py              (edit) export the new names
src/phenotypic/correction/__init__.py   (edit) export + __all__
```

`correction/` has no `CLAUDE.md`; add one covering this subpackage, matching the
pattern in `measure/` and `enhance/`.

### Reuse, not reimplementation

| needed | reuse |
|---|---|
| reference load + Bradford adaptation | `_load_reference_data` |
| solver, outlier rejection, diagnostics | `ColorCheckerProfile._fit_from_measured` |
| geometric median | `phenotypic.util.robust_color_center` |
| ΔE2000 | `colour.difference.delta_E_CIE2000` |
| apply matrix to whole image | `ColorCorrector` |
| Plotly diagnostics | `ColorCheckerProfile.report` |
| EXIF compatibility | `CaptureMetadata` |

Port from `SnP-ColorCorrection/chipdetect/`: the lattice model, the rigid and ECC
refinements, the Hungarian identity assignment with orientation hypotheses, the
impurity / robust-shift / clipping statistics, and the QC limits. Leave behind:
the rig-specific constants in `chiprig.py`, the linear-RGB-into-`rgb2lab`
measurement path, and `M3_blob` / `M23_cascade` (1.419 and 1.825 s/band against
the rigid snap's 0.357 — 4.0× and 5.1×, or 24× and 31× against ECC's 0.058 —
with no measured gain).

### Naming

`CalibrateColorRpcc` is the settled name. It reads as what the operation does —
*calibrate* (fit from the card in this frame) rather than merely *correct*
(apply a matrix you already have), and *Rpcc* names the method, keeping it
distinct from the existing `ColorCorrector` in both autocomplete and prose.

The distinction to hold onto when writing docstrings: `CalibrateColorRpcc`
derives the correction and applies it; `ColorCorrector` only applies one.

### Registration checklist

- Export from `_color_correction/__init__.py` and `correction/__init__.py`.
- `tests/smoke/test_operation.py` and `test_serialization.py` enumerate
  operations — the new class must construct from defaults **plus** `rois`
  (there is no valid default `rois`, so check how the smoke suite supplies
  required fields and add a fixture if needed).
- `extra="forbid"` and `validate_assignment=True`, matching `ColorCheckerProfile`.
- GUI/SDK operation listing picks up `correction.__all__`; verify no duplicate
  entry appears for the legacy-name `__getattr__` shim.

## Acceptance criteria

1. `CalibrateColorRpcc(rois=[left, right]).apply(frame)` on the four *Rhodotorula*
   plate frames detects 24/24 tiles, passes QC, and yields an in-sample mean
   ΔE2000 within 0.15 of the notebook's per-frame result (1.95).
2. Held-out transfer is reported, not the in-sample diagonal: applying a frame's
   profile to another frame of the same session reproduces ≈ 2.1 ΔE2000, and
   across sessions ≈ 4.0.
3. No tile's measurement mask contains a fabricated pixel. (The notebook route's
   tiles were 49–91 % mirrored, mean 65 %.)
4. Identity is recovered without being told: with `rois` carrying bounds only,
   all 24 patches on all four frames are assigned correctly, at a placement
   margin ≥ 0.29 under the general hypothesis set. A frame with its card
   mounted 180° out is assigned correctly too, not merely flagged. A card with
   three or more tiles occluded is refused on margin rather than mislabelled.
5. Patch colour is deterministic: two runs on the same frame return
   bit-identical patch sRGB values, and the result equals the exhaustive ΔE2000
   medoid over the same core pixels on every tile of the calibration frame.
6. Measurement cost ≤ 5 s per 24-tile frame on the reference machine
   (detection ≈ 0.7 s, measurement ≈ 3.4 s).
7. The configured degree reaches every frame unchanged: processing images with
   differing patch counts uses the same degree throughout and stamps it into
   each frame's `diagnostics["degree"]` and measurement provenance. A frame
   below `min_patches` is warned about and flagged, not corrected differently.
   Each row of the missing-tile warning table fires on a frame constructed to
   trigger it — in particular, a card missing cyan warns by name, and a run
   whose images have unequal patch counts warns once.
8. Fault injection — flipped card, blanked card, 60 px displacement, a tile
   occluded by a bright object — is refused with ≥ 2 QC flags each, while a 2.5×
   exposure change is accepted.
9. A degree-3 fit on fewer than 13 patches raises rather than returning the
   spurious near-zero residual the subset experiment produced.
10. `diagnostics["illuminant"]["bradford_adapted"]` is `True` for a D50 chart
   corrected to D65, and reference Lab matches `_load_reference_data` exactly.
11. A saved `.json.pht-op` round-trips the ROIs and the lattice prior and
    reproduces the same correction matrix. ROIs supplied as
    `[row_min, col_min, row_max, col_max]` lists produce an operation identical
    to the `CheckerRoi` form, serialise in the canonical form, and a
    `regionprops` `bbox` tuple is accepted unchanged; a four-item sequence in
    `(x, y, w, h)` or `(row, col, h, w)` order raises with a message naming the
    expected order rather than silently cropping the wrong rectangle.
12. The validation script exits 0.

## Deferred

- **Pin the white balance in RAW development.** Both `InvertLinearLMMSE.pp3` and
  the chip sidecar use `Setting=Camera` with the camera on auto-WB; the chip
  frame lands at 3840 K against the plates' 3659 K. `Setting=Custom` is the fix,
  and it is upstream of this operation. The 5 % WB-divergence warning above is a
  detector for the problem, not a cure.
- **A chroma-only companion metric.** Absolute ΔE2000 against reference conflates
  illumination falloff with colour accuracy for edge-mounted tiles: on real plate
  frames the correction moved mean ΔE2000 only 19.68 → 15.93, but that is
  dominated by a −21.4 → −16.8 L\* offset the exposure-invariant correction cannot
  address, while chroma distance fell 14.60 → 12.72. A white-normalised metric
  would separate the two.
- **Reconciling the two geometric medians in the repo**
  (`util/_geometric_median.py` in Lab, `_color_correction/_helpers.py` in sRGB).
- **`_fit_ridge` and `ridge_lambda` are dead** on the main path —
  `ridge_lambda` is a stored field reachable only from an `except` fallback.
  Document or remove; out of scope here.

## References

- Finlayson, Mackiewicz & Hurlbert (2015), root-polynomial colour correction.
- Sharma, Wu & Dalal (2005), CIEDE2000 test data — used to validate the ΔE00
  implementation in the validation script.
- Prior sessions: colour-correction overview and validation; automated chip
  detection; geometric median vs ΔE2000 medoid; patch importance and detection
  robustness.
- Prototype: `/Users/alex/Projects/SnP-ColorCorrection/chipdetect/`
  (`chipdetect.py`, `chiprig.py`, `rig_prior.json`).
- Related spec: `docs/superpowers/specs/2026-09-19-weiszfeld-coincident-point-singularity/`.
