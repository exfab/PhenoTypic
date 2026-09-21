# Expansion degree and patch count — evidence

Companion to `README.md` (the `CalibrateColorRpcc` spec). The spec fixes the
root-polynomial degree and warns about missing tiles; this document is the
evidence behind both choices, and the reference for anyone deciding what degree
to run a batch at or how worried to be about a partly-detected card.

Measurements come from the four *Rhodotorula* plate frames
(`d000220_300_021`, `_022`, `_038`, `d000320_300_003`; three imaging sessions)
via `SnP-ColorCorrection/patch_measurements.npz`, plus a 24-patch calibration
chip. They describe this rig and this chart. Treat them as calibrated
expectations for this project, not universal constants.

---

## 1. How much each patch is worth

768 leave-one-out fits: drop one patch's row from the least-squares input, fit,
then score the resulting profile against reference Lab on **all 24 patches**. The
image is never edited and no pixels are masked — a dropped patch is still
measured afterwards, it simply had no vote in the fit.

**At degree 3, one patch dominates.** Removing cyan raises card-wide error from
**1.96 → 3.08 ΔE00** on the same photograph, and **3.04 → 4.23** when the profile
is reused on another frame. Every other patch costs **< 0.44 ΔE00**; blue is the
worst of the rest; the median patch costs **0.09**; half the card costs under
0.10 each.

Why cyan: it is the most saturated patch on the card and sits outside the sRGB
gamut — its own reference value does not round-trip without 6.5 ΔE00 of
clipping. Removing it converts interpolation into extrapolation at the edge of
the fit's range. Its physical position is the bottom tile of the right
half-card's **outer** column, which the frame border clips — i.e. the single
most valuable patch is on one of the hardest tiles to detect. That is the whole
reason the spec warns about it by name.

**At degree 2 the effect vanishes**: the worst removal (yellow green) costs
0.25 ΔE00.

A caution on a tempting but wrong reading: *hard to predict* is not *important
to detect*. The `_ownpatch` variant — how well a left-out patch predicts
**itself** — ranks cyan 31.8, blue 14.1, magenta 9.4 ΔE00, yet magenta's
card-wide cost is only 0.31. Use the card-wide number when deciding what a
detector must find.

## 2. How error grows as patches go missing

18 240 subset fits over 30 random removal orders (seed 1729).

| patches in fit | degree 3 | degree 2 |
|---|---|---|
| 24 | 1.96 | 2.79 |
| 20 | 2.43 | — |
| 18 | 3.34 | — |
| 16 | 4.66 | ~3.1 |
| 14 | 9.78 | — |
| 13 | — | 3.27 |
| 11 | — | 3.73 |
| 8 | — | 4.58 |
| 7 | — | 7.53 |
| 6 | — | 13.24 |

Degree 3 degrades steeply and is highly **order-dependent** — the IQR across the
30 removal orders at 18 patches is 1.65 ΔE00, against 0.36 for degree 2. Which
patches are missing matters a great deal at degree 3 and barely at all at
degree 2. Degree 2 overtakes degree 3 at **18 patches** on the same photograph,
**19** when the profile is reused on another frame.

**Miss budget** — how many patches may be absent and still stay within tolerance:

| correction | within 0.1 ΔE00 | within 0.5 ΔE00 |
|---|---|---|
| degree 3, same photograph | 0 | 4 |
| degree 3, reused cross-frame | 1 | 4 |
| degree 2, same photograph | 6 | 11 |
| degree 2, reused cross-frame | 5 | 11 |

**Determinacy is not accuracy.** Degree 2 has a unique solution down to 6
patches, but its error at 6 is 13.24 ΔE00. The practical floor is **13**. Do not
describe degree 2 as safe below 13 patches.

Conditioning explains the shape of both curves: the degree-3 design matrix has a
median condition number of **9.5×10⁴** on a full 24-patch card against **54** for
degree 2. The spec's validation script reproduces the ratio independently
(2.89×10⁴ vs 29.7, 973×) and confirms degree 3 goes rank-deficient at 12 patches.

Two consequences worth remembering when reading fit output:

- A degree-3 fit on fewer than 13 patches returns a minimum-norm solution whose
  in-sample residual can be near zero (0.09 and 0.00 ΔE00 were observed on
  12-patch subsets, at condition numbers 1.2×10⁹–7×10¹⁰). That number is an
  artifact of exact interpolation, not accuracy. The spec refuses the fit rather
  than reporting it.
- At degree 3, two near-identical fits can have coefficients differing by up to
  48 while their predictions agree. **Compare fits by their predictions, never by
  their matrices.**

## 3. Why the degree must be the same for every frame in a batch

This is the measurement that settled the design. An earlier draft resolved the
degree per image from that image's patch count; it was withdrawn.

Degree 2 and degree 3 do not differ only in accuracy — they differ in a
**reproducible direction**. Fitting both on the same 24 patches of the same
frame and comparing the two corrected outputs of the same pixel:

- median disagreement **2.17 ΔE00**, max 6.45 (red);
- the disagreement points the same way in Lab on every frame — its across-frame
  mean is **5.9×** its across-frame scatter, so it is a bias and does not average
  out over replicate plates;
- it is largest on the saturated patches: red 6.45, orange 3.99, cyan 3.66,
  light skin 2.91, yellow 1.54 ΔE00 — the colours a pigmented yeast occupies.

Apparent colour difference between two frames showing the **same** patch, median
over 12 ordered frame pairs × 24 patches:

| correction policy | median | P90 | max |
|---|---|---|---|
| both frames degree 3, all 24 patches | 0.48 | 1.18 | 3.42 |
| both frames degree 2, all 24 patches | 0.50 | 1.09 | 2.90 |
| both degree 2, one frame fitted on 17 patches | 1.21 | 2.94 | 8.40 |
| both degree 3, one frame fitted on 17 patches | 1.35 | 11.40 | 63.98 |
| **degree 3 vs a frame demoted to degree 2 (17 patches)** | **2.40** | **5.25** | **14.52** |

Three readings:

1. **Mixing degrees is the worst of the five policies** — ~5× inflation of the
   apparent between-plate difference over the matched baseline, concentrated in
   the pigmented colours: 17× on cyan, 16× on magenta, 15× on red.
2. **Which degree you pick barely matters for comparability, provided it is the
   same everywhere.** Batch-wide degree 2 costs **+0.02 ΔE00** of between-frame
   comparability against batch-wide degree 3, while costing 0.98 ΔE00 of
   *absolute* accuracy. These are different objectives: an error common to every
   frame largely cancels in a between-frame contrast, which is what a screen
   measures.
3. **If any frame in the batch will be short of patches, run the batch at
   degree 2.** Holding degree 3 through a 7-patch loss has a similar median but a
   catastrophic tail (P90 11.40, max 63.98) — the conditioning problem again.
   Degree 2 degrades gracefully (P90 2.94, max 8.40).

Fixing the degree removes the model-class bias. It does **not** make a
partly-detected card as good as a fully-detected one: at fixed degree 2, one
frame missing 7 patches still doubles the apparent between-frame difference
(1.21 vs 0.50 median). Chasing 24/24 detection remains worthwhile, which is what
the spec's missing-tile warnings are for.

## 4. Practical guidance

- **Default to degree 3** and fix it for the run when every frame detects 24/24,
  or at least 20.
- **Drop the whole batch to degree 2** — do not mix — if any frame will be short.
  Record the choice; it is a property of the batch, not of a frame.
- **Never accept a frame whose card is missing cyan** without deciding
  explicitly. It is the only single patch whose loss exceeds 0.5 ΔE00, and it
  lives on the hardest column to detect.
- **Report the held-out figure, not the in-sample diagonal**, when quoting
  accuracy: per-image in-sample is 1.95 ΔE00, cross-frame held out 3.03 (2.09
  within a session, 3.98 across sessions), against 15.93 for a transferred chip
  profile.
- **Within-plate comparisons are unaffected** by any of this. A treatment and its
  control on the same plate share one correction, so it cancels. The concern in
  §3 applies to between-plate and between-session contrasts.

## 5. Provenance

| number | source |
|---|---|
| §1, §2 | patch-importance session — `patch_importance_loo.csv`, `patch_importance_summary.csv`, `patch_reduction_sweep.csv`, `patch_reduction_summary.csv`, `patch_removal_permutations.csv` (seed 1729) |
| §3 | `degree_policy_between_frame.csv`, `degree_effect_per_patch.csv`, computed from `patch_measurements.npz` |
| conditioning, rank | `../../logic_validation_scripts/2026-09-21-in-frame-checker-color-correction/checker_color_correction.py`, claim C4 |
| transfer figures | per-image profile grid, `per_image_profile_grid.csv` |

The fast fit/eval substrate used for §1–§2 reproduces the full
`ColorCorrector` + `MeasureColor` pipeline to 0.020 ΔE00 mean / 0.071 worst
across all 16 frame pairs, and its RPCC call matches
`colour.characterisation.matrix_colour_correction_Finlayson2015` to 5.5×10⁻⁹.
