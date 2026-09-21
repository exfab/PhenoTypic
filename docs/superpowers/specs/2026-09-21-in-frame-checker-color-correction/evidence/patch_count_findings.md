# How many ColorChecker patches does the correction actually need?

**Question.** The rig photographs every plate with two 2×6 half-cards of a 24-patch
ColorChecker at the frame edges, and the colour correction is fitted per frame from
those patches. If we are going to run this at scale, we need to know whether a
detector that misses a few patches is acceptable — and whether some patches are
worth more effort to find than others.

**Answer, in one line.** At the setting the pipeline uses now (`ColorCheckerProfile`
with `degree=3`), a detector may miss up to **4 arbitrary patches** before the colour
error on the card rises 0.5 ΔE₀₀ above the full-card result — but it must not miss
**cyan**, which alone costs 1.1 ΔE₀₀. Dropping to `degree=2` buys a much larger miss
budget (11 patches) and removes the dependence on any single patch, at the cost of
0.83 ΔE₀₀ when the full card is available.

---

## What was done

Dropping a patch means dropping its row from the least-squares fit. No pixels were
masked and the images were never modified — a dropped patch is still measured
afterwards, it simply had no vote in the fit. Every fitted correction is scored on
all 24 card tiles, so the "cost" of a removal includes whatever damage it does to
patches that were never dropped.

Fitting is root-polynomial colour correction (RPCC, Finlayson 2015), the same call
`ColorCheckerProfile` makes internally, at two fixed expansions: `degree=2`
(6 free parameters per output channel) and `degree=3` (13). Colour error is CIEDE2000
between each tile's robust colour centre after correction and that patch's published
reference value, the same drift metric as the existing transfer grid. Four plate
photographs are the replicates — three from one session (2025-09-27, 2025-09-30) and
one four months later (2026-02-02). Results are split by scope:

| scope | meaning | n |
|---|---|---|
| same-image | correction fitted and used on the same photograph — the detector-miss case | 4 photographs |
| cross-frame | correction fitted on one photograph and used on another | 12 ordered pairs |

Two experiments: every single-patch removal (768 fits), and a reduction sweep from 24
patches down to 6 over 30 random removal orders (18,240 fits).

---

## 1. Only cyan is load-bearing

![Ranked cost of removing each patch]({{artifact:f9ce0a82-3437-41a7-b27f-8359096f1848}})

At `degree=3`, same photograph (baseline with all 24 patches: 1.96 ΔE₀₀):

| patch removed | colour error over all 24 tiles | cost vs full card |
|---|---|---|
| cyan | 3.08 | **+1.11** (SEM 0.23) |
| blue | 2.40 | +0.44 |
| magenta | 2.27 | +0.31 |
| green | 2.23 | +0.27 |
| bluish green | 2.20 | +0.24 |
| *median of all 24* | 2.05 | +0.09 |

Cross-frame reuse gives the same ordering and nearly the same costs (cyan +1.19,
blue +0.46), so this is a property of the fit rather than of profile transfer.
Twelve of the 24 patches cost under 0.10 ΔE₀₀ to lose.

At `degree=2` the effect disappears: the most costly removal is yellow green at
+0.25 ΔE₀₀ and no patch exceeds it. A 6-parameter fit has enough redundancy across
24 patches that no single one anchors it.

**Why cyan.** It is the most saturated patch on the card and sits outside the sRGB
gamut — its reference value does not even round-trip through linear sRGB without
clipping (6.5 ΔE₀₀ of clipping error, the largest of the 24). It therefore defines
the edge of the colour range the fit has seen. Remove it and the 13-parameter
expansion is extrapolating over that corner of colour space rather than
interpolating.

**Being hard to predict is not the same as being important.** A left-out patch is
reproduced badly for *itself* — cyan at 31.8 ΔE₀₀, blue 14.1, magenta 9.4, against
0.6–3.8 when they are in the fit — but that own-patch error barely moves the
card-wide mean for most of them. Magenta is reproduced at 9.4 ΔE₀₀ when unseen yet
costs only 0.31 ΔE₀₀ card-wide. So a large own-patch error is not by itself a reason
to require a patch; `fig_patch_importance_ownpatch.png` separates the two readings.

**What four photographs cannot settle.** Cyan and blue separate clearly. Below them
the ranking is not established: the remaining 22 bars span 0.44 ΔE₀₀ at same-image
scope and most neighbouring pairs sit within one standard error of each other. At
cross-frame scope the median standard error of a bar (0.33 ΔE₀₀) is as large as the
entire non-cyan spread, so there only cyan, and marginally blue, stands out.

---

## 2. The cost of fitting on fewer patches

![Colour error against number of patches used]({{artifact:f8621c0a-d434-4469-9676-b144a4def746}})

Miss budget — the largest number of patches a detector may miss while the median
colour error stays within tolerance of the full-card result:

| correction | scope | within 0.1 ΔE₀₀ | within 0.5 ΔE₀₀ |
|---|---|---|---|
| `degree=3` (current) | same photograph | 0 patches | **4 patches** |
| `degree=3` (current) | reused on another | 1 patch | **4 patches** |
| `degree=2` | same photograph | 6 patches | **11 patches** |
| `degree=2` | reused on another | 5 patches | **11 patches** |

Median colour error by patch count, same photograph:

| patches used | 24 | 23 | 22 | 21 | 20 | 19 | 18 | 17 | 16 | 15 | 14 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `degree=3` | 1.96 | 2.09 | 2.24 | 2.31 | 2.43 | 2.65 | 3.34 | 3.50 | 4.66 | 6.45 | 9.78 | 17.3 |
| `degree=2` | 2.79 | 2.81 | 2.82 | 2.86 | 2.86 | 2.88 | 2.88 | 2.92 | 3.01 | 3.10 | 3.11 | 3.27 |

`degree=3` degrades steeply and then collapses; `degree=2` is nearly flat over the
same range. The two cross at **18 patches** on the same photograph and **19** on
reuse: with fewer than about 19 usable patches the 6-parameter fit is the better
choice, and below 16 it is better by a wide margin. The spread across removal orders
tells the same story — at 18 patches the interquartile range over 30 orders is
0.36 ΔE₀₀ at `degree=2` against 1.65 at `degree=3`, so with the 13-parameter fit it
matters a great deal *which* patches went missing, and with the 6-parameter fit it
barely matters at all.

The underlying reason is visible in the fit conditioning: the design matrix has a
median condition number of 54 at `degree=2` and 9.5 × 10⁴ at `degree=3` even with all
24 patches. The 13-parameter fit is already near-degenerate on a full card, which is
why removing rows hurts it so quickly.

**The shaded region.** Below 13 patches a `degree=3` fit has fewer observations than
free parameters and the solution is no longer unique (the minimum-norm solution is
reported). Those boxes are marked in the figure and are not an accuracy level; the
apparent *improvement* below 13 patches is the minimum-norm solution shrinking
towards zero, not recovered accuracy. `degree=2` stays determined down to 6 patches.

---

## Consistency check

The two experiments measure the same quantity at 23 patches by different routes — one
exhaustively over all 24 single removals, the other over 30 random removal orders.
They agree to within 0.013 ΔE₀₀ on the mean in every degree × scope combination
(largest discrepancy: `degree=3` cross-frame, 3.179 vs 3.192). The 30 random orders
happened never to drop cyan first, which is the kind of gap the exhaustive
leave-one-out covers.

Separately, the fast fit/eval path used here was checked against the saved pipeline
results (`per_image_profile_grid.csv`) on all 16 frame pairs: the fit matrix is
identical to `colour.characterisation.matrix_colour_correction_Finlayson2015` to
5.5 × 10⁻⁹, and frame-mean colour error agrees with the full
`ColorCorrector` + `MeasureColor` path to 0.020 ΔE₀₀ on average and 0.071 ΔE₀₀ at
worst.

---

## What this means for the detector

1. **Require cyan.** At the current `degree=3` setting it is the one patch whose loss
   costs more than the frame-to-frame spread. Measured from the reconstructed checker
   canvas of frame `d000220_300_021`, it is the **bottom tile of the right half-card's
   outer column** (chart row 3; tile centre at x 836, y 1484 in a 1670 × 972 canvas) —
   the column clipped by the frame's right border, so also one of the harder tiles to
   segment. Worth handling explicitly rather than leaving to a generic detector. Blue,
   the second most costly removal at +0.44 ΔE₀₀, is the top tile of that same column.
2. **Four arbitrary misses are affordable**, and one or two are nearly free
   (23 patches costs 0.13 ΔE₀₀, 22 costs 0.28). A detector does not need to be
   perfect.
3. **If a detector routinely finds fewer than ~19 patches, change the expansion
   rather than fighting the detector.** `degree=2` on 13 patches (3.27 ΔE₀₀) beats
   `degree=3` on 16 (4.66). Its useful range ends around 13 patches, though: the
   `degree=2` median same-photograph error is 3.27 at 13 patches, 3.73 at 11, 4.58 at
   8, 7.53 at 7 and 13.24 at 6, against a 2.79 full-card baseline — so "determined
   down to 6 patches" is a statement about the fit being unique, not about it being
   accurate. Treat 13 as the practical floor, which is also where the 0.5 ΔE₀₀ miss
   budget of 11 patches runs out. This is relevant to the standing proposal to fit on
   the two clear columns only: 12 patches at `degree=2` gives 3.47 ΔE₀₀
   same-photograph — 0.68 above the full card, usable but not free — whereas
   `degree=3` on 12 patches is underdetermined and meaningless.
4. **The budget assumes misses are arbitrary.** Removal orders here were uniformly
   random. A detector that systematically loses one colour family — for instance all
   six neutrals, or the whole wall-occluded inner column — is not described by these
   medians and would need its own test.

## Caveats

- Only four plate photographs exist, three from one session and one four months
  later, so the cross-frame scope mixes within-session and cross-session reuse and
  the between-plate spread is poorly resolved. Per-patch ranking below cyan and blue
  is not established.
- The half-cards sit at the frame edges, so scoring their tiles against absolute
  reference L\*a\*b\* mixes how brightly that corner of the plate was lit with how
  accurate the colour is. This affects the baseline level in every row above, not the
  paired comparisons between patch counts.
- Outlier rejection was disabled so that "n patches" means exactly n. The pipeline's
  own fit rejects patches beyond `outlier_sigma`; on these frames it rejects none at
  24 patches, but it could reject on small subsets.
- There is no ground truth for plate content, only for the card. Everything here is
  colour accuracy measured on the card tiles.

## Files

- `patch_importance_loo.csv` — every leave-one-out fit (768 rows)
- `patch_importance_summary.csv` — per-patch means, SEM and cost vs baseline
- `patch_reduction_sweep.csv` — the full sweep (18,240 rows)
- `patch_reduction_summary.csv` — median, IQR and degeneracy fraction per patch count
- `patch_removal_permutations.csv` — the 30 removal orders (numpy default_rng seed 1729)
- `patch_measurements.npz` — per-patch substrate for the four frames
- `rpcc_subset.py`, `build_substrate.py` — the fit/eval path and how the substrate was built
