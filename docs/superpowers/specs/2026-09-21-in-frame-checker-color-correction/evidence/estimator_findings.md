# Geometric median vs ΔE2000 medoid: does the tile statistic matter?

**Short answer.** No — swapping one for the other moves a tile's measured colour
by 0.05 ΔE2000 (median), changes the drift you would report by at most
0.06 ΔE2000 out of 10–20, and changes the corrected colour by 0.02 ΔE2000. All
of that is far below the ~1 ΔE2000 a person can see.

**But there is an algorithm issue**, and it is a bigger effect than the question
that prompted the search: the "geometric median" that `ColorCheckerProfile.fit`
uses to reduce each checker tile to one colour is, on 116 of 120 tiles, not a
geometric median. It is a single raw pixel.

---

## What was run

Five frames, 24 ColorChecker patches each (n = 120 tiles):

| frame | what it is | tile pixels (min–max) |
|---|---|---|
| calibration card | the free-standing 24-patch card, cropped out of the flat-field frame | 21k–27k |
| 27 Sep, 30 Sep a, 30 Sep b, 2 Feb | SnP plate photographs; the checker is rebuilt from the two edge-mounted half-cards, keeping only real (non-mirrored) columns | 2.7k–9k |

Every estimator sees byte-identical pixels for a given tile, so any difference
between them is the estimator and not the input. Colours were measured with
`MeasureColor` on a labelled object map, corrections were fitted with
`ColorCheckerProfile` (root-polynomial, degree 3) and applied with
`ColorCorrector` — the shipped code throughout.

Two pixel sets are in play, because the two shipped estimators do not use the
same one:

- **all tile pixels, unfiltered** — what `MeasureColor` sees.
- **core pixels of the median-filtered image** (`core_fraction=0.5`,
  `median_filter_size=10`) — what `ColorCheckerProfile.fit` sees. 1.3k–11k pixels.

Estimators compared on the core set (so the comparison is like-for-like):
geometric median in Lab, ΔE2000 medoid, the shipped profile estimator, a
converged Weiszfeld geometric median in sRGB, and the plain mean.

---

## 1. Does the estimator change the drift you detect?

No. On identical core pixels, geometric median minus medoid:

- per tile: median 0.052, mean 0.063, 95th percentile 0.136, max 0.342 ΔE2000
- per-tile accuracy against the reference card: mean difference −0.004 ΔE2000
  (Wilcoxon signed-rank p = 0.81, n = 120)
- per-frame mean drift changes by ≤ 0.06 ΔE2000 against a signal of 9.9
  (calibration card) to 20.5 (plate half-cards); the ranking of frames by drift
  is identical under either statistic.

On all tile pixels (the `MeasureColor` columns `ColorLab_*GeoMedian` vs
`ColorLab_*Medoid`) the two differ a little more — median 0.088, max 0.608 —
and the disagreement tracks how noisy the tile is (Spearman ρ = 0.46 against
within-tile ΔE2000 spread). It is largest on the calibration card, whose tiles
are both larger and noisier (mean within-tile spread 3.34 vs 0.95–1.16 on the
plate half-cards), and it never reaches the visible limit.

## 2. Does the estimator change the correction?

No. Fitting five profiles per frame that differ only in the tile colour handed
to the fitter:

| statistic used for the fit | in-sample tile error after correction (ΔE2000, mean of 120 tiles) | chip-fit applied to plate frames (held out) |
|---|---|---|
| geometric median (Lab) | 2.185 | 16.173 |
| ΔE2000 medoid | 2.177 | 16.155 |
| shipped profile estimator | 1.909 | 15.921 |
| converged geometric median (sRGB) | 2.211 | 16.190 |
| plain mean | 1.953 | 15.916 |

Uncorrected, the same tiles sit at 9.95 (card) and 18.1–20.5 (plates). Paired
per tile, the geometric-median fit and the medoid fit differ by +0.009 ΔE2000 on
average (p = 0.11, n = 120; largest single-tile difference 0.30). The fitted
matrices differ by 1–8 % in their largest element, but in directions the data do
not probe: the corrected colours are the same. No patch was rejected by the
outlier gate under any estimator; condition numbers stayed in 26–117 throughout.

Within-tile colour spread rises after correction under every estimator (1.51 →
2.00–2.09 ΔE2000 mean), as it did in earlier runs — the correction applies a
luminance gain and amplifies per-pixel noise with it. That is why the pipeline
runs `DenoiseBlockMatch` after `ColorCorrector`, and it is estimator-independent.

## 3. The algorithm issue

### 3.1 The profile fit's geometric median is usually a single pixel

> **Fixed, 19 Sep 2026.** `_color_checker_profile.py` now reduces each patch
> with `phenotypic.util.robust_color_center`, and `_helpers.geometric_median`
> delegates to that same solver, so the package has one geometric median.
> `util.geometric_median` now defaults to `method="weiszfeld"`. Re-fitting the
> calibration card: 0/24 patches return a raw pixel (was 24/24), and the
> in-sample residual moves 1.108 → 1.139 ΔE2000, exactly as predicted below.
> The rest of this section describes the behaviour before that change.

`ColorCheckerProfile._fit_from_rois` reduces each tile with
`_helpers.geometric_median` (`correction/_color_correction/_color_checker_profile.py:593`).
That function (`correction/_color_correction/_helpers.py:379-421`) runs Weiszfeld
from the mean with `eps=1e-3`, `max_iter=20`, and this early exit:

```python
distances = np.linalg.norm(points - guess, axis=1)
if np.any(distances < eps):
    return points[distances.argmin()].copy()      # _helpers.py:410-414
```

`eps` is doing two jobs — convergence tolerance and coincides-with-a-data-point
test — in units of gamma-encoded sRGB on [0, 1]. A checker tile holds thousands
of near-identical pixels, so after the first iteration some pixel is essentially
always within 1e-3 of the running estimate and the function returns *that pixel*.

**Measured: this fired on 116 of 120 tiles (96.7 %).** So the value the
correction is fitted on is a raw pixel near a two-iteration Weiszfeld
iterate — which is to say, a medoid-like estimate seeded at the mean. It sits
0.40 ΔE2000 (median; mean 0.49, max 1.98) from a converged geometric median of
the same pixels, and only 0.08 ΔE2000 (median) from the plain mean. That is
four to eight times the geometric-median-vs-medoid difference this experiment
set out to measure.

The iteration cap compounds it: a converged Weiszfeld on these tiles needs 25–109
iterations (median 48), so `max_iter=20` would truncate even without the early
exit.

**Direction of the effect.** On this data the shortcut is not harmful — it is
marginally *better* (1.909 vs 2.211 ΔE2000 in-sample), because it lands near the
mean, and with dust and specular pixels already removed by the median filter and
the core mask, the mean is the better centre for a least-squares fit. Treat this
as a correctness and reproducibility defect, not an accuracy loss: the code does
not compute what its name, its docstring and its caller all say it computes, and
what it does compute depends on pixel ordering and on an absolute tolerance in
sRGB units that will behave differently on a darker or brighter capture.

Suggested fix: separate the two roles of `eps` — keep a small coincidence guard
(or drop it and clamp distances, as the other implementation does), give the
convergence test its own tolerance, and raise `max_iter` to ~200. If the
mean-like behaviour is actually wanted, make it an explicit choice.

### 3.2 MeasureColor's geometric median is correct

`robust_color_center` (`util/_robust_color_stats.py:17-45`) delegates to
`weiszfeld_median` (`util/_geometric_median.py:1097`), which clamps distances
rather than snapping to a pixel. Its output matched a fully converged Weiszfeld
to within 2.6e-4 ΔE2000 across all 120 tiles, so the defaults
(`geomedian_max_iter=50`, `geomedian_tol=1e-4`) are not binding in practice.

Note that the repo now holds two different geometric medians — different space
(Lab vs sRGB), different initialisation behaviour, different tolerance, one with
a snap-to-pixel exit and one without. Worth consolidating on the correct one.

### 3.3 The medoid's subsample is noisier than the estimator choice

`medoid_ciede2000` picks the medoid from a seeded random subsample of at most
1000 pixels (`util/_robust_color_stats.py:86-103`). Re-drawing that subsample
with nine other seeds moves the reported medoid by 0.12 ΔE2000 (median of the
per-tile mean; worst tile 0.81), and raising the cap to 5000 pixels moves it by
0.08 (max 0.35). Both exceed the geometric-median-vs-medoid difference itself.

The seed is fixed at 0, so a given run is reproducible; but the medoid columns
carry sampling noise of the same order as the quantity under study, and the
noise scales with within-tile spread (ρ = 0.75). If the medoid is used for
anything quantitative, raise `medoid_max_pixels` — 5000 costs ~2.7 s per tile
against 0.1 s at 1000, which is acceptable for 24 checker tiles and not for
thousands of colonies.

### 3.4 An unrelated snag worth recording

The calibration frame shows the free-standing card *and* one of the rig's
edge-mounted half-cards, so whole-frame segmentation finds 36 chips and the
strict 24-chip gate refuses it. The fit needs an explicit `rois=` bounding the
central card (rows 1450–3074, columns 2836–3815 for this frame).

---

## Recommendation

Keep whichever statistic you prefer for the tile colour — on this evidence the
choice is not worth an argument, and the ΔE2000 medoid's main disadvantage is
its subsample noise rather than its central value. Fix `_helpers.geometric_median`
instead, and decide deliberately whether the fit should use a mean, a geometric
median, or a medoid, because right now it silently uses something between the
first two.

## What this does not cover

- One capture rig, one camera, one development profile; 4 plate frames from 3
  sessions. Estimator differences that only appear under a different noise
  character (higher ISO, JPEG, a different sensor) would not show up here.
- Degree-3 root-polynomial correction only; no other correction family tested.
- Held-out transfer was tested in one direction only (card-fitted correction
  applied to plate frames). The large held-out error (≈16 ΔE2000) is the
  previously documented luminance deficit of the edge-mounted half-cards, not
  an estimator effect — it is the same for all five.
- Colony measurements were not touched. Whether the estimator choice matters for
  pigmented colonies — which are smaller, rounder and more heterogeneous than a
  checker tile — is a separate question, and the noise-dependence in §1 suggests
  it would be worth checking before assuming the answer carries over.
