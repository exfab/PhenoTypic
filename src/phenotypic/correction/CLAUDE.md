# correction/

Whole-image transformations: every operation here is an `ImageCorrector`, which
means it rewrites `rgb`, `gray` and `detect_mat` together rather than touching
one component. Read `abc_/CLAUDE.md` first for that contract.

## What lives here

| Area | Operations |
|---|---|
| Colour | `CalibrateColorRpcc`, `ColorCorrector`, `ColorCheckerProfile`, `ColorDenoise` |
| Noise | `DenoiseBlockMatch`, `BayesShrinkCorrector`, `VisuShrinkCorrector` |
| Geometry | `GridAligner`, `CropImage`, `PadImage` |

## Colour correction: which entry point

- **`CalibrateColorRpcc`** — the chart is *in the plate photograph*. Give it the
  rectangles where the cards appear and it locates the tiles, works out which
  chart patch each one is, measures them, fits a profile and applies it. Use
  this for a rig with edge-mounted half-cards.
- **`ColorCheckerProfile.fit(image)`** — the chart is free-standing and fully
  visible, photographed on its own. Border-fill segmentation does a better job
  than lattice detection when the whole card with intact gutters is present.
- **`ColorCorrector(profile=...)`** — a profile already exists and only needs
  applying.

Fitting from the frame's own card is most of the accuracy, not a convenience:
the root-polynomial expansion is homogeneous of degree 1, so a same-image fit
absorbs an exposure difference as a matrix scaled by 1/k, while a transferred
profile is exposure-invariant by construction and preserves the deficit. Via
the earlier notebook extraction, a transferred chip profile measured 15.93 mean
ΔE2000 on the reference set's in-frame tiles against 1.95 for a per-image fit.

**`CalibrateColorRpcc` does not yet reach that figure.** On the same three
frames it measures 3.40 / 2.88 / 4.74 in-sample and 4.37–11.35 held out. The
median per-patch error is 2.11, so the mean is carried by a few tiles — purple,
foliage and the neutrals, which also have the highest within-tile spread,
pointing at core-box centring rather than the fit. Task 7 of
`docs/superpowers/plans/2026-09-21-in-frame-checker-color-correction/plan.md`
carries the diagnosis; do not quote 1.95 as this operation's accuracy.

Run `DenoiseBlockMatch` *after* correction: correction improves accuracy while
mildly amplifying per-pixel noise (within-tile ΔE2000 spread 3.32 → 3.63 median
on the calibration chip).

## Invariants a change here must not break

- **Reference values are Bradford-adapted** from the chart's own illuminant to
  the working illuminant before any comparison, in `_load_reference_data`. Never
  compare measured colour against unadapted chart values, and never re-derive
  reference values in new code.
- **All Lab comes from `Image.color.Lab`.** Feeding linear RGB to a converter
  that expects sRGB yields a self-consistent contrast measure, not calibrated
  CIE — every threshold expressed in those units then means something other
  than it says.
- **The polynomial degree is fixed across a run.** Never adapt it per frame,
  and never expose it to a tuner. Degree 2 and degree 3 differ in a
  reproducible direction (median 2.17 ΔE2000, 5.9× its across-frame scatter),
  so mixing them puts a bias into every between-frame contrast. A short card
  warns; only rank-insufficiency raises.
- **Patch identity is derived, never declared**, and decided by scoring whole
  rigid placements. A free per-tile assignment lets a label wander onto
  whichever reference fits it best, silently mislabelling an occluded tile.
- **`CheckerRoi` carries only a rectangle.** If you find yourself adding a
  field that names chart patches or a layout, re-read the spec first —
  `test_roi_declares_nothing_about_its_contents` guards it deliberately.

## Module map

`_color_correction/` splits the calibration pipeline one stage per file:
`_checker_roi` (rectangles and lattices, no image access) → `_checker_detect`
(plateau counting, lattice fit, rigid refinement; ECC as a building block
only) → `_checker_identity` (placement enumeration and scoring) →
`_checker_measure` (core trimming, deterministic ΔE2000 medoid, contamination
statistics) → `_checker_qc` (gate) → `_calibrate_color_rpcc` (orchestration).
`_color_checker_profile` owns the solver both entry points share.

Design spec and evidence:
`docs/superpowers/specs/2026-09-21-in-frame-checker-color-correction/`.
