# Code review — `CalibrateColorRpcc` working tree (2026-09-21)

**Reviewed:** the uncommitted working tree on `feat/calibrate-color-rpcc`
(`_calibrate_color_rpcc.py` and `_checker_{detect,identity,measure,qc,roi}.py`
under `src/phenotypic/correction/_color_correction/`), by `/code-review high`.
The reviewer read the code and ran nothing.

**Verified:** every finding now has a failing test in
`tests/unit/correction/test_calibrate_color_rpcc_review.py`, numbered to match.
All of them failed against the reviewed code for the reason stated below. The
fixture control (`test_the_fixture_calibrates_cleanly`) passes. Where running
the test showed the reviewer's mechanism was wrong or overstated, the entry
says so under **Verified behaviour**.

Fix plan: `docs/superpowers/plans/2026-09-21-in-frame-checker-color-correction/review-fixes.md`.

| # | Severity | Finding | Test |
|---|---|---|---|
| 1 | medium | An empty tile crashes identity scoring and bypasses `on_qc_fail` | `test_an_empty_tile_never_turns_the_fit_into_nan` |
| 2 | medium | The rotation ECC recovers is never applied to the measurement boxes | `test_a_rotated_card_is_measured_on_rotated_boxes` |
| 3 | medium | The rank guard counts patches before outlier rejection | `test_rank_is_checked_after_outlier_rejection` |
| 4 | medium | `patch_census` catches `ValueError`, but `apply()` raises `RuntimeError` | `test_patch_census_records_a_failing_frame_instead_of_crashing` |
| 5 | medium | `reference_bands` is `exclude=True` and lost on every rebuild | `test_reference_bands_survive_a_{model_dump,json}_round_trip` |
| 6 | medium | A border-clipped column votes in the anchor-disagreement check | `test_a_clipped_column_does_not_refuse_an_in_range_shift` |
| 7 | medium | Detection failures raise directly and skip the `on_qc_fail` policy | `test_an_expect_tiles_mismatch_is_skipped_under_skip`, `test_an_roi_with_no_card_is_skipped_under_skip` |
| 8 | low | A skipped frame keeps the previous frame's `fitted_profile` | `test_a_skipped_frame_clears_the_previous_profile` |
| 9 | low | `min_patches` has two homes and one silently overwrites the other | `test_min_patches_has_one_home` (+ pin `test_the_operation_min_patches_is_honoured`) |
| 10 | low | Per-ROI lists are not length-checked at construction | `test_a_short_lattice_prior_is_rejected_at_construction`, `test_short_reference_bands_are_rejected_at_construction` |
| 11 | low | ROI Lab is computed under D65 whatever the image's illuminant | `test_roi_lab_uses_the_image_illuminant` |

## 1. Empty tile — `_calibrate_color_rpcc.py:258`

**Reported (high):** an empty tile's NaN sRGB passes every gate, is fitted, and
turns the corrected image into NaN.

**Verified behaviour:** the NaN never gets as far as the fit. It enters
`observed`, and `identity_features` normalises luminance by `values[:,1].max()`,
which is NaN, so every feature becomes NaN. Then
`scipy.optimize.linear_sum_assignment` in `_hungarian_agreement`
(`_checker_identity.py:303`) raises `ValueError: matrix contains invalid numeric
entries`. The result is an opaque crash that bypasses `on_qc_fail`, not silently
wrong output. I lowered the severity to medium for that reason. A stale
`lattice_prior` on a shorter ROI triggers it.

## 2. Rotation dropped — `_calibrate_color_rpcc.py:206`

**Reported (high):** `boxes()` is never given `rot=lattice.rot`. Its rotation
also turns the opposite way to OpenCV's warp and pivots on the lattice centroid
rather than the ROI origin.

**Verified behaviour:** confirmed. On a card rotated 5°, ECC recovers
−5.000° at correlation 1.000, but the tiles are measured on unrotated boxes, and
tile (ROI 1, row 3, col 0) reads 63.1 ΔE00 away from its colour. ECC's
origin-pivot translation (dy 9.7, dx −22.2 for a rotation about the band
centre) is also reported as the card's displacement, which inflates `shift_px`.
Severity is lowered to medium: the spec measured rig rotation at ≤ 0.084°, where
the box error is below a pixel. Any `refine_method="ecc"` user with a rotated
card is still exposed.

## 3. Rank guard before rejection — `_calibrate_color_rpcc.py:340`

Confirmed: with 24 measured patches, degree 4, and three neutrals painted green,
outlier rejection leaves 21 rows for 22 terms, and the fit runs without any
warning.

## 4. `patch_census` exception type — `_calibrate_color_rpcc.py:432`

Confirmed: a blank frame escapes the census as `RuntimeError`.

## 5. `reference_bands` lost on rebuild — `_calibrate_color_rpcc.py:137`

Confirmed for both `model_validate(model_dump())` and `from_json(to_json())`.
The spec declares the field as `list[NdArrayField] | None` with no `exclude`.

## 6. Clipped anchor column — `_calibrate_color_rpcc.py:376`

Confirmed: with column 0 flush to the frame edge and a 25 px shift, the
disagreement is 12.5 px against a 12 px limit, so the ROI is refused.

## 7. Detection failures bypass the policy — `_calibrate_color_rpcc.py:247`

Confirmed for an `expect_tiles` mismatch and for "No patch columns found". The
placements-below-2 branch has the same structure but no test: no chart shape
reaches it without an impossible lattice.

## 8. Stale profile on skip — `_calibrate_color_rpcc.py:332`

Confirmed.

## 9. `min_patches` overwritten — `_calibrate_color_rpcc.py:342`

Confirmed. The spec (README §Operation interface) puts `min_patches` on the
operation only, so the fix removes the `QcLimits` copy rather than choosing a
precedence between the two.

## 10. Per-ROI list lengths — `_calibrate_color_rpcc.py:237`

Confirmed: neither list is checked when the operation is constructed.

## 11. Illuminant dropped — `_calibrate_color_rpcc.py:194`

Confirmed: on a D50 image, the ROI Lab differs from `image.color.Lab` by up to
4.99 in a\*/b\*.

## Noticed while verifying (not a review finding)

- The spec's `insufficient_patches` **QC flag**, which should participate in
  `on_qc_fail` (README §Missing-tile warnings), is not implemented. Below
  `min_patches` the operation only warns. This is left out of the fix plan
  because it changes the gate's behaviour and is a separate decision.
- `fit_lattice` cannot bootstrap on the synthetic rig-shaped fixture: a
  low-contrast column falls below `COLUMN_THRESHOLD_FRACTION`. The fixture uses a
  prior instead. That is consistent with the spec's advice to supply `grid`, but
  there is no synthetic end-to-end test of the bootstrap path.

## Decision after verification (2026-09-21)

The user chose to **remove `refine_method="ecc"` and `reference_bands` from the
operation** rather than make the bands serialisable, which would cost about
40 MB of JSON per band in every pipeline and every provenance journal. That
retires finding 5, the `reference_bands` half of finding 10, and the
operation-level half of finding 2. Their guards in the review test file are
replaced by `test_ecc_is_not_offered_by_the_operation`. The rotation bug in
`refine_ecc`/`CheckerLattice.boxes` is still fixed, and is guarded by
detect-level unit tests (plan Task 5).
