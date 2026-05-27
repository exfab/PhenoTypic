# Cross-channel stddev ROI mask for color-checker patch detection

**Status:** Draft — awaiting user review
**Author:** Alex Nguyen + Claude (brainstorming)
**Date:** 2026-05-26
**Module:** `src/phenotypic/correction/_color_correction/`

## Problem

`ColorCheckerProfile._fit_from_rois` feeds the entire pre-processed ROI to
`lab_checker_cluster_masks`, which performs a nearest-neighbour assignment in
Lab space against the 24 reference patches. Each pixel is assigned to the
closest reference patch unless its Lab distance exceeds
`border_distance_threshold` (default 12.0).

The checker card's dark frame (outer border, central divider, inter-swatch
gutters) sits at Lab ≈ (10–30, 0, 0). The reference F4 "black" patch is at
Lab (20.64, 0.07, -0.46). Border pixels are within ΔE 12 of F4 and get
assigned to it. The F4 patch mask returned by clustering is therefore *the
dark frame itself*, not the F4 swatch. Downstream
`compute_core_mask` + `geometric_median` then measure the frame's color in
place of the patch's, contaminating the correction matrix fit.

The bug presents as: F4 patch "detected" but its measured color matches the
frame, not the swatch.

## Goal

Stop dark-frame pixels from entering the Lab-NN clustering pool, so each
patch mask contains only true swatch-interior pixels.

## Approach

Build a 2-D boolean ROI mask from per-row and per-column cross-channel Lab
stddev magnitude, then pass it to `lab_checker_cluster_masks(roi_mask=...)`.
Pixels whose row or column is uniform (border-like) are excluded before
clustering. The mechanism is the same primitive
(`find_cross_channel_stddev_magnitude`) already used by
`center_and_pad_checker` for centering, generalised to both axes and combined
into a 2-D mask.

Rationale: dark frame regions have low cross-channel Lab stddev along their
spanning axis (uniform color); swatch interiors have high cross-channel
stddev along *both* axes (varying colors across patches). The intersection
of high-stddev rows and high-stddev columns is exactly the swatch grid.

This approach was prototyped (along axis=0 only, for centering) in
`SnP-ColorCorrection/ColorCorrectionAutomated-SnP.ipynb`.

## Design

### 1. New helper in `_helpers.py`

```python
def compute_swatch_roi_mask(
    lab_image: np.ndarray,
    stddev_mag_threshold: float = 15.0,
    filter_size: int = 10,
) -> np.ndarray:
    """Return a 2-D bool mask True for swatch-interior pixels.

    Computes cross-channel Lab stddev magnitude along both axes via
    ``find_cross_channel_stddev_magnitude``. Columns and rows with magnitude
    ``<= stddev_mag_threshold`` are classified as border (uniformly dark
    checker frame, center divider, inter-swatch gutters). The returned mask
    is True only at pixels whose row AND column are above threshold — i.e.
    only patch interiors survive.
    """
    col_mag = find_cross_channel_stddev_magnitude(
        lab_image, axis=0, filter_size=filter_size
    )  # shape (1, W)
    row_mag = find_cross_channel_stddev_magnitude(
        lab_image, axis=1, filter_size=filter_size
    )  # shape (H, 1)
    col_swatch = (col_mag > stddev_mag_threshold).ravel()  # (W,)
    row_swatch = (row_mag > stddev_mag_threshold).ravel()  # (H,)
    return row_swatch[:, None] & col_swatch[None, :]       # (H, W)
```

Reuses the existing `find_cross_channel_stddev_magnitude` primitive. Private
to the color-correction package; not re-exported.

### 2. Extend `_RoiPreprocessing` NamedTuple

`_color_checker_profile.py:42` gains one field:

```python
class _RoiPreprocessing(NamedTuple):
    original: np.ndarray
    trimmed: np.ndarray
    filtered: np.ndarray
    padded: np.ndarray
    padded_normed: np.ndarray
    lab: np.ndarray
    swatch_roi_mask: np.ndarray  # NEW: (H, W) bool, True for non-border pixels in `padded`
```

### 3. Compute the mask in `_preprocess_roi`

After the canonical Lab is produced (`_color_checker_profile.py:324`):

```python
swatch_roi_mask = compute_swatch_roi_mask(
    sub_image.color.Lab[:],
    stddev_mag_threshold=self.stddev_mag_threshold,
    filter_size=self.median_filter_size,
)
if not swatch_roi_mask.any():
    logger.warning(
        "Cross-channel stddev ROI mask is empty for ROI; falling back to "
        "full-image clustering. Consider lowering stddev_mag_threshold."
    )
    swatch_roi_mask = np.ones(sub_image.shape[:2], dtype=bool)
```

Returned as the new field of `_RoiPreprocessing`.

### 4. Pass mask into clustering in `_fit_from_rois`

`_color_checker_profile.py:407`:

```python
cluster_result = lab_checker_cluster_masks(
    lab_padded,
    ref_Lab_tuples,
    border_distance_threshold=self.border_distance_threshold,
    roi_mask=prep.swatch_roi_mask,   # NEW
    include_labels=True,
)
```

`lab_checker_cluster_masks` already accepts `roi_mask` and respects it — no
helper change needed.

### 5. Post-cluster area-outlier safety net

After clustering returns `(masks, bboxes, labels)`, compute the median
non-border mask area across all detected patches (oversized candidates are
included in the median — the naive single-pass approach is sufficient given
the stddev mask already excludes most border pixels; iterative refinement
is out of scope for v1). Drop any patch whose mask area is more than
`2 × median_area` from the measurement pool, and append a human-readable
string to `diagnostics["warnings"]`. The patch's entry in
`diagnostics["patches"][name]` records `oversized_dropped: True` (default
`False` when the field is omitted, for backwards compatibility with the
existing patch-diagnostics consumers) so the dashboard can flag it.

This catches edge cases where the stddev mask leaks part of a frame region
into the clustering pool (e.g., a card without a clean frame–swatch
transition).

### 6. Dashboard integration (`_diagnostic_dashboard.py`)

Two touch-points:

1. **Line 292** — the segmentation panel re-runs `lab_checker_cluster_masks`
   to visualize the result. Pass `roi_mask=prep.swatch_roi_mask` so the
   panel shows the same clustering the fit used.

2. **Pipeline stages panel** (lines 215–222) — currently displays
   `trimmed → filtered → padded`. Append a fourth stage:
   `padded × swatch_roi_mask[..., None]` (or a 50%-darken overlay) so the
   user can visually confirm the mask boundary on the card.

### 7. Parameter surface

**No new pydantic fields.** The new behavior is gated by existing fields on
`ColorCheckerProfile`:

| Field | Role | Default |
|---|---|---|
| `stddev_mag_threshold` | Border vs swatch threshold (now used by both centering and ROI mask) | 15.0 |
| `median_filter_size` | Kernel for the stddev pre-filter | 10 |
| `border_distance_threshold` | In-swatch outlier safety net during Lab-NN (unchanged role) | 12.0 |

`border_distance_threshold` stays because its role changes from "primary
border filter" (where it was insufficient) to "in-swatch outlier rejector"
(specular highlights etc.), which it handles fine at 12.0.

## Behavior change summary

| Path | Before | After |
|---|---|---|
| `pad_checker=True`, framed card | F4 mask = entire dark frame | F4 mask = real F4 swatch |
| `pad_checker=False`, hand-tight ROI | F4 mask leaks outer frame | F4 mask clean |
| Frameless / pure-grid input | unchanged | unchanged (mask is all-True) |
| Card with frame breaks | unchanged or worse | safety net drops contaminated patch |

## Backwards compatibility

- **Public API:** `ColorCheckerProfile` constructor signature, `model_dump`,
  `model_json_schema` — all unchanged.
- **Persisted profiles:** the only persisted numeric field is
  `correction_matrix`. Existing JSON dumps round-trip unchanged.
- **Internal API:** `_RoiPreprocessing` and `compute_swatch_roi_mask` are
  private. The dashboard is the only consumer of the new field besides
  `_fit_from_rois`.
- **Existing fit results will change.** Any test asserting on specific Lab
  measurements, ΔE values, or correction-matrix entries needs to be
  re-baselined. To re-baseline as part of the implementation rather than
  ahead of it.

## Test plan (`tests/unit/correction/test_color_corrector.py`)

**Helper unit tests for `compute_swatch_roi_mask`:**
- Synthetic 3×4 swatch grid with 20-px Lab≈(20, 0, 0) frame → mask True only
  inside the grid, False on every frame row/col.
- All-uniform image → mask all-False (degenerate input handled at caller).
- Pure swatch grid with no frame → mask all-True (no border detected).

**Integration regression test in `TestEdgeNoiseHandling`:**
- Build a synthetic 24-patch checker image with a thick (≥50-px)
  Lab≈(20, 0, 0) frame around the swatch grid.
- Fit the profile.
- Assert F4 measured Lab is within ΔE 2000 < 3 of the reference (currently
  this would fail because F4's measurement collapses to the frame color).
- Assert F4's `core_fraction_used` is plausible (mask wasn't dominated by
  frame).

**Safety-net test:**
- Construct a scenario where one patch mask is > 2 × median patch area
  (either via a crafted fixture or by monkeypatching
  `lab_checker_cluster_masks`).
- Assert that patch is recorded as `oversized_dropped: True` in
  `diagnostics["patches"]` and a corresponding string appears in
  `diagnostics["warnings"]`.

**Fallback test (empty mask):**
- Force the helper to return all-False (very high threshold) → assert
  `_preprocess_roi` falls back to all-True, logs warning, fit still
  succeeds.

**Dashboard regression test:**
- Extend an existing `TestColorCorrectionDashboard` test using a
  framed-checker fixture; assert the segmentation panel renders without
  error when `swatch_roi_mask` flows through.

Expected: ~8 new tests, all in `test_color_corrector.py`. No new fixture
files — synthetic generation inline, following the existing
`make_synthetic_checker` helper.

## Files touched

| File | Change |
|---|---|
| `src/phenotypic/correction/_color_correction/_helpers.py` | Add `compute_swatch_roi_mask` |
| `src/phenotypic/correction/_color_correction/_color_checker_profile.py` | Add `swatch_roi_mask` field to `_RoiPreprocessing`; compute in `_preprocess_roi`; pass to `lab_checker_cluster_masks` in `_fit_from_rois`; add area-outlier safety net + diagnostics fields |
| `src/phenotypic/correction/_color_correction/_diagnostic_dashboard.py` | Pass `roi_mask` to dashboard clustering call; append mask-overlay stage to pipeline panel |
| `tests/unit/correction/test_color_corrector.py` | ~8 new tests covering helper, integration, safety net, fallback, dashboard |

## Out of scope

- Adaptive thresholding per ROI (one threshold per fit is enough).
- Detecting checker card geometry from raw images (caller still supplies
  ROIs as today).
- Changing the Finlayson 2015 fitting routine or the post-fit
  Δ E 2000 outlier rejection.
- Migrating other parts of the package that consume
  `find_cross_channel_stddev_magnitude`.
