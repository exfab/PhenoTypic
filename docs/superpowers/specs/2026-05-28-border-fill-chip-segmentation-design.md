# Border-fill chip segmentation for color-checker profile fit

**Status:** Draft — awaiting user review
**Author:** Alex Nguyen + Claude (brainstorming)
**Date:** 2026-05-28
**Module:** `src/phenotypic/correction/_color_correction/`

## Problem

`ColorCheckerProfile._fit_from_rois` segments checker chips by **per-pixel
Lab nearest-neighbour clustering** (`lab_checker_cluster_masks`): each pixel
is assigned to whichever of the 24 reference patches it is closest to in Lab
space, unless that distance exceeds `border_distance_threshold` (12.0).

This is fragile:

- A dark/ambiguous chip can be *stolen* by a neighbour or by leftover frame
  pixels, because membership is decided by absolute colour proximity, not by
  which physical cell a pixel sits in. The motivating bug was the dark
  checker frame being labelled as the black F4 patch.
- A chip whose measured colour drifts more than ΔE 12 from every reference is
  silently dropped.

The segmentation depends on the colours being approximately right — which is
backwards, since the colours are exactly what we are trying to measure and
correct.

## Goal

Replace colour-based segmentation with **geometry-based segmentation**: use
the already-computed background mask to isolate each chip as an enclosed
region, fill it into a solid binary mask, then assign each region to a
reference patch by an optimal one-to-one colour match.

Chip *membership* becomes geometric (which cell between the grid lines);
colour is used only for *labelling* (which of 24 names), and only as a global
optimal assignment rather than a per-pixel absolute cutoff.

## Approach

`compute_swatch_roi_mask` already returns a 2-D boolean mask that is `True`
on swatch interiors and `False` on the frame, central divider, and
inter-swatch gutters (low cross-channel Lab stddev along a row or column).
Its `True` regions are exactly the chips, already separated by `False` grid
lines.

1. Fill enclosed holes, then connected-component label the mask → one blob
   per chip.
2. Size-filter stray noise, then require exactly the expected chip count
   (strict gate).
3. Hungarian-match each blob's median Lab against the 24 reference Labs to
   assign names.
4. Measure each labelled blob's colour exactly as today (core mask +
   geometric median) and feed `_fit_from_measured` unchanged.

## Design

### 1. Connected-component segmentation (`_fit_from_rois`)

Per ROI, after `_preprocess_roi` produces `prep`:

```python
from scipy.ndimage import binary_fill_holes, label as ndi_label

mask = binary_fill_holes(prep.swatch_roi_mask)   # fill specular/noise holes
labeled, n_components = ndi_label(mask)           # default 4-connectivity
```

`binary_fill_holes` fills only `False` regions *not* connected to the array
border. Every gutter runs to the outer frame, which touches the image edge,
so gutters are never filled — only truly enclosed holes inside a chip are.
Each enclosed chip interior therefore becomes one labelled blob.

### 2. Size filter + strict count gate

```python
areas = np.bincount(labeled.ravel())[1:]          # component areas, skip bg=0
median_area = float(np.median(areas)) if areas.size else 0.0
keep = [i + 1 for i, a in enumerate(areas)
        if median_area > 0 and a >= self.min_swatch_area_frac * median_area]

if len(keep) != n_expected:
    raise ValueError(
        f"Border-fill segmentation found {len(keep)} chips in ROI "
        f"{roi_idx}, expected {n_expected}. Gutters may have merged "
        f"(try raising stddev_mag_threshold) or the card is partially "
        f"occluded."
    )
```

`n_expected` is `len(patch_names)` (24 for ColorChecker24). The strict gate
fails loudly rather than silently producing a partial fit — the chosen
behaviour.

### 3. Hungarian labelling

```python
observed_Lab = np.vstack([
    np.median(prep.lab[labeled == lbl], axis=0) for lbl in keep
])  # (n_expected, 3)
mapping = hungarian_match_swatches(observed_Lab, ref_Lab_tuples)
#   -> {patch_name: row_index_into_observed_Lab}
blob_to_name = {keep[row]: name for name, row in mapping.items()}
```

Reuses `hungarian_match_swatches` (`_helpers.py:562`), previously dead code:
it builds a ΔE 2000 cost matrix and solves the linear-sum assignment for the
globally optimal one-to-one mapping. Because it is a relative optimal
assignment, a chip whose measured Lab is off by a large ΔE is still labelled
correctly as long as it remains the best available match for that reference —
the per-pixel ΔE-cutoff failure mode is gone.

### 4. Per-chip measurement (unchanged)

For each labelled blob, exactly as today:

```python
blob_mask = labeled == lbl
core = compute_core_mask(blob_mask, core_fraction=self.core_fraction)
_, warnings = validate_patch_shape(core)
core_fraction_used = float(core.sum()) / max(float(blob_mask.sum()), 1.0)
patch_srgb = geometric_median(prep.padded_normed[core])   # sRGB [0,1]
measured_srgb[blob_to_name[lbl]].append(
    (patch_srgb, roi_idx, core_fraction_used, warnings)
)
```

Multi-ROI pooling (best `core_fraction_used` per patch) and
`_fit_from_measured` are untouched.

### 5. Parameter / API surface

**New field on `ColorCheckerProfile`:**

| Field | Type | Default | Meaning |
|---|---|---|---|
| `min_swatch_area_frac` | `float` | `0.3` | Components below this fraction of the median component area are discarded as noise before the strict count gate. Validated `0 < x <= 1`. |

**Removed field:** `border_distance_threshold` (the per-pixel ΔE cutoff) is
deleted along with its references. To keep already-serialised profiles
deserialisable, add `extra="ignore"` to the model config:

```python
model_config = ConfigDict(
    arbitrary_types_allowed=True,
    validate_assignment=True,
    extra="ignore",          # tolerate the removed border_distance_threshold key
)
```

**Removed helper:** `lab_checker_cluster_masks` (`_helpers.py`) and its
dedicated unit tests are deleted — nothing calls it after this change.
`hungarian_match_swatches` flips from dead code to the labelling path and
stays.

**Unchanged:** `stddev_mag_threshold` (now even more central — it defines the
gutters that separate blobs), `median_filter_size`, `core_fraction`,
`outlier_sigma`, `ridge_lambda`, `degree`, the entire `_fit_from_measured`
routine, `_RoiPreprocessing` (still carries `swatch_roi_mask`), and the
serialisation contract for `correction_matrix`.

### 6. Dashboard (`_diagnostic_dashboard.py`)

`_segmentation_section` (~line 292) currently calls
`lab_checker_cluster_masks`. It migrates to the same connected-component +
Hungarian pipeline so the panel mirrors the fit:

- `binary_fill_holes` → `label` → size-filter → Hungarian-label the blobs.
- Overlay each labelled blob in a distinct colour, annotated with the matched
  patch name (replaces the per-pixel cluster overlay).
- The dashboard must never raise on inspection: if the strict-24 gate would
  fail, render whatever blobs were found with a visible warning banner
  instead of raising. (Diagnosis is precisely when the count is wrong.)

The pipeline panel's "5. Border Mask" stage stays — it visualises
`prep.swatch_roi_mask`, now the literal segmentation source.

## Behaviour change summary

| Scenario | Before | After |
|---|---|---|
| Dark frame near a chip colour | Frame pixels stolen into that chip | Frame is not a blob; impossible |
| Chip drifts > ΔE 12 from reference | Chip silently dropped | Still labelled via optimal match |
| Two chips merge across a thin gutter | Quietly mismatched / merged | `ValueError` (strict count gate) |
| Card partially occluded / < 24 chips | Partial best-effort fit | `ValueError` (strict count gate) |
| Frameless / pure-grid input | Works | Likely one merged blob → raises (out of scope; this method assumes visible gutters) |

## Test plan (`tests/unit/correction/test_color_corrector.py`)

Reuses the synthetic `make_synthetic_framed_checker_image` fixture.

- **CC segmentation unit test** — `binary_fill_holes` + `label` on a framed
  grid yields exactly 24 blobs; each blob centroid lands in the correct
  swatch cell.
- **Hungarian labelling unit test** — blobs presented in scrambled spatial
  order each map to the correct reference name.
- **Strict-count raise test** — a `gutter=0` fixture (or a stddev threshold
  low enough to merge chips) makes `fit()` raise `ValueError` naming the
  count found.
- **Integration / regression test** — framed-checker `fit()`: all 24 patches
  measured; black chip `measured_lab` within ΔE 2000 < 5 of reference (the
  original motivating bug, now solved structurally).
- **Serialization back-compat test** — `ColorCheckerProfile.model_validate`
  on a dict containing a stray `border_distance_threshold` key succeeds and
  ignores it.
- **Dashboard test** — segmentation panel renders without error using the new
  pipeline.

Removed: `lab_checker_cluster_masks` dedicated tests; the
`border_distance_threshold == 12.0` assertion in `test_valid_defaults`.

## Files touched

| File | Change |
|---|---|
| `src/phenotypic/correction/_color_correction/_color_checker_profile.py` | Replace clustering with CC + Hungarian in `_fit_from_rois`; add `min_swatch_area_frac` field + validator; remove `border_distance_threshold`; add `extra="ignore"`; swap helper imports |
| `src/phenotypic/correction/_color_correction/_helpers.py` | Remove `lab_checker_cluster_masks` |
| `src/phenotypic/correction/_color_correction/_diagnostic_dashboard.py` | Migrate `_segmentation_section` to CC + Hungarian; warning banner instead of raise |
| `tests/unit/correction/test_color_corrector.py` | New CC/Hungarian/strict/regression/back-compat/dashboard tests; remove obsolete tests + assertion |

## Out of scope

- Frameless or gutterless cards (this method requires visible grid lines to
  separate chips; such inputs raise the strict-count error).
- Watershed splitting of merged chips (strict gate raises instead).
- Auto-detecting checker geometry from raw images — the caller still supplies
  ROIs.
- Rotation/perspective correction of the card.
- Changes to the Finlayson 2015 fit or the post-fit ΔE 2000 outlier
  rejection.
