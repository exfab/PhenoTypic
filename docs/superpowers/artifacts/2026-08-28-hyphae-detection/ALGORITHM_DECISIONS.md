# Final hyphae-detection algorithm decisions

## Decision status

These are the final pipeline choices for the present branch-orientation study and
its figures. They are based on 16 labeled crops from the same evaluation set:
four *Neurospora crassa* on menadione, four *N. crassa* on xylan, and eight
*Ganoderma* on glucose plus yeast extract. They are not independently validated
production defaults, and no production detector code was changed during this
study.

The annotations are subjective Sholl-style colony envelopes rather than
pixel-accurate branch masks. The decisions therefore prioritize all-crop visual
enclosure, detection reliability, sparse-margin coverage, and convex-envelope
coverage together. The numerical scores are comparative diagnostics.

## Selected pipelines

| Species and medium | Selected detector pipeline | Decision |
|---|---|---|
| *N. crassa*, menadione | Current `TwoKFilamentousDetector` pipeline | Retain the simpler current pipeline. All four TwoK variants had median coverage 1.000, while the current path succeeded on 4/4 crops with minimum coverage 1.000. The evaluation does not justify adding an enhancer for this phenotype. |
| *N. crassa*, xylan | `TwoKFilamentousDetector` with monogenic phase congruency inside `branch_base` | Use the phenotype-specific monogenic branch enhancement. It had the highest median overall coverage, 0.883, and median sparse-zone coverage, 0.865, with 4/4 successful detections. |
| *Ganoderma*, glucose plus yeast extract | Grayscale plus oriented phase-congruency-transform composite, then SAM2 | Use `max(stretched_gray, oriented_PCT)` without background subtraction. It had median overall coverage 0.909, minimum coverage 0.835, median sparse-zone coverage 0.903, and 8/8 successful detections. |

There is no selected universal pipeline. Species and medium, or an equivalent
phenotype classification supplied by the analysis configuration, must choose the
detector path.

## Exact tested configurations

### *N. crassa* on menadione

Use the current TwoK branch base:

1. `FlattenIllumination(sigma=300.0)`.
2. `ContrastStretching(lower_percentile=70, upper_percentile=99)`.
3. `TwoKFilamentousDetector` with `max_colony_radius_px` equal to half the
   smaller crop dimension, `min_branch_width_px=3`, and
   `reconnect_scope="branches"`.

Do not add local contrast, unsharp masking, or phase congruency by default for
this condition. This is a parsimony decision under a coverage tie, not evidence
that those alternatives are inferior on an independent dataset.

### *N. crassa* on xylan

Use the same TwoK settings, but append
`FocusEdgeMonogenicPhase(output="pc")` inside `branch_base`, after illumination
flattening and contrast stretching. The tested monogenic operation used its
defaults: `n_scale=4`, `min_wavelength=3.0`, `mult=2.1`, `sigma_onf=0.55`,
`k=3.0`, `deviation_gain=1.5`, `cutoff=0.5`, `g=10.0`, and
`noise_method=-1.0`.

The enhancer belongs inside `branch_base`. External enhancement also alters the
center/body path and was less reliable in the representative screens.

### *Ganoderma* on glucose plus yeast extract

Construct the SAM2 input as follows:

1. Apply `ContrastStretching()` to `detect_mat`, using its tested defaults of
   the 2nd and 98th percentiles.
2. From that stretched image, compute
   `FocusEdgePhase(n_orient=8, min_wavelength=6.0, k=6.0,
   output="pc_sum")`.
3. Form `composite = maximum(stretched_gray, pct).clip(0, 1)`.
4. Pass the composite as `detect_mat` to SAM2.

The tested SAM2 configuration was `model_size="tiny"`,
`points_per_side=32`, `points_per_batch=8`, `pred_iou_thresh=0.5`,
`stability_score_thresh=0.5`, `min_mask_region_area=100`,
`crop_n_layers=0`, `device="cpu"`, and `input_layer="detect_mat"`.

Do not apply Gaussian background subtraction before the selected composite.
Adding subtraction to the same grayscale/PCT construction reduced median
overall coverage from 0.909 to 0.703 and median sparse-zone coverage from 0.903
to 0.587. Current subtraction plus SAM2 had median overall coverage 0.561 and
median sparse-zone coverage 0.144.

## Operational rules and unresolved deployment issue

- Ganoderma label 3 (`DenseCoreUnresolved`) may be included with the dense/core
  region. A separate inoculum-core boundary is not required for the downstream
  orientation-window objective.
- Historical Neurospora label 3 is normalized to sparse label 4 before scoring.
- Detector failure is scored as zero, so method selection accounts for both
  coverage and reliability.
- The evaluation selected a SAM2 proposal using overlap with human label 1.
  That annotation is unavailable in production. The image preprocessing and
  detector choice are selected, but a deployable rule for choosing among nested
  SAM2 proposals remains unresolved. It should prefer a plausible
  center-containing outer proposal and must be validated without annotation
  access.
- For a species, medium, or phenotype not represented here, do not silently
  reuse one of these condition-specific choices. Require detector QA or an
  explicit configured fallback.

## Evidence files

- `scratch/equal_pipeline_coverage_outputs/equal_pipeline_coverage_summary.csv`
  contains the equal-treatment summary for eight pipelines applied to all 16
  crops. In the detector-only archive, it is under
  `results/equal_pipeline_coverage_outputs/`.
- `scratch/equal_pipeline_coverage_outputs/equal_pipeline_coverage_rows.csv`
  contains all 128 crop-pipeline evaluations. The archive uses the same
  `results/equal_pipeline_coverage_outputs/` directory.
- `scratch/ganoderma_gray_pct_composite_outputs/ganoderma_gray_pct_composite_summary.csv`
  contains the targeted Ganoderma composite comparison. In the archive, it is
  under `results/ganoderma_gray_pct_composite_outputs/`.
- `scratch/ganoderma_grayscale_pct_composite_evaluation.ipynb` is the executed
  final notebook. In the archive, it is under `notebooks/`.
