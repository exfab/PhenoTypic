# Ganoderma detector input for orientation-zone measurement

Status: selected diagnostic pipeline, 2026-09-03.

## Scope

This document records the detector input selected for the evaluated
*Ganoderma* crops grown on 1% glucose plus 1.2% yeast extract at pH 4. It is a
species x medium configuration, not a universal fungal detector default.

The selection is based on eight labeled crops. The human masks are subjective
Sholl-style colony envelopes, not pixel-accurate branch masks. The reported
scores therefore compare coarse radial enclosure and availability rather than
claiming exact biological segmentation. The complete decision record and
limitations are in the
[hyphae-detection algorithm decisions](../../artifacts/2026-08-28-hyphae-detection/ALGORITHM_DECISIONS.md).

## Selected image construction

The SAM2 input is

```text
stretched_gray = P2/P98 contrast stretch(grayscale)
oriented_pct   = phase congruency(stretched_gray)
detect_mat     = clip(max(stretched_gray, oriented_pct), 0, 1)
objmap         = SAM2(detect_mat)
```

Do not apply Gaussian background subtraction before this composite. In the
same eight-crop evaluation, adding background subtraction reduced median
convex-envelope coverage from 0.909 to 0.703 and median sparse-zone coverage
from 0.903 to 0.587. The selected no-subtraction pipeline detected all eight
crops, with minimum convex-envelope coverage 0.835.

## Reproducible PhenoTypic configuration

The following public operations reproduce the evaluated continuous input map.
The two `CompositeEnhance` branches are intentional: one returns stretched
grayscale, while the other applies phase congruency to an independently
stretched copy of the same grayscale input.

```python
from phenotypic import ImagePipeline
from phenotypic.detect.nn import Sam2
from phenotypic.enhance import (
    CompositeEnhance,
    ContrastStretching,
    FocusEdgePhase,
    SetDetectMode,
)


ganoderma_detection = ImagePipeline(
    name="ganoderma-gray-pct-sam2",
    reset=False,
    ops=[
        SetDetectMode(mode="gray"),
        CompositeEnhance(
            ops=[
                ContrastStretching(
                    lower_percentile=2,
                    upper_percentile=98,
                ),
                ImagePipeline(
                    name="ganoderma-pct-from-stretched-gray",
                    reset=False,
                    ops=[
                        ContrastStretching(
                            lower_percentile=2,
                            upper_percentile=98,
                        ),
                        FocusEdgePhase(
                            n_scale=4,
                            n_orient=8,
                            min_wavelength=6.0,
                            mult=2.1,
                            sigma_onf=0.55,
                            k=6.0,
                            cutoff=0.5,
                            g=10.0,
                            noise_method=-1.0,
                            output="pc_sum",
                        ),
                    ],
                ),
            ],
            mode="max",
            include_gray=False,
            norm="clip",
        ),
        Sam2(
            model_size="tiny",
            points_per_side=32,
            points_per_batch=8,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.5,
            min_mask_region_area=100,
            crop_n_layers=0,
            device="cpu",
            input_layer="detect_mat",
        ),
    ],
)
```

`include_gray=False` is required in this construction. Setting it to `True`
would add the immutable raw grayscale layer as a third response; the evaluated
mathematics uses stretched grayscale instead. A direct numerical check on a
fixed synthetic image found zero pixel difference between this composable form
and the notebook expression
`clip(max(stretched_gray, oriented_pct), 0, 1)`.

The evaluation explicitly fixed the parameters shown above. Other SAM2 fields
used the defaults serialized by PhenoTypic 0.19.0. `device="cpu"` records the
evaluated execution device; a different device should be rechecked for any
numerical or proposal-order differences.

## Connection to canonical Method B

After a final Ganoderma object is selected:

1. Preserve the selected SAM2 object as the final `objmap` mask.
2. Preserve the grayscale/PCT composite as `detect_mat`.
3. Run `MeasureOrientationZones` and `MeasureSymZones` on that same image.

Canonical Method B then uses the final object mask for radial extent and the
composite `detect_mat` for orientation support. Its center detector remains an
independent operation and does not replace either array.

## Unresolved deployment limitation

The evaluation chose one SAM2 proposal per crop using maximum overlap with
human label 1. That label is unavailable during production inference. Thus the
preprocessing and SAM2 proposal-generator configuration are selected, but the
end-to-end detector is not yet independently validated for automatic proposal
selection.

Until a label-free selection rule is validated, use this pipeline for the
scoped segmentation diagnostic or retain all SAM2 proposals for explicit
review. Do not report its same-set scores as end-to-end production detector
performance. A production selector should prefer a plausible center-containing
outer proposal and must be evaluated without access to the human annotations.
