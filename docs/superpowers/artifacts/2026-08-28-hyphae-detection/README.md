# Hyphae detection and edge-enhancement evaluation

This artifact records the detector-only experiments used to revisit branch-orientation zoning for *Neurospora* and *Ganoderma*. The executable notebook is intentionally kept in [`scratch/hyphae_detection_enhancement_evaluation.ipynb`](../../../../scratch/hyphae_detection_enhancement_evaluation.ipynb), beside the orientation notebook. It does not change production detector code.

## Final decisions

The completed equal-treatment and grayscale/PCT experiments supersede the
interim recommendations below. The final study pipelines are documented in
[`ALGORITHM_DECISIONS.md`](ALGORITHM_DECISIONS.md): current TwoK for
*N. crassa* on menadione, TwoK with monogenic phase inside `branch_base` for
*N. crassa* on xylan, and grayscale plus oriented PCT without background
subtraction followed by SAM2 for *Ganoderma*.

## Evaluation scope

The complete evaluation contains 16 crop/mask pairs: four Neurospora Menadione, four Neurospora Xylan, and eight Ganoderma. Human masks are treated as subjective Sholl-style colony envelopes, not pixel-accurate ground truth. The reported qualitative recall, containment, and 95th-percentile radial enclosure ratio are comparative diagnostics only.

The complete per-crop measurements are in [`detection_ablation_results.csv`](detection_ablation_results.csv). Cyan contours in the figures show the selected detector instance. Dashed magenta contours show the nonzero human-labeled envelope.

## Paths tested

Representative screening includes every preprocessing path tested in this investigation:

- Neurospora external preprocessing: current input, unsharp masking, local contrast, local contrast plus unsharp masking, Frangi filtering, monogenic phase congruency at `k=3` and `k=5`, and color phase congruency at `k=3` and `k=5`.
- Neurospora `TwoKFilamentousDetector.branch_base`: current pipeline, unsharp masking, local contrast, local contrast plus unsharp masking, monogenic phase congruency, and color phase congruency.
- Ganoderma SAM2 input: current Gaussian subtraction, raw input, unsharp masking, local contrast, local contrast plus unsharp masking, contrast stretching plus unsharp masking, illumination flattening, local edge denoising plus unsharp masking, monogenic phase congruency, and a subtraction/phase-congruency maximum composite.

The selected four-way ablation was then run on every crop:

- Neurospora: current, branch unsharp, branch local contrast, and branch monogenic phase congruency.
- Ganoderma: current subtraction, raw, local contrast, and contrast stretching plus unsharp masking.

## Findings

- The current Neurospora path remains the most consistent Menadione option. Its median qualitative recall was 0.981. Local contrast failed on one Menadione crop because its input contained floating-point values outside the enhancer's accepted range.
- For Xylan, monogenic phase congruency inside `branch_base` increased median qualitative recall from 0.264 to 0.448 and median radial enclosure from 0.911 to 1.047. Median containment decreased from 0.987 to 0.954, consistent with some detections extending beyond the subjective envelope.
- Current Gaussian subtraction for Ganoderma had median qualitative recall 0.550 and median radial enclosure 0.752, confirming systematic loss of the sparse margin.
- Raw and local-contrast Ganoderma inputs had higher median recall, 0.819 and 0.793, but each collapsed to a small nested SAM2 instance on at least one crop. Their minimum recalls were 0.014 and 0.003.
- Contrast stretching plus unsharp masking was less expansive but more consistent. Its median recall was 0.662, minimum recall was 0.606, median radial enclosure was 0.845, and minimum radial enclosure was 0.788.
- The Ganoderma failure is therefore not solely an enhancement problem. Preprocessing changes which nested SAM2 proposal is selected. A later controlled experiment selected grayscale plus oriented PCT without background subtraction as the best tested input; proposal selection without annotation access remains unresolved.

These observations are same-set prototype findings. They require validation on independent crops before a production default is selected.

## Paper supporting figure

The paper-oriented composite compares the three verified species × medium groups:

- *Neurospora crassa* on menadione (`n=4`).
- *N. crassa* on xylan (`n=4`).
- *Ganoderma* on 1% glucose plus 1.2% yeast extract at pH 4 (`n=8`). The medium definition was recovered from `/Volumes/T9/exfab/UCR-033-E-D_LinzerGanoderma/Results/frame00_discriminability/deliverables/metadata.csv`.

Panel A shows one candidate-pipeline crop per group, selected deterministically as the crop closest to that group's median qualitative recall. Panel B shows median qualitative-envelope recall with every crop overlaid. Panel C shows the detector-to-human-envelope 95th-percentile radial enclosure ratio, with 1 indicating equal radial reach. Gold identifies the species × medium candidate, and a blue border identifies the current baseline.

The bars cover the four detector paths evaluated on every crop. The additional representative-only enhancement screens remain in the notebook and are not presented as group-level performance estimates.

Publication exports:

- [600 dpi PNG](paper_hyphae_pipeline_by_species_media.png)
- [vector PDF](paper_hyphae_pipeline_by_species_media.pdf)
- [editable SVG](paper_hyphae_pipeline_by_species_media.svg)
- [per-crop plotted scores](paper_pipeline_score_rows.csv)
- [species × medium score summary](paper_pipeline_score_summary.csv)

The notebook includes a draft figure caption. The claim should remain phrased as detector performance varying across species × medium groups. The present same-set, qualitative evaluation does not establish general performance on unobserved crops.

## Figures

- [`neurospora_full_detection_ablation.png`](neurospora_full_detection_ablation.png)
- [`ganoderma_full_detection_ablation.png`](ganoderma_full_detection_ablation.png)
- [`representative_neurospora_external_part1.png`](representative_neurospora_external_part1.png) and [`part2`](representative_neurospora_external_part2.png)
- [`representative_neurospora_branch_part1.png`](representative_neurospora_branch_part1.png) and [`part2`](representative_neurospora_branch_part2.png)
- [`representative_ganoderma_part1.png`](representative_ganoderma_part1.png) and [`part2`](representative_ganoderma_part2.png)
