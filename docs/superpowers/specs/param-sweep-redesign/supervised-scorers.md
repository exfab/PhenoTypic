# Ground-Truth-Based (Supervised) Segmentation, Detection & Counting Metrics — A Companion Catalogue

**Purpose & scope.** This document is a developer-facing reference catalogue of
**ground-truth-based** (a.k.a. *supervised*, *reference-based*) quality metrics for
scoring a colony segmentation / detection / count **against ground-truth (GT)
annotations**. It synthesizes a five-lane literature review — region/overlap metrics,
boundary/distance metrics, instance/object-level & detection metrics, counting &
localization metrics, and partition-agreement metrics + metric-selection guidance — and
maps each family onto PhenoTypic's domain: **a regular grid of ~circular microbial
colonies on agar, frequently touching**. It is written so a developer can decide *which*
metrics to implement for the `SupervisedScorer`, *how* to match predicted colonies to GT,
*which* GT modality each metric requires, and *how* the result feeds the engine's
meta-validation gate.

**Pointer back to the master spec.** This is the companion to
[`2026-06-01-parameter-tuning-engine-design.md`](2026-06-01-parameter-tuning-engine-design.md),
elaborating **§4** (the `Scorer` Protocol and specifically `SupervisedScorer` —
"ground truth → count error / IoU / Dice / F1 / adjusted Rand"), and **decision D1**
(the objective function is pluggable; `QCScorer` is the Phase-1 default, the
`ReferenceFreeScorer` is gated behind meta-validation against a small GT set). It also
expands the master's parenthetical "metric choice matters: F-measure / QR / SEI track the
visual optimum best for combined over/under-segmentation — Jozdani 2020."

**Pointer to the reference-free companion — and this doc's load-bearing second role.**
The `SupervisedScorer` is not only the objective when annotations exist. It is **the
reference signal that the reference-free meta-validation gate is correlated against**
(see [`reference-free-segmentation-metrics.md`](reference-free-segmentation-metrics.md),
§E, and master §4 / D1). The reference-free doc's gate "correlates the no-reference proxy
against a small ground-truth set, records the correlation, and warns/abstains if the
correlation is weak"; **this** document defines what that ground-truth side of the
correlation actually computes. That dual role is why even a *few* annotated plates are
high-leverage: they both score the objective directly **and** certify whether the
cheaper reference-free proxy can be trusted to drive optimization at all. The
reference-free §E.1 explicitly names Jozdani & Chen (2020) as telling PhenoTypic "which
*reference* metric to use as the gate's GT side" — that choice is made here.

> **The colony-critical point, stated up front.** Touching colonies plus a *known grid*
> make **instance-aware** metrics (family C) and **count / grid** metrics (family D)
> central. Pixel-level Dice/IoU alone (family A) is **split/merge-blind**: merging two
> adjacent colonies into one blob, or splitting one colony into two, barely changes the
> pooled foreground pixel mask, so global Dice scores both the right answer and the wrong
> answer almost identically (Maier-Hein et al., 2024; the whole motivation for families
> C–E). A colony `SupervisedScorer` must therefore pair an overlap term with an
> instance/count term and — because IoU-based one-to-one matching itself *breaks down*
> when colonies touch — a **matching-free partition guard** (ARI / VI, family E), per the
> "Metrics Reloaded" recommendation.

---

## Taxonomy / map of the field

Supervised quality metrics split into five families by *what kind of agreement they
measure* and *what GT modality they consume*. The cheapest GT modalities first (per the
master's framing): per-plate **count**; per-grid-cell **present/absent**; **centroids**;
per-colony **instance masks**.

| Family | Lane | One-line characterization | GT modality | Split/merge aware? |
|---|---|---|---|---|
| **(A) Region / overlap (volumetric)** | SR1 | Pixel-set overlap of foreground vs reference: Dice/F1, Jaccard/IoU, Tversky, volumetric similarity, precision/recall | mask (or grid-cell occupancy vector) | **No** (partition-blind) |
| **(B) Boundary / distance** | SR2 | How cleanly the predicted rim traces the GT rim: HD/HD95, ASSD/MASD, BF-score, NSD, Boundary IoU, Mahalanobis | mask (contour extracted) | Indirectly (bridge artifacts) |
| **(C) Instance / object-level & detection** | SR3 | Match objects, then count + score them: object F1@IoU, PQ (=SQ×RQ), AJI/AJI+, SEG/DET, object-Dice, AP/mAP | mask (or centroid → detection-only) | **Yes** (the point of the lane) |
| **(D) Counting & localization** | SR4 | How many and where, no mask overlap needed: count error (MAE/RMSE/MAPE), agreement (Bland–Altman, CCC, ICC), centroid P/R/F1@τ, FROC, per-grid-cell confusion | count / centroid / grid-cell | Partly (localization breaks count cancellation) |
| **(E) Partition-agreement + metric selection** | SR5 | Score the *partition* with no one-to-one matching: Rand/ARI, VI, covering, USE/BR, boundary F_b — plus the meta-guidance on which metrics to choose and which traps to avoid | mask / object-IDs | **Yes**, *without* needing a matcher |

**How they relate.** Family A measures *whether the right pixels are foreground* but not
*how they are grouped into objects*. Family B measures *boundary fidelity* but is blind to
a missed interior. Family C *counts objects and penalizes split/merge* but requires a
one-to-one match that touching colonies break. Family D scores *count and position* from
the cheapest GT and exploits the known grid to collapse detection to a per-cell binary
trial. Family E scores the *partition* directly (no matcher), recovering split/merge
sensitivity exactly when family C's matcher fails, and supplies the meta-rules
("complement overlap with boundary," "don't combine Dice and IoU," "use partition metrics
when matching is infeasible") that govern the whole composite. The recurring cross-lane
verdict: **no single metric covers split, merge, boundary, count, and small-colony errors
at once** — the scorer must be a *small complementary panel* (Maier-Hein et al., 2024;
Taha & Hanbury, 2015; Pont-Tuset & Marqués, 2013).

---

# (A) Region / overlap (volumetric) metrics — SR1

*Pixel-set overlap tools: they need a binary (or fuzzy) reference foreground and reward
correct pixel placement while penalizing over-/under-growth of the colony footprint.
Being computed on the **pooled** pixel set, they are intrinsically **partition-blind** —
two fused colonies and one large colony score identically. That blindness is the central
caveat for touching-colony grids and is why this family must be paired with an
instance-aware (C) or localization (D) metric.*

**Shared notation.** `A` = predicted foreground pixel set, `B` = reference foreground
pixel set, `|·|` = pixel/voxel count, `∩` intersection, `∪` union. Confusion counts on the
foreground-vs-background pixel classification: `TP` (foreground in both), `FP` (predicted
foreground, reference background), `FN` (reference foreground, predicted background), `TN`
(background in both).

## A.1 Dice Similarity Coefficient (DSC) / F1-score (and the Dice ≡ F1 equivalence)

- **Use cases.** The de-facto default overlap metric in biomedical segmentation — "by far
  the most widely used metric in the field of medical image analysis" (Maier-Hein et al.,
  2024). Scores a predicted foreground mask against a reference; ranges 0 (no overlap) to
  1 (perfect).
- **Mathematical foundation.**
  - Set form: `DSC(A,B) = 2|A ∩ B| / (|A| + |B|)`.
  - Confusion form: `DSC = 2·TP / (2·TP + FP + FN)`.
  - **Dice ≡ F1.** With precision `P = TP/(TP+FP)` and recall `R = TP/(TP+FN)`,
    `F1 = 2PR/(P+R) = 2TP/(2TP+FP+FN)` — algebraically identical to DSC. The segmentation
    "Dice" and the detection/classification "F1" are the same number on the pixel
    confusion matrix (Reinke et al., 2024; Maier-Hein et al., 2024 treat them in one
    family).
  - **Fuzzy / soft Dice.** For probabilistic maps, replace the hard `∩` with a t-norm
    (`min(p₁,p₂)`; Taha & Hanbury, 2015; Crum et al., 2006). The differentiable training
    surrogate writes `|A∩B| = ⟨ỹ, y⟩` with relaxed `ỹ ∈ [0,1]ᵈ`:
    `soft-Dice = 2⟨ỹ,y⟩ / (‖ỹ‖ + ‖y‖)` (Eelbode et al., 2020).
- **Advantages.** Single bounded [0,1] figure; jointly rewards size and localization
  agreement (Zijdenbos et al., 1994 via Eelbode et al., 2020); **robust to extreme
  foreground/background imbalance because it ignores TN** — exactly right for agar plates
  where colonies are a tiny pixel fraction. A "more adequate indicator for perceptual
  quality" than pixel accuracy (Eelbode et al., 2020); a special case of kappa when
  background voxels dominate (Zijdenbos et al., 1994).
- **Limitations.** Shape-unaware / boundary-insensitive ("not able to capture the object
  shape properly," Reinke et al., 2024) → pair with a boundary metric (B). Single-pixel
  sensitivity, especially with high inter-rater variability (Maier-Hein et al., 2024),
  and dependence on absolute TP size penalizes small colonies disproportionately (Seghier,
  2024). **Undefined (0/0) for empty reference + empty prediction** — handle explicitly
  (define as 1, or exclude; Seghier, 2024). Not a true metric (no triangle inequality).
  **Aggregation bias:** pooling all pixels ("micro Dice") lets large images/colonies
  dominate — see A.4.
- **Colony relevance.** *High* as the per-colony footprint-overlap term: directly measures
  over-/under-growth of colony area and is imbalance-robust. **But** on a touching-colony
  grid, global Dice cannot tell a clean split from a merged blob — it **must** be reported
  alongside an instance (C) or localization (D) metric. For per-grid-cell present/absent
  GT, DSC over the occupancy vector reduces exactly to detection F1.

## A.2 Jaccard Index / Intersection-over-Union (IoU)

- **Use cases.** The other canonical overlap metric (a.k.a. IoU, Tanimoto). Reported
  interchangeably with Dice; **the standard threshold variable for instance-matching**
  (family C's AP@IoU, PQ's >0.5 gate) — that *use* is C's, but the quantity is this lane's.
- **Mathematical foundation.** `J(A,B) = IoU(A,B) = |A∩B| / |A∪B| = TP/(TP+FP+FN)`.
  **Dice ↔ Jaccard monotone bijection:** `J = D/(2−D)` and `D = 2J/(1+J)` (Imran et al.,
  2018; Eelbode et al., 2020). Because the map is strictly monotone on [0,1], **Dice and
  Jaccard always induce the same ranking** of competing segmentations (Eelbode et al.,
  2020, Prop. II.1). Optimizing the Dice loss minimizes an upper bound on the Jaccard loss
  via `φ(x)=2x/(1+x)` (Eelbode et al., 2020).
- **Advantages.** Same imbalance-robustness and TN-exclusion as Dice; `1 − J` is a *proper
  metric* (triangle inequality holds, unlike `1 − D`); penalizes under-/over-segmentation
  *more severely* than Dice for the same error (Müller et al., 2022).
- **Limitations.** Inherits all of Dice's failure modes. **Because it ranks identically to
  Dice, reporting both is redundant for ranking** (and Taha & Reinke et al., 2021 warn
  against combining mathematically-related metrics — see E.B3). Choose one as the
  optimization target; emit the other only for literature comparability. Its harsher
  penalty makes absolute scores lower, so "70% is good" heuristics differ between the two.
- **Colony relevance.** *High*, functionally equivalent to Dice for footprint overlap —
  **pick one.** IoU is the natural choice if PhenoTypic *matches* predicted colonies to GT
  per grid cell with an IoU threshold (that matching is C's lane; the IoU value feeding it
  is this metric).

## A.3 Tversky index (asymmetric FP/FN weighting; generalizes Dice & Jaccard)

- **Use cases.** A one-parameter-family generalization that weights false positives vs
  false negatives asymmetrically — useful when over-growth and under-detection should not
  be penalized equally.
- **Mathematical foundation.** `T(A,B) = TP / (TP + α·FP + β·FN)`, `α,β ≥ 0`.
  **`α = β = ½` ⇒ Tversky = Dice; `α = β = 1` ⇒ Tversky = Jaccard** (Eelbode et al.,
  2020). The Dice-approximation error is **0 at α=β=½ and degrades monotonically** as
  weights move away (Eelbode et al., 2020, Prop. II.3) — only the symmetric settings are
  "unbiased" surrogates.
- **Advantages.** Tunable recall/precision trade-off in one overlap formula; same
  TN-exclusion and [0,1] range; smoothly interpolates the two canonical metrics.
- **Limitations.** Free parameters must be justified (arbitrary α,β = arbitrary scoring);
  the Dice↔Jaccard ranking guarantees are lost away from the symmetric settings. Same
  shape-unawareness and small-structure issues as A.1–A.2.
- **Colony relevance.** *Medium–high* as the **configurable core**: a single Tversky
  kernel with α,β exposed subsumes Dice and Jaccard and lets the sweep prefer (e.g.)
  penalizing merged-colony over-growth (FP) more than missed faint colonies (FN), or vice
  versa. Default to α=β=½ (=Dice) unless the task demands asymmetry.

## A.4 Aggregation: micro (pooled) vs macro (per-image/per-object) averaging

- **Use cases.** How to turn many per-pixel overlaps into one dataset-level number — "the
  single most error-prone *application* step" (Maier-Hein et al., 2024).
- **Mathematical foundation.** Three schemes:
  - **Micro / global / pooled:** `D_micro = 2·ΣᵢTPᵢ / Σᵢ(2TPᵢ + FPᵢ + FNᵢ)` — large
    images/colonies dominate.
  - **Macro / per-image then average:** `D_macro = (1/N)·Σᵢ Dᵢ` — each image weighted
    equally.
  - **Per-object:** Dice per matched colony, then average (the bridge into instance
    metrics — SEG/AJI, family C).
  Crum et al. (2006) give a principled **generalized overlap** single figure-of-merit
  fusing multiple labels/images with volume- or equal-weighting; the weighting choice
  changes the ranking.
- **Advantages.** Macro/per-image respects the **hierarchical data structure** ("pixels of
  the same image are highly correlated… compute per image then aggregate," Maier-Hein et
  al., 2024). Per-object averaging gives every colony equal say, counteracting Dice's
  absolute-TP-size bias.
- **Limitations.** **Mixing schemes silently is a documented pitfall** — global Dice and
  mean-per-image Dice can rank methods differently (Reinke et al., 2024). Equal-weighting
  over-weights tiny colonies; volume-weighting over-weights large ones; empty-image
  handling interacts badly with averaging.
- **Colony relevance.** A plate is a natural hierarchy (pixels → colonies → cells → plate →
  batch). The scorer should **compute Dice/IoU per plate (or per colony) then average**,
  not pool across plates. For per-grid-cell GT, equal-weighting per cell is clean. Decide
  micro-vs-macro **once** and document it — it is a ranking-determining hyperparameter of
  the scorer itself.

## A.5 Volumetric Similarity & the precision/recall (four-cardinality) family

- **Use cases.** Metrics derived directly from TP/FP/TN/FN: Volumetric Similarity,
  Sensitivity (Recall/TPR), Specificity (TNR), Precision (PPV), FPR, Accuracy. The
  components from which Dice/Jaccard/Tversky are assembled, also reported standalone (Taha
  & Hanbury, 2015).
- **Mathematical foundation.**
  - `Sensitivity = Recall = TPR = TP/(TP+FN)`; `Precision = PPV = TP/(TP+FP)`;
    `Specificity = TNR = TN/(TN+FP)`; `FPR = FP/(FP+TN)`;
    `Accuracy = (TP+TN)/(TP+TN+FP+FN)`.
  - **Volumetric Similarity:** `VS = 1 − |FN − FP| / (2TP + FP + FN)` (Taha & Hanbury,
    2015). Measures agreement of the two masks' **total volumes** *independent of spatial
    overlap* — VS is high whenever the masks have similar size even if they barely overlap.
- **Advantages.** Decomposable and diagnostic: precision/recall separate over-growth (low
  precision) from under-detection (low recall), naming *why* a colony segmentation is
  wrong. VS is the right metric when the downstream quantity is colony **size/area**
  (a primary phenotype) rather than exact placement.
- **Limitations.** **TN-dependence is the headline pitfall.** Any metric using TN
  (Specificity, Accuracy, FPR, ROC/AUC, plain kappa) is "biased against the ratio between
  foreground and background voxels" — the class imbalance (Taha & Hanbury, 2015). On agar
  plates background ≫ colony pixels, so Specificity/Accuracy/AUC are **near-1 and
  uninformative** — exclude them from the objective. Single-sided metrics are gameable
  (Recall maximized by predicting all-colony; Precision by predicting one safe pixel) —
  must be paired (which is Dice/F1). **VS ignores localization entirely** — two
  non-overlapping equal-area masks get VS=1; never use VS alone.
- **Colony relevance.** Precision and Recall (per plate or per colony) are excellent
  **diagnostic companions** to the headline Dice/IoU — they tell the tuner whether the
  pipeline is over-growing (merging → low precision) or under-detecting (faint colonies
  missed → low recall). **Volumetric Similarity** is directly meaningful because colony
  area is a phenotype. **Do not** put Specificity/Accuracy/AUC in the objective.

## A.6 Soft / probabilistic Dice & the optimization-bias caveat

- **Use cases.** When predictions/references are probabilistic rather than hard binary.
  Two distinct things share the name: a *fuzzy evaluation* metric (Taha & Hanbury, 2015;
  Crum et al., 2006) and a *training loss surrogate* (soft-Dice/Lovász).
- **Mathematical foundation.** Fuzzy overlap replaces hard `∩` with a t-norm and `∪` with
  the t-conorm; the loss surrogate uses `|A∩B| = ⟨ỹ,y⟩` (see A.1).
- **Advantages.** Consumes confidence maps without thresholding; mathematically continuous.
- **Limitations.** **Volumetric bias under uncertainty:** optimizing *soft-Dice as a loss*
  yields near-binary maps with a volume bias proportional to inherent uncertainty (Bertels
  et al., 2021 *Med. Image Anal.*; 2020 workshop) — a soft-Dice-trained pipeline may
  systematically over/under-estimate colony volume even while its hard Dice looks good.
  This is a *training* phenomenon; for a pure *scorer* it is a caution against using
  soft-Dice as the sweep objective if colony area is a reported phenotype. Hard-threshold
  the prediction before scoring to sidestep it.
- **Colony relevance.** *Low for the current pipeline* — PhenoTypic detection masks are
  effectively binary, so plain hard Dice/IoU is the expected path; soft variants matter
  only if the scorer consumes probability maps.

### Cross-cutting takeaways from family (A)

1. **One overlap metric, not two.** Dice and Jaccard rank identically (Eelbode et al.,
   2020); a **Tversky** kernel with α,β knobs generalizes both (α=β=½ recovers Dice) and
   is the cleanest configurable core.
2. **Exclude TN-based metrics from the objective** — Specificity/Accuracy/ROC-AUC/(plain)
   kappa saturate under plate-background imbalance (Taha & Hanbury, 2015). Keep
   Precision/Recall as decomposed diagnostics and Volumetric Similarity as a size-fidelity
   companion.
3. **Overlap is partition-blind** — cannot tell a clean split from a merged blob covering
   the same pixels. Pair with an instance-aware metric (C) for merges/splits and a
   count/localization metric (D) for grid-cell present/absent and centroid hits
   (Maier-Hein et al., 2024; Reinke et al., 2024).
4. **Aggregation is a ranking-determining hyperparameter** — decide micro vs macro up
   front (Reinke et al., 2024); for a plate grid, per-plate-then-average (or per-grid-cell
   equal weighting) respects the hierarchy.
5. **Handle empty masks explicitly** (define Dice/IoU = 1 for empty/empty; Seghier, 2024).

---

# (B) Boundary / distance-based metrics — SR2

*How cleanly the segmentation traces each colony rim, and how well it resolves
touching/merging colonies. Overlap metrics saturate for large, well-filled blobs and are
nearly blind to a one-pixel rim error on a big colony — but that rim error is exactly what
shifts an area/radius measurement. Boundary-distance metrics are scale-balanced and
shape-aware: they directly penalize rim displacement and the bridge artifacts where two
colonies touch. **Shared blind spot:** pure distance metrics ignore region content — a
prediction with a hole in the middle can score perfectly — so they are paired with an
overlap metric (family A), not used alone (Taha & Hanbury, 2015; Maier-Hein et al.,
2024).*

**Shared notation.** `A`, `B` = boundary/surface point sets of prediction and GT; `a, b` =
boundary points; `‖a − b‖` = Euclidean distance; `min_{b∈B}‖a−b‖` = nearest-point distance
from `a` to `B`. Unless noted, all need binary masks from which a contour is extracted (via
a distance transform); BF-score and Boundary IoU can work more directly from contours.

## B.1 Hausdorff Distance (HD)

- **Use cases.** The classic worst-case boundary-discrepancy metric; second-most-used
  segmentation metric after Dice (~47% of challenge tasks vs ~92% for Dice; Maier-Hein et
  al., 2018). Reported when the single largest boundary error matters (a spike, leaked
  rim, missed protrusion).
- **Mathematical foundation.** Directed `h(A,B) = max_{a∈A} min_{b∈B} ‖a−b‖`; symmetric
  `HD(A,B) = max( h(A,B), h(B,A) )` (Taha & Hanbury, 2015; Huttenlocher et al., 1993).
  Also called the **Maximum Symmetric Surface Distance** (Yeghiazaryan & Voiculescu,
  2018). Units of length. Correspondence-free (Huttenlocher et al., 1993).
- **Advantages.** Intuitive (a length), shape-aware, scale-balanced across object sizes
  (unlike Dice/IoU); captures the *worst* local failure that overlap metrics hide.
- **Limitations.** **Maximal outlier sensitivity** — one noisy pixel or a single
  touching-colony bridge sets the entire score (Taha & Hanbury, 2015; Maier-Hein et al.,
  2024). "Shape-only": a prediction missing the whole interior can score HD=0 if its
  outline matches (Maier-Hein et al., 2024, Fig. 61). Unbounded above → ill-behaved
  averaging (Maier-Hein et al., 2018).
- **Colony relevance.** *Low as a primary; useful as a guardrail.* HD flags the single
  worst rim error (catch catastrophic leaks/bridges) but is too brittle as the tuning
  objective — agar speckle or one debris pixel swamps otherwise good segmentations. Use
  HD95 for routine scoring.

## B.2 95th-percentile Hausdorff Distance (HD95)

- **Use cases.** The robust, de-facto-standard variant of HD for benchmarking; the go-to
  "boundary error, outlier-suppressed" number (~5–8% of challenge tasks; Maier-Hein et al.,
  2018).
- **Mathematical foundation.** Take the q-th quantile (q=95%) instead of the maximum over
  the pooled directed nearest-point distances
  `{ min_{b∈B}‖a−b‖ : a∈A } ∪ { min_{a∈A}‖b−a‖ : b∈B }` (Taha & Hanbury, 2015, "quantile
  method"). Symmetrization conventions vary (pooled quantile vs max of per-direction
  quantiles) — **pin it in code**.
- **Advantages.** Keeps HD's scale-balance and shape-awareness while discarding the top 5%
  of distances (isolated stray voxels), far more stable for ranking than raw HD (Maier-Hein
  et al., 2018, 2024).
- **Limitations.** Still "shape-only" (shares HD's missed-interior blindness). The 95%
  cutoff is arbitrary; on tiny objects the percentile is coarsely quantized. Unbounded
  above.
- **Colony relevance.** *Medium–high* — the recommended **boundary guardrail**: penalizes
  consistent rim mis-tracing and most touching-colony bridge errors while ignoring one or
  two speckle pixels. Pair with a per-colony interior/overlap check for the
  "hole-in-the-middle" blind spot.

## B.3 Average Symmetric Surface Distance (ASSD) / Mean Average Surface Distance (MASD)

- **Use cases.** The standard *average* (rather than worst-case) boundary-discrepancy
  metric; the typical rim error in length units; co-reported with HD/HD95.
- **Mathematical foundation.** Directed average `d̄(A,B) = (1/|A|) Σ_{a∈A} min_{b∈B}‖a−b‖`.
  Two non-identical symmetrizations:
  - **ASSD** (= Average Hausdorff Distance, AVD): pool both directions, average over the
    *total* point count:
    `ASSD = ( Σ_{a∈A} min_{b∈B}‖a−b‖ + Σ_{b∈B} min_{a∈A}‖b−a‖ ) / ( |A| + |B| )` — "stable
    and less sensitive to outliers than HD" (Taha & Hanbury, 2015).
  - **MASD:** `MASD = ½ ( d̄(A,B) + d̄(B,A) )` — treats both structures equally; ASSD
    "if one boundary is much larger… will impact the score much more" (Maier-Hein et al.,
    2024).
- **Advantages.** Stable, interpretable (mean rim error in pixels), shape-aware,
  scale-balanced (outliers diluted; Taha & Hanbury, 2015).
- **Limitations.** Pooled ASSD is biased when prediction and GT boundary lengths differ
  (use MASD). Shape-only (missed-interior pitfall). **Ranking error:** plain AVD/ASSD
  mis-ranked 179/200 simulated segmentations vs 52/200 for the **balanced** AVD (bAVD),
  median Kendall 0.89 vs 1.00 (Taha et al., 2021). Unbounded above.
- **Colony relevance.** *Medium–high* — good primary "how far off are the rims on average"
  signal, robust to agar speckle. Prefer MASD (or balanced AVD) over pooled ASSD so a large
  colony's long rim does not dominate a small neighbor's. Combine HD95 (worst-case) +
  ASSD/MASD (typical-case) for a complete boundary picture.

## B.4 Boundary F-score (BF / contour Dice)

- **Use cases.** Contour-matching: "what fraction of each boundary is correctly placed
  within a tolerance band?" From natural-image boundary detection (Martin et al., 2004),
  adapted as the BF-score (Csurka et al., 2013); MATLAB `bfscore`. Good when you want a
  *bounded* [0,1] boundary score.
- **Mathematical foundation.** Match boundary pixels within a distance tolerance θ:
  - Boundary precision `Pᶜ` = (predicted-boundary points within θ of a GT-boundary point) /
    (predicted boundary length).
  - Boundary recall `Rᶜ` = (GT-boundary points within θ of a predicted-boundary point) /
    (GT boundary length).
  - `BF = 2·Pᶜ·Rᶜ / (Pᶜ + Rᶜ)` (Csurka et al., 2013; MATLAB `bfscore`). The
    contour-matching idea comes from Martin et al. (2004) (bipartite matching within a
    pixel-distance threshold). **Default θ = 0.75% of the image-diagonal** (MATLAB
    `bfscore`).
- **Advantages.** Bounded [0,1], scale-balanced, decomposes into precision (leaks) vs
  recall (clipped rims); the tolerance band forgives annotation jitter.
- **Limitations.** **Hard θ-threshold** → discontinuous, less graded than Boundary IoU
  (Cheng et al., 2021). Sensitive to θ (too small penalizes harmless noise; too large
  trivially saturates). The related **Trimap IoU** is asymmetric and "favors predictions
  whose masks are larger than the ground truth" (Cheng et al., 2021) — a reason
  BF/Boundary IoU are preferred.
- **Colony relevance.** *Medium–high* — θ encodes acceptable rim slop (1–2 px or a small %
  of the plate diagonal); the precision/recall split distinguishes bleeding rims (low
  precision) from clipped rims (low recall). Bounded [0,1] → directly usable as a tuning
  objective.

## B.5 Normalized Surface Dice (NSD) / Surface Dice at tolerance τ

- **Use cases.** A "tolerated-boundary" Dice (Nikolov et al., 2021): the fraction of each
  surface within a task-specific tolerance τ of the other. Designed to track *editing
  effort* (how much boundary a human must redraw) and to be robust to inter-observer
  annotation uncertainty.
- **Mathematical foundation.** With border region `B^{(τ)}` = points within τ of a surface,
  `NSD_τ = ( |S_pred ∩ B_gt^{(τ)}| + |S_gt ∩ B_pred^{(τ)}| ) / ( |S_pred| + |S_gt| )`,
  `|·|` = surface area (Nikolov et al., 2021). τ = maximum tolerated deviation (set in the
  original work from inter-observer variability); numerator counts *acceptable* surface on
  both sides; value in [0,1], 1 = every surface point within τ. Symmetric by construction.
  "A hybrid metric between boundary-based and counting-based approaches… acceptable
  deviations… captured by a threshold τ" (Maier-Hein et al., 2024).
- **Advantages.** Bounded [0,1], scale-balanced, shape-aware, robust to annotation
  uncertainty, tunable to a domain-meaningful tolerance; correlates better with human
  correction time than volumetric Dice (Nikolov et al., 2021).
- **Limitations.** Shares the "hole-in-the-middle" blind spot (NSD=1.00 possible with a
  missed interior; Maier-Hein et al., 2024). Result depends entirely on τ (needs
  inter-observer data or a principled rule). Conceptually close to the BF-score (differ in
  surface-area vs point-count normalization).
- **Colony relevance.** *Medium–high*, very strong for **rim fidelity with an explicit
  "acceptable slop" τ** (1–2 px, or set from re-annotation variability of colony rims).
  Because colony GT rims are inherently fuzzy (soft edges, halos), NSD's tolerance band
  avoids over-penalizing biologically meaningless sub-pixel disagreement while still
  catching real leaks/bridges. Bounded [0,1] → directly usable as a tuning objective.

## B.6 Boundary IoU

- **Use cases.** A boundary-focused IoU (Cheng et al., 2021) fixing Mask IoU's
  insensitivity to boundary errors on *large* objects and over-penalization of small ones;
  drives Boundary AP / Boundary PQ. Best when you want IoU-style scoring but
  boundary-sensitive and scale-balanced.
- **Mathematical foundation.** With masks `G`, `P` and the **boundary region** = mask
  pixels within distance `d` of the contour,
  `Boundary IoU = | (G_d ∩ G) ∩ (P_d ∩ P) | / | (G_d ∩ G) ∪ (P_d ∩ P) |` (Cheng et al.,
  2021). `d` controls sensitivity; **with `d` large enough to include all interior pixels,
  Boundary IoU → Mask IoU**. The authors set `d` from annotator consistency (median
  Boundary IoU between two experts > 0.9 at `d` = 2% of the image diagonal). Symmetric by
  construction (contrasted with the asymmetric Trimap IoU). **Soft/graded** response (IoU
  degrades gracefully as contours diverge), unlike BF's hard cutoff.
- **Advantages.** Bounded [0,1], IoU-familiar, symmetric, scale-balanced, soft response;
  far more sensitive than Mask IoU to boundary errors on large objects without
  over-penalizing small ones (Cheng et al., 2021).
- **Limitations.** Shares the missed-interior pitfall — recommend **min(Boundary IoU, Mask
  IoU)** to resolve it (Maier-Hein et al., 2024, citing Cheng et al.). `d` must be scaled
  to image/object size; on very small colonies the band saturates toward Mask IoU.
- **Colony relevance.** *Medium–high* — with a modest `d`, rewards crisp rim tracing and
  cleanly separated touching colonies while staying scale-balanced across colony sizes on
  one plate. Use `min(Boundary IoU, Mask IoU)` to guard against the hollow-mask artifact.
  Strong default companion to HD95/NSD.

## B.7 Mahalanobis distance (brief — boundary/distance variant)

- **Use cases.** A distance-based metric in Taha & Hanbury's (2015) distance category;
  measures distance between two point sets accounting for their covariance. Rare in modern
  benchmarking.
- **Mathematical foundation.** `D_M = √( (μ_A − μ_B)ᵀ Σ⁻¹ (μ_A − μ_B) )`, with `μ_A, μ_B` =
  centroids of the two point sets and `Σ` = a (common) covariance of voxel coordinates;
  `Σ⁻¹` down-weights high-spread directions. Taha & Hanbury (2015) list MHD among their
  three distance metrics. **The exact Σ/pooling convention is not reproduced verbatim from
  the retrieved excerpts — treat the formula above as the standard Mahalanobis definition,
  not a verbatim source quote** (see Verification status).
- **Advantages.** Accounts for anisotropy/orientation; insensitive to isotropic scaling
  along principal axes.
- **Limitations.** Compares *distributions* (centroid + covariance), not boundary detail —
  far less sensitive to local rim/contour errors than HD/ASSD/NSD; really a
  region-distribution metric.
- **Colony relevance.** *Low* — captures gross position/spread of a colony's pixel cloud,
  not rim quality. Not recommended as a primary boundary scorer; listed for completeness.

### Cross-cutting takeaways from family (B)

1. **Shared "shape-only" blind spot** — all of HD/HD95/ASSD/MASD/NSD/Boundary IoU can
   score perfectly while missing the interior (Maier-Hein et al., 2024, Fig. 61). **Always
   pair a boundary term with an overlap term** (A), or use `min(Boundary IoU, Mask IoU)`
   (Cheng et al., 2021).
2. **Outlier-robustness spectrum:** HD (max — brittle) → HD95 (95th pct) → ASSD/MASD/bAVD
   (mean) → BF/NSD/Boundary IoU (tolerance-banded, bounded [0,1], most robust to jitter).
   Agar speckle and fuzzy rims argue for **HD95 + a tolerance-banded metric**.
3. **Tolerance-band family (τ / θ / d)** — BF (θ, 0.75% diagonal), NSD (τ, from
   inter-observer variability), Boundary IoU (d, ~2% diagonal) are the same idea; **set the
   band from colony-rim re-annotation variability or a fixed small pixel tolerance, not a
   guess.**
4. **Bounded vs unbounded** — prefer a bounded boundary metric (NSD or Boundary IoU) as the
   primary, with HD95 as a worst-case guardrail; HD/HD95/ASSD/MASD are unbounded lengths
   (bad for averaging and as a normalized objective).
5. **Symmetrization matters** — pin it: HD/HD95 (max of directed), ASSD (pooled,
   size-biased) vs MASD (mean of directed means, size-balanced), BF/NSD/Boundary IoU
   (symmetric).
6. **Centroid-only GT → none of these apply** (they all need a reference boundary); that is
   family D's localization lane.
7. **Empirical pointer:** in a vessel-segmentation metric-selection study, distance
   measures (balanced AVD rank 1, AVD rank 2) tracked expert visual rankings better than
   Dice (rank 7), "especially in high-quality segmentations" (Taha et al., 2021b) — i.e.
   boundary-distance metrics discriminate best *once segmentations are already decent*,
   exactly the regime a tuning engine operates in.

---

# (C) Instance / object-level & detection metrics — SR3

*Metrics that explicitly **count objects and penalize split/merge errors** — the failure
modes that matter most for arrayed, often-touching colonies. On an agar grid the dominant
errors are **under-segmentation** (two touching colonies merged into one detection) and
**over-segmentation** (one colony split into two). Global pixel IoU/Dice is nearly blind to
these; only *instance-aware* metrics, which require a one-to-one match between predicted and
GT objects, convert a merge or split into an explicit FN / FP. The **grid prior** (known
nrows × ncols) further anchors the matching and turns "missing object at cell (r,c)" into a
hard, interpretable error.*

## C.0 The matching / assignment problem (foundation for all of C)

- **Use cases.** Every object-level metric first builds a correspondence between predicted
  and GT instances, and whether that match is *unique* is the load-bearing design decision.
  Two families:
  - **Threshold + greedy / "unique by construction":** declare a pred–GT pair a TP iff
    `IoU ≥ τ`. When **τ > 0.5** the match is provably **unique** — no solver needed.
  - **Optimal (Hungarian) assignment:** solve the linear-assignment problem (Kuhn, 1955)
    maximizing total IoU. Needed when τ ≤ 0.5 (matches no longer unique) or for the
    globally optimal pairing under ambiguity.
- **Mathematical foundation (the IoU > 0.5 uniqueness theorem).** For GT segment `g` and
  two non-overlapping predictions `p₁, p₂`: `IoU(pᵢ,g) ≤ |pᵢ∩g|/|g|`, and summing gives
  `IoU(p₁,g) + IoU(p₂,g) ≤ 1`. Hence if `IoU(p₁,g) > 0.5` then `IoU(p₂,g) < 0.5` — **at
  most one** prediction can exceed 0.5 IoU with any GT (Kirillov et al., 2019). For τ ≤ 0.5
  uniqueness breaks and an explicit Hungarian assignment restores a one-to-one mapping
  (Segebarth et al., 2020, following Caicedo et al., 2019).
- **Advantages.** Greedy/threshold is O(pairs), simple, reproducible; Hungarian is provably
  optimal and removes double-counting.
- **Limitations.** Greedy matching can be order-dependent and double-count below τ=0.5.
  Both families need IoU per object → require **instance labels**, not just a binary mask. A
  single global τ hides the precision/recall trade-off (addressed by AP, C.6).
- **Colony relevance.** A merge → one prediction overlapping two GT colonies → under τ>0.5
  matches at most one, the second GT becomes an unmatched **FN**. A split → two predictions
  for one GT → only one matches, the other is an **FP**. The grid prior makes Hungarian
  especially attractive: restrict candidate matches to predictions near each expected cell
  → faster matching + free "empty cell" / FN detection.

## C.1 Object-level Precision / Recall / F1 (detection F1, F1@IoU)

- **Use cases.** The canonical instance-detection score for nuclei/cells/colonies: how many
  GT objects were correctly found vs how many predictions were spurious. The de-facto
  standard for nucleus and colony detection (Caicedo et al., 2019; AGAR, Majchrowska et al.,
  2021 [preprint]).
- **Mathematical foundation.** Match predictions to GT (C.0), then at IoU threshold τ:
  `Precision = TP/(TP+FP)`, `Recall = TP/(TP+FN)`,
  `F1 = 2·TP/(2·TP+FP+FN) = 2·P·R/(P+R)`, where TP = matched pairs with IoU ≥ τ, FP =
  predictions with no qualifying GT, FN = GT with no qualifying prediction. Caicedo et al.
  (2019) compute F1 **over a range of IoU thresholds** (0.5–1.0) to expose both detection
  (low τ) and contour quality (high τ); Segebarth et al. (2020) match via Hungarian over the
  pairwise IoU matrix.
- **Advantages.** Interpretable, decomposable into precision vs recall, threshold-tunable,
  directly counts objects. F1@IoU sweeps separate "found the colony" from "delineated it
  well."
- **Limitations.** A single τ is a hard cut (IoU 0.49 scores as a total miss); F1 ignores
  *how* a TP overlaps. Small objects make F1 noisy. Does not by itself distinguish a *split*
  (FP+FN pair) from an independent FP and FN elsewhere.
- **Colony relevance.** *High — the most directly actionable colony metric.* **Merge** → 1
  TP + 1 FN (recall↓). **Split** → 1 TP + 1 FP (precision↓). The precision/recall split
  *names the error mode*: low recall ⇒ merging/missing; low precision ⇒ splitting/spurious
  specks. With a known grid, FN at a populated cell and FP between cells are individually
  loggable. AGAR and the Caicedo nucleus framework both use F1@IoU, confirming its fit for
  touching round objects.

## C.2 Panoptic Quality (PQ = SQ × RQ)

- **Use cases.** A single unified instance-segmentation score that simultaneously rewards
  *finding the right objects* and *segmenting them well*, with a clean recognition vs
  segmentation decomposition; the headline metric for nuclear instance segmentation in
  Hover-Net (Graham et al., 2019). **Metrics Reloaded recommends PQ as an alternative to
  F_β for instance segmentation** (Maier-Hein et al., 2024; see E.B1).
- **Mathematical foundation.** Match by the IoU > 0.5 unique rule (C.0), then
  `PQ = ( Σ_{(p,g)∈TP} IoU(p,g) ) / ( |TP| + ½|FP| + ½|FN| )`, factorizing as
  **PQ = SQ × RQ** with `SQ = ( Σ_{(p,g)∈TP} IoU(p,g) ) / |TP|` (mean IoU over matched
  pairs) and `RQ = |TP| / ( |TP| + ½|FP| + ½|FN| )` (**RQ is literally the detection F1**).
  So PQ = (mean-IoU-of-hits) × (detection-F1) (Kirillov et al., 2019). Report DQ (=RQ) and
  SQ separately, since "SQ is calculated only within true positive segments" (Graham et al.,
  2019).
- **Advantages.** One number that *cannot* be gamed by trading detection for segmentation;
  the SQ/RQ split tells you which sub-task failed; symmetric in FP/FN. **Avoids AJI's
  over-penalization** — Hover-Net places "a larger emphasis on PQ" because AJI/DICE2
  over-penalize overlap regions.
- **Limitations.** Hard IoU > 0.5 gate → a discontinuity at the threshold (IoU 0.50
  contributes 0, 0.51 contributes fully). SQ averages only over hits (read it with RQ). All
  FP/FN weighted ½ regardless of size.
- **Colony relevance.** *High.* **Merge** → merged blob matches ≤1 of two GT colonies, the
  other is FN → RQ↓. **Split** → one fragment is FP → RQ↓, and each fragment's IoU is low →
  SQ↓. So PQ penalizes split/merge on *both* axes. RQ ≈ "fraction of grid cells correctly
  resolved into single colonies"; SQ ≈ "how tightly we trace colony boundaries." Pairs
  naturally with the grid (per-cell TP/FP/FN → plate-level PQ).

## C.3 Aggregated Jaccard Index (AJI) and AJI+

- **Use cases.** The original instance-segmentation metric for the MoNuSeg/nuclei benchmark
  (Kumar et al., 2017); penalizes object- *and* pixel-level errors in one ratio; still
  reported for direct comparison with prior nuclei work (Graham et al., 2019).
- **Mathematical foundation.** For each GT object `Gᵢ`, find the predicted object
  maximizing Jaccard, `M(Gᵢ)`; then
  `AJI = ( Σᵢ |Gᵢ ∩ M(Gᵢ)| ) / ( Σᵢ |Gᵢ ∪ M(Gᵢ)| + Σ_{Pⱼ∈U} |Pⱼ| )`, where `U` = predicted
  objects never matched to any GT (their pixels added to the denominator as a penalty)
  (Graham et al., 2019, paraphrasing Kumar et al., 2017). **AJI+** replaces the greedy
  max-Jaccard assignment with a one-to-one (Hungarian-style) assignment to remove
  double-counting artifacts. **The exact AJI+ formula was not retrieved from a standalone
  primary derivation (described only qualitatively in Hover-Net) — treat the exact AJI+
  formula as unverified** (see Verification status); the AJI base formula is verified from
  Kumar 2017 via Hover-Net's excerpt.
- **Advantages.** Single number capturing missed/spurious objects (via U) *and* pixel
  overlap; historically the nuclei comparison standard.
- **Limitations.** **Over-penalization of overlapping regions** — a prediction differing "by
  a few pixels" can get a markedly inferior AJI (and DICE2) "due to over-penalization of the
  overlapping regions" (Graham et al., 2019, Fig. 4). The greedy max-Jaccard rule can
  mis-assign / inflate the FP penalty (the defect AJI+ targets). No detection/segmentation
  split (unlike PQ).
- **Colony relevance.** *Medium — usable but PQ preferred.* AJI does penalize merges/splits,
  but its over-penalization of slight boundary disagreement is a liability for round colonies
  whose edges are diffuse on agar. If reported, prefer **AJI+** over AJI and pair with PQ.

## C.4 SEG and DET scores (Cell Tracking Challenge)

- **Use cases.** The standardized segmentation- and detection-accuracy measures of the Cell
  Tracking Challenge, designed for objective benchmarking and **parameter tuning** of cell
  segmentation algorithms — the *same use case* as our `SupervisedScorer` (Maška et al.,
  2014; Matula et al., 2015; Ulman et al., 2017).
- **Mathematical foundation.**
  - **SEG:** Jaccard over matched reference cells. `J(S,R) = |R∩S| / |R∪S| ∈ [0,1]`.
    **Matching rule:** `S` matches `R` iff `|R∩S| > ½·|R|` (covers more than half of R's
    pixels → unique S per R); no match ⇒ that reference contributes `J = 0`. SEG = **mean
    Jaccard over all reference objects** (Maška et al., 2014; Ulman et al., 2017).
  - **DET:** from the Acyclic Oriented Graph Matching (AOGM) edit-cost framework, nodes
    only: `DET = 1 − min(AOGM-D, AOGM-D₀) / AOGM-D₀`, where AOGM-D = weighted cost of edit
    operations (split / delete / add a node) and AOGM-D₀ = cost of building the reference
    from an empty result (Matula et al., 2015). DET ∈ [0,1], higher better.
- **Advantages.** SEG is a per-object Jaccard with an explicit, *unique*, biologically
  motivated ">½-overlap" rule — simple, reproducible. DET's edit-cost view directly counts
  split/missed/spurious detections weighted by human-correction effort ("how much manual
  fixing remains"). Both are battle-tested for tuning.
- **Limitations.** SEG's ">½ of R" is a hard, asymmetric gate (a 49%-overlap prediction
  scores J=0); SEG conflates detection and segmentation into one number (DET separates
  detection). DET needs AOGM weights chosen (conventional defaults, tunable). Neither yields
  as clean a precision/recall split as object-F1/PQ.
- **Colony relevance.** *High — arguably the closest published analogue to our task* (per-
  instance Jaccard + an explicit detection-edit cost, both built for tuning). A **merged**
  prediction satisfies ">½" for at most one of two GT colonies (the other → J=0 + a DET
  delete/split cost); a **split** leaves one fragment matching and the spurious fragment
  incurs a DET add cost. The grid makes the reference object set deterministic (one R per
  occupied cell) → SEG/DET become per-cell scores. DET's "split-a-node" operation is the
  exact named penalty for over-segmenting a colony.

## C.5 Object-level Dice (object-Dice / "DICE2" / Ensemble Dice)

- **Use cases.** An instance-aware Dice that aggregates a per-object Dice rather than one
  global Dice over the whole foreground; reported for nuclei alongside AJI (Graham et al.,
  2019).
- **Mathematical foundation.** Base `DICE(X,Y) = 2·|X∩Y| / (|X|+|Y|)`. **Object-Dice /
  DICE2** computes Dice **per matched object and aggregates** (typically a size-weighted
  average over GT and, symmetrically, predicted objects), so merging/splitting changes the
  pairings and the score — unlike global Dice (A.1). (Several object-Dice variants exist;
  DICE2 = "Ensemble Dice," Graham et al., 2019.)
- **Advantages.** Familiar Dice scale, made instance-sensitive; penalizes split/merge that
  global Dice misses.
- **Limitations.** Shares AJI's **over-penalization of overlapping regions** (Graham et al.,
  2019, Fig. 4); single number with no detection/segmentation split; exact
  aggregation/weighting differs across papers (definition ambiguity).
- **Colony relevance.** *Medium — lower priority than PQ/SEG.* It reacts to merges/splits,
  but over-penalizes small boundary disagreements on diffuse-edged colonies. If a
  Dice-family number is desired for continuity with global Dice (A), this is the
  instance-aware upgrade — but **PQ's SQ component is a cleaner "mean overlap of correctly
  detected colonies."**

## C.6 Average Precision (AP) / mean AP (mAP) at IoU thresholds — COCO-style

- **Use cases.** The dominant **confidence-threshold-free** detection metric: summarizes the
  full precision–recall trade-off across confidence scores and (in COCO form) across a sweep
  of IoU thresholds. **The published evaluation metric for microbial-colony detection on
  agar** (AGAR, Majchrowska et al., 2021 [preprint]: "we rely on the mean Average Precision
  (mAP) established for the COCO competition").
- **Mathematical foundation.** For fixed IoU threshold τ, rank predictions by confidence,
  trace the PR curve `P(r)`, then `AP(τ) = ∫₀¹ P_interp(r) dr`, `P_interp(r) = max_{r′≥r}
  P(r′)` (COCO uses a 101-point recall grid). **COCO mAP** averages AP over **10 IoU
  thresholds τ ∈ {0.50, 0.55, …, 0.95}** and over classes:
  `mAP = AP@[.5:.95] = (1/|T|·|C|) Σ_{τ∈T} Σ_{c∈C} AP_c(τ)`, with variants AP₅₀, AP₇₅ and
  size-stratified AP_S/AP_M/AP_L (Lin et al., 2014). Predictions matched to GT greedily by
  descending confidence (unique GT per prediction).
- **Advantages.** Confidence-threshold-free (evaluates the whole ranking); AP@[.5:.95] also
  integrates over localization strictness, rewarding tight boundaries without a single hard
  IoU gate; standard and comparable across the detection literature and *the* AGAR colony
  metric.
- **Limitations.** **Requires per-prediction confidence scores** — classical colony
  segmenters (threshold + watershed) emit no ranking, so AP is ill-defined / degenerate (a
  hard segmenter has one operating point → AP collapses toward precision×recall). **mAP is a
  poor proxy for counting accuracy:** "mAP is not a reliable metric to select the best model
  to count animals … a counting-focused metric like the F1-score should be favored" (Moreni
  et al., 2023). Averaging over IoU thresholds can mask whether errors are detection or
  localization.
- **Colony relevance.** *Medium — only when the detector produces confidences* (e.g. a
  Faster/Cascade R-CNN colony detector, as in AGAR). Then AP@[.5:.95] is an excellent
  operating-point-free score (AP₅₀ ≈ "can we localize at all," AP₇₅ ≈ "do we trace
  tightly"). **For a colony *counting* engine, prefer fixed-τ object-F1 / SEG / PQ as the
  primary objective with mAP as a secondary diagnostic** (Moreni et al., 2023); for
  score-free classical segmenters AP is not applicable at all.

### How each instance metric responds to split / merge (SR3 summary)

| Metric | Merge (2 GT → 1 pred) | Split (1 GT → 2 pred) | Detection/seg split? | Needs confidence? |
|---|---|---|---|---|
| Object F1@IoU | 1 TP + 1 FN → recall↓ | 1 TP + 1 FP → precision↓ | Yes (P vs R) | No |
| PQ (=SQ×RQ) | RQ↓ (FN) | RQ↓ (FP) + SQ↓ | Yes (SQ vs RQ) | No |
| AJI / AJI+ | union ratio↓, penalty | unmatched-fragment penalty | No | No |
| Object-Dice (DICE2) | pairing changes → ↓ | pairing changes → ↓ | No | No |
| SEG (CTC) | other GT → J=0 | spurious fragment unmatched | Partly (DET separates) | No |
| DET (CTC, AOGM) | delete/merge edit cost | **split-node** edit cost | Detection only | No |
| AP / mAP@[.5:.95] | recall↓ at all τ | low-conf FP → precision↓ | Averaged away | **Yes** |

Global pixel Dice/IoU (family A) appears in **no** row — it barely moves under merge/split.
That is the motivation for this entire lane.

### Cross-cutting takeaways from family (C)

1. **The matching rule is the central design decision.** Default to **IoU > 0.5 unique
   matching** (no solver, provably one-to-one; Kirillov et al., 2019); switch to
   **Hungarian** only for τ ≤ 0.5 or optimal pairing under ambiguity (Caicedo/Segebarth).
   The **grid prior** constrains candidate matches near each cell → faster matching + free
   FN ("empty cell") detection.
2. **Recommended primary instance score:** **PQ** (single number, SQ/RQ split) or
   **object-F1@IoU swept over τ**; the CTC **SEG + DET** pair is the closest published
   analogue built specifically for *parameter tuning* and a strong co-primary. All three are
   **confidence-score-free**, so they work for classical (threshold/watershed) segmenters.
3. **Avoid AP/mAP as the sole objective for a counting pipeline** — needs confidences (often
   absent) and is an empirically poor proxy for count accuracy (Moreni et al., 2023). Keep
   it diagnostic where scores exist.
4. **AJI and object-Dice over-penalize boundary disagreement** (Graham et al., 2019, Fig. 4)
   — a liability for diffuse-edged colonies. Report only for comparability; prefer AJI+ over
   AJI and PQ's SQ over object-Dice.
5. **Threshold discontinuities** — every hard-IoU metric (F1@τ, PQ, SEG's ">½") has a cliff
   at the threshold; mitigate by **sweeping τ** (report a curve / AP-like integral).
6. **Centroid-only GT (no masks)** degrades these mask-IoU metrics to detection-only —
   family D's lane.

---

# (D) Counting & localization metrics — SR4

*Metrics that score **how many** objects were found and **where** they are, without
requiring pixel-mask overlap — the cheapest, most directly available GT modality for
PhenoTypic: per-plate **count**, per-grid-cell **present/absent**, and/or **centroids**.*

**Shared notation.** For plate `i`: `yᵢ` = GT count, `ŷᵢ` = predicted count, `N` = number of
plates.

## D.1 Count-error metrics: absolute / signed / relative, and MAE / RMSE / MAPE

- **Use cases.** The simplest CFU-counter score (per-plate "how far off") and the de-facto
  metric for deep-learning colony counters (AGAR; synthetic-colony work). A tuning engine
  wants a *single scalar per parameter setting* aggregated over plates — MAE/RMSE/MAPE are
  those scalars.
- **Mathematical foundation.**
  - Per plate: absolute `eᵢ = |ŷᵢ − yᵢ|`; signed/bias `sᵢ = ŷᵢ − yᵢ`,
    `bias = (1/N) Σᵢ (ŷᵢ − yᵢ)` (positive ⇒ over-counting); relative `pᵢ = (ŷᵢ − yᵢ)/yᵢ`
    (undefined when `yᵢ=0`, a real empty-plate edge case).
  - Aggregate: `MAE = (1/N) Σᵢ |ŷᵢ − yᵢ|`; `RMSE = sqrt( (1/N) Σᵢ (ŷᵢ − yᵢ)² )`;
    `MAPE = (100/N) Σᵢ |ŷᵢ − yᵢ| / |yᵢ|`. RMSE penalizes large per-plate misses
    super-linearly; MAE weights every CFU equally; MAPE is scale-free but unstable for
    `yᵢ ≈ 0` (Hyndman & Koehler, 2006; the crowd-counting MAE/MSE convention, Idrees et al.,
    2018). Hyndman & Koehler's **MASE** (scaled error) is the recommended choice across
    series of very different magnitudes (including zeros).
- **Advantages.** One number per configuration, directly optimizable; signed mean separates
  bias from scatter; MAE/RMSE share the unit (colonies); RMSE surfaces catastrophic plates;
  percent error is scale-free.
- **Limitations.** **The cancellation trap:** a single count is lossy — a plate where the
  detector misses 5 true colonies and hallucinates 5 false ones scores a perfect count error
  of 0 (FP and FN cancel) — the pathology FROC was invented to avoid (Bunch et al., 1978).
  RMSE is outlier-dominated; MAPE is asymmetric and undefined for empty plates (sMAPE is not
  truly symmetric either).
- **Colony relevance.** *High as the count-accuracy term.* Brugger et al. (2012) validated
  an automated CFU counter this way (regression slope 1.01 automated vs 0.67 routine manual —
  the human under-counted). AGAR / synthetic-colony work reports **counting MAE ≈ 4.31–4.49
  colonies/plate** alongside detection mAP (Pawłowski et al., 2022). **MAE over the plate set
  is the natural count-accuracy objective**, RMSE as a worst-case guard, MAPE only when no
  plate is empty. **Overlaps the master spec's `QCScorer`** (expected-vs-detected count) —
  reuse, don't duplicate (see Recommendations).

## D.2 Count-agreement statistics vs a (fallible) reference / manual count

*Not "what is the error" but "do the two counting methods agree well enough to be
interchangeable," with explicit treatment of bias, proportional bias, and scatter — the
right framing when the GT is itself a fallible human count.*

### D.2.1 Bland–Altman limits of agreement
- **Use cases.** Method-comparison when both the automated and manual counts are noisy
  estimates of the same CFU quantity; reveals mean bias, limits of agreement, and (via
  difference-vs-mean) whether disagreement grows with colony density (heteroscedasticity).
- **Mathematical foundation.** Differences `dᵢ = ŷᵢ − yᵢ`; mean `d̄`, SD `s_d`; 95% limits
  `d̄ ± 1.96·s_d`; plot `dᵢ` vs `(ŷᵢ + yᵢ)/2` (Bland & Altman, 1986). Assumes roughly normal,
  constant-variance differences (often violated for counts → log/√ transform).
- **Advantages.** Separates bias from precision in one picture; limits in the natural unit;
  does not reward mere correlation (catches a constant offset that correlation misses).
- **Limitations.** Descriptive (an interval, not a pass/fail — pre-specify an acceptable
  difference). Classic limits assume homoscedasticity; counts are heteroscedastic. One
  measurement per method per plate (repeated measures need the extension, Bland & Altman,
  2007).
- **Colony relevance.** *Medium–high* — directly applicable to automated-vs-manual count;
  the difference-vs-mean plot immediately shows the crowding-induced under-count all CFU
  counters suffer at high density.

### D.2.2 Lin's Concordance Correlation Coefficient (CCC)
- **Use cases.** A single scalar in [−1,1] for agreement of paired continuous counts about
  the 45° line — rewards counts that are both correlated *and* unbiased/equal-scale (unlike
  Pearson). A one-number agreement objective.
- **Mathematical foundation.**
  `CCC = ρ_c = 2·ρ·σ_x·σ_y / ( σ_x² + σ_y² + (μ_x − μ_y)² )`, with Pearson `ρ`, variances
  `σ_x², σ_y²`, means `μ_x, μ_y`. The `(μ_x−μ_y)²` term penalizes location shift; `ρ_c = ρ`
  only when means and variances match. Equals the ICC under a two-way model for the
  squared-distance case (Carrasco & Jover, 2003).
- **Advantages.** One bounded interpretable number; decomposes into precision (ρ) × accuracy
  (bias correction); strictly stronger than Pearson for agreement; widely implemented.
- **Limitations.** A summary — hides the density-dependent disagreement Bland–Altman reveals
  (complementary, not substitutes). Inflated by a wide range of true counts (between-plate
  variance dwarfs within-plate disagreement). **The original 1989 variance formulas had
  typographical errors corrected by Lin's 2000 erratum** (documented by Steichen & Cox, 2002
  — see Verification status).
- **Colony relevance.** *Medium–high* — microbiome-reproducibility work adapts CCC to
  microbiology data (Cui et al., 2021), confirming domain transfer; the natural
  single-scalar agreement score between automated and manual counts.

### D.2.3 Intraclass Correlation Coefficient (ICC) as count reliability
- **Use cases.** When the reference is several human counters (or several automated re-runs)
  and you want a reliability coefficient for ≥2 raters on the same plates; also for
  inter-replicate count reliability across plate replicates.
- **Mathematical foundation.** A ratio of ANOVA variance components. For the two-way
  absolute-agreement single-measure form (ICC(A,1)):
  `ICC = (MS_R − MS_E) / ( MS_R + (k−1)MS_E + (k/n)(MS_C − MS_E) )`, with `MS_R` between-plate,
  `MS_C` between-rater, `MS_E` residual mean squares, `k` raters, `n` plates. Different forms
  (Shrout & Fleiss's ICC(1,1)/(2,1)/(3,1) etc.) drop/keep the rater term and treat raters as
  random vs fixed; consistency forms omit `(MS_C − MS_E)`, absolute-agreement forms include it
  (penalizing systematic offsets).
- **Advantages.** Handles >2 raters natively; absolute-agreement form penalizes systematic
  bias; CIs and interpretive bands (Koo & Li: <0.5 poor … >0.9 excellent).
- **Limitations.** The proliferation of forms is a footgun (reporting "ICC" without the form
  is uninformative; ten Hove et al., 2024; Qin et al., 2019). Inflated by a wide between-plate
  count range.
- **Colony relevance.** *Medium–high* — PhenoTypic's Smart-QC roster already ships an ICC
  check (time-as-subject reliability); natural when the GT is a *panel* of manual counts or
  for scoring replicate-to-replicate count stability on the known array. **Overlaps the
  master's `QCScorer` (ICC replicate reliability) — reuse.**

### D.2.4 Pearson / Spearman (weak agreement statistics)
- **Use cases.** Quick sanity check of automated-vs-manual association; Spearman when the
  relationship is monotone-nonlinear or counts are skewed.
- **Mathematical foundation.** Pearson
  `r = Σ(xᵢ−x̄)(yᵢ−ȳ) / sqrt(Σ(xᵢ−x̄)² Σ(yᵢ−ȳ)²)`; Spearman = Pearson on the ranks. Both in
  [−1,1], measuring *association*, not *agreement*.
- **Advantages.** Trivial, universal; Spearman robust to a few mis-counted plates and
  monotone nonlinearity.
- **Limitations.** **High correlation ≠ agreement** — a counter reading exactly half the true
  CFU on every plate has Pearson r = 1.0 yet is useless. This is precisely why Bland–Altman
  (1986) and Lin (1989) exist. Use only as a coarse first look; **never as the primary
  objective.**
- **Colony relevance.** *Low as an objective.* Brugger et al. (2012) found the regression
  *slope*, not the correlation, was informative; arrayed-colony tools report inter-plate
  Pearson of replicate colony sizes (0.88–0.95) as a *stability* signal (Jaeger et al., 2015),
  not accuracy.

## D.3 Point / centroid detection metrics (matched within a distance tolerance) + FROC

*The modality that exploits **centroid** GT: a prediction is a TP iff it lies within a
tolerance τ of an unmatched truth, under one-to-one matching.*

### D.3.1 Centroid Precision / Recall / F1 with a distance tolerance τ
- **Use cases.** Scoring *where* colonies are when GT is centroids and you care about
  split/merge errors that count alone hides. Arrayed plate → small τ (a fraction of the grid
  pitch); free plate → τ ≈ a typical colony radius.
- **Mathematical foundation.** Bipartite matching between predicted points `{p̂ⱼ}` and GT
  points `{gₖ}`: a candidate pair matches iff `‖p̂ⱼ − gₖ‖ ≤ τ`, each used at most once
  (greedy-nearest or Hungarian). Then TP = #matched, FP = #unmatched predictions, FN =
  #unmatched truths; `Precision = TP/(TP+FP)`, `Recall = TP/(TP+FN)`, `F1 = 2PR/(P+R)`.
  Sweeping a confidence threshold traces a PR curve whose area is AP; averaging AP over τ
  values gives an mAP analogue (Wu et al., 2021's `AP_t`). A common convention fixes a
  "golden region" of radius ~6 px around each GT center (Tofighi et al., 2018).
- **Advantages.** Captures split (extra FP) and merge (missed FN) that net count erases;
  localization-aware; reduces to ordinary detection P/R/F1; the PR-curve form is
  threshold-independent; one-to-one matching prevents one prediction covering several truths.
- **Limitations.** Results depend entirely on τ and the matching rule (greedy vs optimal
  disagree near threshold); F1 at one operating point hides the trade-off (report the curve /
  AP); confluent colonies make matching ambiguous; too-large τ lets a sloppy detector pass.
- **Colony relevance.** *High on free plates* — centroid P/R/F1 at a τ tied to grid pitch is
  the most information-rich count-and-locate score and the natural multi-objective companion
  to MAE.

### D.3.2 Mean localization error (MLE)
- **Use cases.** Once matched, *how precisely* are colonies located — a complementary scalar
  reporting residual positional error (relevant if grid assignment / neighbor effects depend
  on accurate centroids).
- **Mathematical foundation.** `MLE = (1/M) Σ_matched ‖p̂ⱼ − g_{m(j)}‖`, with redundant/missing
  predictions given a fixed distance penalty (e.g. 16 px) so detectors are not rewarded for
  omitting hard cases (Wang et al., 2021); the same quantity is "localization error" in
  vertebrae-detection work (Windsor et al., 2020).
- **Advantages.** Direct, interpretable (pixels); isolates positional precision from
  detection completeness.
- **Limitations.** Defined only on matched pairs → must be reported *with* recall/precision
  (a detector matching only its 3 easiest colonies posts a tiny MLE); the fixed unmatched
  penalty is arbitrary; sensitive to the matching rule.
- **Colony relevance.** *Medium — a secondary objective* where sub-pixel centroid accuracy
  aids array indexing; not standalone.

### D.3.3 Free-Response ROC (FROC) and its summary indices
- **Use cases.** The principled "detect-and-locate" framework where each image holds an
  unknown number of targets and the system places an arbitrary number of rated marks —
  characterizes a colony detector at *all* operating points and, unlike count, refuses to let
  an FP cancel an FN.
- **Mathematical foundation.** A mark is a location-level TP iff within an acceptance radius
  of a true colony (the D.3.1 tolerance), else an FP. The FROC curve plots
  `LLF(ξ) = (# correctly localized targets)/(total targets)` vs
  `NLF(ξ) = (# false marks)/(# images)`, parameterized by decision threshold `ξ`; the x-axis
  is an unbounded *rate per image* (no fixed number of negatives). Summary FOMs: the JAFROC
  FOM and the Bandos–Rockette–Song penalized area under the empirical FROC (with a closed-form
  variance for CIs / sample-size planning) (Bunch et al., 1978; Chakraborty, 2013; Bandos et
  al., 2009).
- **Advantages.** Purpose-built for many-objects-per-image detection; eliminates count's
  FP/FN cancellation (Bunch et al., 1978); threshold-free summary; rigorous inference with
  public software.
- **Limitations.** Heavier machinery than P/R/F1; the acceptance radius still drives results;
  summary FOMs compress a curve; needs calibrated per-mark confidences.
- **Colony relevance.** *Medium — on free plates with confidences.* A colony detector on a
  plate is structurally identical to a lesion detector on a radiograph; FROC (or its lighter
  cousin, a centroid PR-curve from D.3.1) is the most defensible localization-aware score
  when per-colony confidences are available.

## D.4 Per-grid-cell present/absent confusion matrix (exploiting the known array)

*PhenoTypic's arrayed layout is the easy case: positions are known, so detection collapses
to a per-cell binary classification — colony at cell (r,c) or not — **with no matching
problem at all**.*

- **Use cases.** Genome-wide knockout/SGA screens, drug-hypersensitivity arrays, any pinned
  96/384/1536-format plate where the readout is "did this strain grow." The cheapest possible
  GT (a human marks empty vs grown cells on the known grid).
- **Mathematical foundation.** Per cell the call is TP (present, detected), TN (empty, called
  empty), FP (empty, called present — dust/bubble), FN (present, missed). Over all cells:
  `Accuracy = (TP+TN)/(TP+TN+FP+FN)`, `Precision = TP/(TP+FP)`,
  `Recall(Sensitivity) = TP/(TP+FN)`, `Specificity = TN/(TN+FP)`, `F1 = 2PR/(P+R)`. Because
  cell positions are fixed, **no distance tolerance and no matching** — each cell is an
  independent binary trial.
- **Advantages.** No τ, no matching ambiguity; class-balanced view via specificity; F1/P/R
  expose the empty-vs-grown error structure; trivially aggregated across plates; per-cell
  labels are extremely cheap to collect.
- **Limitations.** Requires reliable **grid registration** first (a mis-located grid corrupts
  every cell label); on sparse plates the classes are imbalanced (report F1 + specificity or a
  balanced accuracy, not accuracy alone); discards *size/quantity* information (a micro-colony
  and a huge one are both "present") — complements, not replaces, size-based scoring.
- **Colony relevance.** *High — the highest-signal, lowest-ambiguity GT metric on arrayed
  plates.* The arrayed-colony tool ecosystem is built on exactly this grid abstraction
  (**gitter**, **Balony**, **Spotsizer**, **Colony-live**, **pyphe**). **Per-cell
  present/absent F1 against a curated empty/grown map** is the recommended count-only/grid
  GT metric; the known grid is what makes it possible. **Overlaps the master's `QCScorer`
  (expected-vs-detected grid count) — reuse.**

### Cross-cutting takeaways from family (D)

1. **The cancellation trap is the through-line** — a bare count (and MAE/RMSE/MAPE over
   counts) is blind to compensating FP and FN. Pair a count scalar (MAE) with a
   localization-aware scalar (centroid F1@τ on free plates, per-cell F1 on arrays) so a
   parameter setting cannot win by getting the *number* right while getting the *colonies*
   wrong.
2. **Correlation is not agreement** — Pearson/Spearman/ICC-consistency reward a counter off by
   a constant factor; use Bland–Altman, Lin's CCC, and ICC-*absolute-agreement* when the
   reference is a (possibly biased) human count and you want interchangeability.
3. **Counts are heteroscedastic and can be zero** — compute Bland–Altman/CCC on √-transformed
   counts; MAPE/percentage errors are undefined on empty plates (prefer MAE, or MASE for
   plate sets spanning empty→confluent).
4. **The known array is a gift** — on arrayed plates prefer the clean per-cell confusion
   matrix (D.4); reserve τ-based centroid matching (D.3.1) and FROC (D.3.3) for free plates.
   Grid-registration quality is the upstream prerequisite the scorer must guard.
5. **Tolerance / model choice is the hidden hyperparameter** — distance τ, the matching rule,
   and the ICC form each silently change the metric. **Fix and document them**, and for
   FROC/PR-curve metrics report the area/FOM, not a single operating point.

---

# (E) Partition-agreement metrics + metric selection & pitfalls — SR5

*Partition metrics treat a segmentation as a **partition of pixels (or objects) into
clusters** and score agreement between predicted and GT partitions. Their defining advantage
for our problem: they need **no one-to-one instance matching** — critical when colonies touch
and an IoU-based matcher fails. Section E.A is the partition catalogue; E.B is the
meta-guidance that governs which metrics belong in the scorer.*

## E.A Partition / clustering-agreement metrics

### E.A1 Rand Index (RI)
- **Use cases.** Objective comparison of clustering solutions (Rand, 1971); in segmentation,
  the **Probabilistic Rand Index** with multiple GTs (Pont-Tuset & Marqués, 2013).
- **Mathematical foundation.** For every unordered pair of elements, classify by whether
  partitions agree: `P11` (same cluster in both), `P00` (different in both), `P10`, `P01`
  (disagree). `RI = (N11 + N22)/N_total_pairs` = proportion of pairs the two partitions agree
  on (Pont-Tuset & Marqués, 2013). Range [0,1], 1 = identical.
- **Advantages.** Symmetric, simple, bounded; uses no inter-cluster correspondence (robust to
  label permutation and differing cluster counts).
- **Limitations.** **No correction for chance** (random partitions score well above 0).
  **Dominated by `P00`** — RI's value "is determined to a large extent by the number of pairs
  of objects that are not joined in either partition… not clearly indicative of agreement,"
  and Warrens & van der Hoef (2022) "generally recommend against the use of the Rand index."
  For mostly-background plates `P00` swamps the score — the partition analogue of TN inflation
  (Taha & Hanbury, 2015).
- **Colony relevance.** *Low as a pixel-level score* (background domination); higher over
  *objects* (colonies) than pixels, but then **prefer ARI**.

### E.A2 Adjusted Rand Index (ARI)
- **Use cases.** The de-facto standard external clustering-validity index; used directly as a
  **segmentation** metric in Taha & Hanbury's (2015) "pair-counting" group and recommended by
  them for **small-segment** segmentation.
- **Mathematical foundation.** RI corrected for chance under a generalized-hypergeometric
  null with fixed marginals: `ARI = (RI − E[RI]) / (max(RI) − E[RI])` (Hubert & Arabie, 1985).
  Expectation 0 under random labelling, upper bound 1, **can go negative**. ARI **equals
  Cohen's κ** on the fourfold pair-agreement table (Warrens, 2008).
- **Advantages.** Chance-corrected → comparable across cluster counts/sizes; recommended for
  small segments (Taha & Hanbury, 2015); **no instance matching required**.
- **Limitations.** **Cluster-size-imbalance bias** — ARI decomposes into a weighted average
  of per-cluster indices whose weight is ~quadratic in cluster size: "a cluster twice as big…
  will receive four times the weight," so overall measures "primarily reflect agreement on
  the large clusters" (Warrens & van der Hoef, 2022; they give a robust harmonic-mean
  variant). **Spatially blind** — "totally ignores the invaluable spatial locations of
  objects" (Yan et al., 2025 [preprint]). Minimum value depends on cluster sizes (not −1 in
  general; Chacón & Rastrojo, 2022).
- **Colony relevance.** *High as a matching-free partition guard* — catches the merged/split
  cases the IoU-matched instance score silently mishandles, **provided background is handled**
  (compute over foreground colony pixels or over object-IDs, not whole-image pixels) and the
  size-imbalance caveat is documented. For arrayed plates colonies are roughly equal-area,
  which **mutes the Warrens size-imbalance bias** — a point in ARI's favour for this domain.

### E.A3 Variation of Information (VI)
- **Use cases.** Information-theoretic distance between two clusterings; a BSDS region
  benchmark (Arbeláez et al., 2011); **recommended by Metrics Reloaded for instance
  segmentation where one-to-one matching is infeasible** (Maier-Hein et al., 2024).
- **Mathematical foundation.**
  `VI(C, C′) = H(C) + H(C′) − 2 I(C, C′) = H(C|C′) + H(C′|C)` — "the amount of information lost
  and gained in changing from clustering C to C′" (Meilă, 2007). VI is a **true metric** on
  partitions (non-negative, symmetric, triangle inequality), lattice-aligned and convexly
  additive; Meilă's **impossibility result** proves no criterion can be simultaneously
  lattice-aligned, convexly additive, *and* bounded — so VI is **unbounded** (max grows with
  log n).
- **Advantages.** A genuine metric (composable distances). **Decomposes split vs merge
  explicitly:** `H(C′|C)` ≈ over-segmentation/splitting, `H(C|C′)` ≈
  under-segmentation/merging — exactly the directional diagnosis the engine wants.
- **Limitations.** **Unbounded and scale-dependent** (Meilă, 2007) → not directly comparable
  across images without normalization; spatially blind; lower-is-better (must be
  negated/normalized to act as a score).
- **Colony relevance.** *High as a split/merge diagnostic* (which way is the tuner erring —
  splitting colonies vs merging touching ones?), more than as the single headline objective
  (unboundedness). A normalized VI (e.g. ÷ log n) restores boundedness at the cost of Meilă's
  additivity.

### E.A4 Segmentation Covering (BSDS region benchmark)
- **Use cases.** The standard region score in the Berkeley Segmentation Dataset, alongside
  Rand and VI (Arbeláez et al., 2011); equivalent to "achievable segmentation accuracy" (ASA)
  in superpixel evaluation and the asymmetric partition distance (Pont-Tuset & Marqués, 2013).
- **Mathematical foundation.** Size-weighted average over each GT region of its best overlap
  with any predicted region: `C(S→G) = (1/N) Σ_R |R|·max_{R′} O(R,R′)`, `O` = overlap
  (Arbeláez et al., 2011; Giraud et al., 2016). Range [0,1], 1 = perfect.
- **Advantages.** Region-oriented, interpretable ("how well is each true colony captured by
  some predicted region"); an upper bound on object accuracy from merging predicted regions
  (the ASA reading); no global one-to-one matching.
- **Limitations.** **Asymmetric, blind to over-segmentation in the S→G direction** — splitting
  a true colony into fragments is *not* penalized (a known ASA/USE weakness — always reported
  *with* boundary recall and under-segmentation error: Giraud et al., 2016; Buyssens et al.,
  2014). Size-weighted (small colonies contribute little).
- **Colony relevance.** *Medium — a "capture rate per colony" diagnostic.* Penalizes merging
  (a region spanning two colonies lowers a best-overlap); tolerates splitting unless
  symmetrized. **Must be paired** with an under-segmentation-sensitive measure.

### E.A5 Over-/Under-segmentation Error (USE), Boundary Recall (BR), ASA (the superpixel panel)
- **Use cases.** Standard superpixel-vs-GT partition metrics (Neubert & Protzel, 2013, via
  Giraud et al., 2016; Buyssens et al., 2014; Van den Bergh et al., 2013).
- **Mathematical foundation.**
  - **Under-segmentation error (USE/UE):** the percentage of pixels that cross GT boundaries
    — sums each predicted region's "bleeding" across GT-segment boundaries (Giraud et al.,
    2016; Zhang et al., 2016). The dedicated **merge / under-segmentation** penalty.
  - **Boundary Recall (BR):** fraction of GT boundary pixels within ε (typically 2 px) of a
    predicted boundary (Giraud et al., 2016; Buyssens et al., 2014). Pure
    boundary-localization recall.
  - **ASA** = segmentation covering (E.A4).
- **Advantages.** Each isolates **one** failure mode, so reported together they give a
  complete split/merge/boundary picture; the **triplet (BR↑, USE↓, ASA↑)** is the standard
  superpixel panel (Buyssens et al., 2014).
- **Limitations.** BR is **inflated by simply producing more/finer regions** (high BR for
  ERS/SEEDS is partly a low-compactness artifact; Buyssens et al., 2014) → read at matched
  region count / against a precision counterpart. USE can be gamed by over-compact regions;
  ASA tolerates over-segmentation. **Mutual gaming** is why they are reported as a *set*.
- **Colony relevance.** *High* — a colony plate is structurally a "superpixel-style" partition
  (regular regions on a flat background). USE directly flags merged touching colonies;
  symmetric covering flags split colonies; BR flags loose colony edges. **The cleanest
  precedent for our design: report a small panel whose members cover complementary error
  modes, not one number.**

### E.A6 Boundary Precision–Recall F-measure (F_b, BSDS)
- **Use cases.** The dominant boundary/contour benchmark from BSDS (Arbeláez et al., 2011);
  recommended (with F_op) as the **tool of choice** for supervised segmentation evaluation
  by Pont-Tuset & Marqués (2013).
- **Mathematical foundation.** Treat predicted/GT boundary maps as point sets; compute a
  **maximum-weight bipartite matching** allowing a small localization tolerance, yielding
  boundary precision `P_b`, recall `R_b`, and `F_b = 2 P_b R_b / (P_b + R_b)` (Pont-Tuset &
  Marqués, 2013; Arbeláez et al., 2011 at the Optimal Dataset Scale). Range [0,1].
- **Advantages.** **Directly diagnostic of the split/merge axis via the P/R split** — boundary
  PR "statistically reflect[s] that an algorithm is providing too coarse segmentations (low
  recall, high precision) or… too fragmented (low precision, high recall)" (Pont-Tuset &
  Marqués, 2013). Tolerance-aware (robust to sub-pixel jitter). **Meta-evaluation winner:** F_b
  (with F_op) ranks as the measure of choice, beating region summary measures.
- **Limitations.** **Most expensive** of the BSDS measures (bipartite matching; 3.79 ± 2.06
  s/image vs ≥1 order of magnitude faster for the rest; Pont-Tuset & Marqués, 2013).
  Boundary-only (ignores region interior — pair with a region measure). Tolerance ε is a free
  parameter.
- **Colony relevance.** *High and specific* — the **colony-vs-colony separating boundary** is
  the make-or-break structure, and `R_b` directly measures whether the tuner recovers it. For
  touching colonies that contact boundary is exactly the boundary-recall term, making F_b
  unusually well-targeted at our hardest case. A cheaper proxy if F_b's matching is too slow
  per-config: **under-segmentation error** (E.A5).

## E.B Metric selection & pitfalls — the meta-guidance

### E.B1 Metrics Reloaded (Maier-Hein, Reinke et al., 2024, *Nature Methods*) — the consensus framework
A Delphi-consensus framework for **problem-aware** metric selection (with a MONAI reference
implementation), built to stop "choosing metrics on the basis of their popularity rather than
their suitability." Core mechanism — **problem fingerprinting** — captures whether boundary vs
volume vs center matters, structure size relative to the grid, class imbalance, and the
possibility of empty output, then routes to a metric *set*. Recommendations directly relevant
here:
- **Use multiple complementary metrics** — "a single metric typically cannot cover the complex
  requirements… we generally recommend the usage of multiple complementary metrics."
- **Complement an overlap-based metric with a boundary-based metric** (validates pairing family
  A's Dice/IoU with family B's NSD and/or our F_b).
- **Instance segmentation = a detection metric + a per-instance segmentation metric**; **PQ is
  recommended as an alternative to F_β** for instance segmentation (family C).
- **When matching is infeasible, use partition metrics** — "InS problems in which the matching
  of reference and predicted instances is infeasible, causing overlap-based localization
  criteria to fail. Metrics such as the **Rand index and variation of information** address
  this issue by avoiding one-to-one correspondence." **This is the load-bearing citation for
  putting ARI/VI in a touching-colony scorer.**
- **Errors by existence vs distance** — a fundamental selection fork (counting/overlap vs
  distance-based HD/NSD, family B).

### E.B2 Understanding Metric-Related Pitfalls (Reinke et al., 2024, *Nature Methods*) — the pitfall catalogue
The traps that bite a colony `SupervisedScorer`:
- **Overlap metrics can reward missing small colonies** — "the pixel-level DSC of a prediction
  recognizing every structure… is *lower* than that of a prediction that only recognizes one of
  the three." Mitigation: chance-adjusted metrics (ARI/κ) for small segments (Taha & Hanbury,
  2015).
- **Over- vs under-segmentation are not symmetric under overlap metrics** (citing Yeghiazaryan &
  Voiculescu) — a single Dice cannot express that splitting and merging are differently bad;
  need a metric whose split/merge behavior is explicit (VI, boundary P/R).
- **Distance metrics over-penalize single bad pixels on small structures** — "especially…
  Hausdorff distance (HD)… HD-95th percentile (HD95)… was designed to deal with spatial
  outliers" → family B's HD must be the percentile variant for small colonies.
- **Dataset-property pitfalls** — class imbalance (background ≫ foreground), small samples,
  imperfect GT (humans disagree on touching-colony boundaries) directly affect metric values.
- **Ranking instability** — "rankings are highly sensitive to alterations of the metric
  aggregation operators, the underlying dataset or the general ranking method… the winning
  algorithm might be identified by chance." **Directly relevant: our engine ranks parameter
  configs by the supervised score; if the score/aggregation is unstable the chosen config may be
  noise.**

### E.B3 Taha & Hanbury (2015, *BMC Medical Imaging*) — the selection workhorse
A 20-metric survey + selection framework + open tool. Property-based rules:
- **Avoid TN-based metrics under class imbalance** — TN-based metrics "reward segmentations with
  small segments and penalize those with large segments… biased against the class imbalance"
  (also the basis for excluding accuracy/plain RI on colony plates).
- **Chance adjustment for small segments** — prefer KAP / ARI.
- **Outliers ⇒ avoid Hausdorff (or use the quantile)** — otherwise the average distance (AVG)
  and overlap metrics are stable.
- **Metric redundancy** — "IoU and DSC are mathematically related… combining metrics that are
  related will not provide additional information for a ranking" (Taha & Reinke et al., 2021
  [preprint]) → **don't put both Dice and IoU in the panel** (family A's pair is redundant);
  pick complementary axes.

### E.B4 Pont-Tuset & Marqués (2013, CVPR) — meta-measures (how to judge a metric)
The **meta-validation precedent most aligned with our "reference signal" mandate.** They define
three quantitative meta-measures that quantify **how coherent each evaluation measure is with
plausible ground-truth hypotheses** (human judgments, behavior under refinement/swapping); the
**F_b–F_op pair** gives the best meta-scores, region summary measures the worst. Takeaway: **a
metric is only as good as its agreement with the downstream truth you care about, and you can
*measure* that agreement** — exactly what the engine's meta-validation gate does by correlating
the reference-free signal against this supervised score. Argues we should validate the chosen
`SupervisedScorer` against held-out human judgments before trusting its correlation with the
gate.

### E.B5 Jozdani & Chen (2020, *ISPRS J. Photogramm. Remote Sens.*) — metric versatility
Experimental comparison of popular and recent **supervised** segmentation-quality metrics on
building extraction (already referenced in the reference-free doc's §E.1 as telling PhenoTypic
"which *reference* metric to use as the gate's GT side"). Thesis: **no single supervised metric
is uniformly "versatile" across error types; metrics disagree on ranking depending on whether
over- or under-segmentation dominates** — reinforcing E.B1/E.B3 (pick a complementary set, know
each metric's bias). This is the source the master spec §4 cites for "F-measure / QR / SEI track
the visual optimum best." **Retrieved as metadata/abstract only (closed access); full-text
claims flagged as inference** (see Verification status).

### E.B6 Decision-tool corroboration — MIDRC-MetricTree (Drukker et al., 2024)
An interactive decision tree that, for the one-label/one-annotator segmentation branch, routes
to six metric groups "following Taha and Hanbury's work" (overlap, volume/area, **pair-counting**,
**information-theoretic**, probabilistic, spatial-distance) — independent corroboration that the
partition / pair-counting and information-theoretic families (E.A) are first-class recommended
segmentation-metric families, not exotic.

### Cross-cutting takeaways from family (E)
1. **Partition metrics are the right tool exactly when instance matching fails** (touching
   colonies) — ARI (chance-corrected agreement) + VI (split/merge direction), per Metrics
   Reloaded.
2. **Compute ARI/RI over foreground colony pixels or object-IDs**, not whole-image pixels, to
   dodge the `P00`/TN background domination (Warrens & van der Hoef, 2022; Taha & Hanbury, 2015).
3. **Report a small complementary panel, not one number** — the superpixel triplet (BR/USE/ASA)
   and the BSDS F_b–F_op pairing are the precedents; each member covers a different error mode.
4. **Don't combine mathematically-related metrics** (Dice and IoU) — no extra ranking
   information (Taha & Reinke et al., 2021).
5. **Meta-validate the scorer against downstream truth** before trusting its correlation with
   the reference-free gate (Pont-Tuset & Marqués, 2013) — and aggregate per-object/pooled and
   report stability, because parameter-config rankings are unstable to the aggregation operator
   (Reinke et al., 2024).

---

# Summary comparison table

| Family / metric | What it rewards | GT modality (count / centroid / mask) | Penalizes split/merge? | Compute cost | Key failure mode | Colony fit |
|---|---|---|---|---|---|---|
| **(A) Dice / F1** | Footprint pixel overlap | mask (or grid-cell vector) | **No** (partition-blind) | Very low | Split/merge-blind; shape-unaware; 0/0 on empty | High (one term) |
| **(A) Jaccard / IoU** | Footprint overlap (harsher) | mask | No | Very low | Ranks identically to Dice → redundant | High (pick one) |
| **(A) Tversky** | Overlap w/ tunable FP/FN | mask | No | Very low | Arbitrary α,β; loses Dice↔Jaccard guarantees | Med–high (config core) |
| **(A) Volumetric Similarity** | Equal total mask size | mask | No | Very low | Ignores localization (VS=1 for disjoint equal areas) | Med (size phenotype) |
| **(A) Precision / Recall** | Over- vs under-growth | mask | No (diagnostic) | Very low | Single-sided gameable; must be paired | High (diagnostic) |
| **(A) Specificity / Accuracy / AUC** | Background correctness | mask | No | Very low | TN-saturated under plate imbalance | **Low (exclude)** |
| **(B) Hausdorff (HD)** | Worst-case rim error | mask (contour) | Indirect (bridges) | Low | One stray pixel sets the score; unbounded | Low (guardrail only) |
| **(B) HD95** | Robust worst-case rim error | mask (contour) | Indirect | Low | Shape-only (missed interior); unbounded | Med–high (guardrail) |
| **(B) ASSD / MASD / bAVD** | Mean rim displacement | mask (contour) | Indirect | Low–med | Pooled ASSD size-biased; shape-only; unbounded | Med–high |
| **(B) Boundary F-score** | Fraction of rim within θ | mask (contour) | Indirect | Low–med | Hard θ cutoff; θ sensitivity | Med–high |
| **(B) NSD (surface Dice@τ)** | Rim within tolerance τ | mask (contour) | Indirect | Low–med | Hole-in-the-middle; τ dependence | Med–high (rim fidelity) |
| **(B) Boundary IoU** | Boundary-band overlap | mask | Indirect | Low–med | Hole-in-the-middle (use min w/ Mask IoU); d scaling | Med–high |
| **(B) Mahalanobis** | Distribution position/spread | mask | No | Low | Not boundary-specific; Σ convention non-verbatim | Low |
| **(C) Object F1@IoU** | Correct objects, P/R split | mask (centroid → detection-only) | **Yes (P↓ split / R↓ merge)** | Low | Hard τ cut; needs instance labels | High |
| **(C) PQ (=SQ×RQ)** | Detect + segment, decomposed | mask | **Yes (both axes)** | Low–med | IoU>0.5 cliff; SQ over hits only | High |
| **(C) AJI / AJI+** | Object + pixel overlap | mask | **Yes** | Low–med | Over-penalizes overlap regions; AJI+ formula unverified | Med (prefer PQ) |
| **(C) SEG + DET (CTC)** | Per-cell Jaccard + edit cost | mask | **Yes (DET split-node)** | Low–med | SEG ">½" hard gate; AOGM weights | High (built for tuning) |
| **(C) Object-Dice (DICE2)** | Per-object overlap | mask | **Yes** | Low–med | Over-penalizes overlap; definition ambiguity | Med |
| **(C) AP / mAP@[.5:.95]** | Confidence-free PR + localization | mask + confidences | **Yes** | Med | Needs confidences; poor count proxy | Med (only if scored) |
| **(D) Count error (MAE/RMSE/MAPE)** | Correct number | **count** | No (FP/FN cancel) | Very low | Cancellation trap; MAPE undefined at 0 | High (count term) |
| **(D) Bland–Altman / Lin's CCC** | Agreement (bias + scatter) | count | No | Very low | Descriptive / range-inflated; correlation≠agreement | Med–high |
| **(D) ICC** | Multi-rater count reliability | count | No | Very low | Form proliferation; range-inflated | Med–high |
| **(D) Centroid P/R/F1@τ** | Correct number + position | **centroid** | **Yes** | Low | τ dependence; matching ambiguity | High (free plates) |
| **(D) Mean localization error** | Centroid precision | centroid | No | Low | Only on matched pairs; arbitrary penalty | Med (secondary) |
| **(D) FROC + FOM** | Detect-and-locate, all op points | centroid + confidences | **Yes** | Med | Acceptance radius; needs confidences | Med (free plates) |
| **(D) Per-cell present/absent F1** | Grid-cell occupancy | **count / grid-cell** | Partly (no size) | Very low | Needs grid registration; class imbalance | High (arrayed plates) |
| **(E) Rand Index (RI)** | Pair-agreement (no match) | mask / object-IDs | Yes (weakly) | Low–med | No chance correction; P00-dominated | Low (use ARI) |
| **(E) Adjusted Rand Index (ARI)** | Chance-corrected partition agreement | mask / object-IDs | **Yes (matching-free)** | Low–med | Size-imbalance bias; spatially blind | High (partition guard) |
| **(E) Variation of Information (VI)** | Split/merge direction | mask / object-IDs | **Yes (directional)** | Low–med | Unbounded; spatially blind; lower-is-better | High (diagnostic) |
| **(E) Segmentation covering / ASA** | Per-colony capture rate | mask | Merge only (S→G) | Low | Tolerates splitting; size-weighted | Med (pair w/ USE) |
| **(E) USE / BR / ASA panel** | Isolated merge / boundary / split | mask | **Yes (as a set)** | Med | Mutual gaming; read together | High |
| **(E) Boundary F_b (BSDS)** | Contact-boundary recovery | mask | **Yes (P/R split)** | High (bipartite) | Boundary-only; ε free; slowest | High (contact zones) |

*"Compute cost" is per-evaluation in a sweep. "GT modality" is the cheapest annotation the
metric can consume (mask-based metrics degrade to detection-only on centroid GT, as noted).
Colony fit per the lane authors' judgments. Specificity/Accuracy/AUC are listed explicitly to
mark them as **exclude** under plate-background imbalance.*

---

# Recommendations for PhenoTypic's `SupervisedScorer`

Synthesizing SR1–SR5 against master-spec **§4** (the `Scorer` Protocol and `SupervisedScorer`),
**D1** (pluggable objective; `QCScorer` primary default; `ReferenceFreeScorer` gated against a
small GT set), and the **meta-validation gate**. The unifying verdict across all five lanes:
**no single metric covers split, merge, boundary, count, and small-colony errors at once — the
scorer must be a small complementary panel, chosen by GT modality** (Maier-Hein et al., 2024;
Taha & Hanbury, 2015; Pont-Tuset & Marqués, 2013).

## 1. Default composite by GT modality

The scorer adapts to whatever GT exists, cheapest modality first. Every term normalized to a
common **higher-is-better** scale per master §4's `higher_is_better` / normalizer contract
(negate VI and the distance metrics; bound the unbounded ones).

**(a) Count-only GT (per-plate count, or per-grid-cell present/absent):**
- **Per-grid-cell present/absent F1 + specificity** against the curated empty/grown map
  (D.4) — the highest-signal, lowest-ambiguity arrayed-plate metric; *no matching, no
  tolerance* because grid positions are known (Wagih & Parts, 2014; Young & Loewen, 2013).
- **Count-agreement:** MAE (+ RMSE worst-case guard) over the plate set (D.1) **and** a
  bias/agreement statistic vs the reference count — Bland–Altman limits and/or **Lin's CCC**
  (D.2.1–2.2), *not* Pearson (correlation ≠ agreement; Bland & Altman, 1986; Lin, 1989). Use
  ICC when the reference is a *panel* of human counts or for replicate-to-replicate
  reliability (D.2.3).
- **Reuse, don't duplicate:** count and grid-occupancy metrics overlap the master's
  `QCScorer` (expected-vs-detected grid count, ICC replicate reliability). The
  `SupervisedScorer` should **call the same `analysis/` QC checks**, not re-implement them
  (master §4, D1; see the planned `qc-objective-mapping.md` companion).

**(b) Centroid GT (free or arrayed plates):**
- **Matched centroid Precision / Recall / F1 at a distance tolerance τ** (D.3.1), τ tied to
  the grid pitch (arrayed) or a typical colony radius (free); report the PR-curve area / **AP@τ**
  when confidences exist (Wu et al., 2021; Tofighi et al., 2018). The P/R split names the error
  mode (low recall ⇒ merge/miss; low precision ⇒ split/speck).
- Optional **mean localization error** as a secondary positional-precision term, reported *with*
  recall so a detector cannot win by matching only its easiest colonies (D.3.2).
- On free plates with per-colony confidences, **FROC + a summary FOM** (D.3.3) is the most
  defensible all-operating-points score (Bandos et al., 2009).

**(c) Mask GT (per-colony instance masks):**
- **Primary instance term:** **object F1 / Panoptic Quality (PQ = SQ × RQ)** with the IoU > 0.5
  unique-matching rule (C.0–C.2; Kirillov et al., 2019; Graham et al., 2019). PQ's SQ/RQ split
  separates "how tightly we trace boundaries" from "how many cells we resolve." Metrics Reloaded
  recommends PQ as the instance-segmentation metric (Maier-Hein et al., 2024). Sweep τ to avoid
  the hard-threshold cliff (C, cross-cutting note 5).
- **Object-Dice term:** an instance-aware **object-Dice / SEG** (per-colony Jaccard, the CTC
  analogue built for tuning; Maška et al., 2014; Ulman et al., 2017) — but prefer PQ's SQ over
  object-Dice / AJI, which over-penalize diffuse colony rims (Graham et al., 2019).
- **Matching-free partition guard (the key hardening for touching colonies):** **ARI** over
  *foreground colony pixels or object-IDs* (not whole-image pixels — dodge the `P00`/TN
  background domination; Warrens & van der Hoef, 2022; Taha & Hanbury, 2015) plus **VI** reported
  as its two conditional-entropy halves (split vs merge direction; Meilă, 2007). Per **Metrics
  Reloaded**, Rand/VI are precisely the metrics to use when one-to-one instance matching is
  infeasible — which is exactly the touching-colony regime where the IoU matcher breaks
  (Maier-Hein et al., 2024). On arrayed plates colonies are near-equal-area, *muting* ARI's
  size-imbalance bias.
- **Boundary term for rim fidelity / contact zones:** **NSD** (surface Dice at tolerance τ; the
  master's "boundary NSD for rim fidelity"; Nikolov et al., 2021) and/or BSDS **boundary recall
  R_b** of the colony-vs-colony separating contours (Pont-Tuset & Marqués, 2013), with **HD95**
  as a worst-case bridge/leak guardrail (Taha & Hanbury, 2015; Reinke et al., 2024). Per Metrics
  Reloaded, **complement the overlap term with a boundary term** (Maier-Hein et al., 2024). Use
  `min(Boundary IoU, Mask IoU)` or pair NSD with an interior/overlap check to cover the
  hole-in-the-middle blind spot (Cheng et al., 2021).

**Composition.** Combine terms into one scalar (`CompositeScorer`, master §4) or return a dict
for true multi-objective Pareto. **Do not** put both Dice and IoU in the panel — they rank
identically and add no ranking information (Eelbode et al., 2020; Taha & Reinke et al., 2021).
Pick complementary axes (overlap × instance × partition × boundary × count), per the consensus
"use multiple complementary metrics" recommendation (Maier-Hein et al., 2024).

## 2. Matching strategy (grid-cell vs Hungarian)

- **On arrayed plates, prefer the per-grid-cell assignment** — the known `nrows × ncols`
  collapses detection to an independent per-cell binary trial with **no distance tolerance and
  no matching** (D.4). This is the most robust, lowest-ambiguity path and should be the default
  whenever the grid is registered.
- **When instance masks must be matched**, default to **IoU > 0.5 unique matching** (provably
  one-to-one, no solver; Kirillov et al., 2019); switch to **Hungarian** assignment (Kuhn, 1955)
  only for τ ≤ 0.5 or optimal pairing under ambiguity (Caicedo et al., 2019; Segebarth et al.,
  2020). Constrain candidate matches to predictions near each expected grid cell — faster, and
  it makes "empty cell" an explicit FN.
- **When colonies touch and any IoU matcher becomes unreliable, fall back to the matching-free
  partition guard (ARI/VI)** rather than trusting a degenerate match (Maier-Hein et al., 2024).
- **Pin and document** the matching rule, τ, the symmetrization of boundary metrics, the ICC
  form, and the aggregation scheme (micro vs macro) — each silently changes the metric and the
  *ranking* of parameter configs (Reinke et al., 2024; SR1/SR4 cross-cutting notes).

## 3. GT annotation-format tiers (cheapest → richest)

| Tier | Annotation | Cost | Primary metric(s) | Notes |
|---|---|---|---|---|
| 1 | Per-plate **count** | cheapest | MAE/RMSE + Bland–Altman / CCC | Cancellation-blind alone — pair with tier 2/3 |
| 2 | Per-grid-cell **present/absent** | cheap | Per-cell F1 + specificity (D.4) | Needs grid registration; no matching/τ |
| 3 | **Centroids** | moderate | Centroid P/R/F1@τ (+ FROC if scored) | Mask metrics degrade to detection-only here |
| 4 | Per-colony **instance masks** | richest | PQ + object-Dice/SEG + ARI/VI guard + NSD/HD95 | Full split/merge/boundary panel available |

The scorer should **degrade gracefully**: with only counts it runs tier 1 (+ tier 2 if the grid
is registered); given masks it runs the full tier-4 panel. Even a few tier-4 plates are
high-leverage because they double as the gate's reference (next).

## 4. Role as the meta-validation gate's reference (master §4 / D1; reference-free §E)

The `SupervisedScorer` **is the reference signal** the reference-free meta-validation gate
correlates against (master §4, D1; reference-free §E.1, §E.4). Concretely:
- **Reference metric for the gate:** Dice / Jaccard / F-measure (and, for touching colonies, an
  instance metric like PQ or SEG) computed by *this* scorer on the small annotated set plays the
  role Vinet's measure plays in Chabrier et al. (2006), and is the "GT side" of the correlation
  the reference-free doc's §E names (Jozdani & Chen, 2020 informs *which* reference metric).
- **Acceptance test:** the gate requires **rank agreement (Spearman ρ)** between the
  reference-free proxy and this supervised score **plus an argmax test** (the proxy's chosen
  params land near the true best-by-supervised-score params), stratified by quality regime and
  plate region — a single global ρ is insufficient (reference-free §E.4; Pont-Tuset & Marqués,
  2013; Kazakevičiūtė-Januškevičienė 2020 via the reference-free doc). The master §4 floor of
  "≥3–5 annotated plates" and "warns/abstains if the correlation is weak" is the same gate.
- **Why a few plates are high-leverage:** each annotated plate both scores the objective directly
  *and* helps certify whether the cheaper reference-free proxy can be trusted to drive
  optimization — search amplifies a bad objective, so **gate before letting the optimizer exploit
  the proxy** (reference-free §E.3, Deo 2025 via that doc).
- **Meta-validate this scorer too:** before trusting its correlation with the gate, validate the
  `SupervisedScorer`'s own metric choice against held-out human judgments where available
  (Pont-Tuset & Marqués, 2013) — *a metric is only as good as its agreement with the downstream
  truth.* Re-run the gate **per domain** (yeast ≠ bacteria; reference-free §E.4).

## 5. Overlap with `QCScorer` — reuse, don't duplicate

The count and grid-occupancy terms (D.1, D.2.3, D.4) are **already** the master's `QCScorer`
checks (expected-vs-detected grid count, ICC replicate reliability — D1's primary Phase-1
default). The `SupervisedScorer` must **call those existing `analysis/` checks**, not
re-implement them; it *adds* the GT-requiring metrics (overlap, instance, partition, boundary,
centroid-matched) that the QC checks cannot compute without annotations. See the planned
`qc-objective-mapping.md` companion for the QC-as-objective wiring.

## 6. Phasing pointer

GT-driven scoring is the most trustworthy objective and a natural early deliverable: per master
§12, the `SupervisedScorer` slots alongside the Phase-1 `QCScorer` (count/grid metrics it
shares) and becomes the **reference for the Phase-3 `ReferenceFreeScorer` gate**. Start with the
modality-adaptive count + grid F1 + (where masks exist) PQ + ARI panel; add the boundary and
FROC terms as annotation richness grows.

---

# Verification status & caveats

Provenance flags preserved verbatim from the five lane reports — **read before reimplementing
any formula.**

**Formulas mechanism-verified but NOT symbol-verified (do not treat as authoritative closed
forms):**
- **AJI+ (SR3):** the exact AJI+ assignment formula was **not retrieved from a standalone primary
  derivation** — it is described only qualitatively in Hover-Net (Graham et al., 2019) as a
  one-to-one (Hungarian-style) correction to AJI's greedy max-Jaccard rule. **Treat the exact
  AJI+ formula as unverified.** The **AJI base formula is verified** from Kumar et al. (2017) via
  Hover-Net's excerpt.
- **Boundary F-score (SR2):** the `BF = 2·Pᶜ·Rᶜ/(Pᶜ+Rᶜ)` formula and the boundary
  precision/recall definitions are taken from **MATLAB `bfscore` documentation** (consistent with
  Csurka et al., 2013; the contour-matching idea from Martin et al., 2004). The default
  **θ = 0.75% of the image diagonal** is MATLAB's default, not a peer-reviewed cutoff.
- **Mahalanobis distance (SR2):** the `D_M = √((μ_A−μ_B)ᵀΣ⁻¹(μ_A−μ_B))` form is the **standard
  Mahalanobis definition, not a verbatim quote** from Taha & Hanbury (2015); the exact Σ/pooling
  convention (their Eqs. 52–54) was not reproduced from the retrieved excerpts. Treat the
  **Σ-convention as standard-form, not verbatim.**

**Citations via secondary sources with NO DOI (do not fabricate one):**
- **Jaccard (1912) (SR1):** the original *New Phytologist* paper was **not independently retrieved
  with a DOI** — the Jaccard coefficient's priority is cited via the retrieved segmentation
  literature (Eelbode et al., 2020; Crum et al., 2006), **with no DOI assigned here.**
- **Tversky (1977) (SR1):** "Features of similarity," *Psychological Review*, was **not
  independently retrieved with a DOI** — the Tversky-index relationships are cited via Eelbode et
  al. (2020) and Bertels et al. (2019), **with no DOI assigned here.**

**Erratum / version flags:**
- **Lin's CCC (SR4):** the original **1989** *Biometrics* variance formulas contained
  **typographical errors corrected by Lin's 2000 erratum** (documented by Steichen & Cox, 2002,
  which records the numerical effect). Use the corrected variance expressions.
- **Soft-Dice volumetric bias (SR1):** prefer the **peer-reviewed MEDIA 2021** version (Bertels
  et al., 2021) over the MICCAI-workshop / arXiv:1911.02278 preprint.
- **NSD / surface DSC (SR2):** the originating source is the **2018 arXiv preprint** (Nikolov et
  al., 2018, **not peer-reviewed**), superseded by the **2021 JMIR** peer-reviewed version
  (Nikolov et al., 2021) — cite the 2021 version.

**Journal-metadata corrections the researchers made (scite mislabels):**
- **Metrics Reloaded (Maier-Hein et al., 2024) and Understanding Metric-Related Pitfalls (Reinke
  et al., 2024):** scite labels the venue **"nature chemical biology"** — this is a **scite
  metadata error**. Both are in ***Nature Methods*** vol. 21 (Feb 2024), pp. 195–212 and 182–194
  respectively (verified via nature.com + Google Scholar). No retraction/concern notices.
- **Ulman et al. (2017) (and Stringer et al., 2021):** scite mislabels the journal — these are
  ***Nature Methods***. Corrected here.

**Preprint flags (NOT peer-reviewed):**
- **AGAR (Majchrowska et al., 2021):** arXiv:2108.01234 — cited only as the closest in-domain
  (agar-plate colony) precedent for COCO mAP / counting MAE; **preprint, not peer-reviewed**
  (a peer-reviewed version also exists; cited here as domain context).
- **SR4 preprints:** RCNN-SliceNet (Wu et al., 2021, arXiv:2106.15753); LDC-Net (Wang et al.,
  2021, arXiv:2110.04727); vertebrae detection (Windsor et al., 2020, arXiv:2007.02606); pyphe
  (Kamrad et al., rs.3.rs-401914/v1 — a peer-reviewed book-chapter version exists at
  10.1007/978-1-0716-2257-5_21). All **preprints, not peer-reviewed.**
- **SR1 preprint:** Bertels et al. (2020) soft-Dice MICCAI workshop has an arXiv:1911.02278
  not-peer-reviewed version — prefer the MEDIA 2021 journal version.
- **SR5 preprints:** Yan et al. (2025) spatially-aware ARI (bioRxiv 10.1101/2025.03.25.645156);
  Taha & Reinke et al. (2021) "Common limitations" (arXiv:2104.05642, the precursor to the 2024
  pitfalls paper). Both **preprints, not peer-reviewed.**

**Closed-access formulas verified from OA sources:** Kumar et al. (2017) and Kirillov et al.
(2019) are closed-access, but their formulas were verified from OA sources (Hover-Net green PDF;
arXiv:1801.00868). Jozdani & Chen (2020) was retrieved as **metadata/abstract only** (closed
access) — full-text claims (E.B5) flagged as inference.

**No retractions or editorial concerns** were found on any cited peer-reviewed paper across all
five lanes (checked via scite metadata). Citation tallies in the source reports are scite's
(approximate).

**Independent citation audit (2026-06-01).** All 80 reference DOIs were verified to resolve via
scite/Crossref; **no retractions or editorial concerns** surfaced. The four scite venue-mislabel
corrections above were independently confirmed (Metrics Reloaded = *Nature Methods* 21:195–212;
Reinke pitfalls = *Nature Methods* 21:182–194; Ulman 2017 = *Nature Methods* 14:1141–1152;
Stringer/Cellpose = *Nature Methods* 18:100–106), as were Kuhn (1955) and the Lin's-CCC 2000
erratum; and the load-bearing PQ = SQ × RQ uniqueness theorem (Kirillov 2019), AJI (Kumar 2017),
and SEG ">½ overlap" rule (Ulman 2017) were confirmed from primary text. Three bibliographic
entries were corrected after the audit: **Neubert & Protzel (2013)** carried a DOI that
misdirected to a different paper → corrected to BMVC 2013 (`10.5244/C.27.39`); **Seghier (2024)**
issue `34(3)` → `34(6)`; **ten Hove et al. (2024)** completed to `29(5):967–979`.

---

# References (deduplicated across all five lane reports)

*Reconciled to one canonical entry where lanes overlapped: **Taha & Hanbury (2015)** (SR1 + SR2 +
SR5); **Maier-Hein et al. "Metrics Reloaded" (2024)** (SR1 + SR2 + SR5); **Reinke et al.
"Understanding… pitfalls" (2024)** (SR1 + SR5); **Jozdani & Chen (2020)** (SR5 + the reference-free
companion §E); **Lin et al. / COCO (Lin et al., 2014)**, **Caicedo et al. (2019)**, **Ulman et al.
(2017)**, and **Majchrowska et al. "AGAR" (2021)** (across SR3/SR4/SR5). Preprints are marked
**[PREPRINT — not peer-reviewed]**. DOIs render as https://doi.org/DOI.*

**Overlap / region & probabilistic metrics (A)**
- Dice, L. R. (1945). Measures of the amount of ecologic association between species. *Ecology*, 26(3), 297–302. https://doi.org/10.2307/1932409
- Zijdenbos, A. P., Dawant, B. M., Margolin, R. A., & Palmer, A. C. (1994). Morphometric analysis of white matter lesions in MR images: method and validation. *IEEE Transactions on Medical Imaging*, 13(4), 716–724. https://doi.org/10.1109/42.363096
- Crum, W. R., Camara, O., & Hill, D. L. G. (2006). Generalized overlap measures for evaluation and validation in medical image analysis. *IEEE Transactions on Medical Imaging*, 25(11), 1451–1461. https://doi.org/10.1109/tmi.2006.880587
- Crum, W. R., Camara, O., Rueckert, D., Bhatia, K. K., Jenkinson, M., & Hill, D. L. G. (2005). Generalised overlap measures for assessment of pairwise and groupwise image registration and segmentation. *MICCAI 2005*, LNCS 3749, 99–106. https://doi.org/10.1007/11566465_13
- Bertels, J., Eelbode, T., Berman, M., Vandermeulen, D., Maes, F., Bisschops, R., & Blaschko, M. (2019). Optimizing the Dice score and Jaccard index for medical image segmentation: theory and practice. *MICCAI 2019*, LNCS 11765, 92–100. https://doi.org/10.1007/978-3-030-32245-8_11
- Eelbode, T., Bertels, J., Berman, M., Vandermeulen, D., Maes, F., Bisschops, R., & Blaschko, M. B. (2020). Optimization for medical image segmentation: theory and practice when evaluating with Dice score or Jaccard index. *IEEE Transactions on Medical Imaging*, 39(11), 3679–3690. https://doi.org/10.1109/tmi.2020.3002417
- Bertels, J., Robben, D., Vandermeulen, D., & Suetens, P. (2021). Theoretical analysis and experimental validation of volume bias of soft Dice optimized segmentation maps in the context of inherent uncertainty. *Medical Image Analysis*, 67, 101833. https://doi.org/10.1016/j.media.2020.101833
- Bertels, J., Robben, D., Vandermeulen, D., & Suetens, P. (2020). Optimization with soft Dice can lead to a volumetric bias. *MICCAI BrainLes 2019 workshop*, LNCS 11992, 89–97. https://doi.org/10.1007/978-3-030-46640-4_9 **[PREPRINT — not peer-reviewed (arXiv:1911.02278); prefer the MEDIA 2021 version above]**
- Müller, D., Soto-Rey, I., & Kramer, F. (2022). Towards a guideline for evaluation metrics in medical image segmentation. *BMC Research Notes*, 15, 210. https://doi.org/10.1186/s13104-022-06096-y
- Seghier, M. L. (2024). Image segmentation evaluation with the Dice index: methodological issues. *International Journal of Imaging Systems and Technology*, 34(6), e23203. https://doi.org/10.1002/ima.23203
- Jaccard, P. (1912). The distribution of the flora in the alpine zone. *New Phytologist*, 11(2), 37–50. **No DOI retrieved — cited via secondary sources (Eelbode et al., 2020; Crum et al., 2006); do not fabricate a DOI.**
- Tversky, A. (1977). Features of similarity. *Psychological Review*, 84(4), 327–352. **No DOI retrieved — cited via secondary sources (Eelbode et al., 2020; Bertels et al., 2019); do not fabricate a DOI.**

**Boundary / distance metrics (B)**
- Huttenlocher, D. P., Klanderman, G. A., & Rucklidge, W. J. (1993). Comparing images using the Hausdorff distance. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 15(9), 850–863. https://doi.org/10.1109/34.232073
- Yeghiazaryan, V., & Voiculescu, I. (2018). Family of boundary overlap metrics for the evaluation of medical image segmentation. *Journal of Medical Imaging*, 5(1), 015006. https://doi.org/10.1117/1.JMI.5.1.015006
- Taha, A. A., Aydin, O. U., Hilbert, A., et al. (2021). On the usage of average Hausdorff distance for segmentation performance assessment: hidden error when used for ranking. *European Radiology Experimental*, 5, 4. https://doi.org/10.1186/s41747-020-00200-2
- Taha, A. A., et al. (2021b). [Vessel-segmentation metric-selection study.] *BMC Medical Imaging*. https://doi.org/10.1186/s12880-021-00644-x
- Martin, D. R., Fowlkes, C. C., & Malik, J. (2004). Learning to detect natural image boundaries using local brightness, color, and texture cues. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 26(5), 530–549. https://doi.org/10.1109/TPAMI.2004.1273918
- Csurka, G., Larlus, D., & Perronnin, F. (2013). What is a good evaluation measure for semantic segmentation? *Proc. BMVC*, 32.1–32.11. https://doi.org/10.5244/C.27.32
- Nikolov, S., Blackwell, S., Zverovitch, A., et al. (2021). Clinically applicable segmentation of head and neck anatomy for radiotherapy: deep learning algorithm development and validation study. *Journal of Medical Internet Research*, 23(7), e26151. https://doi.org/10.2196/26151
- Nikolov, S., Blackwell, S., Zverovitch, A., et al. (2018). Deep learning to achieve clinically applicable segmentation of head and neck anatomy for radiotherapy. *arXiv*:1809.04430. https://doi.org/10.48550/arXiv.1809.04430 **[PREPRINT — not peer-reviewed; superseded by Nikolov et al., 2021]**
- Cheng, B., Girshick, R., Dollár, P., Berg, A. C., & Kirillov, A. (2021). Boundary IoU: improving object-centric image segmentation evaluation. *Proc. IEEE/CVF CVPR*, 15334–15342. https://doi.org/10.1109/CVPR46437.2021.01508
- Maier-Hein, L., Reinke, A., et al. (2018). Why rankings of biomedical image analysis competitions should be interpreted with care. *Nature Communications*, 9, 5217. https://doi.org/10.1038/s41467-018-07619-7

**Instance / object-level & detection metrics (C)**
- Kirillov, A., He, K., Girshick, R., Rother, C., & Dollár, P. (2019). Panoptic segmentation. *Proc. IEEE/CVF CVPR*, 9404–9413. https://doi.org/10.1109/CVPR.2019.00963
- Graham, S., Vu, Q. D., Raza, S. E. A., et al. (2019). Hover-Net: simultaneous segmentation and classification of nuclei in multi-tissue histology images. *Medical Image Analysis*, 58, 101563. https://doi.org/10.1016/j.media.2019.101563
- Kumar, N., Verma, R., Sharma, S., et al. (2017). A dataset and a technique for generalized nuclear segmentation for computational pathology. *IEEE Transactions on Medical Imaging*, 36(7), 1550–1560. https://doi.org/10.1109/TMI.2017.2677499
- Caicedo, J. C., Roth, J., Goodman, A., et al. (2019). Evaluation of deep learning strategies for nucleus segmentation in fluorescence images. *Cytometry Part A*, 95(9), 952–965. https://doi.org/10.1002/cyto.a.23863
- Segebarth, D., Griebel, M., Stein, N., et al. (2020). On the objectivity, reliability, and validity of deep learning enabled bioimage analyses. *eLife*, 9, e59780. https://doi.org/10.7554/eLife.59780
- Maška, M., Ulman, V., Svoboda, D., et al. (2014). A benchmark for comparison of cell tracking algorithms. *Bioinformatics*, 30(11), 1609–1617. https://doi.org/10.1093/bioinformatics/btu080
- Matula, P., Maška, M., Sorokin, D. V., et al. (2015). Cell tracking accuracy measurement based on comparison of acyclic oriented graphs. *PLoS ONE*, 10(12), e0144959. https://doi.org/10.1371/journal.pone.0144959
- Ulman, V., Maška, M., Magnusson, K. E. G., et al. (2017). An objective comparison of cell-tracking algorithms. *Nature Methods*, 14(12), 1141–1152. https://doi.org/10.1038/nmeth.4473 *(scite mislabels the journal; this is Nature Methods.)*
- Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014*, LNCS 8693, 740–755. https://doi.org/10.1007/978-3-319-10602-1_48
- Moreni, M., Théau, J., & Foucher, S. (2023). Do you get what you see? Insights of using mAP to select architectures of pretrained neural networks for automated aerial animal detection. *PLoS ONE*, 18(4), e0284449. https://doi.org/10.1371/journal.pone.0284449
- Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1–2), 83–97. **(Primary assignment-algorithm reference; cited in SR3 for Hungarian matching.)**

**Counting & localization metrics (D)**
- Brugger, S. D., Baumberger, C., Jost, M., et al. (2012). Automated counting of bacterial colony forming units on agar plates. *PLoS ONE*, 7(3), e33695. https://doi.org/10.1371/journal.pone.0033695
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001
- Idrees, H., Tayyab, M., Athrey, K., et al. (2018). Composition loss for counting, density map estimation and localization in dense crowds. *ECCV 2018*, LNCS 11210, 544–559. https://doi.org/10.1007/978-3-030-01216-8_33
- Pawłowski, J., Majchrowska, S., & Golan, T. (2022). Generation of microbial colonies dataset with deep learning style transfer. *Scientific Reports*, 12, 5212. https://doi.org/10.1038/s41598-022-09264-z
- Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, 327(8476), 307–310. https://doi.org/10.1016/S0140-6736(86)90837-8
- Bland, J. M., & Altman, D. G. (2007). Agreement between methods of measurement with multiple observations per individual. *Journal of Biopharmaceutical Statistics*, 17(4), 571–582. https://doi.org/10.1080/10543400701329422
- Lin, L. I-K. (1989). A concordance correlation coefficient to evaluate reproducibility. *Biometrics*, 45(1), 255–268. https://doi.org/10.2307/2532051 *(2000 erratum corrected typographical errors in the variance formulas.)*
- Steichen, T. J., & Cox, N. J. (2002). A note on the concordance correlation coefficient. *The Stata Journal*, 2(2), 183–189. https://doi.org/10.1177/1536867X0200200206
- Carrasco, J. L., & Jover, L. (2003). Estimating the generalized concordance correlation coefficient through variance components. *Biometrics*, 59(4), 849–858. https://doi.org/10.1111/j.0006-341X.2003.00099.x
- Cui, Y., Peng, L., & Hu, Y.-J. (2021). Assessing the reproducibility of microbiome measurements based on concordance correlation coefficients. *Journal of the Royal Statistical Society Series C*, 70(5), 1027–1043. https://doi.org/10.1111/rssc.12497
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in assessing rater reliability. *Psychological Bulletin*, 86(2), 420–428. https://doi.org/10.1037/0033-2909.86.2.420
- McGraw, K. O., & Wong, S. P. (1996). Forming inferences about some intraclass correlation coefficients. *Psychological Methods*, 1(1), 30–46. https://doi.org/10.1037/1082-989X.1.1.30
- ten Hove, D., Jorgensen, T. D., & van der Ark, L. A. (2024). Updated guidelines on selecting an intraclass correlation coefficient for interrater reliability. *Psychological Methods*, 29(5), 967–979. https://doi.org/10.1037/met0000516
- Qin, S., Nelson, L., McLeod, L., et al. (2019). Assessing test–retest reliability of patient-reported outcome measures using intraclass correlation coefficients. *Quality of Life Research*, 28(4), 1029–1033. https://doi.org/10.1007/s11136-018-2076-0
- Jaeger, P. A., McElfresh, C., Wong, L. R., & Ideker, T. (2015). Beyond agar: gel substrates with improved optical clarity and drug efficiency and reduced autofluorescence for microbial growth experiments. *Applied and Environmental Microbiology*, 81(16), 5639–5649. https://doi.org/10.1128/AEM.01327-15
- Wu, L., Han, S., Chen, A., et al. (2021). RCNN-SliceNet: a slice and cluster approach for nuclei centroid detection in 3D fluorescence microscopy images. *arXiv*:2106.15753. https://doi.org/10.48550/arXiv.2106.15753 **[PREPRINT — not peer-reviewed]**
- Tofighi, M., Guo, T., Vanamala, J. K. P., & Monga, V. (2018). Deep networks with shape priors for nucleus detection. *IEEE ICIP 2018*, 719–723. https://doi.org/10.1109/ICIP.2018.8451797
- Wang, Q., Han, T., Gao, J., & Yuan, Y. (2021). LDC-Net: a unified framework for localization, detection and counting in dense crowds. *arXiv*:2110.04727. https://doi.org/10.48550/arXiv.2110.04727 **[PREPRINT — not peer-reviewed]**
- Windsor, R., Jamaludin, A., Kadir, T., & Zisserman, A. (2020). A convolutional approach to vertebrae detection and labelling in whole spine MRI. *arXiv*:2007.02606. https://doi.org/10.48550/arXiv.2007.02606 **[PREPRINT — not peer-reviewed]**
- Bunch, P. C., Hamilton, J. F., Sanderson, G. K., & Simmons, A. H. (1978). A free response approach to the measurement and characterization of radiographic observer performance. *Proc. SPIE*, 0127, 124–135. https://doi.org/10.1117/12.955926
- Chakraborty, D. P. (2013). A brief history of free-response receiver operating characteristic paradigm data analysis. *Academic Radiology*, 20(7), 915–919. https://doi.org/10.1016/j.acra.2013.03.001
- Bandos, A. I., Rockette, H. E., Song, T., & Gur, D. (2009). Area under the free-response ROC curve (FROC) and a related summary index. *Biometrics*, 65(1), 247–256. https://doi.org/10.1111/j.1541-0420.2008.01049.x
- Wagih, O., & Parts, L. (2014). gitter: a robust and accurate method for quantification of colony sizes from plate images. *G3: Genes|Genomes|Genetics*, 4(3), 547–552. https://doi.org/10.1534/g3.113.009431
- Young, B. P., & Loewen, C. J. R. (2013). Balony: a software package for analysis of data generated by synthetic genetic array experiments. *BMC Bioinformatics*, 14, 354. https://doi.org/10.1186/1471-2105-14-354
- Bischof, L., Převorovský, M., Rallis, C., et al. (2016). Spotsizer: high-throughput quantitative analysis of microbial growth. *BioTechniques*, 61(4), 191–201. https://doi.org/10.2144/000114459
- Kamrad, S., Rodríguez-López, M., Cotobal, C., et al. High-throughput, high-precision colony phenotyping with pyphe. *Research Square* (preprint). https://doi.org/10.21203/rs.3.rs-401914/v1 **[PREPRINT — not peer-reviewed; peer-reviewed book-chapter version at https://doi.org/10.1007/978-1-0716-2257-5_21]**

**Partition-agreement metrics (E.A)**
- Rand, W. M. (1971). Objective criteria for the evaluation of clustering methods. *Journal of the American Statistical Association*, 66(336), 846–850. https://doi.org/10.1080/01621459.1971.10482356
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193–218. https://doi.org/10.1007/BF01908075
- Meilă, M. (2007). Comparing clusterings — an information based distance. *Journal of Multivariate Analysis*, 98(5), 873–895. https://doi.org/10.1016/j.jmva.2006.11.013
- Arbeláez, P., Maire, M., Fowlkes, C., & Malik, J. (2011). Contour detection and hierarchical image segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(5), 898–916. https://doi.org/10.1109/TPAMI.2010.161
- Pont-Tuset, J., & Marqués, F. (2013). Measures and meta-measures for the supervised evaluation of image segmentation. *Proc. IEEE CVPR*, 2131–2138. https://doi.org/10.1109/CVPR.2013.277
- Warrens, M. J., & van der Hoef, H. (2022). Understanding the adjusted Rand index and other partition comparison indices based on counting object pairs. *Journal of Classification*, 39(3), 487–509. https://doi.org/10.1007/s00357-022-09413-z
- Chacón, J. E., & Rastrojo, A. I. (2022). Minimum adjusted Rand index for two clusterings of a given size. *Advances in Data Analysis and Classification*, 17(1), 125–133. https://doi.org/10.1007/s11634-022-00491-w
- Warrens, M. J. (2008). On the equivalence of Cohen's kappa and the Hubert–Arabie adjusted Rand index. *Journal of Classification*, 25(2), 177–183. https://doi.org/10.1007/s00357-008-9023-7
- Pinto, F. R., Carriço, J. A., Ramirez, M., & Almeida, J. S. (2007). Ranked adjusted Rand: integrating distance and partition information in a measure of clustering agreement. *BMC Bioinformatics*, 8, 44. https://doi.org/10.1186/1471-2105-8-44
- Yan, Y., Feng, X., & Luo, X. (2025). Spatially aware adjusted Rand index for evaluating spatial transcriptomics clustering. *bioRxiv*. https://doi.org/10.1101/2025.03.25.645156 **[PREPRINT — not peer-reviewed]**
- Buyssens, P., Gardin, I., & Ruan, S. (2014). Eikonal-based region growing for efficient clustering. *Image and Vision Computing*, 32(12), 1045–1054. https://doi.org/10.1016/j.imavis.2014.10.002
- Giraud, R., Ta, V.-T., & Papadakis, N. (2016). SCALP: superpixels with contour adherence using linear path. *Proc. ICPR*, 2374–2379. https://doi.org/10.1109/ICPR.2016.7899991
- Van den Bergh, M., Roig, G., Boix, X., Manen, S., & Van Gool, L. (2013). Online video SEEDS for temporal window objectness. *Proc. IEEE ICCV*, 377–384. https://doi.org/10.1109/ICCV.2013.54
- Zhang, Y., Li, X., Gao, X., & Zhang, C. (2016). A simple algorithm of superpixel segmentation with boundary constraint. *IEEE Transactions on Circuits and Systems for Video Technology*, 27(7), 1502–1514. https://doi.org/10.1109/TCSVT.2016.2539839
- Neubert, P., & Protzel, P. (2013). Evaluating superpixels in video: metrics beyond figure-ground segmentation. *Proc. British Machine Vision Conference (BMVC) 2013* (canonical USE/ASA reference). https://doi.org/10.5244/C.27.39 **(Cited via Giraud et al., 2016 & Zhang et al., 2016. The previously-listed `10.1016/j.patrec.2013.09.013` was a wrong DOI — it resolves to Schick et al. 2014 — corrected per citation audit.)**

**Metric selection & pitfall meta-guidance (E.B) — shared anchors across SR1/SR2/SR5**
- Maier-Hein, L., Reinke, A., Godau, P., et al. (2024). Metrics reloaded: recommendations for image analysis validation. *Nature Methods*, 21(2), 195–212. https://doi.org/10.1038/s41592-023-02151-z *(scite mislabels the venue as "nature chemical biology" — it is Nature Methods. No retraction/concern.)*
- Reinke, A., Tizabi, M. D., Baumgartner, M., et al. (2024). Understanding metric-related pitfalls in image analysis validation. *Nature Methods*, 21(2), 182–194. https://doi.org/10.1038/s41592-023-02150-0 *(same venue mislabel; corrected to Nature Methods.)*
- Taha, A. A., & Hanbury, A. (2015). Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. *BMC Medical Imaging*, 15, 29. https://doi.org/10.1186/s12880-015-0068-x
- Taha, A. A., Reinke, A., Tizabi, M. D., et al. (2021). Common limitations of image processing metrics: a picture story. *arXiv*:2104.05642. https://doi.org/10.48550/arXiv.2104.05642 **[PREPRINT — not peer-reviewed; precursor to the 2024 pitfalls paper]**
- Jozdani, S., & Chen, D. (2020). On the versatility of popular and recently proposed supervised evaluation metrics for segmentation quality of remotely sensed images: an experimental case study. *ISPRS Journal of Photogrammetry and Remote Sensing*, 160, 275–290. https://doi.org/10.1016/j.isprsjprs.2020.01.002 *(closed access; retrieved as metadata/abstract only — full-text claims flagged as inference.)*
- Drukker, K., Sahiner, B., Hu, T., et al. (2024). MIDRC-MetricTree: a decision-tree-based tool for recommending performance metrics. *Journal of Medical Imaging*, 11(2), 024504. https://doi.org/10.1117/1.JMI.11.2.024504

**Domain grounding (colony phenotyping — context, not metric primaries)**
- Majchrowska, S., Pawłowski, J., Guła, G., et al. (2021). AGAR: a microbial colony dataset for deep learning detection. *arXiv*:2108.01234. https://doi.org/10.48550/arXiv.2108.01234 **[PREPRINT — not peer-reviewed; cited as the closest in-domain agar-plate precedent for COCO mAP / counting MAE]**

---

*Companion to [`2026-06-01-parameter-tuning-engine-design.md`](2026-06-01-parameter-tuning-engine-design.md)
(master §4 `SupervisedScorer`, D1) and
[`reference-free-segmentation-metrics.md`](reference-free-segmentation-metrics.md)
(this scorer is the meta-validation gate's reference signal — §E). See also the bundle index
[`README.md`](README.md).*
