# Reference-Free Segmentation-Quality Metrics — A Companion Catalogue

**Purpose & scope.** This document is a developer-facing reference catalogue of
**reference-free** (a.k.a. no-reference / unsupervised / "empirical-goodness")
segmentation-quality metrics: ways to score *how good a segmentation is* **without
a ground-truth mask**. It synthesizes a five-lane literature review (classic
region statistics, object/GEOBIA + spatial statistics, bioimaging/cell/colony
metrics, learning- & uncertainty-based predictors, and meta-evaluation/tuning) and
maps each family onto PhenoTypic's domain: **a regular grid of ~circular microbial
colonies on agar**. It is written so a developer can decide *which* metrics to
implement for the `ReferenceFreeScorer` and *how* to combine and validate them.

**Pointer back to the master spec.** This is the companion to
[`2026-06-01-parameter-tuning-engine-design.md`](2026-06-01-parameter-tuning-engine-design.md),
elaborating **§4** (the `Scorer` Protocol and specifically `ReferenceFreeScorer`),
**§2** (the literature basis for reference-free scoring), and the **meta-validation
gate** that decisions **D1** ("`ReferenceFreeScorer` is gated behind
meta-validation") and **D8** (fANOVA importance) depend on. Where the master spec
summarizes "intra-colony homogeneity vs. background contrast, boundary gradient,
shape regularity (Zhang 2008 / Chen 2021 style)," this document supplies the
catalogue, the formulas, the caveats, and a concrete combination + gating recipe.

> **Master-spec correction noted here (see §E and Verification status):** the
> master spec cites **Chen & Murphy (2021)**. That is the bioRxiv preprint date;
> the peer-reviewed journal version is **Chen & Murphy (2023), *Molecular Biology
> of the Cell*** (`10.1091/mbc.e22-08-0364`). The master spec's reference should be
> corrected to **2023**.

---

## Taxonomy / map of the field

Every reference-free metric reduces to the same shape: a **homogeneity / intra-region
term** (regions should be internally uniform) traded against a **counter-term**
(separation between regions, a region-count penalty, a boundary/edge term, or a
shape/structure prior) — because the homogeneity term alone is trivially minimized
by over-segmenting into single-pixel regions. The families differ in *which*
homogeneity measure, *which* counter-term, *how* they combine, and whether they
need training or a probabilistic segmenter.

| Family | Lane | One-line characterization | GT at build? | Probabilistic segmenter? |
|---|---|---|---|---|
| **(A) Hand-crafted region-statistics** | R1 | Classic grayscale/color "goodness" — uniformity vs. contrast / count penalty (Otsu, F/Q, Zeboudj, entropy E) | No | No |
| **(B) Object/GEOBIA + spatial statistics** | R2 | Treat segments as objects; intra-variance vs. inter-segment spatial autocorrelation (Global Score, q-statistic, Moran's I) | No | No |
| **(C) Bioimaging / cell / colony-specific** | R3 | Coverage, shape-prior plausibility, replicate homogeneity, expected-vs-detected grid count (Chen & Murphy, OpenCFU, gitter/SGA) | No | No |
| **(D) Learning- & uncertainty-based predictors** | R4 | Regress Dice/IoU from (image, mask), or read off predictive uncertainty (RCA, degrade-and-train, EvanySeg, MC-dropout/entropy) | Usually yes | Some require it |
| **(E) Meta-evaluation, reliability & tuning** | R5 | *Cross-cutting:* how to judge a metric, when metrics mislead, and how to use them as a tuning objective (MSET, Chabrier-vs-Vinet, Deo "metrics that matter", USPO) | — (the validation layer) | — |

Families A–D are catalogues of *metrics*; family E is the cross-cutting concern
that decides whether any chosen metric can be **trusted to drive optimization** —
the substance of the master spec's meta-validation gate.

---

# (A) Hand-crafted region-statistics metrics

*Lane R1. The "empirical goodness" tradition (Zhang, Fritts & Goldman 2008, after
Haralick & Shapiro): score a segmentation from the image + result alone via
**intra-region uniformity** balanced against **inter-region disparity** and/or an
**over-segmentation penalty**.*

**Shared notation.** Image `I`, size `S_I = I_h × I_w` pixels; segmentation = `N`
regions; region `j` is `R_j`, area `S_j = |R_j|`; `C_x(p)` = value of component `x`
(R/G/B or luminance) at pixel `p`; region mean `Ĉ_x(R_j) = (Σ_{p∈R_j} C_x(p)) / S_j`;
**squared color error** `e²_x(R_j) = Σ_{p∈R_j} (C_x(p) − Ĉ_x(R_j))²`. `N(a)` = number
of regions of area exactly `a`; `MaxArea` = largest region area.

## A.1 Levine & Nazif — performance vector: region uniformity & contrast

- **Use cases.** General-purpose region-segmentation goodness; the conceptual
  template for nearly all later goodness metrics; driving parameter selection in
  classic region-growing / split-merge systems.
- **Mathematical foundation.** Gray-level uniformity
  `U = 1 − (1/Z) Σ_{j=1}^{N} e²_gl(R_j)·W_j`, where `e²_gl(R_j)` is the squared
  gray-level error of region `j`, `W_j` a per-region (typically area-proportional)
  weight, and `Z` a normalizer so `U ∈ [0,1]` (higher = more uniform). Region
  contrast (inter-region disparity) sums per-region contrasts
  `c_i = Σ_adj p_{ij}·|Ĉ(R_i) − Ĉ(R_j)| / (Ĉ(R_i) + Ĉ(R_j))`, optionally weighted
  by a human contrast-sensitivity function (higher = better separation).
- **Advantages.** No reference image; intuitive; decomposes cleanly into
  homogeneity vs. contrast; defined the field's "characteristic criteria."
- **Limitations.** Several free weights/normalizers (`W_j`, `Z`) are
  unspecified/heuristic; the contrast term needs an adjacency graph; sensitive to
  texture (textured regions read as non-uniform); no built-in over-segmentation
  penalty.
- **Colony relevance.** Direct — colony interiors should be uniform (high `U`),
  colony-vs-agar contrast high; area-weighting downweights specks. Good "homogeneity
  half" of a composite. (medium)

## A.2 Weszka & Rosenfeld — busyness & threshold evaluation

- **Use cases.** Automatic threshold selection (minimize busyness); evaluating
  bilevel (foreground/background) thresholding.
- **Mathematical foundation.** **Busyness** = fraction of object↔background
  adjacencies from the gray-level co-occurrence matrix (sum of off-diagonal
  entries representing transitions across the object/background boundary;
  equivalently the sum of absolute 4-/8-neighbor Laplacian responses). **Lower
  busyness ⇒ smoother ⇒ better.** Discrepancy-after-thresholding
  `Discrepancy = Σ_i Σ_j (C_gl(i,j) − L(i,j))`, with `C_gl(i,j)` the original gray
  level and `L(i,j)` the thresholded/segmented gray level.
- **Advantages.** Extremely cheap (one co-occurrence-matrix pass); fully
  reference-free; strong when objects are compact and weakly textured.
- **Limitations.** Requires a threshold value, so it only evaluates threshold-type
  algorithms; assumes smooth, untextured objects (fails on texture); global, not
  per-region; binary-oriented.
- **Colony relevance.** High — arrayed colonies *are* compact, weakly-textured
  blobs on smooth agar, the ideal busyness regime; penalizes ragged/speckled masks;
  cheap enough to evaluate every candidate threshold in a sweep. (high)

## A.3 Otsu — within-/between-class variance (degenerate two-region objective)

- **Use cases.** Canonical global thresholding; baseline binarization; a free
  per-threshold reference-free score for two-class problems.
- **Mathematical foundation.** With object `o`, background `b`, areas `S_o, S_b`
  (`S_o + S_b = S_I`): within-class variance
  `σ²_W = (S_b/S_I)·e²_gl(R_b) + (S_o/S_I)·e²_gl(R_o)` (intra-uniformity, lower =
  better); between-class variance
  `σ²_B = (S_b/S_I)·(S_o/S_I)·(Ĉ_gl(R_o) − Ĉ_gl(R_b))²` (separation, higher =
  better). Total `σ²_T = σ²_W + σ²_B` is constant, so maximizing `σ²_B ≡` minimizing
  `σ²_W ≡` maximizing `η = σ²_B / σ²_T`.
- **Advantages.** Closed-form, parameter-free, blazingly fast (single histogram
  scan); the cleanest illustration of intra-vs-inter balance.
- **Limitations.** Two classes only; assumes a bimodal histogram; **ignores spatial
  structure entirely** (a salt-and-pepper mask ties a clean one at the same
  gray-level split); breaks under uneven illumination or very unbalanced class
  sizes.
- **Colony relevance.** The workhorse for colony/agar binarization, and its
  `σ²_B/σ²_T` is a *free* per-threshold score — but being histogram-only it **must**
  be paired with a spatial term (busyness, Zeboudj, F/Q penalty). (high, as one term)

## A.4 Liu & Yang — evaluation function F

- **Use cases.** Color-segmentation goodness; automatic color-space / parameter
  selection; the baseline against which F′, Q and later goodness functions are
  benchmarked.
- **Mathematical foundation.** `F(I) = √N · Σ_{j=1}^{N} ( e²_j / √S_j )`, where `N`
  = number of regions, `S_j` = area, `e²_j` = squared color error of region `j`. The
  leading `√N` grows with region count to discourage trivial over-segmentation.
  **Smaller F ⇒ better.** (The commonly published normalized form divides by
  `1000·S_I`; the constant does not change ranking.)
- **Advantages.** No user parameters; content-independent; single cheap scalar;
  intuitive (error × over-segmentation penalty).
- **Limitations.** Documented **bias toward over-segmentation** — the `√N` penalty
  is too weak, so `F` still favors many small regions (Borsotti 1998; Alkama 2015).
  Squared error alone says nothing about inter-region separation or shape.
- **Colony relevance.** Weak as-is: the soft `√N` penalty won't strongly punish
  shattering one colony into fragments, and colony arrays have a *known* region count
  (rows×cols) a domain-aware penalty exploits far better. Prefer F′/Q. (low)

## A.5 Borsotti, Campadelli & Schettini — F′ and Q (improved F)

- **Use cases.** De-facto standard reference-free color-segmentation scores;
  automatic parameter/color-space selection (e.g. adapted to microscopy by
  Meas-Yedid et al.); benchmarking; metaheuristic (PSO/GA) fitness functions.
- **Mathematical foundation.**
  `F'(I) = (1/(1000·S_I)) · √( Σ_{a=1}^{MaxArea} [N(a)]^{1+1/a} ) · Σ_{j=1}^{N}( e²_j/√S_j )`
  — the `√(Σ_a N(a)^{1+1/a})` factor blows up when many regions share the same small
  area (the signature of noise over-segmentation).
  `Q(I) = ( √N/(1000·S_I) ) · Σ_{j=1}^{N} [ e²_j/(1+log S_j) + ( N(S_j)/S_j )² ]`,
  where `N(S_j)` = number of regions sharing region `j`'s area. First term penalizes
  under-segmentation (large heterogeneous regions); second term penalizes
  over-segmentation (small regions whose area is shared by many). **Smaller F′/Q ⇒
  better.**
- **Advantages.** Parameter-free; `Q` penalizes **both** failure modes (over- *and*
  under-segmentation), unlike F/F′; consensus "best of the F-family."
- **Limitations.** The same-area terms (`N(a)`, `N(S_j)`) are brittle to exact-area
  coincidences/quantization; the `1000` is arbitrary scaling; still intra-region +
  count-based (no explicit inter-region contrast or boundary/shape term); can
  misrank when correct regions genuinely vary in size.
- **Colony relevance.** Strong. `Q`'s two-sided penalty matches colony failure
  modes exactly (split colony → many small same-area regions; merged colonies →
  large heterogeneous region). Likely the single most directly reusable classic
  metric. (high)

## A.6 Zeboudj contrast measure

- **Use cases.** Reference-free ranking of region segmentations; parameter
  auto-tuning; one of the six criteria in the Chabrier/Rosenberger comparative
  benchmarks.
- **Mathematical foundation.** Pixel contrast `c(s,t) = |I(s) − I(t)|/(L−1)`,
  `L = max(I) − min(I)`. Inner contrast of `R_i`:
  `I_i = (1/A_i) Σ_{s∈R_i} max{ c(s,t) : t ∈ W(s) ∩ R_i }` (low = uniform). Outer
  contrast: `E_i = (1/l_i) Σ_{s∈F_i} max{ c(s,t) : t ∈ W(s), t ∉ R_i }` over the
  boundary `F_i` of length `l_i` (high = well-separated). Per-region
  `C(R_i) = 1 − I_i/E_i` if `0 < I_i < E_i`; `= E_i` if `I_i = 0`; `= 0` otherwise.
  Global `C(I) = (1/A) Σ_i A_i·C(R_i)`, area-weighted. **Higher ⇒ better.**
- **Advantages.** Couples intra-uniformity **and** inter-separation in one
  normalized `[0,1]` score; local/neighborhood-based (spatially aware, unlike
  Otsu/F); area-weighted (dominated by large meaningful regions).
- **Limitations.** "Does not correctly take into account strongly textured regions"
  (internal contrast high inside texture → wrongly read as non-uniform); needs
  region adjacency/boundary extraction; `max`-of-neighbors is noise-sensitive; the
  `I_i = 0` special case is a discontinuity.
- **Colony relevance.** Very high — arguably the best-balanced classic single score
  for colonies: low-texture interiors and agar (texture weakness barely bites), low
  internal contrast inside colonies, high external contrast at the rim. Strong
  complement to a count-aware `Q`. **Provenance flag: primary source is an
  untraceable 1988 PhD thesis** (see Verification status). (high)

## A.7 Entropy-based metric E (Zhang, Fritts & Goldman)

- **Use cases.** General reference-free goodness for gray/color segmentation; the
  metric the CVIU survey recommends and benchmarks; a common "E" baseline in
  remote-sensing/SAR evaluation.
- **Mathematical foundation** (MDL-grounded; base-10 log, base only rescales):
  per-region entropy `H(R_j) = − Σ_{m∈V_j} (L_j(m)/S_j)·log(L_j(m)/S_j)` (`V_j` =
  luminance values in region `j`, `L_j(m)` = pixel count at luminance `m`; `= 0` when
  uniform). Expected region entropy `H_r(I) = Σ_{j=1}^{N} (S_j/S_I)·H(R_j)`
  (uniformity term, → 0 under over-segmentation). Layout entropy
  `H_ℓ(I) = − Σ_{j=1}^{N} (S_j/S_I)·log(S_j/S_I)` (increases with region count,
  counteracting over-segmentation). Combined `E = H_ℓ(I) + H_r(I)`. **Smaller E ⇒
  better.**
- **Advantages.** Fully parameter-free (no magic `1000`, no weights); principled
  MDL/coding-theory justification; intrinsically balances over- vs.
  under-segmentation; works on any segmentation (not just thresholding).
- **Limitations.** Layout entropy depends only on the *size distribution*, not
  spatial separation (two segmentations with identical size histograms but different
  layouts tie); no boundary/contrast/shape term; luminance-only by default; entropy
  sensitive to intensity quantization/binning.
- **Colony relevance.** Good and cheap — low region entropy when each colony mask is
  intensity-uniform; layout entropy penalizes shattering. Parameter-free → robust
  across plates. Weakness: no inter-region term, so pair with Zeboudj/contrast. (medium)

## A.8 Rosenberger / Chabrier — adaptive (FRC) & combination metrics

- **Use cases.** Parameter auto-selection and segmentation-by-optimization (GA
  fitness); robust evaluation of **textured** images where Zeboudj/F fail; the
  comparative benchmarks that indicate which classic metric best matches human
  judgment.
- **Mathematical foundation.** Intra-region disparity (non-textured mode)
  `D(I) = (1/N) Σ_{j=1}^{N} (S_j/S_I)·e²_x(R_j)` (lower = more homogeneous; textured
  mode applies the same form to texture-attribute vectors). Inter-region disparity
  per pair `D(R_i,R_j) = |Ĉ_gl(R_i) − Ĉ_gl(R_j)|/N_G` (`N_G` = number of gray
  levels), area-averaged to `D̄(I)` (higher = better; textured variant uses
  barycenter distance `d(B_i,B_j)/(|B_i|+|B_j|)`). Rosenberger's criterion combines
  by **difference**, `(intra-uniformity) − (inter-disparity)` (minimize); SVM/GA
  variants instead *learn* a combination of several literature criteria.
- **Advantages.** Adaptive uniform/textured handling (its key edge); explicit intra
  **and** inter terms; the comparative studies (validated against Vinet's measure on
  an 8,400-image database) give empirical metric-choice guidance; learned
  combinations outperform any single criterion.
- **Limitations.** More complex/slower (texture clustering per region); the
  textured mode has several internal choices; combination/SVM variants need training
  data and lose the parameter-free appeal; mostly validated on synthetic gray-level
  images.
- **Colony relevance.** Adaptive texture handling is mostly unnecessary for
  low-texture plates (a plus — stay in the cheap uniform mode). The real value is
  *methodological*: the **intra − inter** combination is a clean template, and the
  comparative finding — *no single classic metric dominates; learned/weighted
  combinations track human judgment best* — directly motivates building the colony
  scorer as a **small weighted ensemble** rather than one metric. (medium / methodological)

### Cross-cutting takeaways from family (A)

1. **Every metric = (homogeneity) traded against (over-segmentation control).** The
   counter-term is a `√N` factor (F), a same-area penalty (F′, Q), explicit
   inter-region contrast (Levine–Nazif, Zeboudj, Rosenberger, Otsu `σ²_B`), or
   size-distribution entropy (E's `H_ℓ`). **For colonies the strongest counter-term
   is domain knowledge of the expected region count (rows×cols)** — which no generic
   metric has.
2. **Four combination forms:** sum/complement (E), report separately (PV), **ratio**
   intra÷inter (Otsu `η`, Zeboudj), **difference** intra−inter (Rosenberger).
   Ratio/difference are self-normalizing — a good default for a single scalar.
3. **Size-weighting** of per-region terms (Otsu, FRC, Zeboudj, E) is the most
   defensible aggregation for colonies (suppresses tiny spurious regions).
4. **Spatial vs. histogram-only.** Otsu and the entropy terms are histogram
   statistics — blind to layout. Busyness, Zeboudj, and boundary/contrast terms are
   spatially aware. A robust colony score needs **≥1 spatially-aware term**.
5. **Empirical guidance (Chabrier/Rosenberger):** **Borsotti `Q` and Zeboudj are the
   strongest single classic criteria; no single metric dominates; learned/weighted
   combinations beat any one.**
6. **Watch the signs.** *Lower is better:* F, F′, Q, E, `σ²_W`, Rosenberger intra `D`.
   *Higher is better:* Zeboudj `C(I)`, Levine–Nazif `U`/contrast, Otsu `σ²_B`/`η`.
   Any ensemble must declare that natural sense so the shared scorer boundary can
   orient every term to lower-is-better cost (master §4).
7. **Cost.** Otsu/F/F′/Q/E are essentially free per evaluation; Zeboudj/Levine-contrast/
   Rosenberger need region adjacency or boundary extraction (modest extra cost).

---

# (B) Object / GEOBIA metrics + spatial statistics

*Lane R2. The GEOBIA (Geographic Object-Based Image Analysis) tradition: treat
segments as objects and score **intra-segment homogeneity** vs. **inter-segment
heterogeneity**, built on spatial-statistics machinery (Moran's I, Geary's C, the
q-statistic). Used for **automatic parameter / scale selection without ground
truth** — exactly PhenoTypic's use case.*

## B.1 Espíndola Global Score (GS) — weighted variance + Moran's I

- **Use cases.** Automatic selection of the multiresolution/region-growing **scale
  parameter**; ranking candidate segmentations of one scene.
- **Mathematical foundation.** Area-weighted variance (intra, lower = better)
  `v = ( Σ_i a_i·σ_i² ) / ( Σ_i a_i )` (`a_i` = area, `σ_i²` = band variance of
  segment `i`). Global Moran's I (inter, lower = better)
  `I = ( n·Σ_i Σ_j w_ij (y_i − ȳ)(y_j − ȳ) ) / ( ( Σ_i (y_i − ȳ)² )·( Σ_i Σ_{j≠i} w_ij ) )`,
  with `y_i` = segment mean, `w_ij = 1` if segments share a boundary else `0`. Both
  min–max normalized to `[0,1]` over the tested set and summed:
  `GS = v_norm + I_norm` (lower = better).
- **Advantages.** Simple, interpretable; needs only the segmentation + source
  band(s); captures both error directions in one scalar; the de-facto GEOBIA
  standard.
- **Limitations.** **Range-dependence is severe** — `X_min`/`X_max` come from the
  tested set, so the optimum *shifts when you change which segmentations you compare*
  (see B.3). Single-band, spectral-only; assumes adjacency/RAG; Moran's I needs ≥ a
  handful of segments.
- **Colony relevance.** High — weighted variance rewards uniform colony interiors;
  Moran's I rewards colony-vs-agar contrast. Caveat: with a fixed grid the segment
  count barely changes, so the min–max range can be narrow — prefer fixed-reference
  or rank-based normalization. (high)

## B.2 Johnson & Xie GS — the widely-cited implementation

- **Use cases.** Optimal-scale selection for object-based land-cover
  classification; iterative refinement; the standard "JM" baseline.
- **Mathematical foundation.** Same `GS = WV_norm + MI_norm` as B.1, with (i) WV a
  **weighted sum of per-band variances** (multispectral) and (ii) Moran's I over all
  segments. Validated via a clever reference-free sanity check: an **expert manual
  digitization should score well (low GS)**.
- **Advantages.** Multiband; ties scale selection to downstream classification
  accuracy; the refinement step fixes locally bad segments.
- **Limitations.** Inherits GS range-dependence (B.3); Moran's I over a RAG is
  O(edges); a global scalar can mask spatially-varying optimal scale.
- **Colony relevance.** High and the most copyable baseline — plates are
  multi-channel (RGB/Lab) so multiband WV applies directly; "score the partition,
  pick the min, optionally refine bad colonies" is exactly a `ReferenceFreeScorer`
  loop, and the **validation trick (a hand-labeled colony mask should score well)**
  is a cheap acceptance test for any PhenoTypic implementation. (high)

## B.3 Böck et al. — instability of the Global Score (critical caveat)

- **Use cases.** A design constraint for *any* GS-style scorer: how you normalize
  determines reproducibility.
- **Mathematical foundation.** The flaw is in `X_norm = (X − X_min)/(X_max − X_min)`:
  since `v` rises and `I` falls monotonically with scale, `X_min`/`X_max` are the
  finest/coarsest tested members; add or remove a candidate and the **argmin moves**.
  Remedy: normalize against **fixed external bounds** (theoretical / single-pixel /
  whole-image extremes), decoupling the optimum from the search grid.
- **Advantages (of the critique).** Explains contradictory results across GS papers;
  gives a concrete fix.
- **Limitations.** The fix needs sensible external bounds, which are problem-specific.
- **Colony relevance.** **Directly actionable** — PhenoTypic's sweep *is* a parameter
  grid, so a naive min–max-over-grid GS would make the "best" parameters depend on
  grid endpoints. Use **fixed normalization** (variance relative to whole-image
  variance; Moran's I in its natural `[−1,1]`; q in `[0,1]`) or rank/Mahalanobis-based
  combination instead. (high / cautionary)

## B.4 Corcoran & Winstanley — Spatial Unsupervised (SU) metric

- **Use cases.** Benchmark UE metric for OBIA; one of the metrics against which
  Troya-Galvis validate their UOA metric.
- **Mathematical foundation.** A weighted combination of an **intra-region
  uniformity** term and an **inter-region contrast** term computed **at shared object
  boundaries in the spatial domain** (modelling the human visual system seeking
  boundary contrast). *Exact equations could not be verified from a retrieved source
  — flagged (paper paywalled).*
- **Advantages.** Spatially aware (boundary contrast); explicitly perception-motivated;
  less reliant on global spectral statistics than GS.
- **Limitations.** Boundary terms sensitive to ragged edges; formula not openly
  available; heavier to compute than GS.
- **Colony relevance.** Moderate–high — colony edges are the salient feature, so a
  boundary-contrast metric is well-matched; but treat SU as a **design idea ("score
  boundary contrast, not just global variance")** rather than a copy-ready equation.
  (medium–high)

## B.5 Gao et al. — q-statistic + Moran's I, combined by Mahalanobis distance

- **Use cases.** Scale/parameter selection on VHR optical imagery; shown to
  correlate with supervised metrics better than plain GS.
- **Mathematical foundation.** Spatial stratified heterogeneity q-statistic (intra,
  higher = better): `q = 1 − ( Σ_{h} Σ_{k} (Y_hk − Ȳ_h)² ) / ( Σ_{i} (Y_i − Ȳ)² )`
  = fraction of total variance explained by the segmentation (`q ∈ [0,1]`). Global
  Moran's I at segment level (inter, low |MI| = better). Combination via Mahalanobis
  distance to the worst-case point:
  `d_M(X_o, X_s) = sqrt( (X_o − X_s)ᵀ Σ⁻¹ (X_o − X_s) )`, with `X_s = (|MI|_s, q_s)`,
  `X_o = (1, 0)` the worst case, `Σ` = covariance of all quality points (larger
  `d_M` = better). Computed on a **fused spectral + texture (bilateral-filtered)**
  feature set.
- **Advantages.** Multi-feature (robust on textured objects); q has a clean
  variance-explained interpretation; Mahalanobis combination is
  scale/correlation-invariant (partially addresses Böck instability).
- **Limitations.** `Σ` is **still estimated from the tested set** (residual
  range-dependence); feature extraction adds cost; needs enough segmentations to
  estimate `Σ`. *Eqs. verified from PMC full text.*
- **Colony relevance.** High — the q-statistic = "fraction of pixel-intensity variance
  explained by the colony partition" is a very natural reference-free colony score;
  texture fusion helps if colonies have internal structure; Mahalanobis is a good
  template for fusing homogeneity + boundary terms without arbitrary weights. (high)

## B.6 Troya-Galvis et al. — Under-/Over-segmentation Aware (UOA) local metric

- **Use cases.** Diagnosing *where* and *which way* a segmentation fails; selecting
  parameters that trade over- vs. under-segmentation; driving local refinement.
- **Mathematical foundation.** Per-segment: with homogeneity criterion `H(R_i)` and
  threshold `δ`, a neighbor that *should* merge (`H(R_i ∪ R_j) ≤ δ`) signals
  **over-segmentation**; an internally heterogeneous segment (`H(R_i) > δ`) signals
  **under-segmentation**. `H` is a pluggable meta-parameter (variance, entropy,
  contrast, CIE L\*a\*b\* contrast, cohesion, spectral angle, GLCM texture). Two
  global aggregations: `UOA_Σ` (weighted sum of local scores) and `UOA_L2` (L2
  aggregation of the two components). *Aggregation equations partially retrieved; the
  H/δ classification logic verified from full text.*
- **Advantages.** Local → respects object-size variability; **separates the two error
  directions** (most scalars conflate them); `H` is swappable. (The authors note all
  six tested `H` indices correlate with segment size — a confound they expose.)
- **Limitations.** Needs a homogeneity threshold `δ` (tuning); per-segment
  neighborhood analysis is costlier; the H–size correlation can bias toward
  particular scales.
- **Colony relevance.** High and conceptually apt — "over-segmented colony" (one
  split) vs "under-segmented colony" (two merged) are *the* two colony failure modes,
  and the grid gives an expected colony count/size to set `δ`. A per-colony UOA-style
  score is arguably more useful than any global scalar. (high)

## B.7 Zhao et al. — Fast Global Score (FGS): WV + Difference-To-Neighbor-Pixels

- **Use cases.** Fast scale-parameter selection on VHR imagery; large-scene sweeps
  where Moran's-I-over-RAG is too slow.
- **Mathematical foundation.** Area-weighted variance
  `WV = ( Σ_i a_i·v_i ) / ( Σ_i a_i )`, `v_i = (1/m) Σ_b v_ib` (per-band variance,
  equal band weight). **DTNP** replaces Moran's I: for each object `i`, take its
  bounding box enlarged by `d` (= 1 pixel), and measure the mean spectral difference
  between the object's pixels and the surrounding ring `B_i(d) \ object_i`,
  area-weighted across objects — **no RAG**. Combined `FGS` (their Eq. 6) =
  normalized homogeneity + normalized heterogeneity. *DTNP closed form (Eqs. 4–5) not
  fully transcribed; bounding-box-ring mechanism verified from full text.*
- **Advantages.** **No RAG construction** → cheap and simple; DTNP stays sensitive to
  both error directions where some global heterogeneity measures saturate; drop-in
  intra-term.
- **Limitations.** Bounding-box ring is a crude neighborhood (leaks across nearby
  objects when packing is dense); still GS-style normalization (range caveat);
  spectral-only.
- **Colony relevance.** High and practical — sparse, compact, near-circular blobs on
  a uniform background are *ideal* for a bounding-box-ring "difference to surrounding
  agar" heterogeneity term, far cheaper than a colony adjacency graph. Watch the
  dense-grid case: size `d` to the colony gap so neighbor rings don't overlap. (high)

## B.8 Yu et al. — SAR unsupervised evaluation (GHO + GHE + EVI → G)

- **Use cases.** Reference-free parameter selection for SAR segmentation (region
  smoothing, FCM, MRF, DNN-based segmenters).
- **Mathematical foundation.** Extract heterogeneous features — intensity (edge-
  preserving IFEE, Gamma + MAP speckle suppression), Gabor texture, multi-scale edge
  — then three indicators fused (their Eq. 19) into **G**: **GHO** (global
  intra-segment homogeneity, lower = better); **GHE** (global inter-segment
  heterogeneity via a **Bhattacharyya-coefficient** intensity-histogram distance plus
  a **Canberra-distance** texture term); **EVI** (edge validity index: agreement
  between segment boundaries and a multi-scale Prewitt edge map). *Symbol-level
  equations (Eqs. 10, 14, 18, 19) not transcribed; composition verified from full
  text. Reported mean correlation with supervised metrics > 0.67 (> 0.99 in one
  setting).*
- **Advantages.** Multi-feature (intensity + texture + edge) → robust to speckle and
  texturally-distinct objects; an explicit **edge-validity** term the optical GS
  family lacks; validated across four segmenters.
- **Limitations.** Heavy feature-extraction pipeline; many sub-parameters;
  SAR-specific assumptions (Gamma intensity) not all transferable; lightly cited
  (new).
- **Colony relevance.** Moderate — the speckle/Gamma machinery is overkill for
  colonies, *but* the **three-term design (homogeneity + heterogeneity + explicit
  edge-validity) is the most complete template here** and maps well to colonies
  (uniform interior + contrast to agar + sharp circular boundary). The **EVI** idea
  ("does the segment boundary sit on a real intensity edge?") is especially relevant
  for crisp colony rims and is missing from the GS/FGS families. (medium)

## B.9 Spatial-statistics underpinnings

- **Global Moran's I** — `I = (N/W)·( Σ_i Σ_j w_ij (x_i − x̄)(x_j − x̄) ) / ( Σ_i (x_i − x̄)² )`,
  `W = Σ_i Σ_j w_ij`, contiguity weights `w_ij = 1` if adjacent else `0`;
  `I ∈ [−1,1]` (positive = similar neighbors; ≈0 = random; negative = contrast).
  The inter-segment term in B.1/B.2/B.5; low I means colonies stand out from agar.
- **Geary's C** — `C = ((N−1)/(2W))·( Σ_i Σ_j w_ij (x_i − x_j)² ) / ( Σ_i (x_i − x̄)² )`;
  `C ∈ [0, ~2]` (`<1` positive autocorrelation, `1` none, `>1` negative). More
  sensitive to **local/short-range** differences than Moran's I — its boundary/local
  sensitivity could suit colony-edge contrast.
- **q-statistic (geographical detector)** — formula as in B.5; `q ∈ [0,1]` = fraction
  of total variance explained by the strata. Arguably the **cleanest single
  reference-free "is this partition good?" score for colonies**: directly computable,
  bounded, no normalization gymnastics.

### Cross-cutting takeaways from family (B)

1. **One shared recipe:** `score = combine( intra homogeneity , inter heterogeneity )`,
   often + an edge/boundary term (Corcoran SU, Yu EVI). A sensible colony default is
   **q-statistic (homogeneity) + a bounding-box-ring DTNP or Geary's C
   (heterogeneity) + an edge-validity term**, combined Mahalanobis- or rank-style.
2. **Normalization is the #1 trap (Böck).** Min–max-over-the-tested-set makes the
   argmin depend on the parameter grid you sweep — **fatal for a tuning engine that
   *is* a grid sweep.** Use fixed/external normalization or covariance/rank-based
   combination. (This is the single most important transfer caveat for master §4.)
3. **RAG cost & the colony grid.** Moran's I, Geary's C, and Corcoran SU need a
   region-adjacency graph; FGS's DTNP and the q-statistic do **not**. PhenoTypic's
   known grid makes adjacency nearly free, but q/DTNP skip it entirely — cheaper and
   no dense-packing leakage.
4. **Strong prior = extra signal these metrics ignore.** GEOBIA assumes *unknown*
   object count/layout; PhenoTypic knows the grid geometry, expected count, and
   approximate size. That makes **UOA-style over/under-segmentation classification**
   especially powerful — deviations from one-blob-per-grid-cell are direct,
   almost-supervised quality signals.
5. **Edge-validity is under-used in optical UE** (only Corcoran and Yu reward
   boundaries sitting on real edges) — a high-value, cheap term for crisp colony rims.

---

# (C) Bioimaging / cell / colony-specific reference-free metrics

*Lane R3. Reference-free scoring of biological-object segmentations. The colony-array
setting is unusually friendly because three strong GT-free priors are available:
(1) **expected count and grid geometry**, (2) **shape priors** (compact, convex,
~circular), and (3) **homogeneity/replicate structure** (sibling colonies should look
alike).*

## C.1 Chen & Murphy reference-free cell-segmentation metric suite (the anchor)

- **Use cases.** Choosing the best segmenter for a new tissue/modality without
  annotating it (ranked 14 methods across 4 multiplexed modalities × 5 tissues);
  driving an automatic post-processing repair step; deployed in HuBMAP.
- **Mathematical foundation.** **14 reference-free metrics** combined by **PCA** into
  one "cell segmentation quality score," in two families:
  - *Coverage:* **NC** = `N_cells / (A_µm²/100)` (cells per 100 µm²); **FFC** =
    `(pixels in cell masks ∩ foreground)/(foreground pixels)`; **FMCN** = fraction of
    cells with a one-to-one matched nucleus.
  - *Uniformity:* CV-of-marker-intensity within inferred cell-type clusters folded to
    a bounded score: for `CV = σ/μ`, the metric is `1/(ACVC + 1) ∈ (0,1]` (so lower
    variability → higher, bounded score); **AS** = average silhouette of a cell-type
    clustering (see C.4); **FPCC** a first-PC homogeneity term.
  A matched cell↔nucleus preprocessing step removes mismatched cells first
  (Otsu-thresholded nuclear mask substituted when none is emitted). The 14 z-scored
  metrics are PCA-reduced; the leading PC(s) form the score, which the paper reports
  "highly correlates with three quality benchmarks that use expert annotations" — the
  key validation that a reference-free score can stand in for GT-based F1/Jaccard.
  *Several exact closed forms (precise foreground definition, ACVC cluster weighting,
  PC count/weights) summarized but not quoted verbatim — flagged.*
- **Advantages.** No human reference (avoids cost + observer variability); single
  scalar comparable across methods; empirically tracks GT benchmarks; the
  coverage/uniformity split is clean and partially modality-agnostic.
- **Limitations.** Uniformity terms rely on **multichannel marker** info — a plain
  RGB/grayscale colony plate has no marker panel, so that family does **not** transfer
  directly. Shape plausibility is only handled indirectly (authors flag shape
  descriptors as the missing future-work piece). Nucleus-matching assumes a paired
  nuclear channel. PCA weighting is dataset-dependent.
- **Colony relevance.** High via three transferable ideas: (1) **coverage metrics
  translate almost verbatim** — FFC → fraction of plate foreground occupied by
  colonies; NC → detected colony count vs. expected grid count (no marker channel
  needed); (2) the **`1/(x+1)` bounded-score trick** folds any dispersion (e.g. colony-
  size CV) into `[0,1]`; (3) the **z-score-then-PCA-leading-component(s) composition
  recipe** is directly reusable to fuse colony coverage + shape + count + replicate-
  homogeneity terms, with the paper's own GT-benchmark validation as precedent. The
  uniformity *spirit* transfers if **replicate colonies of one strain** are treated as
  "cells of the same type" (low-variance feature vectors = marker-free ACVC analogue).
  (high)

## C.2 Morphology / shape-prior plausibility (circularity, solidity, etc.)

- **Use cases.** Per-object accept/reject and plausibility scoring across phenomics,
  nuclear-morphology disease scoring, particle analysis, and colony counters (C.5)
  that reject artifacts.
- **Mathematical foundation** (ImageJ/scikit-image conventions; `A` = area, `P` =
  perimeter, `A_convexhull` = convex-hull area, `L_major`/`L_minor` = fitted-ellipse
  axes): **Circularity** `C = 4πA/P²` (1.0 = perfect circle, → 0 elongated/ragged);
  **Solidity** `S = A/A_convexhull` (1.0 convex; drops with concavities / merged
  neighbors); **Roundness** `R = 4A/(π·L_major²)` (insensitive to boundary
  roughness); **Aspect ratio** `AR = L_major/L_minor`; **Eccentricity**
  `e = sqrt(1 − (L_minor/L_major)²)`; **Extent** `A/A_boundingbox`; **convexity**
  `P_convexhull/P`; Feret diameters. All scale-, translation-, rotation-invariant.
- **Advantages.** Cheap, interpretable, fully reference-free, computed from the binary
  mask alone (no intensity/marker channel), and **exactly aligned with the colony
  shape prior**. Population summaries (median + CV of circularity/solidity) give a
  plate-level number.
- **Limitations.** Circularity is very sensitive to perimeter quantization at small
  sizes (a few-pixel colony is unreliable); solidity and circularity are strongly
  **collinear** (don't double-count); descriptors don't catch *under*-coverage (a
  too-small but circular mask scores well); they assume the true object is convex/round
  (fails on irregular/filamentous/swarming colonies).
- **Colony relevance.** The single most directly transferable family — per-colony
  **solidity** flags merged-neighbor/concave segments; **circularity** flags
  perimeter roughness; **eccentricity/aspect ratio** flag smears/streaks; aggregate
  (median, fraction passing) for a plate score. Chen & Murphy explicitly flag shape
  descriptors as the missing piece of marker-based scoring — so this is the
  complementary signal that makes a *marker-free* colony scorer viable. (high)

## C.3 Size-CV & replicate-homogeneity (size-uniformity priors)

- **Use cases.** Detecting implausibly heterogeneous object sizes (a sign of
  merged/split errors); flagging/normalizing plate edge effects; QC gating before
  fitness scoring.
- **Mathematical foundation.** Object-size CV `CV = std(size)/mean(size)`, folded
  with the `1/(x+1)`-style bounded transform (Chen & Murphy include a cell-size-
  dispersion term). Galardini plate **border correction**:
  `correction = median(S_outer)/median(S_inner)`, where `S_outer` = colony sizes on
  the outer border ring and `S_inner` = interior sizes; far from 1 quantifies
  edge-effect inflation (border colonies grow larger from reduced competition). *Both
  verified from text.*
- **Advantages.** Marker-free, needs only the mask; the **within-replicate** form
  exploits the array's known replicate structure (a near-perfect marker-free analogue
  of Chen & Murphy's cell-type uniformity); border-ratio is one interpretable number.
- **Limitations.** Real biology produces genuinely heterogeneous sizes (different-
  fitness mutants), so global size-CV must be applied **within groups expected to be
  homogeneous** (replicates / same strain), not across the whole plate; edge
  correction needs enough colonies to estimate interior vs. border medians robustly.
- **Colony relevance.** Direct — PhenoTypic knows the grid layout and (often) the
  replicate map, so **within-strain size CV** is a reference-free homogeneity score
  and **median(border)/median(interior)** an edge-effect QC flag; both are standard in
  SGA pipelines and pure functions of mask + grid metadata. (high)

## C.4 Average silhouette as an internal cluster-validity term

- **Use cases.** Choosing `k` / validating clustering with no labels; in Chen &
  Murphy, scoring marker-based cell-type separability as a downstream-quality proxy.
- **Mathematical foundation** (Rousseeuw 1987): for point `i`, `a(i)` = mean
  dissimilarity to its own cluster, `b(i)` = smallest mean dissimilarity to any other
  cluster; `s(i) = (b(i) − a(i)) / max(a(i), b(i)) ∈ [−1,1]` (`≈1` well-clustered,
  `≈0` boundary, `<0` likely mis-assigned). The **average silhouette width**
  summarizes a clustering; pick the `k` maximizing it.
- **Advantages.** Bounded, interpretable, label-free, decades of use; fits the
  "objects of the same type should be coherent" prior.
- **Limitations.** Requires a feature space + a clustering step (extra machinery,
  `O(n²)` distances naively); needs ≥2 clusters; assumes compact-ish clusters.
- **Colony relevance.** Usable but heavier than C.2–C.3 — if a plate has multiple
  strains/conditions, the average silhouette of colonies clustered in
  (size, color, texture) space scores whether the segmentation yields separable
  phenotype groups (a marker-free AS analogue). Likely a **secondary** term, behind
  coverage/shape/count. (medium)

## C.5 Reference-free colony counters (OpenCFU, CHiTA, CFUCounter, ColTapp)

- **Use cases.** CFU enumeration robust to agar bubbles, cracks, dust, dish edges;
  time-lapse colony growth (ColTapp).
- **Mathematical foundation.** **OpenCFU** iterates a threshold; at each level a
  **particle filter** validates each component using relationships among **area,
  perimeter, convexity, aspect ratio and hollowness** to decide if it is a valid
  circular region, incrementing a per-pixel **score-map** ("how recurrently a pixel is
  part of a circular region across thresholds"). A second pass labels components
  invalid / individual / multiple (watershed-split on the distance transform, then
  re-validated); optionally a normal distribution is fit to object color intensities.
  **CHiTA** uses the **circular Hough transform**; **ColTapp** uses circular-Hough +
  morphology over time-lapse; **CFUCounter** adds local-minima watershed with explicit
  **plate-border handling**.
- **Advantages.** Battle-tested reference-free artifact rejection on *exactly the
  colony domain*; OpenCFU is the field standard and open-source; the **score-map**
  (stability of a region across thresholds) is itself a reference-free per-colony
  confidence measure.
- **Limitations.** Tuned for *counting* (validity), not boundary-accuracy scoring;
  struggle with dense overlapping/touching colonies; shape priors assume circularity
  (fail on irregular/swarming colonies); border colonies remain a recurring failure
  mode.
- **Colony relevance.** The most domain-matched prior art — OpenCFU's particle-filter
  feature set (**area, perimeter, convexity, aspect ratio, hollowness**) is a
  ready-made per-colony validity vector, and its **threshold-stability score-map** a
  reusable per-colony confidence signal (a colony stable across many thresholds is
  trustworthy; a one-threshold bubble is not). CHiTA/ColTapp confirm the
  circular-Hough/circularity prior as the right backbone for agar arrays. (high)

## C.6 Arrayed-colony / SGA pipelines: grid-regularity, expected-vs-detected count, plate QC

- **Use cases.** Genome-scale fitness / genetic-interaction screens; colony size as a
  fitness proxy; automated rejection of bad plates before quantification (Galardini
  discards < 5% of ~4,200 images via these rules).
- **Mathematical foundation.** **Grid fitting (gitter)** via a Radon transform
  `R(r, α) = Σ_{i,j} I_{ij}·δ(r − i·cosα − j·sinα)` (lines through colony rows/columns
  have large sums; the periodic peak structure recovers the grid; its regularity is
  itself a reference-free grid-quality signal). **Plate-QC heuristics**
  (Galardini/Viéitez): a plate is flagged **"poor overall quality" if no colony size
  is reported for > 5% of grid positions**, and **"potential grid misalignment" if no
  colony size is reported for > 90% of a row or column**; **known-empty positions**
  catch mislabeled plates. **Spatial normalization (SGAtools)** corrects row/column/
  edge artifacts before scoring. *All verified from text.*
- **Advantages.** Reference-free, cheap, robust, **proven at genome scale**; the
  count/grid rules need only grid metadata + which positions yielded a measurable
  colony; produce per-plate accept/reject decisions.
- **Limitations.** Assume a known regular array with known empty positions; thresholds
  (5%, 90%) are heuristic/lab-tuned; grid-fitting can fail on heavily morphed colonies,
  rotation, or low contrast; designed for *binary* plate QC, not graded scoring.
- **Colony relevance.** **The most directly portable QC for PhenoTypic** — adopt nearly
  as-is: (1) expected-vs-detected count per plate and per row/column (the 5% / 90%
  rules); (2) grid-regularity from the Radon/peak structure or fitted-lattice
  residuals; (3) known-empty-position checks; (4) SGAtools-style edge/spatial-bias
  checks (cf. C.3 border ratio). Because PhenoTypic already knows `nrows`/`ncols` and
  grid geometry, these are essentially free and the **strongest GT-free signals
  available in the agar-array setting**. (high)

## C.7 Self-supervised / annotation-free cell segmentation (mostly conceptual support)

- **Use cases.** Segmenting 3D cleared-tissue / spatial-transcriptomics data where GT
  is scarce; cases where shape priors substitute for labels.
- **Mathematical foundation.** **CellSeg3D / WNet3D** — a self-supervised W-Net with a
  soft-normalized-cut clustering objective (quality enforced by an internal
  reconstruction/clustering loss, not a labeled mask). **BIDCell** — biologically-
  informed loss tying gene-expression to morphology, evaluated with "metrics in five
  complementary categories." **StarDist** — star-convex polygon parameterization is
  itself a shape prior (a reference-free plausibility constraint). *Learned methods;
  exact loss formulas not all retrieved — flagged.*
- **Advantages.** Demonstrate that morphology/shape priors alone can drive segmentation
  and (implicitly) its evaluation; the "structure inferable from unlabeled data" thesis
  underwrites a marker-free scorer.
- **Limitations.** These are *segmentation* methods, not standalone scorers; their QC is
  entangled in training; the deep-learning ones need compute/data; transfer to a simple
  agar plate is indirect.
- **Colony relevance.** Mostly conceptual — they validate the premise that GT-free
  shape/structure priors are a legitimate quality signal (StarDist's star-convexity is
  apt for round colonies), and BIDCell's "five complementary metric categories" is a
  useful template for a multi-axis score (coverage + shape + count + homogeneity +
  spatial), echoing Chen & Murphy's PCA composition. **Preprint flag:** BIDCell is a
  bioRxiv preprint; CellSeg3D is now peer-reviewed in *eLife* (prefer the eLife
  version). (low / conceptual)

### Cross-cutting takeaways from family (C) — directly usable for agar arrays

Ranked by directness for the `ReferenceFreeScorer`:

1. **Count / grid priors — strongest and cheapest.** Adopt gitter + Galardini/Viéitez
   rules nearly verbatim: expected-vs-detected count, the **> 5% unmeasurable →
   poor-quality** and **> 90%-of-a-row/column-empty → grid-misalignment** flags,
   known-empty-position checks, and a grid-regularity term. Field-standard, validated
   at genome scale, need only mask + grid metadata.
2. **Shape-prior plausibility — second pillar, exactly matches "round colony."**
   Per-colony solidity (merged-neighbor detector), circularity, eccentricity; aggregate
   to plate level. OpenCFU's particle-filter vector + threshold-stability score-map are
   a ready-made per-colony validity + confidence design.
3. **Homogeneity priors translate via replicate structure.** The marker-uniformity
   family doesn't transfer (no marker panel), but its spirit does: within-group CV of
   size/color/texture (`1/(x+1)`-folded) + the median(border)/median(interior) ratio;
   average silhouette over a (size, color, texture) clustering as an optional heavier
   term.
4. **Composition recipe.** Chen & Murphy's **z-score-then-PCA-leading-component(s)** is
   the precedent for fusing the above into one scalar; their GT-benchmark validation
   justifies trusting such a fused score. BIDCell's "five categories" reinforces the
   multi-axis layout.
5. **Coverage metrics port verbatim.** FFC → fraction of plate foreground occupied by
   colonies; NC → colonies per unit area / per grid cell.

**Watch-outs.** (a) Small colonies make circularity/perimeter noisy — gate by minimum
size. (b) Solidity and circularity are collinear — don't double-weight. (c) Global
size-CV must be computed *within* homogeneity groups. (d) Border colonies are a
persistent artifact class — handle explicitly. (e) Shape priors assume convex/round —
provide an escape hatch for irregular/swarming/filamentous morphologies.

---

# (D) Learning- & uncertainty-based quality predictors

*Lane R4. Methods that **predict a per-case quality score (Dice/IoU) at test time
without GT**, either by (a) a learned regressor mapping (image, segmentation) → a
quality metric, or (b) a surrogate read off the model's predictive uncertainty.*

**The hard gating variable.** Methods split on **"does the segmenter expose per-pixel
probabilities?"**
- **Need only a trained regressor + (image, mask) — work on ANY mask, including a
  deterministic classical pipeline's:** RCA (D.1–D.2), Robinson CNN proxy (D.3),
  Galdran degrade-and-train (D.4), EvanySeg (D.5), Li contrastive, Luan radiomics-QA.
  **These fit PhenoTypic's current classical colony pipeline.**
- **Need a probabilistic / Bayesian segmenter (softmax or MC dropout):** Wang TTA/MC
  dropout (D.6), Roy structure-wise uncertainty (D.7), DeVries two-stage (D.8), da
  Cruz entropy (D.9), Outeiral network-score (D.10). **Not usable on a deterministic
  threshold/watershed pipeline** unless a learned probabilistic colony segmenter is
  introduced. (TTA is a partial exception — repurposable as a label-free
  *prediction-stability-under-perturbation* heuristic, but that yields a stability
  score, not a calibrated Dice.)

## D.1 Reverse Classification Accuracy (RCA) — Valindria et al.

- **Use cases.** Per-case QC in clinical pipelines; large-scale population imaging;
  failure detection.
- **Mathematical foundation.** Treat the predicted segmentation `S_I` of a new image
  `I` as pseudo-GT, train a reverse classifier `f_{I,S_I}` on that single (image,
  pseudo-GT) pair, then apply it to a small reference DB with GT; predicted quality
  `ρ̂(S_I) = max_k ρ( f_{I,S_I}(J_k), G_k )` over reference images `{J_k}` with GTs
  `{G_k}` (`ρ` = Dice/DSC etc.). Realized via Atlas Forests, single-atlas label
  propagation (most accurate), or constrained CNNs (least suitable). Needs **no
  good/bad-segmentation training set** — only the GT reference DB.
- **Advantages.** No labeled good/bad training set; per-instance; segmentation-method-
  agnostic; aggregating per-case predictions gives a method-level estimate.
- **Limitations.** Computationally heavy (~660 s/case); overlap metrics (DSC/Jaccard)
  predicted well but **distance metrics (Hausdorff, ASD) poorly**; needs a
  representative GT reference DB + registration/atlas step (awkward for translation/
  rotation-free arrayed plates — the Atlas-Forest variant is the more plausible fit).
- **Colony relevance.** *Could* score a colony segmentation with no inference-time GT,
  but the registration assumption is awkward for grids and it is heavyweight for a
  sweep scoring thousands of variants. Better used as a **label generator** to
  bootstrap a fast regressor. (low as inner loop; medium as label generator)

## D.2 RCA at scale — Robinson et al. (cardiac MRI, UK Biobank)

- **Use cases.** Removing invalid segmentations from population-imaging biomarker
  pipelines before downstream analysis.
- **Mathematical foundation.** RCA with a single-atlas registration classifier
  (center-of-mass + non-linear registration of reference images to the test image,
  warp reference GTs, overlap predicts the metric); thresholds (e.g. MSD > 2.0 mm)
  convert predicted scores to pass/fail.
- **Advantages.** Strong validation at scale: 99% binary low/high accuracy on the
  initial set; r ∈ [0.95, 0.99] between predicted and real metrics; ~95–98% on 4,800
  scans; agreement with manual QC on 7,250 scans. Inherits RCA's no-training-set
  property.
- **Limitations.** Same RCA cost; registration-dependent (plates lack a canonical
  atlas, though the regular grid partly helps); per-structure variance differs.
- **Colony relevance.** Demonstrates RCA is **production-viable for automated QC at
  population scale** — the closest analog to scoring a large parameter sweep. The
  registration step is the main transfer risk. (medium / proof-of-viability)

## D.3 Real-Time Prediction of Segmentation Quality — Robinson et al. (learned CNN proxy)

- **Use cases.** Real-time in-scanner QC; high-throughput screening where RCA is too
  slow.
- **Mathematical foundation.** A fully-3D CNN ingests an (image + 4 masks) stack →
  regresses the DSC, in ~40 ms (GPU) / 600 ms (CPU) — ~10,000× faster than RCA.
  Trained either on (1) **GT-labeled** DSC targets (MAE 0.03, 97% binary accuracy;
  12,880 train samples) or (2) **RCA-pseudo-labeled** targets — no manual data (MAE
  0.14, 91%).
- **Advantages.** Orders of magnitude faster than RCA → feasible for scoring an entire
  sweep; the RCA-labeled regime **bootstraps a quality regressor without manual masks**.
- **Limitations.** The accurate regime needs ~12,880 GT-labeled (image, mask, DSC)
  samples; the label-free regime inherits RCA noise (MAE 0.03 → 0.14); fixed-channel
  design assumes a known class layout — domain shift requires retraining.
- **Colony relevance.** Highly relevant pattern — a CNN ingesting (plate image +
  candidate colony mask) → predicted Dice *is* a `ReferenceFreeScorer`; cost lands at
  build time (labeled plates or RCA-bootstrapped labels); the fixed-channel design must
  generalize for variable colony counts. (medium–high, build-time cost)

## D.4 No-Reference Quality Metric for vessel segmentation — Galdran et al. (degrade-and-train)

- **Use cases.** No-reference scoring of binary structure segmentations; **automatic
  per-image threshold / hyperparameter selection** for an unsupervised segmenter — the
  closest published analog to a parameter-tuning objective.
- **Mathematical foundation.** Take expert masks, **artificially degrade them by known
  amounts**, train a CNN to predict the similarity between a (degraded) mask and its
  clean GT; at inference the CNN scores a new segmentation with no reference. *Source
  PDF non-extractable, so per-symbol formulas are not verifiable from a retrieved
  source — not reproduced (flagged).* Validated by picking an operating threshold for an
  independent unsupervised vessel segmenter: chosen thresholds beat ROC-derived ones by
  **+2.67% F1 and +3.11% MCC** (significant).
- **Advantages.** The "degrade GT to synthesize a labeled quality-regression dataset"
  trick **removes the need to collect real bad segmentations** — the full quality
  spectrum is generated from a handful of clean masks. Directly demonstrated to improve
  a downstream segmenter's threshold choice — exactly the param-sweep use case.
- **Limitations.** Only as good as the **realism of the synthetic degradation model** —
  if real failure modes differ, the predictor mis-ranks (degradation shift); needs clean
  expert masks at build time; vessel topology, transfer to blobby colonies unproven.
- **Colony relevance.** **Strong methodological template** — PhenoTypic already has
  `load_synth_yeast_plate()`; degrade synthetic colony masks (erode/dilate/merge/split/
  drop) and train a CNN proxy to predict overlap, then drive the sweep — no real GT at
  inference. The degradation library must mimic genuine colony errors. (high / template)

## D.5 EvanySeg — ground-truth-free evaluation of any segmentation

- **Use cases.** Flag poorly segmented objects; benchmark models without GT;
  **select the best of N candidate masks** (the param-sweep objective).
- **Mathematical foundation.** Input = object crop with the segmentation blended into
  one channel: `x[:,:,0] = 0.5·x[:,:,0] + 0.5·ŷ`, cropped to the prompt's ROI → 3-channel
  244×244. A backbone `ψ` (ResNet101 / ViT) + regression head `τ` predicts
  `q = π(ŷ, y)` = the Dice (and IoU) against training-time GT, trained with **MSE loss**
  over 3 SAM variants (ViT > ResNet). Trained on 107,055 images / 206,596 masks /
  619,044 object-level pairs across 10 modalities; Pearson ~0.80–0.90 in-distribution.
- **Advantages.** Broad coverage; object-level granularity matches per-colony scoring;
  the "pick best of N" mode is directly the sweep objective.
- **Limitations.** (1) **Local ranking errors** — fine ordering of near-equal Dice can
  be wrong (problematic when a sweep chooses between *similar* good variants); (2)
  **prompt-dependence** (needs a bounding box; 2D only); (3) **domain shift** —
  Pearson dropped to ~0.51 on a non-SAM neuro-pathology set; (4) harder when quality
  variance is low. **Preprint, not peer-reviewed.**
- **Colony relevance.** Conceptually ideal (object-level, GT-free, "select best mask"),
  but the published weights are tuned to SAM-family medical outputs — for classical-
  pipeline colony masks one would **retrain EvanySeg-style on colony data** (its own
  results warn cross-segmenter transfer is weak). The bounding-box prompt maps naturally
  onto per-colony grid cells. (medium, after retraining)

## D.6 Test-Time Augmentation & MC Dropout — Wang et al. (uncertainty foundations)

- **Use cases.** Voxel/structure uncertainty maps; reducing overconfident wrong
  predictions; flagging unreliable structures.
- **Mathematical foundation.** Pass a test image through the network `N` times under
  (a) random spatial transforms + noise from an image-acquisition model prior (TTA)
  and/or (b) active dropout (MC dropout); final label = expectation over the `N`
  samples; **uncertainty = variance or predictive entropy** of the samples. First
  consistent mathematical formulation of TTA via an image-acquisition model.
- **Advantages.** TTA needs **no labels and no architecture change** (works on any
  probability-emitting segmenter); TTA-based aleatoric uncertainty beat MC-dropout-
  alone for flagging errors; useful with small training sets.
- **Limitations.** Requires **per-pixel class probabilities** (softmax) — which classical
  thresholding/watershed colony segmenters do not provide; `N` forward passes = N×
  cost; uncertainty correlates with but ≠ error; calibration is dataset-dependent and
  degrades under domain shift.
- **Colony relevance.** The most plausible label-free uncertainty route *iff* the
  segmenter is (made) probabilistic. For the current deterministic pipeline, TTA can
  still be applied as **prediction stability under input perturbations** (segment under
  small rotations/intensity jitter, measure mask agreement) — a heuristic quality proxy,
  not a calibrated Dice. (low for current pipeline; medium if probabilistic)

## D.7 Structure-wise uncertainty as a Dice proxy — Roy et al. (Bayesian QuickNAT)

- **Use cases.** Scan-level and per-structure QC; confidence-weighted downstream
  statistics; flagging out-of-distribution scans.
- **Mathematical foundation.** Keep dropout active at test time → `N` MC samples
  (N=15). Four structure-wise scalars: coefficient of variation `CV_s` of volume across
  samples, mean pairwise Dice `d^MC_s`, **IoU over all MC samples `IoU_s`** (the best
  Dice proxy), and an entropy-based measure. **Headline: mean IoU over MC samples is a
  suitable proxy for the Dice score.** Validated across 4 out-of-sample datasets.
- **Advantages.** Quality estimate is **free of quality-labeled training data** — only a
  dropout-enabled segmenter is needed; per-structure granularity; `IoU_s` simple to
  compute; generalizes across 4 unseen datasets.
- **Limitations.** Requires a **dropout/Bayesian segmenter** (architecture constraint);
  correlation with Dice, not exact prediction; calibration breaks under strong domain
  shift; N forward passes cost.
- **Colony relevance.** The `IoU/Dice-agreement over stochastic passes ≈ Dice` insight
  is reusable: a dropout-enabled learned colony segmenter would yield **per-colony
  agreement across MC samples as a label-free per-colony quality score**. Not applicable
  to the deterministic classical pipeline. (low for current pipeline)

## D.8 Two-stage uncertainty → quality regression — DeVries & Taylor

- **Use cases.** Refer uncertain cases to manual inspection or drop them;
  silent-failure detection.
- **Mathematical foundation.** Stage 1 produces a spatial uncertainty map; stage 2
  reasons over it to output an image-level quality prediction (+ a per-pixel "where it
  will fail" map). Pipeline-agnostic in principle. *scite returned no full-text excerpts,
  so exact aggregation/regression equations are not verifiable from a retrieved source —
  not reproduced (flagged).* Demonstrated on skin-lesion segmentation. **Preprint, not
  peer-reviewed.**
- **Advantages.** Decouples uncertainty estimation from quality prediction (modular);
  produces both a failure-localization map and a scalar; segmentation-pipeline-agnostic
  in principle.
- **Limitations.** Still relies on uncertainty estimates (probabilistic segmenter);
  stage-2 regressor needs quality-labeled training data; limited external validation
  (preprint); domain-shift behavior uncharacterized.
- **Colony relevance.** The "uncertainty-map → scalar quality" pattern is reusable only
  once a probabilistic colony segmenter exists — lower priority than the
  regressor-on-(image, mask) family. (low for current pipeline)

## D.9 Predictive entropy as a posterior quality estimate — da Cruz et al.

- **Use cases.** Cheap (single-forward-pass) quality screening of any softmax-producing
  segmenter; active-learning sample selection.
- **Mathematical foundation.** Per-pixel entropy `H = − Σ_{c∈C} p_c log p_c`, normalized
  by `log₂|C|` to `[0,1]`; summing pixel entropies gives global segmentation
  uncertainty. Several entropy-derived indices (SAR/SER, MEI/MSI families) are validated
  by AUC of an ROC separating "good" vs "bad" segmentations (thresholded on the *true*
  Jaccard/Dice/ASSD) + Pearson/Spearman correlation. Caveat the authors prove: for >2
  classes, max-softmax `σ_max` is qualitatively but not quantitatively faithful when
  `σ_max < 1/e` — entropy preferred.
- **Advantages.** Applicable to ordinary CNNs (no Bayesian net, no MC sampling) — only
  the softmax output; trivially cheap (one forward pass); model-internals-agnostic.
- **Limitations.** Still requires **per-pixel class probabilities**; entropy↔quality is
  correlational and calibration-/dataset-dependent; tested on a single skin task (limited
  validation, 0 citations at retrieval); over-confident-but-wrong networks defeat it.
- **Colony relevance.** If a softmax-producing colony segmenter is used, global/region
  entropy is the single cheapest no-GT proxy. Not usable for the current non-probabilistic
  pipeline. (low for current pipeline)

## D.10 Network-score metric for QA — Rodríguez Outeiral et al.

- **Use cases.** Triaging which auto-contours a clinician must review; QA of radiotherapy
  target segmentation; threshold/operating-point selection.
- **Mathematical foundation.** Take the network's continuous pre-binarization output
  (nnU-Net soft score map); define the metric as the **mean of the score-map voxels
  above a threshold λ** (mean confidence inside the predicted object). Correlated well
  with **distance-based** contour metrics on two MRI datasets — argued better than
  correlating mean entropy with Dice. (A published comment + response exist — normal
  scholarly exchange, not a concern notice.)
- **Advantages.** Extremely cheap (reuses the network's own output, no extra model, no MC
  sampling); the authors argue mean-confidence-above-λ tracks **boundary** accuracy
  better than entropy-vs-Dice; no GT at inference.
- **Limitations.** Requires a network emitting a continuous score/probability map (again
  not the classical pipeline); `λ` is a tuned hyperparameter; validated on two small
  datasets; reflects model confidence (miscalibration / domain-shift sensitivity, which
  the published comment debates).
- **Colony relevance.** Same gating condition — needs a probabilistic colony segmenter;
  if present, "mean confidence inside each predicted colony" is a near-zero-cost
  per-colony proxy worth including alongside entropy. (low for current pipeline)

## D.11 Background: radiomics-feature QA (Luan et al.) and the original regressor (Kohlberger et al.)

- **Luan et al.** — erode/dilate each predicted contour into inner/outer shells, extract
  **38 radiomics features** from the shells, train 12 ML models to predict DSC pass/fail
  per slice — no gold standard at inference. A **hand-feature analog of EvanySeg**, a
  cheap classical-feature `ReferenceFreeScorer` recipe directly relevant to colonies.
- **Kohlberger et al.** — the historical origin of "train a regressor on features
  extracted from the segmentation to predict Dice"; repeatedly cited as the baseline that
  **requires a fully-annotated training set** (the limitation RCA/uncertainty methods
  avoid). Context only (cite via the citing papers, not directly).

### Cross-cutting takeaways from family (D)

1. **The hard gating variable is "does the segmenter expose per-pixel probabilities?"**
   The (image, mask) regressor family works on the current classical pipeline; the
   uncertainty family does not (TTA only as a stability heuristic).
2. **None needs GT at inference; most need labeled data at build time.** RCA/Robinson-RCA
   uniquely need *no* good/bad training set (only a small GT reference DB). **Galdran's
   degrade-and-train is the standout for PhenoTypic** — it synthesizes the labeled
   quality-regression dataset from a few clean masks, and PhenoTypic already has
   `load_synth_yeast_plate()`.
3. **Domain shift is the universal failure mode.** Every learned predictor degrades off
   its training distribution (EvanySeg ~0.85 → ~0.51; uncertainty calibration breaks
   under shift; Galdran depends on degradations matching real failures). A colony scorer
   must be trained/validated on **colony segmentations and colony-specific error modes**
   (merged neighbors, over-splitting, edge halos, debris), not borrowed weights.
4. **Distance/boundary metrics predict worse than overlap metrics** (RCA predicts
   DSC/Jaccard well, Hausdorff/ASD poorly; Outeiral's contribution targets distance error
   specifically). If colony scoring cares about boundary/radius/area fidelity,
   overlap-only proxies may be insufficient.
5. **Cost/throughput ranking** (for scoring a large sweep): entropy/network-score (one
   pass) < radiomics-QA / learned CNN proxy (one pass + cheap features) < TTA/MC-dropout
   (N passes) < RCA (~660 s/case). The **single-pass learned regressor on (image,
   candidate mask)** is the pragmatic sweet spot; RCA is likely too slow as the inner-loop
   objective but a good *label generator* to bootstrap one.

---

# (E) Meta-evaluation, reliability & use in tuning

*Lane R5. The cross-cutting layer: how to **judge** whether a reference-free metric is
trustworthy, **evidence that metrics mislead**, and how unsupervised metrics have been
**used as a tuning objective** — directly the substance of the master spec's
meta-validation gate (D1).*

## E.1 Meta-evaluation — how to judge an unsupervised metric

The shared pattern: take segmentations whose *true* quality is known (synthetic or
human-annotated), compute the candidate metric on each, and test whether its
ordering/values agree with a supervised reference (correct-classification rate, Dice,
Jaccard, F-measure). Agreement = usable proxy; disagreement = not.

- **MSET — Zhang, Cholleti & Goldman (2006).** The canonical "meta-measure":
  treats *choosing among evaluation measures* as a supervised learning problem — given
  segmentations with known quality, learn/score which standalone unsupervised measure
  ranks them correctly. Directly operationalizes "is this metric any good?". Limitation:
  needs a labeled/known-quality set — it **relocates** the GT need to the validation
  phase, which is precisely the gate's intent.
- **Survey — Zhang, Fritts & Goldman (2008).** The definitive survey of unsupervised
  measures; cross-references MSET as the way to judge them; frames unsupervised
  evaluation as not-yet-solved ("no evaluation criterion appears satisfactory in all
  cases").
- **Chabrier, Emile & Rosenberger (2006) — comparison vs. a supervised reference.** A
  reproducible **template for the gate**: a database of **8,400 synthetic gray-level
  images**, each segmented four ways; **Vinet's measure (correct-classification rate)**
  is the supervised reference; each unsupervised criterion (Borsotti, Zeboudj, two
  Rosenberger, two Levine–Nazif) is judged by *behavioural similarity* to Vinet's
  measure — "the intrinsic quality of the segmentations... is not so important," what
  matters is whether the unsupervised *ranking* matches the GT ranking. Concludes no
  single criterion is universally satisfactory. **Directly transferable: synthetic yeast
  plates have known masks.** *(This is the same EURASIP 2006 paper as R1's
  Rosenberger/Chabrier comparative study — reconciled to one entry.)*
- **Jozdani & Chen (2020) — versatility of (supervised) metrics.** Meta-evaluates **21**
  supervised metrics for which reliably measure quality and which are sensitive to over-
  vs under-segmentation; recommends metrics that store/integrate segment-level
  information. Tells PhenoTypic which **reference** metric to use as the gate's GT side,
  not which reference-free metric to trust. (The master spec cites this for
  "F-measure / QR / SEI track the visual optimum best.")
- **Valindria et al. (2017) — RCA as a *validated* reference-free predictor.** Included
  here for its validation methodology: predict per-image accuracy with no GT, validate
  against real Dice on a small GT reference set — the same construct the gate needs.
  *(Same paper as D.1; reconciled to one entry.)*
- **Sims et al. (2023) — SEG, reference-free cell/nuclei evaluation (closest analogue).**
  Scores each method's segmentation *relative to a weighted ensemble* of many
  segmentations (weights from a model-ablation study to avoid collective bias). The
  gate-relevant step: they **first validate the unsupervised score on a small
  ground-truth-annotated dataset**, and only then apply it to a larger unlabeled set —
  the exact gate workflow. **Preprint, not peer-reviewed.**

## E.2 Reliability caveats — when no-reference metrics MISLEAD

- **Deo et al. (2025) — "Metrics that matter."** Stress-tests **16 no-reference
  image-quality metrics** on brain MRI by injecting noise, distribution shifts, and
  **localised morphological alterations**. Finding: many "correlate poorly with
  downstream task suitability and exhibit a profound insensitivity to localised
  anatomical details." Concrete failure: a GAN scored **better on FID/KID** than a VAE
  yet performed **substantially worse on downstream vessel segmentation (≈0.67 vs
  0.86)** — the metric **ranked the worse model higher**. **Implication for the gate:** a
  reference-free score can rank a *worse* configuration *higher*; the insensitivity to
  *localised* detail is acutely relevant to colonies, where the failures that matter are
  local (merged neighbours, lost faint colonies, ragged edges). Recommends a multifaceted
  validation framework; never select on a single metric. **Preprint, not peer-reviewed.**
- **Muthusivarajan / Rajarajeswari et al. (2024) — IQMs vs DL segmentation accuracy.**
  Tests whether 13 MR image-quality metrics relate to brain-tumor Dice. Finding: a
  relationship *can* exist but is **metric-specific and direction-dependent** — only
  *some* IQMs (inhomogeneity / CV measures, PSNR) improved accuracy when used to select
  training scans, and which ones is not obvious a priori. **Implication:** the empirical
  case **for** the correlation gate (and for testing each candidate metric individually),
  not against reference-free scoring wholesale.
- **Kazakevičiūtė-Januškevičienė et al. (2020) — objective vs subjective quality.**
  Correlates objective metrics with human ratings on remote-sensing images. Finding: the
  **global correlation hides regime dependence** — a metric can look adequate on average
  yet be **unreliable exactly where it matters** (poor segmentations). **Implication:** a
  single global correlation coefficient is an insufficient acceptance test; **stratify by
  quality regime** and confirm the proxy still ranks correctly among poor/borderline
  segmentations.

## E.3 Use in automated parameter / threshold tuning

- **Espíndola et al. (2006) — spatial-autocorrelation objective (a positive result).**
  Objective = Moran's I (inter) + intra-segment variance (homogeneity); the parameter
  set maximizing both is chosen. Finding: "segmentations with the highest
  objective-function values also resulted in the highest classification accuracies" — the
  unsupervised optimum **coincided** with the downstream-accuracy optimum. A best-case
  demonstration that an unsupervised objective *can* track true quality. *(Same paper as
  R2 §1's GS — reconciled to one entry.)*
- **Drăguţ et al. (2014) — ESP2, local-variance scale selection.** A self-tuning tool
  that auto-selects the multiresolution scale parameter with no GT: stop when average
  Local Variance stops increasing. Limitation: LV detects *a* scale transition, **not
  necessarily the task-optimal one**; later work found it inferior to an F-measure
  objective.
- **Georganos et al. (2018) — SPUSPO (local USPO).** Optimizes the segmentation objective
  (Global Score / F-measure from Weighted Variance + Moran's I) **per spatial partition**
  because optimal parameters vary across a scene. Local beat global: 90.5% vs 89.5%
  classification accuracy (significant), mean AFI 0.28 vs 0.36. **Implication:** arrayed
  plates are spatially heterogeneous (edge vs centre colonies, illumination gradients), so
  a globally-tuned parameter set may be suboptimal locally — the gate should check the
  proxy across plate regions, and PhenoTypic may need region-aware tuning.
- **Candidate-range sensitivity (Böck via Georganos 2018).** The chosen "best" parameter
  depends on **which candidate segmentations you feed the metric**; Georganos had to
  empirically bound the sweep to a range spanning evident over- and under-segmentation.
  **Implication:** the optimum a reference-free scorer returns is **conditional on the
  sweep grid** — pin and document a sensible candidate range bracketing both error
  directions, and check robustness to it. (Reinforces the Böck instability of R2 §3.)
- **Grybas et al. (2017) — comparison of tuning objectives.** Head-to-head of F-measure
  vs ESP/local-variance vs Global Score: **F-measure was superior**, attributed to its
  sensitivity to over- and under-segmentation. **Implication:** different unsupervised
  tuning objectives pick **different, non-equivalent optima**; the ranking among them is
  empirical — the strongest single argument that the `ReferenceFreeScorer` choice must be
  validated **per-domain**, not inherited (what's best for buildings may not transfer to
  colonies).
- **Metaheuristic search over an unsupervised objective (da Costa et al. 2007).** A GA
  whose fitness expresses similarity to quality criteria searches parameter space; the
  broader USPO literature plugs Moran's I / variance / LV / F-measure objectives into grid
  or metaheuristic search. **Decouples search strategy (cheap to swap) from objective
  trustworthiness (the thing the gate certifies).** Caveat: **search amplifies a bad
  objective** — if the proxy is wrong, optimization finds the *proxy's* optimum, not the
  true one (the Deo risk). **Hence: gate the objective before letting search exploit it.**

## E.4 Implications for PhenoTypic's meta-validation gate

**What the cited evidence establishes:**

1. **The gate is the field-standard method, not a bespoke precaution** — correlate the
   metric's ranking against a supervised GT reference on known-quality data (Chabrier
   2006 vs Vinet; MSET 2006; the 2008 survey).
2. **No-reference metrics demonstrably mis-rank** — a worse model can score better on a
   no-reference metric while losing downstream (Deo 2025: FID/KID better, vessel
   segmentation ≈0.67 vs 0.86), and are "profoundly insensitive to localised details" —
   the colony-relevant failure mode.
3. **Predictiveness is metric-specific, not generic** — only *some* metrics correlate with
   segmentation accuracy, and which is not obvious a priori (Muthusivarajan 2024).
4. **Different unsupervised tuning objectives pick different optima** — F-measure beat
   ESP/LV and Global Score (Grybas 2017); LV finds a scale transition that isn't
   necessarily task-optimal (Drăguţ 2014). Validate, don't inherit.
5. **A single global correlation number is insufficient** — correlation can be high on
   good segmentations and low on bad ones (Kazakevičiūtė-Januškevičienė 2020) and vary
   spatially (Georganos 2018).
6. **The optimum is conditional on the candidate grid** (Böck via Georganos 2018).
7. **It *can* work — and the success criterion is convergence with downstream accuracy**
   (Espíndola 2006); validate-on-a-small-labelled-subset-then-apply-to-unlabeled-data is
   the published workflow (Sims 2023, preprint; Valindria 2017).

**Concrete gate design** (the R5 author's inference, synthesizing the above — not stated
verbatim by any single paper; PhenoTypic's `load_synth_yeast_plate()` makes these
templates directly executable):
- **Reference metric:** Dice/Jaccard/F-measure on synthetic-plate GT (the role Vinet's
  measure plays in Chabrier 2006).
- **Acceptance statistic:** require **rank agreement** (Spearman ρ, robust to monotone
  nonlinearity) **AND** a **cost-argmin agreement test** (the parameters minimizing
  `ReferenceFreeScorer` cost must land within a small tolerance of those minimizing
  supervised reference cost, equivalently maximizing raw Dice). Suggested
  engineering bars (inference, not a cited cutoff): **Spearman ρ ≥ ~0.7 to pass**,
  **≥ ~0.8 to allow unattended auto-tuning**, demote to advisory below ~0.5. *(The master
  spec's §4 floor of "≥3–5 annotated plates" and "warns/abstains if the correlation is
  weak" is the same gate; this document supplies the statistic and thresholds as a
  starting point.)*
- **Stratify the validation** (evidence-backed): compute the statistic separately on
  *poor/borderline* segmentations and across *plate regions* (edge vs centre).
- **Pin the candidate grid** (evidence-backed): fix and document a sweep range bracketing
  clear over- and under-segmentation and report robustness to it.
- **GT-set size:** no cited paper prescribes a minimum N. *Inference:* validate on a
  **large synthetic set** (hundreds of plates spanning density/contrast/edge sharpness —
  free) **and cross-check on a small real annotated set (~10–30 plates)** to catch the
  synthetic-to-real gap.
- **Per-domain re-validation** (evidence-backed): re-run the gate whenever colony type,
  medium, or imaging setup changes — don't assume a metric validated on yeast transfers
  to bacteria.
- **Fail-safe** (evidence-backed by Deo 2025): if the gate fails, do **not** let
  optimization exploit the proxy (search amplifies a wrong objective); fall back to
  GT-driven or human-in-the-loop tuning and keep the no-reference score advisory only.

---

# Summary comparison table

| Family / metric | What it rewards | GT needed? | Training needed? | Compute cost | Key failure mode | Colony fit |
|---|---|---|---|---|---|---|
| **(A) Levine–Nazif U/contrast** | Uniform interiors + region contrast | No | No | Low (+adjacency) | Heuristic weights; texture-sensitive | Med |
| **(A) Weszka–Rosenfeld busyness** | Smooth, compact foreground | No | No | Very low | Threshold-only; fails on texture; global | High |
| **(A) Otsu σ²_B / η** | Two-class mean separation | No | No | Very low | Histogram-only (blind to layout); 2-class | High (one term) |
| **(A) Liu–Yang F** | Low intra-error, light count penalty | No | No | Very low | Biased to over-segmentation | Low |
| **(A) Borsotti F′ / Q** | Low intra-error; both over- & under-seg | No | No | Low | Brittle same-area terms; no contrast/shape | High |
| **(A) Zeboudj C(I)** | Low internal / high external contrast | No | No | Low (+boundary) | Mis-reads texture; noise-sensitive max | High |
| **(A) Entropy E** | Region uniformity + size-dist. balance | No | No | Very low | Layout term ignores spatial separation | Med |
| **(A) Rosenberger FRC** | Adaptive uniform/textured intra−inter | No | SVM/GA variants only | Med | Complex; texture mode has choices | Med (method) |
| **(B) Global Score (GS)** | Low intra-var + low inter-autocorr | No | No | Med (RAG) | Min–max range-dependence (argmin shifts) | High |
| **(B) Gao q + Moran (Mahalanobis)** | Variance explained + distinct neighbors | No | No | Med–high | Σ still from tested set; feature cost | High |
| **(B) UOA (local)** | Per-segment over/under-seg diagnosis | No | No (needs δ) | High | δ tuning; H–size confound | High |
| **(B) FGS (WV + DTNP)** | Homogeneity + bounding-box-ring contrast | No | No | Low (no RAG) | Ring leaks in dense packing; range caveat | High |
| **(B) Yu SAR (GHO+GHE+EVI)** | Homogeneity + heterogeneity + edge-validity | No | No | High | Heavy pipeline; SAR-specific | Med (EVI idea) |
| **(B) q-statistic** | Fraction of variance explained by partition | No | No | Low | No inter/edge term alone | High |
| **(C) Chen & Murphy suite (PCA)** | Coverage + marker uniformity, fused | No (validated vs GT) | PCA fit (dataset) | Med | Marker-dependent uniformity; PCA dataset-specific | High (coverage+recipe) |
| **(C) Shape descriptors** | Circular/convex/compact objects | No | No | Very low | Noisy at small size; collinear; assumes round | High |
| **(C) Size-CV / replicate homogeneity** | Tight within-replicate size/feature dist. | No | No | Very low | Must group; real biology varies | High |
| **(C) Average silhouette** | Coherent phenotype clusters | No | No (clustering) | Med (O(n²)) | Needs feature space + ≥2 clusters | Med |
| **(C) OpenCFU particle filter + score-map** | Valid circular blobs, threshold-stable | No | No | Low–med | Counting-tuned; dense/touching colonies | High |
| **(C) gitter/SGA grid + count QC** | Expected-vs-detected count, grid regularity | No | No | Very low | Heuristic thresholds; needs known grid | High |
| **(D) RCA** | Predicted Dice via reverse classifier | Reference DB w/ GT | No good/bad set | Very high (~660 s) | Registration; distance metrics poorly | Low (loop) / Med (labels) |
| **(D) Robinson CNN proxy** | Regressed Dice from (image, masks) | Yes (or RCA labels) | Yes | Low (~40 ms) | Big labeled set; fixed-channel; domain shift | Med–high |
| **(D) Galdran degrade-and-train** | Predicted overlap of a mask | Clean masks (degraded) | Yes | Low | Degradation realism; topology transfer | High (template) |
| **(D) EvanySeg** | Predicted Dice/IoU per object | Yes | Yes | Low | SAM-distribution; local ranking errors | Med (retrain) |
| **(D) TTA / MC dropout** | Prediction stability ≈ Dice | No | No (needs prob. seg) | High (N passes) | Needs softmax/dropout segmenter | Low (current pipeline) |
| **(D) Entropy / network-score** | Low prediction uncertainty/confidence | No | No (needs prob. seg) | Very low (1 pass) | Needs softmax; calibration/domain shift | Low (current pipeline) |
| **(D) Luan radiomics-QA** | Predicted DSC pass/fail from shell features | Yes | Yes | Low | Labeled set; domain shift | Med |
| **(E) meta-eval / gate (cross-cutting)** | Whether the chosen metric tracks GT | Small GT set (gate) | Optional | Low | A single global ρ can mislead | — (the validator) |

*"Compute cost" is per-evaluation in a sweep. "GT needed?" is at **build/validation**
time (none of D needs GT at inference). Colony fit per the lane authors' judgments.*

---

# Recommendations for PhenoTypic's `ReferenceFreeScorer`

Synthesizing R1–R5 against master-spec **§4** (the `Scorer` Protocol and
`ReferenceFreeScorer`), **D1** (pluggable objective, `ReferenceFreeScorer` gated), **D8**
(fANOVA importance), and the **meta-validation gate**:

**1. Build it as a small weighted/PCA ensemble, not a single metric.** The strongest
empirical guidance across lanes is unanimous: no single classic metric dominates, and
learned/weighted combinations track human/GT judgment best (Chabrier/Rosenberger 2006;
Chen & Murphy 2023). The colony-array setting is *unusually friendly* because three strong
GT-free priors are available (count/grid, shape, replicate homogeneity). Compose along the
multi-axis layout Chen & Murphy and BIDCell both endorse:

- **Count / grid axis (highest-value, cheapest, most colony-specific):** expected-vs-detected
  colony count per plate and per row/column (Galardini/Viéitez **> 5%** and **> 90%** rules),
  grid-regularity from gitter's Radon-peak structure or fitted-lattice residuals,
  known-empty-position checks (R3 §6). PhenoTypic already knows `nrows`/`ncols` — these are
  almost-supervised signals the RS/medical lanes cannot use. This axis also overlaps the
  master spec's `QCScorer` (D1's primary default) — the `ReferenceFreeScorer` should *reuse*
  those checks, not duplicate them.
- **Shape axis:** per-colony **solidity** (merged-neighbor detector), **circularity**,
  **eccentricity/aspect ratio**, aggregated to plate level (median, fraction passing);
  OpenCFU's particle-filter vector (area, perimeter, convexity, aspect ratio, hollowness) +
  its **threshold-stability score-map** as a per-colony confidence (R3 §2, §5). Gate by
  minimum colony size and don't double-weight collinear solidity/circularity.
- **Homogeneity / contrast axis:** a spatially-aware contrast term — **Zeboudj `C(I)`** (R1)
  or the q-statistic + bounding-box-ring **DTNP** (R2 FGS, cheap, no RAG) — plus
  **within-replicate size/feature CV** folded via the `1/(x+1)` transform, and the
  **median(border)/median(interior)** edge ratio (R3 §3). This is the "intra-colony
  homogeneity vs. background contrast / boundary gradient" the master spec §4 names.
- **Optional edge-validity axis:** Yu's **EVI** idea (does the colony boundary sit on a real
  intensity edge?) for crisp rims — under-used in optical UE and cheap to add (R2 §8).

The cited z-score/PCA and Mahalanobis-to-ideal-point recipes motivate using
complementary, externally normalized axes. The implemented scorer declares each
natural term's sense and fixed normalizer, then converts it to bounded cost. The
canonical `CompositeScorer` scalarizes child costs with augmented Tchebycheff by
default (`blend="weighted_mean"` is the compensatory option), or preserves the axes
for true multi-objective optimization. Mind the raw-metric sign conventions in R1
cross-cutting note 6 before cost orientation.

**2. Avoid the GEOBIA normalization trap (the single most important transfer caveat).**
Do **not** use min–max-over-the-tested-set normalization (Espíndola/Johnson Global Score) —
Böck 2017 shows the argmin then depends on the parameter grid you happen to sweep, which is
*fatal for a tuning engine that is a grid sweep*. Use **fixed/external normalization**
(variance relative to whole-image variance; Moran's I / Geary's C in `[−1,1]`; q-statistic in
`[0,1]`) or rank/covariance-based combination. This also informs **D8**: when fANOVA ranks
parameter importance over the optimizer's trials, the scores it consumes must be
grid-independent or the importances inherit the same instability.

**3. Prefer the (image, mask)-regressor family over the uncertainty family for the current
classical pipeline.** Per R4's hard gating variable, the uncertainty methods (TTA, MC
dropout, entropy, network-score) all need a probabilistic segmenter the deterministic
threshold/watershed colony pipeline doesn't expose. If a learned regressor is ever added,
**Galdran's degrade-and-train is the standout**: PhenoTypic's `load_synth_yeast_plate()` can
generate clean masks, degrade them with colony-specific corruptions (erode/dilate/merge/split/
drop, edge halos, debris), and train a CNN proxy to predict overlap — no real GT at inference.
RCA is too slow as an inner-loop objective but is a viable **label generator** to bootstrap
such a regressor. Whatever is learned must be trained/validated on **colony** error modes (R4
cross-cutting note 3).

**4. Gate before trusting it to drive optimization (D1 / master §4 meta-validation).** This
is non-negotiable and the substance of D1's gating clause. Implement the R5 gate:
- Reference = Dice/Jaccard/F-measure on synthetic-plate GT (Chabrier 2006's Vinet role).
- Acceptance = **Spearman rank agreement AND a cost-argmin agreement test** (the
  picked parameters land near the supervised-cost minimum, equivalently the raw-Dice
  maximum). Suggested bars: **ρ ≥ ~0.7 pass**, **≥ ~0.8 for
  unattended tuning**, advisory below ~0.5 (engineering inference; the literature validates
  the *method*, not a numeric cutoff — consistent with master §4's "warns/abstains if the
  correlation is weak" and §14's open question on the threshold).
- **Stratify** by quality regime (poor/borderline) and plate region (Kazakevičiūtė 2020;
  Georganos 2018) — a single global ρ is insufficient.
- **Pin the candidate grid** bracketing over- and under-segmentation and report robustness
  (Böck via Georganos 2018) — the optimum is conditional on the grid (master §9's
  grid-over-conditional regression test should fix this range).
- **Validate on a large synthetic set + cross-check ~10–30 real annotated plates**; **re-run
  the gate per-domain** (yeast ≠ bacteria; Grybas 2017, Muthusivarajan 2024).
- **Fail-safe:** if the gate fails, do not let the optimizer exploit the proxy — search
  amplifies a wrong objective (Deo 2025) — fall back to `SupervisedScorer` / `QCScorer` /
  human-in-the-loop.

**5. Practical default for Phase 3 (`infer_search_space` + `ReferenceFreeScorer`, master §12
phasing).** Start with the cheapest, most colony-specific, fully reference-free terms — **grid/
count QC + shape descriptors + a Zeboudj-or-DTNP contrast term + within-replicate size-CV** —
fused by PCA/Mahalanobis, normalized with fixed bounds, and **gated** as above. Defer the
learned-regressor and uncertainty families to a later phase that introduces a probabilistic or
trainable colony segmenter; they are higher build-time cost and (for the uncertainty family)
not even applicable to today's pipeline.

---

# Verification status & caveats

Collected provenance flags, preserved from the five lane reports — read before reimplementing
any formula:

**Formulas mechanism-verified but NOT symbol-verified (do not treat as authoritative
closed forms):**
- **R2:** Corcoran & Winstanley **SU** (exact equation not openly retrievable — treat as a
  design idea, not a copy-ready formula); **UOA** aggregation equations (`UOA_Σ`, `UOA_L2`)
  partially retrieved (the H/δ classification logic *is* verified); **FGS** DTNP closed form
  (Eqs. 4–5) not transcribed (bounding-box-ring mechanism verified); **Yu SAR** sub-equations
  (Eqs. 10, 14, 18, 19) not transcribed (composition verified). Espíndola/Johnson GS and
  Corcoran are paywalled — formulas corroborated from open secondary sources (Böck 2017, Zhao
  2020, Gao 2017), not read verbatim.
- **R4:** **Galdran** per-symbol formulas not reproduced (source PDF non-extractable; reported
  at abstract/author-summary level only); **DeVries & Taylor** exact aggregation/regression
  equations not reproduced (no full-text excerpts retrieved). Both flagged in their sections.
- **R3:** several **Chen & Murphy** exact closed forms — the precise foreground definition,
  the exact ACVC cluster weighting, and the exact retained-PC count/weights of the composite —
  were summarized, not quoted verbatim; verify against Table 2 / Methods of PMC10208095 before
  reproducing the score exactly. The `1/(x+1)` transform, the "average over 1–10 clusters," and
  the z-score→PCA shape *are* verified.

**Provenance / access flags:**
- **R1 Zeboudj contrast** originates in **R. Zeboudj's 1988 PhD thesis (Univ. Saint-Étienne),
  which has no DOI**; its formulas here come from the Pandore reference implementation
  (`pzeboudj`) and the CVIU survey Appendix A (which agree), operationalized via the DOI'd
  Chabrier 2004 paper. Treat the thesis as an untraceable primary source.
- The R1 anchor survey (Zhang/Fritts/Goldman 2008) and the entropy primary (SPIE 2003/04) are
  closed-access; formulas read from the authors' public preprints (cs.slu.edu) and cross-checked
  against OA secondary sources.
- **R3 access-not-confirmed-OA:** CHiTA (`10.1088/0031-9155/53/21/007`) and Rousseeuw 1987
  (`10.1016/0377-0427(87)90125-7`). The canonical **ImageJ / Abramoff 2004** shape-descriptor
  source was not retrieved as a standalone DOI — cite the papers that use it (Zingaretti 2021,
  Rosero 2019, Gwoździk 2024), do **not** fabricate a DOI.

**Preprints (NOT peer-reviewed) — flagged in the references below:**
- **R3:** BIDCell (bioRxiv `10.1101/2023.06.13.544733`). CellSeg3D is **now peer-reviewed in
  *eLife*** (`10.7554/elife.99848`) — prefer the eLife version over its bioRxiv preprint.
- **R4:** EvanySeg (arXiv `2409.14874`) and DeVries & Taylor (arXiv `1807.00502`).
- **R5:** Sims et al. (bioRxiv `10.1101/2023.02.23.529809`) and Deo et al. (arXiv `2505.07175`).

**Chen & Murphy year reconciliation (master-spec discrepancy):** R3 corrects the team brief's
"2021": **Chen & Murphy is 2023, *Molecular Biology of the Cell* 34(6):ar50,
`10.1091/mbc.e22-08-0364`** (peer-reviewed); **2021 is the bioRxiv preprint
`10.1101/2021.09.17.460800`**. The master spec currently cites "Chen et al. (2021)" with the
same `10.1091/mbc.E22-08-0364` DOI — i.e. it pairs the **preprint year** with the **journal
DOI**. **Recommendation: correct the master spec §2 and §13 reference to 2023.** This
companion uses the 2023 journal version throughout.

**No retractions or editorial concerns** were found on any cited peer-reviewed paper across all
five lanes (checked via scite metadata / `has_retraction:false`). Two items were deliberately
excluded as quality flags: an IEEE "Notice of Removal" ICIP 2016 item (R2) and any unverifiable
formula upgrades. The Outeiral 2023 paper (R4) carries a published comment + author response — a
normal scholarly exchange over generalizability, **not** a concern notice.

**Independent citation audit (2026-06-01).** All 68 reference DOIs were independently
verified to resolve to real records via scite; **no retractions or editorial concerns**
surfaced. Four bibliographic entries were corrected after the audit — **#7** (now
Nazif & Levine 1984, "Low level image segmentation: an expert system," *IEEE TPAMI*
6(5):555–577), **#61** and **#64** (titles), and **#62** (author initial "Jia, Y.") —
all on otherwise-valid DOIs. Six of seven spot-checked load-bearing figures were
confirmed verbatim; the lone exception is **RCA's "~660 s/case"** (D.1 and the summary
table), which is consistent with the literature (≈10,000× the ~40 ms learned proxy) but
was **not confirmed verbatim** in retrieved Valindria 2017 excerpts — treat as approximate.

---

# References (deduplicated across all five lane reports)

*Reconciled to one canonical entry where lanes overlapped: Zhang–Fritts–Goldman 2008 (R1/R2/R5);
Rosenberger/Chabrier–Emile–Rosenberger EURASIP 2006 (R1 FRC comparative study = R5 §1.3
Chabrier 2006); Espíndola 2006 (R2 §1 = R5 §3.1); Valindria/RCA 2017 (R4 §1 = R5 §1.5);
q-statistic Wang 2010 (R2 §5 = §9c); Galardini 2019 (R3 §3 = §6). Preprints marked **[PREPRINT —
not peer-reviewed]**.*

**Foundational / survey / meta-evaluation (A, E)**
1. Zhang, H., Fritts, J. E. & Goldman, S. A. (2008). Image segmentation evaluation: a survey of unsupervised methods. *Computer Vision and Image Understanding* 110(2), 260–280. https://doi.org/10.1016/j.cviu.2007.08.003
2. Zhang, Y. J. (1996). A survey on evaluation methods for image segmentation. *Pattern Recognition* 29(8), 1335–1346. https://doi.org/10.1016/0031-3203(95)00169-7
3. Zhang, H., Cholleti, S. & Goldman, S. A. (2006). Meta-evaluation of image segmentation using machine learning (MSET). *IEEE CVPR*. https://doi.org/10.1109/cvpr.2006.185
4. Zhang, H., Fritts, J. E. & Goldman, S. A. (2003/2004). An entropy-based objective evaluation method for image segmentation. *Proc. SPIE 5307*, 38–49. https://doi.org/10.1117/12.527167
5. Wang, Z., Wang, E. & Zhu, Y. (2020). Image segmentation evaluation: a survey of methods. *Artificial Intelligence Review* 53(8), 5637–5674. https://doi.org/10.1007/s10462-020-09830-9

**Classic region-statistics metrics (A)**
6. Levine, M. D. & Nazif, A. M. (1985). Dynamic measurement of computer generated image segmentations. *IEEE TPAMI* 7(2), 155–164. https://doi.org/10.1109/tpami.1985.4767640
7. Nazif, A. M. & Levine, M. D. (1984). Low level image segmentation: an expert system (intra/inter-class region-analysis formulas). *IEEE TPAMI* 6(5), 555–577. https://doi.org/10.1109/tpami.1984.4767570
8. Weszka, J. S. & Rosenfeld, A. (1978). Threshold evaluation techniques. *IEEE Trans. Systems, Man & Cybernetics* 8(8), 622–629. https://doi.org/10.1109/tsmc.1978.4310038
9. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Trans. SMC* 9(1), 62–66. https://doi.org/10.1109/tsmc.1979.4310076
10. Liu, J. & Yang, Y.-H. (1994). Multiresolution color image segmentation. *IEEE TPAMI* 16(7), 689–700. https://doi.org/10.1109/34.297949
11. Borsotti, M., Campadelli, P. & Schettini, R. (1998). Quantitative evaluation of color image segmentation results (F′, Q). *Pattern Recognition Letters* 19(8), 741–747. https://doi.org/10.1016/s0167-8655(98)00052-x
12. Chabrier, S., Emile, B., Laurent, H., Rosenberger, C. & Marché, P. (2004). Unsupervised evaluation of image segmentation — application to multispectral images (operationalizes Zeboudj contrast). *Proc. ICPR 2004* 1, 576–579. https://doi.org/10.1109/icpr.2004.1334206 — *(original Zeboudj contrast: R. Zeboudj, PhD thesis, Univ. Saint-Étienne, 1988 — no DOI; provenance-flagged)*
13. Rosenberger, C., Chabrier, S. & Emile, B. (2006). Unsupervised performance evaluation of image segmentation / comparison of unsupervised criteria against Vinet's measure (FRC + 8,400-image comparative study). *EURASIP Journal on Advances in Signal Processing* 2006, 96306. https://doi.org/10.1155/asp/2006/96306 — *(R1's FRC comparative study and R5 §1.3 "Chabrier, Emile & Rosenberger 2006" are the same paper)*
14. Chabrier, S., Rosenberger, C. & Laurent, H. (2005). Segmentation evaluation using a support vector machine. *Springer LNCS (ICAPR)*. https://doi.org/10.1007/11551188_46
15. Rosenberger, C., Chabrier, S. & Emile, B. (2008). Optimization-based image segmentation by genetic algorithms. *EURASIP Journal on Image & Video Processing* 2008, 842029. https://doi.org/10.1155/2008/842029
16. Rosenberger, C. & Chehdi, K. (2000). Genetic fusion: application to multicomponent image segmentation (original FRC). *Proc. ICASSP 2000*. https://doi.org/10.1109/icassp.2000.859280

**Object / GEOBIA + spatial statistics (B)**
17. Espíndola, G. M., Câmara, G. & Reis, I. A. (2006). Parameter selection for region-growing segmentation via a spatial-autocorrelation objective (Global Score). *International Journal of Remote Sensing* 27(14), 3035–3040. https://doi.org/10.1080/01431160600617194 — *(R2 §1 and R5 §3.1 are the same paper)*
18. Johnson, B. & Xie, Z. (2011). Unsupervised image segmentation evaluation and refinement using a multi-scale approach (Global Score implementation). *ISPRS Journal of Photogrammetry and Remote Sensing* 66(4), 473–483. https://doi.org/10.1016/j.isprsjprs.2011.02.006
19. Böck, S., Immitzer, M. & Atzberger, C. (2017). On the objectivity of the objective function — problems with unsupervised segmentation evaluation based on the Global Score. *Remote Sensing* 9(8), 769. https://doi.org/10.3390/rs9080769
20. Corcoran, P., Winstanley, A. & Mooney, P. (2010). Segmentation performance evaluation for object-based remotely sensed image analysis (SU metric). *International Journal of Remote Sensing* 31(3), 617–645. https://doi.org/10.1080/01431160902894475
21. Gao, H., Tang, Y., Jing, L., Li, H. & Ding, H. (2017). A novel unsupervised segmentation quality evaluation method for remote sensing images. *Sensors* 17(10), 2427. https://doi.org/10.3390/s17102427
22. Wang, J.-F., Li, X.-H., Christakos, G., Liao, Y.-L., Zhang, T., Gu, X. & Zheng, X.-Y. (2010). Geographical detectors-based health risk assessment (origin of the q-statistic). *International Journal of Geographical Information Science* 24(1), 107–127. https://doi.org/10.1080/13658810802443457
23. Troya-Galvis, A., Gançarski, P. & Passat, N. (2015). Unsupervised quantification of under- and over-segmentation for object-based remote sensing image analysis (UOA). *IEEE JSTARS* 8(5), 1936–1945. https://doi.org/10.1109/jstars.2015.2424457
24. Zhao, W., Meng, J., Zhang, X., Hu, Y., Sun, Z. & Yang, J. (2020). A fast and effective method for unsupervised segmentation evaluation of remote sensing images (FGS). *Remote Sensing* 12(18), 3005. https://doi.org/10.3390/rs12183005
25. Yu, H., Yin, Q., Liu, X., Luo, Y., Hou, B. & Wang, S. (2023). A novel unsupervised evaluation metric based on heterogeneity features for SAR image segmentation. *IEEE JSTARS* 16, 2851–2867. https://doi.org/10.1109/jstars.2023.3257548
26. Yu, H., Yin, Q., Liu, X., et al. (2022). A novel unsupervised evaluation metric for SAR image segmentation results (ICGMRS precursor). https://doi.org/10.1109/icgmrs55602.2022.9849399
27. Moran, P. A. P. (1950). Notes on continuous stochastic phenomena (Moran's I). *Biometrika* 37(1–2), 17–23. https://doi.org/10.1093/biomet/37.1-2.17
28. Geary, R. C. (1954). The contiguity ratio and statistical mapping (Geary's C). *The Incorporated Statistician* 5(3), 115–145. https://doi.org/10.2307/2986645

**Bioimaging / cell / colony (C)**
29. Chen, H. & Murphy, R. F. (2023). Evaluation of cell segmentation methods without reference segmentations. *Molecular Biology of the Cell* 34(6), ar50. https://doi.org/10.1091/mbc.e22-08-0364 — *(peer-reviewed journal version of the 2021 bioRxiv preprint `10.1101/2021.09.17.460800`; the master spec's "2021" should be corrected to 2023)*
30. Zingaretti, L. M., Monfort, A. & Pérez-Enciso, M. (2021). Automatic fruit morphology phenome and genetic analysis (shape-descriptor definitions). *Plant Phenomics* 2021, 9812910. https://doi.org/10.34133/2021/9812910
31. Rosero, A., Granda, L., Pérez, J. L., et al. (2019). Morphometric and colourimetric tools… sweet potato. *Genetic Resources and Crop Evolution* 66(6), 1257–1278. https://doi.org/10.1007/s10722-019-00781-x
32. Gwoździk, M., et al. (2024). Analysis on the morphology and interface of the phosphate coating (circularity/solidity cross-check). *Materials* 17(12), 2805. https://doi.org/10.3390/ma17122805
33. Galardini, M., Busby, B. P., Viéitez, C., et al. (2019). The impact of the genetic background on gene deletion phenotypes in *S. cerevisiae* (border correction + plate-QC heuristics). *Molecular Systems Biology* 15(12), e8831. https://doi.org/10.15252/msb.20198831 — *(R3 §3 and §6 are the same paper)*
34. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics* 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7 — *(access not confirmed OA)*
35. Geissmann, Q. (2013). OpenCFU, a new free and open-source software to count cell colonies and other circular objects. *PLOS ONE* 8(2), e54072. https://doi.org/10.1371/journal.pone.0054072
36. Bewes, J. M., Suchowerska, N. & McKenzie, D. R. (2008). Automated cell colony counting and analysis using the circular Hough image transform algorithm (CHiTA). *Physics in Medicine and Biology* 53(21), 5991–6008. https://doi.org/10.1088/0031-9155/53/21/007 — *(access not confirmed OA)*
37. Bär, J., Boumasmoud, M., Kouyos, R. D., et al. (2020). Efficient microbial colony growth dynamics quantification with ColTapp. *Scientific Reports* 10, 16084. https://doi.org/10.1038/s41598-020-72979-4
38. Zhang, L. (2022). Machine learning for enumeration of cell colony forming units (CFUCounter). *Visual Computing for Industry, Biomedicine, and Art* 5, 26. https://doi.org/10.1186/s42492-022-00122-3
39. Wagih, O. & Parts, L. (2014). gitter: a robust and accurate method for quantification of colony sizes from plate images. *G3: Genes|Genomes|Genetics* 4(3), 547–552. https://doi.org/10.1534/g3.113.009431
40. Wagih, O., Usaj, M., Baryshnikova, A., et al. (2013). SGAtools: one-stop analysis and visualization of array-based genetic interaction screens. *Nucleic Acids Research* 41(W1), W591–W596. https://doi.org/10.1093/nar/gkt400
41. Viéitez, C., et al. (2021). High-throughput functional characterization of protein phosphorylation sites in yeast (reuses gitter plate-QC rules). *Nature Biotechnology* 40, 382–390. https://doi.org/10.1038/s41587-021-01051-x
42. Bischof, L., Převorovský, M., Rallis, C., et al. (2016). Spotsizer: high-throughput quantitative analysis of microbial growth. *BioTechniques* 61(4), 191–201. https://doi.org/10.2144/000114459
43. Takeuchi, R., et al. (2014). Colony-live — a high-throughput method for measuring microbial colony growth kinetics. *BMC Microbiology* 14, 171. https://doi.org/10.1186/1471-2180-14-171
44. Achard, C., Kousi, T., Frey, M., et al. (2025). CellSeg3D: self-supervised 3D cell segmentation for fluorescence microscopy. *eLife* 13, RP99848. https://doi.org/10.7554/elife.99848 — *(preferred over its preprint `10.1101/2024.05.17.594691`)*
45. Fu, X., Lin, Y., Lin, D., et al. (2023). Biologically-informed self-supervised learning for segmentation of subcellular spatial transcriptomics data (BIDCell). *bioRxiv*. https://doi.org/10.1101/2023.06.13.544733 — **[PREPRINT — not peer-reviewed]**

**Learning- & uncertainty-based predictors (D)**
46. Valindria, V. V., Lavdas, I., Bai, W., et al. (2017). Reverse classification accuracy: predicting segmentation performance in the absence of ground truth. *IEEE Transactions on Medical Imaging* 36(8), 1597–1606. https://doi.org/10.1109/tmi.2017.2665165 — *(R4 §1 and R5 §1.5 are the same paper)*
47. Robinson, R., et al. (2019). Automated quality control in image segmentation: application to the UK Biobank cardiovascular MRI study. *Journal of Cardiovascular Magnetic Resonance* 21(1), 18. https://doi.org/10.1186/s12968-019-0523-x
48. Robinson, R., Valindria, V. V., Bai, W., et al. (2017). Automatic quality control of cardiac MRI segmentation in large-scale population imaging. *MICCAI 2017*, LNCS, 720–727. https://doi.org/10.1007/978-3-319-66182-7_82
49. Robinson, R., Oktay, O., Bai, W., et al. (2018). Real-time prediction of segmentation quality. *MICCAI 2018*, LNCS, 578–585. https://doi.org/10.1007/978-3-030-00937-3_66
50. Galdran, A., Costa, P., Bria, A., Araújo, T., Mendonça, A. M. & Campilho, A. (2018). A no-reference quality metric for retinal vessel tree segmentation. *MICCAI 2018*, LNCS 11070, 82–90. https://doi.org/10.1007/978-3-030-00928-1_10
51. Senbi, A., Huang, T., et al. (2024). Towards ground-truth-free evaluation of any segmentation in medical images (EvanySeg). *arXiv:2409.14874*. https://doi.org/10.48550/arXiv.2409.14874 — **[PREPRINT — not peer-reviewed]**
52. Wang, G., Li, W., Aertsen, M., et al. (2019). Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with CNNs. *Neurocomputing* 338, 34–45. https://doi.org/10.1016/j.neucom.2019.01.103
53. Roy, A. G., Conjeti, S., Navab, N. & Wachinger, C. (2019). Bayesian QuickNAT: model uncertainty in deep whole-brain segmentation for structure-wise quality control. *NeuroImage* 195, 11–22. https://doi.org/10.1016/j.neuroimage.2019.03.042
54. DeVries, T. & Taylor, G. W. (2018). Leveraging uncertainty estimates for predicting segmentation quality. *arXiv:1807.00502*. https://doi.org/10.48550/arXiv.1807.00502 — **[PREPRINT — not peer-reviewed]**
55. da Cruz, J.-M. M., Sangalli, M. & Decencière, É. (2024). A posteriori deep learning segmentation quality estimation based on prediction entropy. *Image Analysis & Stereology* 43(2), 121–130. https://doi.org/10.5566/ias.3024
56. Rodríguez Outeiral, R., Silverio, N. F., González, P. J., et al. (2023). A network score-based metric to optimize the quality assurance of automatic radiotherapy target segmentations. *Physics and Imaging in Radiation Oncology* 28, 100500. https://doi.org/10.1016/j.phro.2023.100500 — *(published comment + response: `10.1016/j.phro.2023.100528`)*
57. Li, X., Peng, B., Xie, Z., et al. (2023). Feature contrastive learning for no-reference segmentation quality evaluation. *Electronics* 12(10), 2339. https://doi.org/10.3390/electronics12102339
58. Luan, S., Xue, X., Wei, C., et al. (2023). Machine learning-based quality assurance for automatic segmentation of head-and-neck organs-at-risk in radiotherapy. *Technology in Cancer Research & Treatment* 22. https://doi.org/10.1177/15330338231157936
59. Kohlberger, T., Singh, V., Alvino, C., Bahlmann, C. & Grady, L. (2012). Evaluating segmentation error without ground truth. *MICCAI 2012* — context only (cite via citing papers; not independently retrieved).

**Reliability caveats & use in tuning (E)**
60. Jozdani, S. & Chen, D. (2020). On the versatility of popular and recently proposed supervised evaluation metrics for segmentation quality of remotely sensed images. *ISPRS Journal of Photogrammetry and Remote Sensing* 160, 275–290. https://doi.org/10.1016/j.isprsjprs.2020.01.002
61. Sims, Z., Strgar, L., Thirumalaisamy, D., et al. (2023). SEG: Segmentation Evaluation in absence of Ground truth labels. *bioRxiv*. https://doi.org/10.1101/2023.02.23.529809 — **[PREPRINT — not peer-reviewed]**
62. Deo, Y., Jia, Y., Lassila, T., Smith, W. A. P., Lawton, T., Kang, S., Frangi, A. F. & Habli, I. (2025). Metrics that matter: evaluating image quality metrics for medical image generation. *arXiv:2505.07175*. https://doi.org/10.48550/arXiv.2505.07175 — **[PREPRINT — not peer-reviewed]**
63. Rajarajeswari (Muthusivarajan), R., Celaya, A., Yung, J., et al. (2024). Evaluating the relationship between MR image quality metrics and DL-based segmentation accuracy of brain tumors. *Medical Physics* 51(7), 4898–4906. https://doi.org/10.1002/mp.17059
64. Kazakevičiūtė-Januškevičienė, G., Janušonis, E., Baušys, R., et al. (2020). Assessment of the Segmentation of RGB Remote Sensing Images: A Subjective Approach. *Remote Sensing* 12(24), 4152. https://doi.org/10.3390/rs12244152
65. Drăguţ, L., Csillik, O. & Eisank, C. (2014). Automated parameterisation for multi-scale image segmentation on multiple layers (ESP2). *ISPRS Journal of Photogrammetry and Remote Sensing* 88, 119–127. https://doi.org/10.1016/j.isprsjprs.2013.11.018
66. Georganos, S., Grippa, T., Lennert, M., et al. (2018). Scale matters: spatially partitioned unsupervised segmentation parameter optimization (SPUSPO). *Remote Sensing* 10(9), 1440. https://doi.org/10.3390/rs10091440
67. Grybas, H., Melendy, L. & Congalton, R. G. (2017). A comparison of unsupervised segmentation parameter optimization approaches using moderate- and high-resolution imagery. *GIScience & Remote Sensing* 54(4), 515–533. https://doi.org/10.1080/15481603.2017.1287238
68. da Costa, G. A. O. P., Feitosa, R. Q. & Cazes, T. B. (2007). Genetic adaptation of segmentation parameters. *Springer LNCS*. https://doi.org/10.1007/978-3-540-77058-9_37
