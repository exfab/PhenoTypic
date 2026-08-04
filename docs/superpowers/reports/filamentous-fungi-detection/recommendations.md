# Filamentous-Fungi Hyphae Segmentation on Macroscopic RGB Agar Plates: A Cross-Field Methods Survey and Ranked Recommendations

## 1. Executive Summary

**The task.** Produce a clean, binary, pixel-level mask of filamentous fungi (
hyphae/mycelia) on macroscopic RGB agar-plate images. The target is a *geometry*-defined
object — thin, oriented, branching, curvilinear/ridge structure detected by local
2nd-derivative/phase geometry, not by an intensity threshold. Two regimes are hard: *
*low SNR** (faint hyphae just above agar-grain noise, where the right strategy is to
*integrate evidence along a hypothesized line* rather than decide per pixel) and *
*dense-overlap→gaps** (a topology/connectivity problem where Frangi/Sato-style
vesselness *suppress the very junctions you need to keep*). Confounders are **chromatic
aberration** (color fringes that mimic thin colored filaments) and **agar grain**. The
desired solution is **training-free → small-data**, exposes a **single tunable
sensitivity knob**, and provides **gap-bridging** that feeds or replaces the existing
Dijkstra reconnection. The baseline to beat: **phase congruency + large-σ background
subtraction + hysteresis, with Dijkstra reconnection; chromatic aberration unhandled.**

**The three recommendations (detailed in §5).**

- **(a) Classical-CV baseline replacement — Steger's unbiased line detector, optionally
  fronted by oriented-Gaussian/coherence-enhancing enhancement.** Steger is the closest
  classical match to the brief's exact requirement: it detects line points purely from
  the first and second directional derivatives (no intensity threshold), exposes a
  *single hysteresis knob in the same idiom as the current pipeline*, and — uniquely
  among ridge filters — **emits an explicit lines-and-junctions graph rather than
  erasing crossings** [howardzzh.com/Steger1998] (V). It is training-free, CPU-cheap,
  and feeds the Dijkstra step cleaner, oriented endpoints. The cryo-EM tool **STRIPER**
  packages exactly this pattern (oriented-Gaussian max-response → Steger) with a
  2–3-image threshold grid-search [biorxiv STRIPER; 10.1107/S2059798320007342] (V), and
  is the most directly portable engineering template.

- **(b) Small-data learned, topology-aware — fine-tune a foundation/U-Net segmenter with
  a centerline/topology loss.** The verified path is a **decoder-only SAM2 fine-tune** (
  Colab-scale compute, small data, clean binary mask, "matches leading
  tools") [biorxiv 2025.11.08.687405] (V), defended at junctions/gaps by a *
  *centerline-aware topology loss** — **Skeleton Recall** (verified >90% cheaper than
  clDice, CPU-side target) [arxiv.org/pdf/2404.03010] (V) or **clDice** (proven
  homotopy-preserving) [arxiv 2003.07311] (V). Synthetic pretraining bridges the
  no-large-corpus constraint and is the natural place to bake in chromatic-aberration
  and agar-grain robustness.

- **(c) Strongest cross-field transfer — SOAX-style Stretching Open Active Contours for
  junction-first topology, and/or a Kalman/oriented-voting reconnection to replace
  Dijkstra.** SOAX treats **junctions as first-class outputs** (T-junctions constructed,
  clustered, spliced) and reconnects by **orientation/smoothness coherence** — a
  principled alternative or feeder to Dijkstra, with a single ridge-threshold knob and
  label-free tuning [srep09081] (V mechanism). For the reconnection step specifically,
  the **Kalman filter** (HEP track finding) bridges gaps by predict-and-coast along the
  local ridge orientation with a single gate-size knob,
  training-free [arxiv.org/abs/1601.08245] (V), and **ant-colony oriented voting** (
  seismic) jointly denoises and reconnects while its **multi-attribute (per-channel)
  front end** is the one cross-field lead that speaks to chromatic
  aberration [sciencedirect.com S0098300412002804; library.seg.org tle40070502.1] (V/u).

**The one confounder nobody solves for you.** Across all ten surveyed fields —
astronomy, biomedical vessels, cryo-EM, cytoskeleton, diffusion preprocessing,
foundation segmenters, Hessian vesselness, morphology/path-operators, line-integration
voting, particle/seismic tracking — **chromatic aberration is essentially never
addressed**; every ridge operator is single-channel. It must be handled **upstream** (
per-channel registration or detection on a CA-robust luminance/decorrelated channel)
before any oriented analysis runs. The only positive leads are analog and unverified: a
**training-free cross-channel shearlet prior** [link.springer.com 978-3-030-69532-3_7] (
u), per-channel multi-attribute fusion [library.seg.org tle40070502.1] (u), and
color-native phase congruency (CMPCM) [link.springer.com s11042-018-6617-x] (u). *
*Nothing in this survey was benchmarked on real agar plates** — all cross-domain
performance is transfer-by-analogy and must be validated on plate images before being
trusted.

---

## 2. Problem Recap

The deliverable is a **binary pixel-level mask** of hyphae/mycelia on **macroscopic RGB
agar-plate** images. Five properties of the target and setting drive every method choice
below:

1. **Geometry, not intensity.** Hyphae are thin, oriented, branching curvilinear
   *ridges*. The defining signal is local 2nd-derivative / phase geometry (a ridge has
   one near-zero curvature along its length and a large curvature across it), not a
   brightness level. Methods that threshold intensity fail on faint or contrast-varying
   filaments; methods that read ridge/orientation geometry are the right family.

2. **Low-SNR regime.** Faint hyphae sit just above agar-grain noise. The correct
   response is **evidence-integration-along-a-line** — accumulate weak signal along a
   hypothesized oriented (or curved) path so that a coherent faint filament beats the
   noise floor, where no per-pixel operator can. This favors matched filters, oriented
   accumulators (RHT/Radon), ridge-following trackers, active contours, and Kalman-style
   gated estimators.

3. **Dense-overlap→gaps regime (topology).** Where mycelium overlaps densely, detection
   fragments into gaps; the problem becomes **connectivity and junction preservation**.
   The brief's central warning is verified repeatedly across domains: **Frangi/Sato
   vesselness suppress junctions** because their single-dominant-orientation model
   scores both eigenvalues large (≈ isotropic curvature) at a crossing and collapses the
   response to near zero [arxiv.org/pdf/1709.05495] (V) — *manufacturing the very gaps*
   the downstream reconnection then has to repair. Junction-preserving methods (Steger's
   split-at-crossing, the bowler-hat "fit-a-long-line", SOAX's constructed junctions,
   multi-orientation flux/DOF) are structurally superior here.

4. **Confounders.** **Chromatic aberration** produces colored fringes along
   high-contrast edges that structurally mimic thin colored filaments — directly
   dangerous for a per-channel ridge detector. **Agar grain** is small-scale, isotropic
   texture that leaks into ridge responses.

5. **Constraints.** Prefer **training-free → small-data** (no large hand-labeled
   corpus); expose a **single tunable sensitivity knob** with a sensible default;
   provide **gap-bridging** that feeds or replaces the existing **Dijkstra reconnection
   **.

**Current baseline:** phase congruency (orientation-aware, contrast-invariant ridge
response) + large-σ background subtraction + hysteresis thresholding, with Dijkstra
least-cost-path reconnection. Phase congruency is a good front end — contrast-invariant,
single-threshold, geometry-based — but it carries a documented junction weakness (its
cross-orientation normalization ΣAₙ is dominated by the strong feature, fading out
weaker crossing branches (u)), and **chromatic aberration is unhandled** because it runs
on a single channel.

---

## 3. Methods by Family

### 3.1 Classical CV (training-free)

**Hessian / vesselness (Frangi, Sato, Jerman, Meijering).** The common engine is the
Hessian's eigenvalues λ₁, λ₂: a bright tubular ridge gives one near-zero and one
large-magnitude eigenvalue, and the eigenvector gives orientation for
free [link.springer.com/chapter/10.1007/bfb0056195] (V). Multi-scale max-over-σ matches
each filament's width [10.1016/S1361-8415(98)80009-1]. The **load-bearing negative
result**: at junctions both eigenvalues are large, so vesselness → 0 and **vessel-like
structures are lost at junctions** [arxiv.org/pdf/1709.05495] (V). Variants partially
repair this: **Jerman** uses an eigenvalue *ratio* reported to reinforce bifurcation
response (u); **Meijering neuriteness** recombines eigenvalues as λ′ = λ₁ + α·λ₂ with a
single tunable α (default −1/3 in 2D), purpose-built for low-contrast filaments (u) but
still single-orientation (no junction fix); **Sato** explicitly discriminates
line-vs-sheet-vs-blob, useful against blob-like agar grain (u). Choose empirically via
the **Lamy et al. ICPR 2020** seven-filter benchmark, whose *bifurcation ROI* directly
measures junction behavior [hal.science/hal-02544493v2] (u).

**Bowler-hat transform (morphological, the verified junction-preserver).** Abandons
eigenvalues for long line-shaped structuring elements at multiple orientations; because
**longer line elements still fit within a junction, junctions are enhanced as brightly
as the vessels joining them, unlike many other enhancement methods
** [arxiv.org/pdf/1709.05495] (V). Training-free, CPU-friendly, single knob (line
length/orientation count) — the strongest verified idea to *keep* junctions and
feed/partly replace Dijkstra.

**Phase congruency & flux (the baseline's family + the better-in-family option).** Phase
congruency locates features where Fourier components are maximally in phase — geometry,
contrast-invariant in [0,1], one universal threshold, with a principled Rayleigh-noise
threshold T = µ_R + k·σ_R (k≈2–3) that is *the same* mechanism as the baseline's energy>
noise rule (u). Its junction weakness (cross-orientation normalization dominance) has a
cheap fix — **Freeman per-orientation normalization** (u). **Optimally Oriented Flux (
OOF)** is the strongest *verified* in-family upgrade: it measures gradient flux through
the *boundary* of a local sphere, so it **avoids corruption from adjacent structures** (
resolving closely-located vessels the Hessian merges), **needs no large-σ smoothing (σ
fixed at 1), is more noise-robust, and recovers weak low-SNR vessels the Hessian misses
entirely** [cse.hkust.edu.hk/~achung/eccv08_law_chung.pdf] (V). Its single physical knob
is the sphere radius. The **OCF** extension accumulates along the structure and encodes
*multiple orientations per voxel* on S³ — a tentative (u) route to representing both
branches of a crossing.

**Steger / steerable / matched filters.** **Steger's unbiased line detector** uses only
first/second directional derivatives, declares a line at the perpendicular
second-derivative extremum, **marks junctions and splits lines there into a connected
graph**, and uses upper/lower hysteresis as its single knob [howardzzh.com/Steger1998] (
V) — the closest classical fit to the entire brief. **Steerable filters** synthesize
oriented response at any angle from a small basis in closed form, giving continuous
per-pixel orientation (sharper at crossings) (u) [10.1109/34.93808]. **Matched filters
** (Chaudhuri Gaussian-cross-section, 12-orientation bank) are the canonical
evidence-along-a-line low-SNR booster but over-respond at junctions and have several
coupled knobs (u). **B-COSFIRE** integrates oriented DoG responses by weighted geometric
mean (every term must fire → strong grain suppression), configurable from one
prototype (V).

**Morphology / path operators.** **Path openings/RORPO** filter by
length-of-admissible-path, *explicitly avoiding the local-neighborhood analysis* of
Hessian filters [researchgate/315838176]; the single knob is **path length L** in
physical units; the **parsimonious/incomplete-path variant bridges gaps inline at
linear, length-independent cost** [pubmed 24569442] and implicitly resists chromatic
fringes (short, non-sustained). RORPO **beat Frangi, OOF, and HDCS with up to 8% more TP
and 50% fewer FP** (u, 3D vascular) [10.1109/tpami.2017.2672972] — but, like Frangi, *
*suppresses junctions** (u), so it needs a topology stage.

**Diffusion preprocessing.** **Coherence-enhancing anisotropic diffusion (CED)**
computes the structure tensor and diffuses *along* the dominant orientation, explicitly
to **close interrupted line-like structures**, with a single coherence/contrast knob —
all (V) [S0262885698001024; 10.1023/A:1008009714131]. It is an *enhancer*, not a
binarizer, and reverts to isotropic at junctions (preserves, doesn't enhance them). The
**PCT** variant swaps in a contrast-independent phase-congruency tensor, was **validated
on saprotrophic fungal networks**, and beats intensity-based coherence on CNR under
noise (u) [10.1109/ISBI.2012.6235519; markfricker.org PCT] — but inherits Hessian
junction-ambiguity (u). **Curvelets** offer training-free anisotropic multiscale
denoising+ridge enhancement (u).

### 3.2 Deep learning (small-data)

**Foundation segmenters (SAM/SAM2/SAM3).** Zero-shot **fails** on thin branching
curvilinear structures (SAM cannot segment retinal vessels even with extra
prompts) [10.3390/diagnostics13111947] (V) — the training-free end is foreclosed. The
viable route is small-data transfer: **decoder-only SAM2 fine-tune** gives robust
cross-modality segmentation from small datasets, runs in a single Colab notebook, and *
*matches leading tools** [biorxiv 2025.11.08.687405] (V); LoRA/adapter routes touch **<
5–7% of weights** and a **20-image** rescue raises Dice ≥200% over zero-shot (
u) [arxiv 2510.10288v1; arxiv 2502.18185v1; 10.3390/diagnostics13111947]. The recurring
failure mode is **thin-branch/junction degradation even after fine-tuning
** [arxiv 2510.10288v1] (u) — to be defended with a connectivity module (TPNet (u)) or a
clDice loss. PhenoTypic's own `Sam2` already exposes `pred_iou_thresh`/
`stability_score_thresh` as a natural sensitivity knob and runs through the staged-GPU
engine.

**U-Net + topology/imbalance losses.** For thin extreme-minority targets, **the loss is
the design decision**, not the architecture; under severe imbalance,
weighted-BCE/Focal/Dice/compound losses beat plain CE (u, crack
benchmark) [S0141029623014037]. **clDice/soft-clDice** scores overlap of *skeletons*, *
*provably preserves topology up to homotopy equivalence**, is a **drop-in differentiable
loss for any network**, and **recovers gap false-negatives** soft-Dice misses — all (
V) [arxiv 2003.07311]; its α blend is the single knob (u). Caveats (u): fragile under
noisy labels and can fail on the densest tangles (ASOCA), where **clCE** is the safer
default. **Detector-first crop** (HessianNet) and **class-balanced weighted CE** (
DeepCrack, positive weight = neg/pos ratio) directly attack imbalance (V/u). Synthetic
pretrain → fine-tune is validated (9000 synthetic + ~240 real; real-only fails at Dice ~
0.1) (u) [10.1016/j.cmpb.2020.105420] and is where CA/grain robustness can be baked in.

### 3.3 SOTA (topology-aware losses, tracing nets, learned reconnection)

**Topology losses beyond clDice.** **Skeleton Recall** computes a dilated GT skeleton
target on **CPU** and applies a soft recall loss — **>90% cheaper than clDice** (≈8%
train time / 2% VRAM vs clDice's ≈88%/52%), drop-in, multi-class scalable where clDice
OOMed on an A100 — all (V) [arxiv.org/pdf/2404.03010]; the dilation radius (default 2)
is the single knob. **Betti matching** matches persistence barcodes with *spatial*
correspondence, penalizing a break/merge *where it occurs* — the most principled
junction signal — (V) [arxiv 2407.04683], at higher (persistent-homology) cost. *
*Homotopy Warping** penalizes only topologically-critical pixels on the *binary mask* (
O(n), noise-robust), beating clDice on DRIVE Betti error (u) [neurips 2022/98143953]. *
*Persistence-image** U-Net features cut DRIVE β₀ error ~41% (
V) [arxiv.org/html/2601.18045]. **TopoLoss** (Betti-number matching) is the expensive
reference these improve on (u).

**Tracing nets / learned reconnection.** **TSNet** fuses multiple frames via
temporal-spatial attention to repair low-contrast discontinuities (Dice 0.897, clDice
0.935) — validates evidence-fusion but **needs a temporal stack** (
u) [10.1016/j.compmedimag.2025.102540]. The **unsupervised synthetic-pair reconnecting
regularizer** learns "curvilinear structures should be connected" from
procedurally-generated broken/whole pairs with **no human labels** — the closest learned
**Dijkstra replacement** (u) [arxiv.org/abs/2408.12943].

---

## 4. Cross-Field Analogs — What Transfers to Fungal Hyphae

Each adjacent field solved a structurally identical problem; the table summarizes the
transferable mechanism, then the strongest specifics follow.

| Field                           | Closest analog object                   | Transferable mechanism                                                                                                                                                                                                           | Best for which regime                          | Verif.                            |
|---------------------------------|-----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|-----------------------------------|
| **Cytoskeleton (closest twin)** | Actin/microtubule networks              | **SOAX**: ridge-seeded active contours; **junctions as first-class outputs**; orientation/smoothness reconnection; single τ knob; label-free tuning                                                                              | Dense-overlap→gaps; low SNR                    | V (mech)                          |
| **Astronomy**                   | Coronal loops, ISM/cosmic-web filaments | Matched-filter rotated Gaussian cross-section (max-response, σ knob); **OCCULT-2** ridge-tracer with inline `ngap` gap-bridging; DisPerSE/FilFinder **persistence** as single topological knob; LSD a-contrario FDR thresholding | Low SNR; junctions; self-thresholding          | V (matched filter, LSD); u (rest) |
| **Cryo-EM**                     | Amyloid fibrils, helical filaments      | **STRIPER** = oriented-Gaussian max-response → **Steger** (split-at-junction); 4 params, 2–3-image grid-search; long-rectangle non-local gap-bridging                                                                            | Low SNR; junctions; self-calibration           | V (STRIPER)                       |
| **Biomedical vessels**          | Retinal/coronary trees, guidewires      | **Hessian λ₁/λ₂ enhancement embedded in a net** (HessianNet); detector-first crop; clDice/error-discriminator; **VNR** algorithmic reconnection; synthetic pretrain                                                              | Low SNR; topology; small data                  | V (HessianNet, seg); u (rest)     |
| **Diffusion/CA**                | Flow-like line fields; color fringing   | **CED** along-orientation gap-closing enhancer (single knob); **cross-channel shearlet prior** for CA                                                                                                                            | Low SNR/along-line gaps; CA (only lead)        | V (CED); u (CA)                   |
| **Particle physics**            | Charged-particle tracks                 | **Kalman filter** predict-and-coast gap-bridging along orientation; single gate-size knob; **CA+KF (CATS)** parallel linking handles forks                                                                                       | Gaps; low SNR; junctions (CATS)                | V (KF role); u (CATS)             |
| **Seismic**                     | Fault surfaces                          | **Ant-colony** oriented voting = joint denoise + reconnection + Zhang-Suen thinning → mask; **multi-attribute/per-channel** front end                                                                                            | Low SNR + gaps jointly; **CA via per-channel** | V (ACO); u (multi-attr)           |
| **Remote sensing / cracks**     | Roads, cracks                           | **clDice** topology loss (Betti-0 1.474→0.920); explicit **junction-detection branch**; **energy-fusion** (area+centerline) reconnection; class-balanced CE; single softmax-threshold knob                                       | Topology; imbalance; single knob               | V (clDice); u (rest)              |

**The single most important convergence.** Five independent fields (astronomy's
OCCULT-2, cytoskeleton's SOAX, cryo-EM's STRIPER/Steger, Hessian-family analyses,
morphology's RORPO) all reach the *same* conclusion the brief states: **a strong ridge
detector still leaves junctions/topology as the hard open problem**, and
single-dominant-orientation models (Frangi/Sato/PCT/RORPO) *suppress* exactly the
crossings hyphal networks are built from. The methods that *win* the dense regime are
the ones that treat junctions as objects to construct (Steger split-at-crossing; SOAX
T-junction cluster-and-splice; bowler-hat fit-a-long-line; multi-orientation
flux/FilDReaMS DOF), not defects to filter away.

**The second convergence — low SNR = integrate along a line.** Astronomy's matched
filter (V) and OCCULT-2 flux-summing tracer (u), cryo-EM's long-rectangle convolution (
u) and oriented-Gaussian max-response (V), cytoskeleton's contour energy (V), seismic
ant-direction-consistency (V), particle-physics Kalman gating (V), and the RHT's ρ=0
accumulation (V) are *all the same idea*: a faint filament whose signal is geometric
beats the noise floor only when evidence is summed along its hypothesized path. The
shared single knob is set from **hyphal width** (matched-filter σ, OOF radius, path
length L, STRIPER filament width) or **gap tolerance** (OCCULT-2 `ngap`, RHT Z, KF
gate).

**Reconnection alternatives to Dijkstra, ranked by transfer confidence.** (1) **Kalman
filter** — highest-confidence (V), training-free, single gate knob, seeds from baseline
fragment endpoints, propagates along the phase-congruency orientation field; weak only
at branches (needs multi-hypothesis). (2) **SOAX orientation/smoothness merge** (V
mechanism) — reconnects by "is this the same filament continued," reconstructing
connectivity *inside* the model. (3) **Ant-colony oriented voting** (V) — jointly
denoises and reconnects, yields a thinned binary mask, and its multi-attribute front end
is the CA lead. (4) **Tensor voting** (analog/u) — gap-bridging *and* junction
*detection* (ball-tensor = junction), single scale knob. (5) **Energy-fusion** of area
mask + centerline (u, RoadCorrector). (6) **Unsupervised synthetic-pair regularizer** (
u) — learned drop-in. (7) **OCCULT-2 inline `ngap`** / **incomplete-path openings** —
absorb *short* gaps during detection so Dijkstra handles only long-range residuals.

---

## 5. Ranked Recommendations vs. the Phase-Congruency + Dijkstra Baseline

Three concrete options, each a different point on the training-free→small-data axis,
each integrable as a Python image-operation in the existing pipeline.

### (a) Classical-CV baseline — **Steger unbiased line detector + oriented-Gaussian /
CED enhancement** (training-free, single knob)

**Why it beats the baseline.** Steger detects line points from first/second directional
derivatives (geometry, not intensity), localizes the centerline to sub-pixel accuracy,
and **uniquely emits a connected lines-and-junctions graph** — it marks a junction and
*splits* the line there rather than collapsing the response, which is the precise
failure of the phase-congruency/Hessian junction normalization the baseline
suffers [howardzzh.com/Steger1998] (V). Its **single knob is upper/lower hysteresis on
the second-derivative response — the same idiom the current pipeline already uses**, so
it is a near drop-in for the response→hysteresis stage. The cryo-EM **STRIPER**
packaging (oriented-Gaussian max-response front end → Steger, four params, **threshold
grid-search from 2–3 annotated images**) is the ready engineering
template [biorxiv STRIPER; 10.1107/S2059798320007342] (V).

- **Low SNR:** Front the detector with an **oriented-Gaussian max-response** enhancer (
  STRIPER, V) or **coherence-enhancing diffusion** (CED, V) so evidence is integrated
  along the line *before* the ridge decision; CED additionally **closes short along-line
  gaps** in the continuous domain [S0262885698001024] (V), shrinking the gap count
  Dijkstra must span. Salience is curvature magnitude, so faint filaments survive
  without an intensity cutoff.
- **Junction/gap:** Steger's split-at-junction gives a topologically-aware graph
  natively (V); for the densest crossings, pair with the **bowler-hat** "
  fit-a-long-line" enhancer, which **lights junctions as brightly as the vessels joining
  them** [arxiv.org/pdf/1709.05495] (V). Steger does not bridge *true* gaps — it **feeds
  ** Dijkstra well-defined oriented endpoints (better cost terms via known `n(t)`), or
  hands them to a Kalman/tensor-voting stage (see (c)).
- **Chromatic aberration:** **Not handled by Steger** (single-channel). Correct
  upstream — per-channel registration or run on a CA-robust luminance/decorrelated
  channel — *before* the enhancer. As a cheap hardening, run the detector per RGB
  channel and **intersect** (a one-channel fringe is dropped). Agar grain is suppressed
  by the enhancer's orientation requirement + a larger σ + a minimum-length filter on
  output segments.
- **Integration (Python op):** A new `ImageOperation` whose `_operate` (i) optionally
  builds the CA-robust channel, (ii) runs the oriented-Gaussian/CED enhancement on
  `detect_mat`, (iii) runs Steger ridge extraction (sub-pixel lines + junction nodes), (
  iv) rasterizes/dilates lines to hyphal width for the binary `objmap`, exposing *
  *one `sensitivity` field** mapped to the hysteresis pair (with width/σ set from
  calibration). The emitted endpoint+orientation graph plugs directly into the existing
  Dijkstra reconnection. Training-free, CPU, GPU-portable (separable derivatives).

### (b) Small-data learned, topology-aware — **decoder-only SAM2 (or U-Net) fine-tune +
centerline topology loss + synthetic pretrain**

**Why it beats the baseline.** A learned segmenter emits a clean binary mask directly
and can learn agar-grain/CA appearance away from real plates — but **only with
adaptation** (zero-shot SAM fails on thin branching
structure [10.3390/diagnostics13111947] (V)). The verified, low-barrier route is a *
*decoder-only SAM2 fine-tune**: robust cross-modality segmentation from small data,
single Colab notebook, **matches leading tools** [biorxiv 2025.11.08.687405] (V) — a
credible *replacement* for the geometric front end. Defend the thin/junction regime (the
documented post-fine-tune failure [arxiv 2510.10288v1] (u)) with a **centerline topology
loss**: **Skeleton Recall** (verified >90% cheaper than clDice, CPU-side skeleton
target, drop-in) [arxiv.org/pdf/2404.03010] (V) is the default; **clDice** (proven
homotopy-preserving, recovers gap false-negatives) [arxiv 2003.07311] (V) when a formal
guarantee is wanted; **clCE** if labels are noisy (u).

- **Low SNR:** Indirect — the topology loss rewards completing faint centerlines, and
  attention integrates weak evidence over long spans (u). If the rig can capture a *
  *growth time-lapse / multi-exposure stack**, a **TSNet**-style temporal-spatial fusion
  realizes true evidence-integration (u) [10.1016/compmedimag.2025.102540].
- **Junction/gap:** This is the loss's job. Skeleton Recall / clDice push the network to
  *not break* the hypha, **replacing or feeding** Dijkstra; **Betti matching** adds
  spatially-localized junction penalties (V) [arxiv 2407.04683] for the densest mats; a
  **TPNet** connectivity module (u) is the architectural option.
- **Chromatic aberration:** The one route that can *learn* CA away — bake **simulated
  per-channel fringe + agar texture into synthetic pretraining** (9000-synthetic →
  fine-tune is validated; real-only fails at Dice ~0.1 (
  u) [10.1016/j.cmpb.2020.105420]). This is a *hypothesis to validate on plates*, not a
  guarantee.
- **Data/compute:** ~20–700 annotated plates by analog evidence (u); LoRA touches <5–7%
  of weights; adaptation is Colab-scale (V). Class-balanced weighted CE (pos weight =
  neg/pos) [DeepCrack] and detector-first crop [HessianNet] (V) handle the severe
  imbalance.
- **Integration (Python op):** `Sam2` is already a `GpuDetector` exposing
  `pred_iou_thresh`/`stability_score_thresh` as the **single inference knob** and runs
  through the staged-GPU engine (preprocess→GPU→measure, objmap sidecar). Fine-tuned
  weights drop in behind that wrapper; training lives offline. Small-data, GPU at
  inference, clean binary mask native.

### (c) Strongest cross-field transfer — **SOAX junction-first contours and/or
Kalman/oriented-voting reconnection (replace Dijkstra)**

**Why it beats the baseline.** This option attacks the regime the baseline is weakest
on — dense-overlap→gaps — with the cytoskeleton **closest-twin** method and the
highest-confidence reconnection import. **SOAX** seeds active contours on intensity
ridges and **constructs junctions as first-class outputs** (T-junctions clustered,
contours cut-and-spliced so they neither end nor bend sharply at junctions),
reconnecting by **orientation/smoothness coherence** — a principled alternative or
feeder to Dijkstra, with **evidence-integration-along-the-contour** for low SNR, a *
*single ridge-threshold τ** knob, and a **label-free F-function tuning**
method [srep09081] (V mechanism; tuning u). For the reconnection step alone, the *
*Kalman filter** is the highest-confidence transfer: it bridges gaps by predicting along
the local ridge orientation and coasting through dropouts, gated by a **single
Mahalanobis gate-size knob**, training-free, seeded from baseline fragment
endpoints [arxiv.org/abs/1601.08245] (V).

- **Low SNR:** SOAX contour energy and KF gating both integrate evidence along the
  line (V); SOAX needs local SNR ≈ 5 and τ-tuning near the floor (V/u). **Ant-colony
  oriented voting** (seismic) **jointly reduces noise and improves continuity** in one
  pass [sciencedirect.com S0098300412002804] (V).
- **Junction/gap:** SOAX is the verified junction-*constructor* (V); KF alone follows
  one branch (needs multi-hypothesis or a CA-cellular-automaton front (CATS, u) so forks
  survive); **tensor voting** both bridges gaps and *detects* junctions (ball-tensor) as
  an analog option (u).
- **Chromatic aberration:** Best cross-field lead is seismic **multi-attribute /
  per-channel** fusion — feed R/G/B (or spectral) ridge evidence to the tracker so
  channel-disagreement (the CA signature) is rejected or exploited rather than corrupted
  by grayscale collapse [library.seg.org tle40070502.1] (u); pair with the training-free
  **cross-channel shearlet prior** [link.springer.com 978-3-030-69532-3_7] (u). All CA
  leads are analog/unverified.
- **Output/cost:** SOAX emits a **centerline+junction vector network** (rasterize/dilate
  to mask), CPU-iterative, cost scaling with contour count; KF/ACO are cheap,
  training-free, and ACO + Zhang-Suen yields a thinned binary mask directly (u).
- **Integration (Python op):** Either (i) a **reconnection operator** that
  replaces/augments Dijkstra — consume the baseline's fragment endpoints + orientation
  field, run KF predict/update (gate-size knob) or oriented-voting, emit a reconnected
  `objmap`; lowest-risk, fastest prototype, highest-confidence (V); or (ii) a **SOAX
  topology operator** that takes the enhanced ridge image, traces contours + junctions (
  τ knob, F-function auto-tune), and rasterizes to mask — higher upside on dense mats,
  more implementation effort, (u)-tier specifics.

**Recommended sequencing.** Ship **(a) Steger** first (verified, training-free, single
knob, drop-in, junction-aware graph). In parallel prototype **(c) Kalman reconnection**
as a direct Dijkstra replacement (verified, cheap, single knob) and evaluate the *
*seismic multi-attribute idea as the CA handler**. Pursue **(b)** once even a small
labeled/synthetic set exists, leading with the decoder-only SAM2 fine-tune + Skeleton
Recall loss, with synthetic pretraining carrying CA/grain robustness. A defensible
composite end-state: **Steger/oriented-enhancement ridge evidence → SOAX or
Kalman/tensor-voting junction+gap handling → Dijkstra reserved for residual long-range
reconnections**, with CA corrected upstream throughout.

---

## 6. Caveats & Open Gaps

**Provenance.** Claims are flagged **(V)** verified (survived adversarial verification)
or **(u)** unverified (sourced, often single-source/abstract-level — cite direction
confidently, magnitude tentatively).

**Firmly verified (V) load-bearing claims.**

- Frangi/Sato **suppress junctions** (both eigenvalues large at crossings → vesselness ≈
  0) [arxiv.org/pdf/1709.05495]. This is the brief's central premise, independently
  corroborated across families.
- **Bowler-hat** enhances junctions as brightly as the vessels joining
  them [arxiv.org/pdf/1709.05495].
- **OOF** is adjacency-robust, needs no large-σ smoothing, and recovers weak low-SNR
  vessels the Hessian misses [cse.hkust.edu.hk/~achung/eccv08_law_chung.pdf].
- **Steger** uses only 1st/2nd directional derivatives, preserves junctions by split,
  single hysteresis knob [howardzzh.com/Steger1998]; **STRIPER** packages
  oriented-Gaussian→Steger with 2–3-image grid-search [biorxiv STRIPER].
- **CED** closes interrupted line-like structures via along-orientation diffusion,
  single knob [S0262885698001024; 10.1023/A:1008009714131].
- **SOAX** constructs/clusters/splices junctions and targets low-SNR images [srep09081].
- **clDice** preserves topology up to homotopy equivalence, drop-in differentiable loss,
  recovers gap false-negatives [arxiv 2003.07311]; **Skeleton Recall** is >90% cheaper
  with a CPU-side target [arxiv.org/pdf/2404.03010]; **Betti matching** does spatial
  barcode matching [arxiv 2407.04683]; **persistence-image** features cut DRIVE β₀ ~
  41% [arxiv.org/html/2601.18045].
- Zero-shot SAM **fails** on retinal vessels [10.3390/diagnostics13111947]; *
  *decoder-only SAM2 fine-tune** matches leading tools at Colab
  scale [biorxiv 2025.11.08.687405].
- **Kalman filter** is the dominant, robust HEP track-finder [arxiv.org/abs/1601.08245];
  **ant-colony** jointly denoises + improves fault
  continuity [sciencedirect.com S0098300412002804]. **LSD** is parameter-free with
  a-contrario FDR control [ipol2012; 10.1109/tpami.2008.300]; **RHT** Z-knob with
  built-in gap tolerance [arxiv.org/pdf/1312.1338].

**Tentative (u) — validate before relying on.** Jerman bifurcation-reinforcement;
Meijering α-knob low-SNR; OCF multi-orientation S³; PCT fungal validation +
junction-ambiguity; RORPO's 8%-TP/50%-FP margins and junction suppression; OCCULT-2
inline `ngap` + cross-domain transfer; FilDReaMS multi-orientation DOF + Monte-Carlo
significance; crYOLO small-data (but **junction-omitting** — wrong sign); TPNet
connectivity; clDice noisy-label fragility + ASOCA failure + clCE remedy; LoRA <5–7%
budgets + 20-image rescue; synthetic-pretrain numbers;
VNR/energy-fusion/CATS/tensor-voting/unsupervised-reconnect mechanisms; all CA leads (
shearlet prior, multi-attribute fusion, CMPCM, learned fringe channel).

**Open gaps.**

1. **Chromatic aberration is essentially unsolved by every surveyed field** — all ridge
   operators are single-channel; CA must be corrected upstream (per-channel
   registration / CA-robust luminance / per-channel intersect). The only positive leads
   are analog and (u): cross-channel shearlet prior, seismic multi-attribute per-channel
   fusion, color-native CMPCM, learned fringe isolation. None demonstrated on hyphae.
2. **Agar grain** is only partially mitigated (oriented smoothing, geometric-mean/length
   selectivity, persistence/path-length pruning); never validated against agar texture
   specifically.
3. **Nothing here was benchmarked on real agar plates.** Every cross-domain result comes
   from retinal/coronary vessels, cryo-EM, cytoscaffold microscopy, roads, cracks,
   astronomy, physics, or seismic data — the *geometry* transfers (thin branching
   curvilinear networks with β₀/β₁ structure), but the *imaging conditions* (macroscopic
   scale, agar grain, chromatic fringe, plate lighting) do **not**. DRIVE/STARE/road
   Betti and Dice deltas are indicative, not transferable; re-validate on plate images.
4. **Single-knob vs. reality.** The cleanest single knobs are classical (Steger
   hysteresis, OOF radius, path length L, SOAX τ, KF gate, RHT Z, matched-filter σ);
   learned methods expose the knob only at *training* (loss-mix α / dilation radius) or
   via post-hoc thresholds (SAM2 `pred_iou_thresh`) — verify a true inference-time
   single knob exists before claiming it.
5. **Junctions remain the residual hard problem** even with the best ridge front end; no
   single method is fully turnkey for dense branching mycelium — the topology stage (
   SOAX / bowler-hat / Betti loss / tensor voting) is where the highest-upside,
   lowest-confidence engineering effort should concentrate, and must be validated on
   hand-labeled hyphal crossings (e.g., via the Lamy bifurcation-ROI protocol).
