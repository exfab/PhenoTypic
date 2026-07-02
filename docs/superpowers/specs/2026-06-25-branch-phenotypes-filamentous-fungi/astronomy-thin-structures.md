# Astronomy analog — thin elongated structures in noisy, low-SNR backgrounds

**Date:** 2026-06-25 · Companion to [branch-phenotypes-catalog.md](branch-phenotypes-catalog.md)

**Why this field:** Astronomy routinely extracts sub-pixel-faint filaments and streaks from
extremely noisy, low-SNR frames — the same regime as thin, faint, dense, overlapping hyphae
on textured agar at poor resolution. Unlike the four biology/physics analogs (which mostly
assume cleaner imaging), astronomy's entire methodology is built around the noise problem.
This file is the standalone deliverable from the `analog-astro` research agent; its highest-value
contributions are folded into §3.2/§3.3 and §5 of the main catalog.

**Assumption tags.** *Noise model:* most astro methods assume Gaussian + Poisson noise with a
locally estimable background; fungal agar texture is structured/correlated noise, which several
methods (Morse persistence, RHT, matched filter) tolerate better than naive thresholding.
*Resolution:* ISM-filament tools assume the filament is **resolved transversely** (several px
across) — the single biggest transfer risk for poorly-resolved hyphae. *Separability:*
point-process / cosmic-web methods assume **sparse, well-separated** tracers — the opposite of
dense overlapping hyphae; image-domain ridge/Hessian/Steger methods handle density far better.

---

## Table: metrics/methods

| Name | What it computes / how it works | Type | Input required | Noise & resolution robustness | Fungal-branch equivalent / use | Software/tool | Key citation(s) |
|---|---|---|---|---|---|---|---|
| **Medial-axis skeletonization (FilFinder spine)** | Adaptive-threshold mask → reduce to 1-px medial axis → graph-prune to spines | Tracing | Raw grayscale → mask | Med — wide brightness dynamic range; needs a usable mask first | Hyphal centerline / spine extraction for length & topology | FilFinder | Koch & Rosolowsky 2015; doi:10.1093/mnras/stv1521 |
| **FilFinder radial width profiling** | Perpendicular intensity profiles along spine; Gaussian/Plummer fit → FWHM width | Metric | Grayscale + spine | Med — width needs transverse resolution (≥3–4 px) | **Hyphal diameter** along each branch | FilFinder | Koch & Rosolowsky 2015; doi:10.1093/mnras/stv1521 |
| **Filament length / curvature / branch-count** | Walk skeleton graph: longest path, intersections (branch pts), local curvature | Metric | Skeleton graph | High — graph metrics robust once skeleton exists | Hyphal length, branching frequency, tortuosity per colony | FilFinder | Koch & Rosolowsky 2015; doi:10.1093/mnras/stv1521 |
| **Plummer / Gaussian radial profile fit** | Fit ρ(r)=ρ_c/[1+(r/R_c)²]^{p/2} to transverse cut | Metric | Grayscale + spine | Med — assumes resolved transverse profile | Hyphal cross-section model; wide vs thin hyphae | FilFinder/custom | Arzoumanian et al. 2011; doi:10.1051/0004-6361/201116596 |
| **DisPerSE (discrete Morse + persistence)** | Delaunay density → critical points → filaments = ascending 1-manifolds; persistence thresholds prune noise | Detection + tracing | Point set or grid | **High** — topological persistence is provably noise-robust; parameter/scale-free | **Noise-robust skeleton**; persistence = per-branch confidence | DisPerSE | Sousbie 2011; doi:10.1111/j.1365-2966.2011.18394.x |
| **Persistence-pair noise pruning (persistent homology)** | Rank features by birth–death; discard low-persistence = noise | Denoise/metric | Scalar field / image | **High** — explicit significance per feature | Reject spurious agar-texture "branches" | DisPerSE / topology libs | Sousbie 2011; doi:10.1111/j.1365-2966.2011.18394.x |
| **Rolling Hough Transform (RHT)** | Per-pixel local Hough over a disk → probability of coherent linearity + orientation | Detection (orientation) | Raw grayscale (bit mask) | **High** — pulls coherent linearity from diffuse noisy HI; tolerant of faint structure | **Hyphal anisotropy / orientation field in noise**; tip-orientation | RHT | Clark, Peek & Putman 2014; doi:10.1088/0004-637X/789/1/82 |
| **getfilaments (multiscale decomposition)** | Decompose into spatial scales; isolate filaments per scale; clean noise per scale | Detection + denoise | Raw grayscale | High — multiscale separation suppresses noise; no free parameters | Separate thin hyphae from wider structures & background by scale | getfilaments/getsf | Men'shchikov 2013; doi:10.1051/0004-6361/201321885 |
| **getsf (structural-component separation)** | Separate sources, filaments, background into distinct components before extraction | Detection + denoise | Raw grayscale (multi-band) | High — parameter-free, decorrelates background from filaments | Disentangle overlapping hyphae from textured agar | getsf | Men'shchikov 2021; doi:10.1051/0004-6361/202039913 |
| **NEXUS+ / MMF multiscale Hessian** | Smooth at many scales; Hessian eigenvalues → filament signature; scale-independent combine | Detection | Grayscale / density grid | **High** — flags elongated structures across widths | **Multiscale tube/ridge enhancement** for hyphae of varying thickness | NEXUS+ / MMF | Cautun et al. 2013; doi:10.1093/mnras/sts416 |
| **Frangi-style multiscale Hessian "vesselness"** | Eigenvalue ratios flag locally tubular structures | Denoise/enhance | Raw grayscale | High — enhances tubes, suppresses blobs/noise | **Faint-hypha enhancement** before thresholding (shared w/ biomedicine) | scikit-image/ITK | Frangi et al. 1998; doi:10.1007/BFb0056195 |
| **Steger unbiased curvilinear detector** | 2nd directional derivative (Hessian of Gaussian) → subpixel line position **and width** | Tracing + metric | Raw grayscale | **High** — explicit line model, subpixel, removes asymmetry bias | **Subpixel hyphal centerline + width** at poor resolution; best low-res tracer | Steger/OpenCV ridge | Steger 1998; doi:10.1109/34.659930 |
| **SCMS density ridges** | Gradient-ascent constrained orthogonal to ridge → 1-D ridges; bootstrap uncertainty | Detection + tracing | Point set (or density) | High on noise (KDE smooths); **assumes sparse points** | Ridge skeleton from detected-pixel cloud; **uncertainty per branch** | SCONCE-SCMS | Chen et al. 2015; doi:10.1093/mnras/stv1996 |
| **Bisous marked point process** | MCMC config of small cylinders; orientation-coherent neighbours chain into filaments | Detection (statistical) | Point set | High on noise (stochastic averaging); **assumes sparse tracers**, slow | Probabilistic hyphal-tract tracing from sparse detections | Bisous | Tempel et al. 2014; doi:10.1093/mnras/stt2454 |
| **Minimum Spanning Tree (MST) extraction** | MST over tracer points; prune edges by length/branching → filamentary graph | Tracing | Point set | Med — sensitive to noise points; cheap | Connect detected hyphal fragments into a network graph | MiSTree/custom | Barrow, Bhavsar & Sonoda 1985; doi:10.1093/mnras/216.1.17 |
| **T-web (tidal tensor) classification** | Eigenvalues of tidal tensor above threshold → void/sheet/filament/knot | Detection (classify) | Density field | Med — needs smoothed field; threshold-dependent | Per-pixel "in a hyphal filament" label from a smoothed density map | custom | Forero-Romero et al. 2009; doi:10.1111/j.1365-2966.2009.14885.x |
| **V-web (velocity shear) classification** | Eigenvalues of velocity-shear tensor; finer scales than T-web | Detection (classify) | Velocity/flow field | Med — needs a flow field | Conceptual; growth-flow analog if time-lapse | custom | Hoffman et al. 2012; doi:10.1111/j.1365-2966.2012.21789.x |
| **COWS (Hessian + skeleton hybrid)** | Hessian classifier → medial-axis skeleton of spine + length | Tracing | Density grid | High — Hessian robustness + clean skeleton | Spine-extraction pipeline pattern for hyphal networks | COWS | Pfeifer et al. 2022; doi:10.1093/mnras/stac1382 |
| **Hough transform (classic line)** | Vote edge pixels into (ρ,θ) accumulator; peaks = straight lines | Detection | Edge map / binary | Med — integrates faint colinear signal but assumes straight lines | Detect straight hyphal segments / fast-growing leaders | OpenCV/scikit-image | Duda & Hart 1972; doi:10.1145/361237.361242 |
| **Radon transform streak detection (FRT)** | PSF matched-filter then Radon integrate along all lines → faint streak peaks | Detection | Raw grayscale | **High** — detects streaks invisible to eye, no prior on angle/position | **Faint straight-hypha detection** by line integration | radon (Nir) | Nir et al. 2018; doi:10.3847/1538-3881/aaddff |
| **Matched filter (optimal source detection)** | Cross-correlate with template matched to signal shape → maximizes SNR | Detection/denoise | Raw grayscale + template | **High** — provably optimal SNR for known shape in Gaussian noise | **Faint-hypha enhancement** with a thin-line/PSF kernel | photutils/custom | Vio & Andreani 2016; doi:10.1051/0004-6361/201527925 |
| **ML ultra-faint streak detection** | ML over line params (matched to broadened line), tuned for sub-noise streaks | Detection | Raw grayscale | **High** — built for sub-visual streaks | Recover faintest individual hyphae below per-pixel SNR | custom | Nir et al. 2018; doi:10.3847/1538-3881/aaddff |
| **ASTRiDE (boundary-trace streak)** | Background-subtract → contour map → morphology (circularity, elongation) per border → keep streaks | Detection | Raw grayscale | Med — relies on contouring; background-sensitive | Shape-based separation of elongated hyphal objects from round colonies | ASTRiDE | Kim 2016; ascl:1605.009 |
| **DeepStreaks (CNN classifier)** | CNN ensemble labels streak cutouts (real vs artifact) | Detection (DL) | Image cutouts + labels | High (learned) — robust if training matches domain | **Learned faint-hypha vs artifact** classifier on patches | DeepStreaks | Duev et al. 2019; doi:10.1093/mnras/stz1096 |
| **U-Net trail segmentation (+Hough refine)** | Encoder-decoder pixel segmentation; Combo (BCE+Dice) loss for class imbalance; Hough cleans line | Detection (DL seg) | Image + pixel masks | High (learned) — Dice/Combo loss handles thin sparse positives | **Pixel-wise hyphal segmentation**; imbalance-aware loss transfers directly | U-Net (astro) | Jeong et al. 2024; doi:10.1051/0004-6361/202451663 |
| **LA Cosmic (Laplacian edge CR rejection)** | Laplacian flags sharp cosmic rays vs smoother PSF; iterative | Denoise | Raw grayscale | High — discriminates sharp artifacts from real structure at high σ | Reject sharp speckle/debris artifacts from hyphae | astroscrappy | van Dokkum 2001; doi:10.1086/323894 |
| **SExtractor thresholding + multi-thresh deblending** | Background mesh → σ-threshold → connected components → flux multi-thresholding splits blends | Detection + deblend | Raw grayscale | Med — robust background; thin faint structure near threshold fragile | Separate **overlapping/touching hyphae** via multi-threshold deblending | Source Extractor | Bertin & Arnouts 1996; doi:10.1051/aas:1996164 |
| **à trous / starlet wavelet denoising** | Isotropic undecimated wavelet → per-scale noise thresholding → reconstruct | Denoise | Raw grayscale | **High** — per-scale noise modeling; preserves faint structure | **Pre-denoise agar texture** while keeping thin hyphae | iSAP/MR | Starck, Fadili & Murtagh 2007; doi:10.1109/TIP.2006.887733 |
| **Multiscale Vision Model (MVM)** | Wavelet-space object detection: connect significant coefficients across scales | Detection + denoise | Raw grayscale | High — detects objects in denoised wavelet space, PSF-agnostic | Detect hyphal objects across scales without raw-px threshold | iSAP | Starck & Murtagh 2006 (book); doi:10.1007/978-3-540-33025-7 |
| **Curvelet / ridgelet transform** | Anisotropic multiscale basis tuned to lines/curves | Denoise/enhance | Raw grayscale | **High** for elongated features — anisotropic atoms match thin curves | Anisotropic denoise matched to **curved thin hyphae** | Curvelab/iSAP | Starck, Candès & Donoho 2002; doi:10.1109/TIP.2002.1014998 |
| **Mexican-hat / multiscale matched filter** | Convolve with Mexican-hat (LoG) at multiple scales; peaks = matched-scale sources | Detection | Raw grayscale | High — scale-matched, suppresses large-scale background | Blob/tip detection; scale-matched faint-feature enhancement | custom | López-Caniego et al. 2006; doi:10.1111/j.1365-2966.2006.10639.x |
| **DRUID (persistent-homology detect/deblend)** | Persistent homology → hierarchy of peaks; deblend via topological lifetimes | Detection + deblend | Raw grayscale | **High** — persistence gives noise-robust, threshold-free deblending | Topological deblending of **dense overlapping hyphae** | DRUID | Whitehead et al. 2024; arXiv:2410.22508 |
| **RANSAC line fitting** | Random sample consensus: fit line to inliers, reject outliers | Tracing | Point set / edge px | Med — robust to outliers; straight segments only | Fit straight hyphal segments amid spurious detections | scikit-image/custom | Fischler & Bolles 1981; doi:10.1145/358669.358692 |
| **Sigma-clip / median trail removal** | Iteratively clip pixels >Nσ across a stack/row → remove transient linear features | Denoise | Image stack / single | Med — needs redundancy or strong contrast | Remove scratches/fibers/transient linear artifacts | astropy sigma_clip | (std; van Dokkum 2001 context) doi:10.1086/323894 |
| **StreakDet (faint streak pipeline)** | Segmentation + clustering + line-param fit for faint/short streaks of unknown velocity | Detection | Raw grayscale | High — built for very faint short trails | Detect short faint hyphal fragments of unknown orientation | StreakDet | Virtanen et al. 2016; doi:10.1016/j.asr.2015.09.024 |

## How astronomy extracts thin faint structures from low-SNR noisy backgrounds

The recurring trick is that **integration beats thresholding** — robust methods accumulate faint
signal over many pixels (along a line, across scales, or over topology) *before* committing to a
detection, so SNR grows as √(pixels integrated) rather than being lost per-pixel.

- **Matched filtering / template integration (optimal SNR).** Cross-correlating with a target-shaped
  template is the provably optimal linear detector in Gaussian noise — it sums the whole feature's
  flux instead of relying on low-SNR edge pixels (Vio & Andreani 2016). Transfers very well: a
  thin-line kernel matched to hyphal width is a cheap, principled faint-hypha enhancer.
- **Radon/Hough accumulation (faint straight features).** The Radon transform integrates along every
  possible line; a per-pixel-invisible streak becomes a clear peak because sub-noise pixels add
  coherently (Nir et al. 2018). Transfers to straight/gently-curved hyphae and growth leaders.
- **Rolling Hough Transform — orientation in noise.** RHT runs a local Hough per pixel and outputs
  the probability of belonging to a coherent linear structure plus orientation (Clark et al. 2014).
  It asks "is there local linear coherence?" not "is this pixel bright?", so faint coherent hyphae
  survive while incoherent agar texture averages out. One of the strongest analogs — a hyphal
  anisotropy/orientation field at low SNR.
- **Morse-theory + persistent-homology thresholding.** DisPerSE finds filaments as ridge lines
  between critical points and prunes by **persistence** (density contrast birth→death), giving each
  branch a statistical significance and removing noise features with a confidence rather than a
  brightness cut (Sousbie 2011). DRUID applies the same idea to *deblend* dense overlapping sources
  (Whitehead et al. 2024). A noise-robust skeletonizer that won't hallucinate branches from texture.
- **Multiscale Hessian ridge / tube enhancement.** NEXUS+/MMF (Cautun 2013), Frangi vesselness
  (1998), and Steger's detector (1998 — adds **subpixel position + width**) flag tubular geometry by
  eigenvalue ratios, suppressing blobs and isotropic noise across a scale ladder. Steger is the best
  bet for subpixel hyphal centerline + diameter at poor resolution (already standard in biomedical
  vessel/neurite tracing).
- **Wavelet/starlet per-scale denoising.** The à trous transform models noise per scale and thresholds
  via a multiresolution support (Starck et al. 2007); curvelets/ridgelets add anisotropic atoms matched
  to thin curved edges (Starck et al. 2002). A pre-denoise step that strips agar texture while keeping
  thin hyphae.
- **Multi-threshold deblending.** SExtractor splits blended objects at saddle points via a flux
  multi-thresholding tree (Bertin & Arnouts 1996) — separating touching hyphae by relative-brightness
  topology; fragments near the noise floor, so pair with wavelet denoising or persistence deblending.
- **Statistical/stochastic models.** Bisous (Tempel 2014) and SCMS (Chen 2015) average over many
  configurations and yield per-branch uncertainty, but assume **sparse, well-separated tracers** — the
  weakest assumption for dense hyphae, so use only on sparse high-confidence detection clouds.
- **Deep learning with imbalance-aware loss.** U-Net trail segmentation with **Combo (BCE+Dice) loss**
  is built for thin sparse positives (Jeong et al. 2024); the loss design transfers directly to
  pixel-wise hyphal segmentation, and the U-Net→Hough refinement (learned mask → classical line cleanup)
  is a strong template.

**Best transfers to dense, overlapping, faint hyphae (ranked):**
1. **Steger unbiased curvilinear detector** — subpixel centerline + width at poor resolution, density-tolerant. *Top pick for tracing.*
2. **Rolling Hough Transform** — noise-robust orientation/anisotropy field at low SNR. *Top pick for orientation.*
3. **Multiscale Hessian ridge (NEXUS+/Frangi-style)** — variable-width tube enhancement before segmentation.
4. **DisPerSE persistence + DRUID topological deblending** — confidence-ranked skeleton and deblending that won't invent branches from texture.
5. **Starlet/curvelet denoise + matched/Radon line filtering** — the SNR-lifting front-end pair.
6. **U-Net with Dice/Combo loss** — the learned route once labels exist.

*Lower-fit (assumption mismatch):* Bisous, SCMS, MST, T-/V-web assume sparse separated tracers and
degrade on dense overlapping fields.

## Tools / pipelines

- **FilFinder** — skeletonize ISM filaments, profile widths, length/curvature/branching. https://github.com/e-koch/FilFinder
- **DisPerSE** — Morse-theory + persistence filament/cosmic-web extraction; noise-robust, parameter-free.
- **RHT** — per-pixel linear-coherence + orientation in noise. https://github.com/seclark/RHT
- **getsf / getfilaments** — parameter-free multiscale source+filament extraction with background separation.
- **NEXUS+ / MMF / COWS** — multiscale-Hessian cosmic-web filament classifiers (+ skeleton in COWS).
- **SCONCE-SCMS** — density-ridge filament finder with per-branch uncertainty. https://github.com/zhangyk8/sconce-scms
- **Bisous** — marked-point-process stochastic filament finder.
- **Source Extractor (SExtractor)** — thresholding + multi-threshold deblending. https://github.com/astromatic/sextractor
- **ASTRiDE** — boundary-trace streak detection. https://github.com/dwkim78/ASTRiDE
- **DeepStreaks** — CNN-ensemble streak classifier. https://github.com/dmitryduev/DeepStreaks
- **astroscrappy / L.A.Cosmic** — Laplacian-edge sharp-artifact rejection. https://github.com/astropy/astroscrappy
- **iSAP / MR** — starlet/curvelet denoising + Multiscale Vision Model. http://www.cosmostat.org/software/isap
- **DRUID** — persistent-homology source detection + deblending. https://github.com/RhysAlfShaw/DRUID

## References (APA)

- Arzoumanian, D., André, P., Didelon, P., et al. (2011). Characterizing interstellar filaments with *Herschel* in IC 5146. *A&A, 529*, L6. https://doi.org/10.1051/0004-6361/201116596
- Barrow, J. D., Bhavsar, S. P., & Sonoda, D. H. (1985). Minimal spanning trees, filaments and galaxy clustering. *MNRAS, 216*(1), 17–35. https://doi.org/10.1093/mnras/216.1.17
- Bertin, E., & Arnouts, S. (1996). SExtractor: Software for source extraction. *A&AS, 117*, 393–404. https://doi.org/10.1051/aas:1996164
- Cautun, M., van de Weygaert, R., & Jones, B. J. T. (2013). NEXUS: Tracing the cosmic web connection. *MNRAS, 429*(2), 1286–1308. https://doi.org/10.1093/mnras/sts416
- Chen, Y.-C., Ho, S., Freeman, P. E., Genovese, C. R., & Wasserman, L. (2015). Cosmic web reconstruction through density ridges. *MNRAS, 454*(1), 1140–1156. https://doi.org/10.1093/mnras/stv1996
- Clark, S. E., Peek, J. E. G., & Putman, M. E. (2014). Magnetically aligned H I fibers and the Rolling Hough Transform. *ApJ, 789*(1), 82. https://doi.org/10.1088/0004-637X/789/1/82
- Duda, R. O., & Hart, P. E. (1972). Use of the Hough transformation to detect lines and curves in pictures. *CACM, 15*(1), 11–15. https://doi.org/10.1145/361237.361242
- Duev, D. A., Mahabal, A., Ye, Q., et al. (2019). DeepStreaks. *MNRAS, 486*(3), 4158–4165. https://doi.org/10.1093/mnras/stz1096
- Fischler, M. A., & Bolles, R. C. (1981). Random sample consensus (RANSAC). *CACM, 24*(6), 381–395. https://doi.org/10.1145/358669.358692
- Forero-Romero, J. E., Hoffman, Y., Gottlöber, S., Klypin, A., & Yepes, G. (2009). A dynamical classification of the cosmic web. *MNRAS, 396*(3), 1815–1824. https://doi.org/10.1111/j.1365-2966.2009.14885.x
- Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A. (1998). Multiscale vessel enhancement filtering. *MICCAI 1998, LNCS 1496*, 130–137. https://doi.org/10.1007/BFb0056195
- Hoffman, Y., Metuki, O., Yepes, G., et al. (2012). A kinematic classification of the cosmic web. *MNRAS, 425*(3), 2049–2057. https://doi.org/10.1111/j.1365-2966.2012.21789.x
- Jeong, M., et al. (2024). Automated detection of satellite trails using U-Net and Hough transform. *A&A, 692*, A106. https://doi.org/10.1051/0004-6361/202451663
- Kim, D.-W. (2016). *ASTRiDE: Automated Streak Detection for Astronomical Images* [Software]. ascl:1605.009. https://ascl.net/1605.009
- Koch, E. W., & Rosolowsky, E. W. (2015). Filament identification through mathematical morphology. *MNRAS, 452*(4), 3435–3450. https://doi.org/10.1093/mnras/stv1521
- Libeskind, N. I., van de Weygaert, R., Cautun, M., et al. (2018). Tracing the cosmic web. *MNRAS, 473*(1), 1195–1217. https://doi.org/10.1093/mnras/stx1976
- López-Caniego, M., Herranz, D., González-Nuevo, J., et al. (2006). Comparison of filters for the detection of point sources. *MNRAS, 370*(4), 2047–2063. https://doi.org/10.1111/j.1365-2966.2006.10639.x
- Men'shchikov, A. (2013). getfilaments: A multi-scale filament extraction method. *A&A, 560*, A63. https://doi.org/10.1051/0004-6361/201321885
- Men'shchikov, A. (2021). getsf: extraction of sources and filaments. *A&A, 649*, A89. https://doi.org/10.1051/0004-6361/202039913
- Nir, G., Ofek, E. O., Gal-Yam, A., et al. (2018). Optimal and efficient streak detection in astronomical images. *AJ, 156*(5), 229. https://doi.org/10.3847/1538-3881/aaddff
- Pfeifer, S., Libeskind, N. I., Hoffman, Y., et al. (2022). COWS: A filament finder for Hessian cosmic web identifiers. *MNRAS, 514*(1), 470–485. https://doi.org/10.1093/mnras/stac1382
- Sousbie, T. (2011). The persistent cosmic web — I. Theory and implementation. *MNRAS, 414*(1), 350–383. https://doi.org/10.1111/j.1365-2966.2011.18394.x
- Starck, J.-L., Candès, E. J., & Donoho, D. L. (2002). The curvelet transform for image denoising. *IEEE TIP, 11*(6), 670–684. https://doi.org/10.1109/TIP.2002.1014998
- Starck, J.-L., Fadili, J., & Murtagh, F. (2007). The undecimated wavelet decomposition and its reconstruction. *IEEE TIP, 16*(2), 297–309. https://doi.org/10.1109/TIP.2006.887733
- Steger, C. (1998). An unbiased detector of curvilinear structures. *IEEE TPAMI, 20*(2), 113–125. https://doi.org/10.1109/34.659930
- Tempel, E., Stoica, R. S., Martínez, V. J., et al. (2014). Detecting filamentary pattern in the cosmic web (SDSS). *MNRAS, 438*(4), 3465–3482. https://doi.org/10.1093/mnras/stt2454
- van Dokkum, P. G. (2001). Cosmic-ray rejection by Laplacian edge detection. *PASP, 113*(789), 1420–1427. https://doi.org/10.1086/323894
- Vio, R., & Andreani, P. (2016). A statistical analysis of the "detection problem" via the matched filter. *A&A, 589*, A20. https://doi.org/10.1051/0004-6361/201527925
- Virtanen, J., Poikonen, J., Säntti, T., et al. (2016). Streak detection and analysis pipeline for space-debris optical images. *AdSpR, 57*(8), 1607–1623. https://doi.org/10.1016/j.asr.2015.09.024
- Whitehead, R. A. S., et al. (2024). DRUID: Source detection and deblending with persistent homology. *arXiv:2410.22508*. https://doi.org/10.48550/arXiv.2410.22508

**Confidence note (from source agent):** FilFinder, DisPerSE, RHT, getsf, LA Cosmic, Nir streak, Tempel
Bisous, Chen SCMS, COWS, and the cosmic-web review DOIs were verified by literature-index lookup. The
remaining canonical references are widely-cited standards reported from established records; a few exact
volume/article numbers (Arzoumanian L6, Jeong A&A 692/A106, getfilaments A&A 560/A63) should be confirmed
against NASA ADS before any external publication. No citations were fabricated.
