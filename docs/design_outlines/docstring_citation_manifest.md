# Docstring Enrichment — Literature & Range Manifest

- Run date: 2026-06-05
- Scope: all families
- Policy: peer-reviewed sources only; no preprints/archives. Every literature claim triple-verified (2 blind verifiers + opus adjudicator, strict consensus).

## Citations (59)

| Family | Class | Claim | DOI | Resolved title | Verdict | Reviewed |
|---|---|---|---|---|---|---|
| threshold-detector | CannyDetector | Canny edge detection applies Gaussian smoothing, gradient estimation, non-maximum suppression, and dual-threshold hysteresis to produce thin edge pixels; this implementation then labels connected components of the edge or inverted-edge map. | 10.1109/TPAMI.1986.4767851 | A Computational Approach to Edge Detection | SURVIVES for the four-stage Canny algorithm — DOI resolves to IEEE Xplore (peer-reviewed IEEE TPAMI 1986, not an archive); connected-component labeling of the edge/inverted-edge map is implementation-specific and should not be attributed to Canny as closed-contour filling. | ☐ |
| threshold-detector | CannyDetector | A 2:1 to 3:1 high-to-low threshold ratio is effective for moderately noisy images. | 10.1109/TPAMI.1986.4767851 | A Computational Approach to Edge Detection | SURVIVES — both verifiers cite the Edinburgh CVonline summary of Canny (1986) confirming the 2:1–3:1 recommendation; widely corroborated (OpenCV docs). One independent secondary mirror (HIPR2) did not state the ratio explicitly, but strict consensus across my verdict + both verifiers is supported; wording kept conservative ('effective for moderately noisy images'). | ☐ |
| threshold-detector | WatershedDetector | Compact-watershed compactness regularises segment shapes; higher values produce more geometrically regular, convex segments (lower values let boundaries follow the Sobel gradient freely). | 10.1109/ICPR.2014.181 | Compact Watershed and Preemptive SLIC: On Improving Trade-offs of Superpixel Segmentation Algorithms | SURVIVES (as already worded) — DOI resolves to IEEE Xplore (peer-reviewed ICPR 2014 proceedings, not an archive). Both verifiers flagged the DOSSIER's 'geodesic distance' framing as overstated (skimage uses Euclidean distance to seed), but the ACTUAL docstring never says 'geodesic' — it uses the conservative 'shape-regularisation penalty ... more geometrically regular, convex segments,' which both verifiers explicitly endorse. No revision needed. | ☐ |
| threshold-detector | WatershedDetector | Otsu threshold is computed only from non-zero pixels when ignore_zeros=True; structural zeros from black borders or pre-masked regions would otherwise bias the bimodal histogram assumed by Otsu's method. | 10.1109/TSMC.1979.4310076 | A Threshold Selection Method from Gray-Level Histograms | SURVIVES — DOI is genuine IEEE TSMC 1979 (peer-reviewed, not archive); both verifiers confirm Otsu's bimodal assumption and that structural zeros form a spurious mode biasing the threshold. Code (lines 160-174) confirms masked-array Otsu on non-zero pixels. Strict consensus: supported. | ☐ |
| detector | ChanVeseDetector | mu is the contour-length penalty weight in the Chan-Vese energy functional; higher values produce shorter, smoother outlines | 10.1109/83.902291 | Active contours without edges | SURVIVES — both verifiers + adjudicator agree supported; mu weights the Length(C) term in the energy functional (peer-reviewed, IEEE TIP, not archive). | ☐ |
| detector | ChanVeseDetector | mu typical range 0--1 (lower for diffuse mucoid edges, higher for noisy plates) | 10.5201/ipol.2012.g-cv | scikit-image chan_vese documentation / Getreuer Chan-Vese Segmentation | SURVIVES as library-derived — the 0--1 range is NOT in Chan & Vese (2001) but is explicitly documented by scikit-image ('typical values for mu are between 0 and 1'). Verified live against scikit-image docs. Range retained; attribution treated as library, not the 2001 paper. | ☐ |
| detector | ChanVeseDetector | lambda1 = lambda2 = 1.0 are the standard default fidelity weights (now: standard default recommended by Chan & Vese 2001) | 10.1109/83.902291 | Active contours without edges | SURVIVES — both verifiers + adjudicator agree supported; Chan & Vese state 'We will take lambda1 = lambda2 = 1', confirmed by scikit-image docs ('typical values for lambda1 and lambda2 are 1'). | ☐ |
| detector | ChanVeseDetector | lambda2 default 1.0; increasing lambda2 relative to lambda1 penalises background heterogeneity more strongly (follows directly from the energy functional); useful when agar is uniform but colony texture is heterogeneous | 10.5201/ipol.2012.g-cv | Chan-Vese Segmentation | SURVIVES as REVISED — the directional effect is presented as a derivation from the energy functional, not an empirical claim; default 1.0 confirmed by source. Original 'enforces tighter background homogeneity' softened to 'penalises background heterogeneity more strongly (follows directly from the energy functional)'. | ☐ |
| detector | ChanVeseDetector | init_level_set checkerboard (sin x sin pattern) provides fast multi-front convergence well-suited to arrayed plates | 10.5201/ipol.2012.g-cv | Chan-Vese Segmentation | SURVIVES — both verifiers + adjudicator agree supported; Getreuer documents checkerboard initialisation has fast convergence; multi-front/arrayed-plate framing is a sound application extension. | ☐ |
| detector | FilamentousFungiDetector | edge_noise_threshold is the k multiplier for phase congruency: features accepted only when phase energy exceeds the noise mean plus k standard deviations of noise energy | 10.1007/s004260000024 | Phase congruency: A low-level image invariant | SURVIVES as REVISED — Kovesi (2000, Psychological Research, peer-reviewed, not archive) defines k as 'number of standard deviations of noise energy beyond the mean'. V2 flagged 'k × noise floor' as conflating mean with floor; revised to 'noise mean plus k standard deviations', matching the source and the code's own dossier formula (mean + k·sigma). | ☐ |
| detector | FilamentousFungiDetector | Orientation coherence measures local directional alignment via the structure tensor; larger coherence_window_radius captures longer-range directional consistency | 10.1023/A:1008009714131 | Coherence-Enhancing Diffusion Filtering | SURVIVES — both verifiers + adjudicator agree supported; Weickert (1999, IJCV, peer-reviewed) defines coherence from structure-tensor eigenvalues; integration-window scale controls the spatial range of directional assessment. | ☐ |
| detector | FilamentousFungiDetector | Disconnected branch fragments are reconnected via multi-source Dijkstra minimum-cost pathfinding on a composite cost surface | 10.1007/BF01386390 | A note on two problems in connexion with graphs | SURVIVES — both verifiers + adjudicator agree supported; Dijkstra (1959, Numerische Mathematik, peer-reviewed) is the correct seminal shortest-path reference. | ☐ |
| refiner | TrimAsymmetry | The PELT inoculum-core penalty is on the BIC scale, of order log(n_annuli) for the mean-change (l2) model. | 10.1080/01621459.2012.737745 | Optimal Detection of Changepoints With a Linear Computational Cost | survived (revised to conservative wording): the specific '≈4.6 floor / 5.0 just above it' numeric framing and the 'moderately conservative' characterization were CUT because both blind verifiers flagged them as overstated/permissive. Independent verification of ruptures CostL2 (centre-borelli docs) confirms it estimates only the mean (1 free parameter), so log(n) is the correct base BIC factor for this model — retained as 'order log(n_annuli) for the mean-change (l2) model', which survives all three assessments. | ☐ |
| enhancer | BayesShrinkEnhancer | Per-subband BayesShrink threshold T_B = sigma^2/sigma_signal with soft shrinkage minimising Bayesian risk | 10.1109/83.862633 | Adaptive wavelet thresholding for image denoising and compression | supported - both verifiers + adjudicator agree; primary BayesShrink source, peer-reviewed IEEE TIP | ☐ |
| enhancer | BayesShrinkEnhancer | Noise sigma auto-estimated via MAD of finest-scale HH subband, median(\|coeff\|)/0.6745 | 10.1093/biomet/81.3.425 | Ideal spatial adaptation by wavelet shrinkage | supported - both verifiers + adjudicator agree; Donoho & Johnstone Biometrika, peer-reviewed | ☐ |
| enhancer | BayesShrinkEnhancer | GAT with gain/mu/read_sigma maps Poisson-Gaussian noise to ~unit-variance Gaussian; exact unbiased inverse applied after denoising | 10.1109/TIP.2012.2202675 | Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise | supported - both verifiers + adjudicator agree; Makitalo & Foi IEEE TIP, peer-reviewed | ☐ |
| enhancer | BM3DDenoiser | BM3D groups similar patches and filters in 3-D transform domain; two-stage HT-then-Wiener yields higher PSNR/sharper boundaries than HT alone | 10.1109/TIP.2007.901238 | Image denoising by sparse 3-D transform-domain collaborative filtering | supported - both verifiers + adjudicator agree; primary BM3D source, peer-reviewed IEEE TIP | ☐ |
| enhancer | BM3DDenoiser | GAT with gain/mu/read_sigma maps Poisson-Gaussian noise to ~unit-variance Gaussian; exact unbiased inverse applied after denoising | 10.1109/TIP.2012.2202675 | Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise | supported - both verifiers + adjudicator agree; Makitalo & Foi IEEE TIP, peer-reviewed | ☐ |
| enhancer | FocusEdgeFrangi | Frangi defaults alpha=0.5, beta=0.5 (authors' experimental values); gamma adaptive as half max Hessian Frobenius norm per scale; vesselness = max across scales | 10.1007/BFb0056195 | Multiscale vessel enhancement filtering | supported - both verifiers + adjudicator agree; primary Frangi source, peer-reviewed MICCAI/Springer LNCS | ☐ |
| enhancer | FocusEdgeFrangi | In 2-D, alpha has no numerical effect (plate-sensitivity ratio undefined/omitted from 2-D formula) | 10.1007/BFb0056195 | Multiscale vessel enhancement filtering | supported-after-revision - verifier1 supported, verifier2 flagged 'reduces to infinity' as post-hoc; revised docstring to 'undefined and omitted from the 2-D vesselness formula', which both assessments accept | ☐ |
| enhancer | FocusEdgeFrangi | Vesselness is per-pixel max across scales, so adding sigmas is monotone-additive (can only raise response) | 10.1007/BFb0056195 | Multiscale vessel enhancement filtering | supported - both verifiers + adjudicator agree; logical consequence of max-across-scales formulation | ☐ |
| enhancer | FocusEdgeHessian | Frangi defaults alpha=0.5, beta=0.5 were the authors' experimental settings | 10.1007/BFb0056195 | Multiscale vessel enhancement filtering | supported - both verifiers + adjudicator agree on the alpha/beta=0.5 portion (peer-reviewed). The skimage fixed gamma=15 detail is a library fact and is NOT attributed to Frangi 1998 in the shipped docstring prose (gamma Arg states Default:15 without citation). | ☐ |
| enhancer | FocusEdgeHessian | In 2-D, alpha has no numerical effect (plate-sensitivity ratio undefined/omitted from 2-D formula) | 10.1007/BFb0056195 | Multiscale vessel enhancement filtering | supported-after-revision - verifier1 supported, verifier2 overstated; revised to 'undefined and omitted from the 2-D formula' | ☐ |
| enhancer | FocusEdgeMeijering | Meijering neuriteness uses analytic shape parameter alpha = -1/(ndim+1) = -1/3 for 2-D, maximally suppressing blob-like/isotropic structures while favouring elongated ridges | 10.1002/cyto.a.20022 | Design and validation of a tool for neurite tracing and analysis in fluorescence microscopy images | supported - both verifiers + adjudicator agree; primary Meijering source, peer-reviewed Cytometry Part A; corroborated by skimage meijering() docs | ☐ |
| enhancer | FocusEdgeMeijering | Per-pixel maximum taken across scales, so sigmas tuple should span the full expected filament-width range | 10.1002/cyto.a.20022 | Design and validation of a tool for neurite tracing and analysis in fluorescence microscopy images | supported - both verifiers + adjudicator agree; consistent with the multi-scale ridge-filter framework and skimage implementation | ☐ |
| enhancer | FocusEdgePhase | Phase congruency detects features where Fourier components come into phase / local energy is maximal; response invariant to image amplitude | 10.1016/0167-8655(87)90013-4 | Feature detection from local energy | supported - both verifiers + adjudicator agree; Morrone & Owens, peer-reviewed Pattern Recognition Letters | ☐ |
| enhancer | FocusEdgePhase | Local Energy model: salient features (edges/lines) occur where Fourier components are maximally in phase; local energy maxima coincide | 10.1098/rspb.1988.0073 | Feature detection in human vision: a phase-dependent energy model | supported - both verifiers + adjudicator agree; Morrone & Burr, peer-reviewed Proc R Soc B | ☐ |
| enhancer | FocusEdgePhase | Phase congruency is contrast-invariant; Rayleigh noise-energy statistics underpin k and noise_method compensation | 10.1007/s004260000024 | Phase congruency: A low-level image invariant | supported - both verifiers + adjudicator agree; Kovesi 2000, peer-reviewed Psychological Research | ☐ |
| enhancer | FocusEdgePhase | Log-Gabor filters (Field 1987) provide the filter-bank basis used in FocusEdgePhase | 10.1364/JOSAA.4.002379 | Relations between the statistics of natural images and the response properties of cortical cells | supported - shipped docstring reference [4] cites Field for log-Gabor motivation only; verifier1's overstatement concern (attributing the sigma_onf parameterisation itself to Field) does not appear in the shipped prose, so the citation as used is supported by both assessments | ☐ |
| enhancer | FocusEdgeSato | Each sigma responds maximally to ridges whose cross-sectional half-width matches that value (multi-scale Sato tubeness). | 10.1016/S1361-8415(98)80009-1 | Three-dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medical images | SURVIVES — both verifiers and my own check agree the Sato 1998 multi-scale Hessian scale-space property supports sigma matching ridge half-width; peer-reviewed (Med. Image Anal.), not a preprint. | ☐ |
| enhancer | LocalEdgeDenoise | Bilateral filter primary method reference (spatial + intensity Gaussian weighting). | 10.1109/ICCV.1998.710815 | Bilateral filtering for gray and color images | SURVIVES — DOI resolves to Tomasi & Manduchi ICCV 1998 (peer-reviewed conference); both verifiers concur. | ☐ |
| enhancer | LocalEdgeDenoise | Optimal sigma_color is approximately proportional to the image noise standard deviation; GAT most beneficial when shot (Poisson) noise dominates. | 10.1109/TIP.2012.2202675 | Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise | SURVIVES — IEEE TIP, peer-reviewed; GAT/Poisson-Gaussian framing supported. (sigma_color~noise proportionality additionally backed by 10.1109/TIP.2008.2006658, also peer-reviewed and confirmed by both verifiers.) | ☐ |
| enhancer | NonLocalMeansDenoiser | Rule of thumb: h approximately equals the noise standard deviation; h is the most critical NLM parameter. | 10.5201/ipol.2011.bcm_nlm | Non-Local Means Denoising | SURVIVES — IPOL is peer-reviewed; DOI independently resolved; h=k*sigma rule supported by both verifiers and my fetch. | ☐ |
| enhancer | NonLocalMeansDenoiser | GAT wrapping benefits low-light fluorescence imaging where Poisson photon noise dominates. | 10.1109/ICIP.2010.5653394 | Poisson NL means: Unsupervised non local means for Poisson noise | SURVIVES — IEEE ICIP 2010 peer-reviewed conference (not preprint); both verifiers confirm; verifier 2 notes confocal fluorescence demonstration directly supports the scene claim. | ☐ |
| enhancer | StructureSmoothing | Two-scale structure tensor: noise scale sigma for gradient computation, integration scale rho for tensor smoothing; alpha is the minimum diffusivity (alpha<<1, typ 0.001) maximising anisotropy. | 10.1023/A:1008009714131 | Coherence-Enhancing Diffusion Filtering | SURVIVES — Weickert IJCV 1999, peer-reviewed; both verifiers confirm the two-scale tensor and alpha minimum-diffusivity formulation. This is the only literature DOI cited in the file. | ☐ |
| enhancer | VisuShrinkEnhancer | Universal threshold T = sigma*sqrt(2*log(N)) is near-minimax optimal for Gaussian white noise; soft thresholding produces continuous output without Gibbs-like ringing. | 10.1093/biomet/81.3.425 | Ideal spatial adaptation by wavelet shrinkage | SURVIVES — Donoho & Johnstone 1994 Biometrika, peer-reviewed; both verifiers confirm the universal threshold and soft-threshold properties. | ☐ |
| enhancer | VisuShrinkEnhancer | GAT stabilizes Poisson-Gaussian noise variance to ~1, making VisuShrink's Gaussian assumption valid. | 10.1109/TIP.2012.2202675 | Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise | SURVIVES — IEEE TIP, peer-reviewed; both verifiers confirm variance stabilization framing. | ☐ |
| corrector | BayesShrinkCorrector | BayesShrink estimates a separate per-subband threshold from the data, using the MAD (median(\|coeff\|)/0.6745) of the finest detail subband for noise estimation. | 10.1109/83.862633 | Adaptive wavelet thresholding for image denoising and compression | survived — strict consensus supported (both verifiers + adjudicator); IEEE TIP, peer-reviewed, not archive | ☐ |
| corrector | BayesShrinkCorrector | MAD-based noise estimation: sigma = median(\|detail_coeffs\|)/0.6745 from the finest-scale subband (auto sigma=None path). | 10.1093/biomet/81.3.425 | Ideal spatial adaptation by wavelet shrinkage | survived — strict consensus supported; Biometrika, peer-reviewed, not archive | ☐ |
| corrector | BayesShrinkCorrector | Soft thresholding produces smoother, more continuous output (consistent with BayesShrink's Bayesian risk derivation); hard thresholding retains amplitude but introduces discontinuities that can appear as ringing near sharp boundaries. | 10.1109/83.862633 | Adaptive wavelet thresholding for image denoising and compression | survived AS REVISED — softened from 'Gibbs-like ringing at colony boundaries' to 'discontinuities that can appear as ringing near sharp boundaries' (one verifier flagged overstated; conservative general-wavelet-theory wording retained, no over-specific colony-edge attribution) | ☐ |
| corrector | BayesShrinkCorrector | GAT converts Poisson-Gaussian mixed noise to approximately unit-variance Gaussian; sigma retargeted to 1.0 in the stabilized domain; the exact unbiased inverse GAT restores the original intensity scale. | 10.1109/TIP.2012.2202675 | Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise | survived — strict consensus supported; IEEE TIP, peer-reviewed, not archive | ☐ |
| corrector | ColorDenoise | CBM3D decorrelates color into a luminance-chrominance space and computes patch grouping once on the luminance channel for reuse across the chrominance channels. | 10.1109/ICIP.2007.4378954 | Color image denoising via sparse 3D collaborative filtering with grouping constraint in luminance-chrominance space | survived AS REVISED — both verifiers flagged 'opponent space' and 'preventing color fringing' as overstated/interpretive; revised to paper-faithful 'luminance-chrominance space' + 'grouping reused across chrominance channels' (the supported core both verifiers confirmed). IEEE ICIP, peer-reviewed, not archive | ☐ |
| corrector | ColorDenoise | After forward GAT the RGB channel noise is approximately unit-variance Gaussian, so sigma_psd=1.0 is correct in the stabilized domain; exact unbiased inverse GAT restores the signal. | 10.1109/TIP.2012.2202675 | Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise | survived — strict consensus supported; IEEE TIP, peer-reviewed, not archive | ☐ |
| corrector | StableDenoise | Forward GAT converts Poisson-Gaussian mixed noise to approximately unit-variance Gaussian so BM3D runs at sigma_psd=1.0; exact unbiased inverse GAT restores the original scale. | 10.1109/TIP.2012.2202675 | Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise | survived — strict consensus supported; IEEE TIP, peer-reviewed, not archive | ☐ |
| corrector | StableDenoise | BM3D two-stage pipeline: hard-thresholding produces a basic estimate used as the pilot for second-stage Wiener collaborative filtering. | 10.1109/TIP.2007.901238 | Image denoising by sparse 3-D transform-domain collaborative filtering | survived — strict consensus supported; IEEE TIP, peer-reviewed, not archive | ☐ |
| corrector | VisuShrinkCorrector | VisuShrink applies the universal threshold T = sigma*sqrt(2*log(N)), near-minimax optimal for Gaussian white noise but conservative, tending to over-smooth versus adaptive alternatives. | 10.1093/biomet/81.3.425 | Ideal spatial adaptation by wavelet shrinkage | survived — strict consensus supported; Biometrika, peer-reviewed, not archive | ☐ |
| corrector | VisuShrinkCorrector | Noise auto-estimation via MAD of finest-scale detail coefficients: sigma = median(\|d\|)/0.6745. | 10.1093/biomet/81.3.425 | Ideal spatial adaptation by wavelet shrinkage | survived — strict consensus supported; Biometrika, peer-reviewed, not archive | ☐ |
| corrector | VisuShrinkCorrector | Soft mode produces smoother, more continuous output; hard mode retains sharper edges but introduces discontinuities that can appear as ringing near sharp boundaries. | 10.1093/biomet/81.3.425 | Ideal spatial adaptation by wavelet shrinkage | survived AS REVISED — softened from 'Gibbs-like ringing near high-contrast colony boundaries' (one verifier flagged overstated) to conservative general-theory wording | ☐ |
| corrector | VisuShrinkCorrector | GAT converts Poisson-Gaussian noise to approximately unit-variance Gaussian; sigma retargeted to 1.0 in stabilized domain; exact unbiased inverse GAT restores the signal. | 10.1109/TIP.2012.2202675 | Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise | survived — strict consensus supported; IEEE TIP, peer-reviewed, not archive | ☐ |
| detector | RoundPeaksDetector | Implements the gitter algorithm: threshold, project row/column intensity sums, find periodic peaks, infer grid edges, assign one colony per grid cell. | 10.1534/g3.113.009431 | gitter: A Robust and Accurate Method for Quantification of Colony Sizes From Plate Images | SURVIVES (corrected). Resolved DOI confirms G3 (Bethesda) 4(3):547-552, 2014, peer-reviewed (Genetics Society of America, not a preprint). Source confirms grid detection via row/column foreground-pixel intensity projection and peak finding. Editor/dossier-asserted DOI 10.1534/g3.114.010595 was WRONG (it resolves to an unrelated mouse-genetics paper, 'A Strategy to Identify Dominant Point Mutant Modifiers of a Quantitative Trait,' G3 4(6):1113-1121). The IEEE citation text (vol 4/no 3/pp 547-552) was already correct; added the correct DOI 10.1534/g3.113.009431. | ☐ |
| detector | SinePeakDetector | Extends the gitter row/column projection grid-finding approach; the project adds a sinusoidal template and rank (Spearman) cross-correlation for robustness to outlier colonies. Cites gitter as the foundational method. | 10.1534/g3.113.009431 | gitter: A Robust and Accurate Method for Quantification of Colony Sizes From Plate Images | SURVIVES (corrected). gitter is correctly cited as the basis (peer-reviewed G3 4(3):547-552, 2014). Verified that gitter itself uses row/column intensity projection and correlation (Pearson) against an expected colony profile, but does NOT use a sinusoidal template or Spearman rank cross-correlation — those are this class's own extensions (confirmed in code: rankdata + np.sin template in _estimate_edges/_normalized_cross_correlation). The docstring frames the sine/Spearman parts as the implementation and cites gitter only for the projection-based grid-finding foundation, which is accurate. Corrected DOI from the wrong dossier-asserted 10.1534/g3.114.010595 to 10.1534/g3.113.009431. | ☐ |
| threshold-detector | IsodataDetector | ISODATA thresholding iteratively partitions pixels into foreground/background by class means until convergence. | 10.1109/TSMC.1978.4310039 | Picture thresholding using an iterative selection method | SURVIVES — IEEE Trans. SMC 8(8):630-632, 1978; peer-reviewed journal (resolves to IEEE Xplore doc 4310039), not a preprint; supports the iterative class-mean refinement claim exactly. | ☐ |
| threshold-detector | LiDetector | Li's minimum cross-entropy thresholding iteratively refines a threshold minimising information loss between the original and binarised intensity distributions. | 10.1016/0031-3203(93)90115-D | Minimum cross entropy thresholding | SURVIVES — Pattern Recognition 26(4):617-625, 1993; peer-reviewed Elsevier journal, not a preprint; supports the cross-entropy minimisation claim exactly. | ☐ |
| threshold-detector | OtsuDetector | Otsu's method finds the global threshold by maximising between-class (minimising intra-class) intensity variance. | 10.1109/TSMC.1979.4310076 | A threshold selection method from gray-level histograms | SURVIVES — IEEE Trans. SMC 9(1):62-66, 1979; peer-reviewed journal, not a preprint; supports the variance-criterion claim exactly. | ☐ |
| threshold-detector | SecondaryOtsuDetector | Per-object Otsu re-thresholding applies Otsu's variance-minimisation principle independently to each detected region. | 10.1109/TSMC.1979.4310076 | A threshold selection method from gray-level histograms | SURVIVES — IEEE Trans. SMC 9(1):62-66, 1979; peer-reviewed, not a preprint; the per-object application is a sound extension of the cited Otsu criterion and the docstring frames it as building on Otsu, not as a separate published result. | ☐ |
| threshold-detector | TriangleDetector | Triangle thresholding computes the threshold at the base of the triangle formed between the histogram peak, minimum, and maximum. | 10.1177/25.7.70454 | Automatic measurement of sister chromatid exchange frequency | SURVIVES — J. Histochem. Cytochem. 25(7):741-753, 1977 (PubMed 70454); peer-reviewed journal, not a preprint; this is the canonical origin of the triangle/geometric thresholding method and supports the claim. | ☐ |
| threshold-detector | YenDetector | Yen's method selects a threshold maximising the squared correlation coefficient between the original intensity image and its binarised version. | 10.1109/83.366472 | A new criterion for automatic multilevel thresholding | SURVIVES — IEEE Trans. Image Process. 4(3):370-378, 1995 (PubMed 18289986); peer-reviewed journal, not a preprint; this is the source skimage.threshold_yen cites and supports the correlation-criterion claim. | ☐ |
| refiner | RefineBySineFit | Prior peer-reviewed work on correlation-based grid-colony quantification from plate images (gitter); class docstring now explicitly states the sinusoidal cross-correlation and rank-transform steps are NOT drawn from this paper. | 10.1534/g3.113.009431 | gitter: A Robust and Accurate Method for Quantification of Colony Sizes From Plate Images | SURVIVES (revised to conservative wording). DOI resolves to G3 (Bethesda) vol. 4(3) pp. 547-552, Mar. 2014 — peer-reviewed, not a preprint; metadata matches exactly. However the paper uses Radon transform + Pearson correlation against a circular-colony intensity model, NOT a sinusoidal template or rank-based Spearman cross-correlation. Original framing ('gitter-faithful', attributing the sinusoidal method to the paper) was a misattribution and was removed/qualified. | ☐ |
| enhancer | EnhanceLocalContrast | Adaptive histogram equalization / CLAHE reference (Pizer et al. 1987) | 10.1016/S0734-189X(87)80186-X | Adaptive histogram equalization and its variations | SURVIVES — peer-reviewed journal article (Computer Vision, Graphics, and Image Processing, vol. 39 no. 3 pp. 355-368, Sep. 1987); citation metadata (vol/issue/pages/year) matches docstring exactly; not a preprint; 3322 cites | ☐ |

## Documented ranges (342)

| Family | Class | Param | Documented range | Source | Matches code | Reviewed |
|---|---|---|---|---|---|---|
| threshold-detector | CannyDetector | sigma | 0.5--3.0, default 1.0 | code | ✅ | ☐ |
| threshold-detector | CannyDetector | low_threshold | 0.05--0.2 (quantile mode), default 0.1 | code | ✅ | ☐ |
| threshold-detector | CannyDetector | high_threshold | 0.1--0.4 (quantile mode), must exceed low_threshold, default 0.2 | code | ✅ | ☐ |
| threshold-detector | CannyDetector | use_quantiles | True/False, default True | library | ✅ | ☐ |
| threshold-detector | CannyDetector | min_size | 20--500 px, default 50 | code | ✅ | ☐ |
| threshold-detector | CannyDetector | invert_edges | True/False, default True | code | ✅ | ☐ |
| threshold-detector | CannyDetector | connectivity | 1 (4-conn) or 2 (8-conn), default 2 | library | ✅ | ☐ |
| threshold-detector | WatershedDetector | footprint | None (default); int diamond radius 5--50 px; 'auto' = floor(half well pitch) from grid spacing | code | ✅ | ☐ |
| threshold-detector | WatershedDetector | min_size | 20--200 px, default 50 | code | ✅ | ☐ |
| threshold-detector | WatershedDetector | compactness | 0.0001--0.1, default 0.001 | code | ✅ | ☐ |
| threshold-detector | WatershedDetector | connectivity | 1 (4-conn) or 2 (8-conn), default 1 | library | ✅ | ☐ |
| threshold-detector | WatershedDetector | relabel | True/False, default True | code | ✅ | ☐ |
| threshold-detector | WatershedDetector | ignore_zeros | True/False, default False | code | ✅ | ☐ |
| detector | ChanVeseDetector | mu | 0--1, default 0.25 | library | ✅ | ☐ |
| detector | ChanVeseDetector | lambda1 | default 1.0 (no numeric range; symmetric default) | literature | ✅ | ☐ |
| detector | ChanVeseDetector | lambda2 | default 1.0 (no numeric range; symmetric default) | literature | ✅ | ☐ |
| detector | ChanVeseDetector | max_num_iter | 100--1000, default 500 | library | ✅ | ☐ |
| detector | ChanVeseDetector | tol | 1e-5--1e-2, default 1e-3 | library | ✅ | ☐ |
| detector | ChanVeseDetector | dt | 0.1--1.0, default 0.5 | library | ✅ | ☐ |
| detector | ChanVeseDetector | init_level_set | checkerboard\|disk\|small disk, default checkerboard | library | ✅ | ☐ |
| detector | ChanVeseDetector | min_size | 10--500, default 50 | code | ✅ | ☐ |
| detector | ChanVeseDetector | connectivity | 1 or 2, default 2 | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | max_colony_radius_px | 50--400, default 250.0 | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | min_branch_width_px | 2--8, default 3 | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | edge_noise_threshold | 2.0--10.0, default 6.0 | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | reconnection_tolerance | 1.5--4.0, default 2.5 | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | max_gap_length | 10--100, default 30 | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | border_margin_px | 0--150, default 50 | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | frag_reach_px | 5--40, default 10 | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | gap_crossing_penalty | 1.0--10.0, default 4.0 | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | gauss_sigma | 50--600, default None (auto 1.2×R) | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | tile_size | 200--3000, default None (auto 4.8×R) | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | tile_overlap | 50--1500, default None (auto 2.4×R) | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | pct_min_wavelength | 2--20 (Nyquist floor 2), default None (auto 2.0×w) | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | mad_window | 3--21 odd, default None (auto 2w+1 odd) | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | path_dilation_radius | 1--10, default None (auto max(1,0.5w)) | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | snr_margin | 1--8, default None (auto max(2,0.5w)) | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | coherence_window_radius | 5--50, default None (auto 5.0×w) | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | ignore_borders | bool, default True | code | ✅ | ☐ |
| detector | FilamentousFungiDetector | inoculum_detector | ObjectDetector\|ImagePipeline\|None, default None | code | ✅ | ☐ |
| refiner | TrimAsymmetry | symmetry_threshold | [0,1] hard bounds; practical window 0.33-0.83; default 0.5 (3/6) | code | ✅ | ☐ |
| refiner | TrimAsymmetry | n_angular_bins | typical 4-12; default 6 | code | ✅ | ☐ |
| refiner | TrimAsymmetry | n_annuli | auto-clamped max(6, min(n_annuli, max_pixel_radius)); typical 10-200; default 100 | code | ✅ | ☐ |
| refiner | TrimAsymmetry | pelt_penalty | typical 1.0-20.0; BIC scale order log(n); default 5.0 | code | ✅ | ☐ |
| refiner | TrimAsymmetry | smoothing_window | typical 1-10; 1 disables; default 3 | code | ✅ | ☐ |
| refiner | TrimAsymmetry | method | Literal["distance","intensity"]; default "distance" | code | ✅ | ☐ |
| refiner | TrimAsymmetry | beehive_threshold | None or >=0; practical 0.0-0.05; start 0.002; default None | code | ✅ | ☐ |
| refiner | TrimAsymmetry | min_cc_area | typical 1-500; default 50 | code | ✅ | ☐ |
| refiner | TrimAsymmetry | min_object_area | typical 10-10000; default 100 | code | ✅ | ☐ |
| enhancer | BayesShrinkEnhancer | sigma | None (auto); manual 0.01-0.05 | code | ✅ | ☐ |
| enhancer | BayesShrinkEnhancer | wavelet | db1-db8, sym2-sym8; default db2 | library | ✅ | ☐ |
| enhancer | BayesShrinkEnhancer | mode | soft\|hard; default soft | code | ✅ | ☐ |
| enhancer | BayesShrinkEnhancer | wavelet_levels | None (max-3); 2-8 | library | ✅ | ☐ |
| enhancer | BayesShrinkEnhancer | clip | bool; default True | code | ✅ | ☐ |
| enhancer | BayesShrinkEnhancer | rescale_sigma | bool; default True | code | ✅ | ☐ |
| enhancer | BayesShrinkEnhancer | use_gat | bool; default False | code | ✅ | ☐ |
| enhancer | BayesShrinkEnhancer | gat_gain | default 1.0 | code | ✅ | ☐ |
| enhancer | BayesShrinkEnhancer | gat_mu | default 0.0 | code | ✅ | ☐ |
| enhancer | BayesShrinkEnhancer | gat_read_sigma | default 0.0 | code | ✅ | ☐ |
| enhancer | BayesShrinkEnhancer | gat_scale_factor | None auto (255/65535) | code | ✅ | ☐ |
| enhancer | BM3DDenoiser | sigma_psd | 0.01-0.05 moderate, 0.05-0.15 heavy; default 0.02; validator >=0 | code | ✅ | ☐ |
| enhancer | BM3DDenoiser | block_size | 4-16, powers of 2; default 8 | library | ✅ | ☐ |
| enhancer | BM3DDenoiser | stage_arg | all_stages\|hard_thresholding; default all_stages | code | ✅ | ☐ |
| enhancer | BM3DDenoiser | clip | bool; default True | code | ✅ | ☐ |
| enhancer | BM3DDenoiser | use_gat | bool; default False | code | ✅ | ☐ |
| enhancer | BM3DDenoiser | gat_gain | default 1.0 | code | ✅ | ☐ |
| enhancer | BM3DDenoiser | gat_mu | default 0.0 | code | ✅ | ☐ |
| enhancer | BM3DDenoiser | gat_read_sigma | default 0.0 | code | ✅ | ☐ |
| enhancer | BM3DDenoiser | gat_scale_factor | None auto (255/65535) | code | ✅ | ☐ |
| enhancer | FocusEdgeFrangi | sigmas | (0.5,1,1.5) to (1,2,3,4); 2-8px hyphae; default (0.5,1,1.5) | literature | ✅ | ☐ |
| enhancer | FocusEdgeFrangi | alpha | 0.1-1.0; default 0.5; no 2-D effect | literature | ✅ | ☐ |
| enhancer | FocusEdgeFrangi | beta | 0.1-1.0; default 0.5 | literature | ✅ | ☐ |
| enhancer | FocusEdgeFrangi | gamma | None (auto half-max-norm/scale); default None | library | ✅ | ☐ |
| enhancer | FocusEdgeFrangi | black_ridges | bool; default False (bright ridges) | code | ✅ | ☐ |
| enhancer | FocusEdgeHessian | sigmas | (1,2,3) to (1,5); default (1,2,3) | code | ✅ | ☐ |
| enhancer | FocusEdgeHessian | alpha | 0.1-1.0; default 0.5; no 2-D effect | literature | ✅ | ☐ |
| enhancer | FocusEdgeHessian | beta | 0.1-1.0; default 0.5 | literature | ✅ | ☐ |
| enhancer | FocusEdgeHessian | gamma | 10-20 typical, 5-10 low-contrast, 20-25 high-contrast; default 15 | library | ✅ | ☐ |
| enhancer | FocusEdgeHessian | black_ridges | bool; default False (bright ridges) | code | ✅ | ☐ |
| enhancer | FocusEdgeHessian | mode | reflect\|constant\|nearest\|mirror\|wrap; default reflect | library | ✅ | ☐ |
| enhancer | FocusEdgeHessian | cval | float; default 0; only used when mode=constant | library | ✅ | ☐ |
| enhancer | FocusEdgeMeijering | sigmas | (1,2,3); lower bound 0.5 fine filaments, upper 5-8 thick mats; default (1,2,3) | code | ✅ | ☐ |
| enhancer | FocusEdgeMeijering | alpha | None (analytic optimum -1/3 for 2-D); default None | literature | ✅ | ☐ |
| enhancer | FocusEdgeMeijering | black_ridges | bool; default False (bright ridges) | code | ✅ | ☐ |
| enhancer | FocusEdgeMeijering | mode | constant\|reflect\|wrap\|nearest\|mirror; default reflect | library | ✅ | ☐ |
| enhancer | FocusEdgeMeijering | cval | float; default 0; only used when mode=constant | library | ✅ | ☐ |
| enhancer | FocusEdgePhase | n_scale | 3-6; default 4; guard >=1 | code | ✅ | ☐ |
| enhancer | FocusEdgePhase | n_orient | 4-8; default 6; guard >=1 | code | ✅ | ☐ |
| enhancer | FocusEdgePhase | min_wavelength | >=2 (Nyquist guard); default 3.0 | code | ✅ | ☐ |
| enhancer | FocusEdgePhase | mult | >1; default 2.1; even-coverage pairs 0.55/3 and 0.75/1.6 | library | ✅ | ☐ |
| enhancer | FocusEdgePhase | sigma_onf | 0.1-1.0 (guard); default 0.55; smaller=wider bandwidth | library | ✅ | ☐ |
| enhancer | FocusEdgePhase | k | >=0; 0 disables; default 2.0 | library | ✅ | ☐ |
| enhancer | FocusEdgePhase | cutoff | (0,1) exclusive guard; default 0.5 | library | ✅ | ☐ |
| enhancer | FocusEdgePhase | g | >0 guard; default 10.0 | library | ✅ | ☐ |
| enhancer | FocusEdgePhase | noise_method | -1 median, -2 mode, >=0 fixed; default -1.0 | code | ✅ | ☐ |
| enhancer | FocusEdgePhase | output | pc_sum\|M\|m; default pc_sum | code | ✅ | ☐ |
| enhancer | FocusEdgeSato | sigmas | (1, 2, 3) default; span (1,3,5) or range(1,10,2) | literature | ✅ | ☐ |
| enhancer | FocusEdgeSato | black_ridges | bool, default False | code | ✅ | ☐ |
| enhancer | FocusEdgeSato | mode | constant/reflect/wrap/nearest/mirror, default reflect | library | ✅ | ☐ |
| enhancer | FocusEdgeSato | cval | any float, default 0 | library | ✅ | ☐ |
| enhancer | LocalEdgeDenoise | sigma_color | 0.02--0.5 on [0,1]; None auto; 1.0 when use_gat | literature | ✅ | ☐ |
| enhancer | LocalEdgeDenoise | sigma_spatial | 1--50 px, default 15 (validator >0, no negative lower bound) | literature | ✅ | ☐ |
| enhancer | LocalEdgeDenoise | win_size | None auto = max(5, 2*ceil(3*sigma_spatial)+1) | library | ✅ | ☐ |
| enhancer | LocalEdgeDenoise | mode | constant/edge/symmetric/reflect/wrap, default constant | library | ✅ | ☐ |
| enhancer | LocalEdgeDenoise | cval | float, default 0.0 | code | ✅ | ☐ |
| enhancer | LocalEdgeDenoise | clip | bool default True; deferred to False under GAT | code | ✅ | ☐ |
| enhancer | LocalEdgeDenoise | use_gat | bool default False | literature | ✅ | ☐ |
| enhancer | LocalEdgeDenoise | gat_gain | 0.1--10.0 e-/ADU (reconciled), default 1.0 (Field gt=0) | literature | ✅ | ☐ |
| enhancer | LocalEdgeDenoise | gat_mu | float, default 0.0 | literature | ✅ | ☐ |
| enhancer | LocalEdgeDenoise | gat_read_sigma | 0.004--0.02 on [0,1], default 0.0 (Field ge=0) | literature | ✅ | ☐ |
| enhancer | LocalEdgeDenoise | gat_scale_factor | None auto (8-bit 255, 16-bit 65535); Field gt=0 | code | ✅ | ☐ |
| enhancer | NonLocalMeansDenoiser | patch_size | REVISED 5--15 -> 3--11 (skimage default 7); <=5 for fine hyphae; default 5 | literature | ✅ | ☐ |
| enhancer | NonLocalMeansDenoiser | search_dist | REVISED 5--21 -> library default 11 (23x23); 5--7 for crowded plates; default 11 | literature | ✅ | ☐ |
| enhancer | NonLocalMeansDenoiser | h | h~=noise std; default 0.5; 1.0 under GAT | literature | ✅ | ☐ |
| enhancer | NonLocalMeansDenoiser | fast_mode | bool default False; h~=0.8*sigma (fast) / 0.6*sigma (orig) | library | ✅ | ☐ |
| enhancer | NonLocalMeansDenoiser | sigma | >=0, 0.0 disables, default 0.0; 1.0 under GAT | library | ✅ | ☐ |
| enhancer | NonLocalMeansDenoiser | use_gat | bool default False | literature | ✅ | ☐ |
| enhancer | NonLocalMeansDenoiser | gat_gain | 0.1--10.0 e-/ADU, default 1.0 (Field gt=0) | code | ✅ | ☐ |
| enhancer | NonLocalMeansDenoiser | gat_mu | float, default 0.0 | code | ✅ | ☐ |
| enhancer | NonLocalMeansDenoiser | gat_read_sigma | 0.001--0.05 on [0,1], default 0.0 (Field ge=0) | code | ✅ | ☐ |
| enhancer | NonLocalMeansDenoiser | gat_scale_factor | None auto (8-bit 255, 16-bit 65535); Field gt=0 | code | ✅ | ☐ |
| enhancer | StructureSmoothing | num_iter | 5--100, default 20 (validator >=1) | code | ✅ | ☐ |
| enhancer | StructureSmoothing | sigma | 0.5--5.0 px, default 1.5 (validator >0) | code | ✅ | ☐ |
| enhancer | StructureSmoothing | rho | >=sigma; 2--3x sigma; None->sigma (validator rho>=sigma) | code | ✅ | ☐ |
| enhancer | StructureSmoothing | dt | (0, 0.125], default 0.1 (validator enforces) | code | ✅ | ☐ |
| enhancer | StructureSmoothing | alpha | (0,1), 0.001--0.1 practical, default 0.001 (validator 0<alpha<1) | literature | ✅ | ☐ |
| enhancer | StructureSmoothing | C | (0,100] percentile, default 99.0 (validator 0<C<=100) | code | ✅ | ☐ |
| enhancer | VisuShrinkEnhancer | sigma | None auto (MAD) or [0,1]; 0.01--0.05 typical; 1.0 under GAT | literature | ✅ | ☐ |
| enhancer | VisuShrinkEnhancer | wavelet | orthogonal families (db2 default, db4, sym4/sym6) | library | ✅ | ☐ |
| enhancer | VisuShrinkEnhancer | mode | Literal['soft','hard'], default soft | literature | ✅ | ☐ |
| enhancer | VisuShrinkEnhancer | wavelet_levels | None auto (max-3) or 1..floor(log2(min_dim)) | library | ✅ | ☐ |
| enhancer | VisuShrinkEnhancer | clip | bool default True; deferred False under GAT | code | ✅ | ☐ |
| enhancer | VisuShrinkEnhancer | rescale_sigma | bool default True; forced False under GAT | library | ✅ | ☐ |
| enhancer | VisuShrinkEnhancer | use_gat | bool default False | code | ✅ | ☐ |
| enhancer | VisuShrinkEnhancer | gat_gain | 0.1--10.0 e-/ADU (reconciled), default 1.0 (Field gt=0) | literature | ✅ | ☐ |
| enhancer | VisuShrinkEnhancer | gat_mu | float, default 0.0 | literature | ✅ | ☐ |
| enhancer | VisuShrinkEnhancer | gat_read_sigma | >=0, default 0.0 (Field ge=0) | literature | ✅ | ☐ |
| enhancer | VisuShrinkEnhancer | gat_scale_factor | None auto (8/16-bit); Field gt=0 | code | ✅ | ☐ |
| corrector | BayesShrinkCorrector | sigma | None (auto, MAD/0.6745); manual 0.01--0.05, up to 0.1; retargeted to 1.0 under use_gat | code | ✅ | ☐ |
| corrector | BayesShrinkCorrector | wavelet | 'db2' default; 'db4' more vanishing moments/wider support | library | ✅ | ☐ |
| corrector | BayesShrinkCorrector | mode | Literal['soft','hard'], default 'soft' | code | ✅ | ☐ |
| corrector | BayesShrinkCorrector | wavelet_levels | None (max-3); manual 2--6 | library | ✅ | ☐ |
| corrector | BayesShrinkCorrector | convert2ycbcr | bool, default True | library | ✅ | ☐ |
| corrector | BayesShrinkCorrector | rescale_sigma | bool, default True (forced False in GAT region) | code | ✅ | ☐ |
| corrector | BayesShrinkCorrector | clip | bool, default True (deferred False in GAT region) | code | ✅ | ☐ |
| corrector | BayesShrinkCorrector | use_gat | bool, default False | code | ✅ | ☐ |
| corrector | BayesShrinkCorrector | gat_gain | 0.5--10 e-/ADU, default 1.0 (Field gt=0) | code | ✅ | ☐ |
| corrector | BayesShrinkCorrector | gat_mu | any float, default 0.0 (no constraint) | code | ✅ | ☐ |
| corrector | BayesShrinkCorrector | gat_read_sigma | count units, default 0.0 (Field ge=0) | code | ✅ | ☐ |
| corrector | BayesShrinkCorrector | gat_scale_factor | None auto (255/65535), default None (Field gt=0) | code | ✅ | ☐ |
| corrector | ColorDenoise | sigma_psd | 0.01--0.05 moderate, 0.05--0.15 heavy; default 0.02 (validator ge=0) | literature | ✅ | ☐ |
| corrector | ColorDenoise | block_size | 4--16; default 8 (validator gt=0); sets both HT and Wiener block sizes per code | library | ✅ | ☐ |
| corrector | ColorDenoise | clip | bool, default True | code | ✅ | ☐ |
| corrector | ColorDenoise | use_gat | bool, default False | code | ✅ | ☐ |
| corrector | ColorDenoise | gat_gain | 0.5--10 e-/ADU, default 1.0 (validator gt=0) | code | ✅ | ☐ |
| corrector | ColorDenoise | gat_mu | any float, default 0.0 (no constraint) | code | ✅ | ☐ |
| corrector | ColorDenoise | gat_read_sigma | count units, default 0.0 (validator ge=0) | code | ✅ | ☐ |
| corrector | ColorDenoise | gat_scale_factor | None auto (255/65535), default None (validator gt=0) | code | ✅ | ☐ |
| corrector | StableDenoise | block_size | 4--16; default 8 (validator gt via code; sets bs_ht and bs_wiener) | literature | ✅ | ☐ |
| corrector | StableDenoise | stage_arg | Literal['all_stages','hard_thresholding']; hard ~40--50% faster (code/profile-derived, NA verifier) | code | ✅ | ☐ |
| corrector | StableDenoise | gain | 0.5--10 e-/ADU, default 1.0 (validator >0) | code | ✅ | ☐ |
| corrector | StableDenoise | mu | any float, default 0.0 (float-coerced, no bound) | code | ✅ | ☐ |
| corrector | StableDenoise | sigma | count units, default 0.0 (validator >=0) | code | ✅ | ☐ |
| corrector | StableDenoise | scale_factor | None auto (255/65535), default None (validator >0) | code | ✅ | ☐ |
| corrector | VisuShrinkCorrector | sigma | None (auto MAD/0.6745); reduce by 2--4x for sharper; retargeted 1.0 under use_gat | library | ✅ | ☐ |
| corrector | VisuShrinkCorrector | wavelet | 'db2' default; 'db4' more vanishing moments/wider support | library | ✅ | ☐ |
| corrector | VisuShrinkCorrector | mode | Literal['soft','hard'], default 'soft' | code | ✅ | ☐ |
| corrector | VisuShrinkCorrector | wavelet_levels | None (max-3); manual 2--6 | library | ✅ | ☐ |
| corrector | VisuShrinkCorrector | convert2ycbcr | bool, default True | library | ✅ | ☐ |
| corrector | VisuShrinkCorrector | rescale_sigma | bool, default True (deferred False in GAT region) | code | ✅ | ☐ |
| corrector | VisuShrinkCorrector | clip | bool, default True (deferred False in GAT region) | code | ✅ | ☐ |
| corrector | VisuShrinkCorrector | use_gat | bool, default False | code | ✅ | ☐ |
| corrector | VisuShrinkCorrector | gat_gain | 0.5--10 e-/ADU, default 1.0 (Field gt=0) | code | ✅ | ☐ |
| corrector | VisuShrinkCorrector | gat_mu | any float, default 0.0 (no constraint) | code | ✅ | ☐ |
| corrector | VisuShrinkCorrector | gat_read_sigma | count units, default 0.0 (Field ge=0) | code | ✅ | ☐ |
| corrector | VisuShrinkCorrector | gat_scale_factor | None auto (255/65535), default None (Field gt=0) | code | ✅ | ☐ |
| detector | CompositeDetector | min_overlap_ratio | 0.0--0.5, default 0.0 | code | ✅ | ☐ |
| detector | CompositeDetector | mode | default 'overlap' (union\|intersection\|overlap) | code | ✅ | ☐ |
| detector | CompositeDetector | detectors | default [OtsuDetector(), RoundPeaksDetector()] | code | ✅ | ☐ |
| detector | InoculumDetector | min_diameter | 5--80 px, default 30.0 | code | ✅ | ☐ |
| detector | InoculumDetector | max_diameter | 50--300 px, default 100.0 | code | ✅ | ☐ |
| detector | InoculumDetector | gmm_separation_threshold | 0.8--1.2, default 0.9 | code | ✅ | ☐ |
| detector | InoculumDetector | gmm_n_components | default 2 | code | ✅ | ☐ |
| detector | RankOtsuDetector | width | None auto-scales to min(height,width)//8 | code | ✅ | ☐ |
| detector | RankOtsuDetector | shape | default 'square' (square\|diamond\|disk) | code | ✅ | ☐ |
| detector | RankOtsuDetector | ignore_zeros | default False | code | ✅ | ☐ |
| detector | RoundPeaksDetector | footprint_width | 4--20 px, default 6 | code | ✅ | ☐ |
| detector | RoundPeaksDetector | noise_radius | 1--3, default 1 | code | ✅ | ☐ |
| detector | RoundPeaksDetector | smoothing_sigma | 0.0--5.0, default 2.0 | code | ✅ | ☐ |
| detector | RoundPeaksDetector | thresh_method | default 'otsu' (otsu\|mean\|local\|triangle\|minimum\|isodata\|li) | code | ✅ | ☐ |
| detector | SinePeakDetector | correlation_threshold | 0.1--0.5, default 0.3 | code | ✅ | ☐ |
| detector | SinePeakDetector | footprint_width | 4--20 px, default 6 | code | ✅ | ☐ |
| detector | SinePeakDetector | noise_radius | 1--3, default 1 | code | ✅ | ☐ |
| detector | SinePeakDetector | smoothing_sigma | 0.0--5.0, default 2.0 | code | ✅ | ☐ |
| threshold-detector | IsodataDetector | ignore_zeros | Default: False | code | ✅ | ☐ |
| threshold-detector | IsodataDetector | ignore_borders | Default: True | code | ✅ | ☐ |
| threshold-detector | LiDetector | ignore_zeros | Default: False | code | ✅ | ☐ |
| threshold-detector | LiDetector | ignore_borders | Default: True | code | ✅ | ☐ |
| threshold-detector | MeanDetector | ignore_zeros | Default: False | code | ✅ | ☐ |
| threshold-detector | MeanDetector | ignore_borders | Default: True | code | ✅ | ☐ |
| threshold-detector | MinimumDetector | ignore_zeros | Default: False | code | ✅ | ☐ |
| threshold-detector | MinimumDetector | ignore_borders | Default: True | code | ✅ | ☐ |
| threshold-detector | OtsuDetector | ignore_zeros | Default: False | code | ✅ | ☐ |
| threshold-detector | OtsuDetector | ignore_borders | Default: True | code | ✅ | ☐ |
| threshold-detector | YenDetector | ignore_zeros | Default: False | code | ✅ | ☐ |
| threshold-detector | YenDetector | ignore_borders | Default: True | code | ✅ | ☐ |
| refiner | ExtractColonyCore | n_components | default 2 | code | ✅ | ☐ |
| refiner | ExtractColonyCore | separation_threshold | 0.5--1.2, default 0.8 | code | ✅ | ☐ |
| refiner | ExtractColonyCore | min_core_area | 10--500, default 30 | code | ✅ | ☐ |
| refiner | ExtractColonyCore | morph_open_radius | 0--5, default 1 | code | ✅ | ☐ |
| refiner | ExtractColonyCore | morph_close_radius | 0--5, default 2 | code | ✅ | ☐ |
| refiner | GridAlignmentRefiner | smoothing_sigma | 0.5--5.0, default 2.0 | code | ✅ | ☐ |
| refiner | GridAlignmentRefiner | min_peak_distance | None auto = half expected spacing | code | ✅ | ☐ |
| refiner | GridAlignmentRefiner | peak_prominence | None auto = 10% of signal range | code | ✅ | ☐ |
| refiner | GridAlignmentRefiner | edge_refinement | default True | code | ✅ | ☐ |
| refiner | GridAlignmentRefiner | selection_mode | dominant\|centered\|regularized, default dominant | code | ✅ | ☐ |
| refiner | GridAlignmentRefiner | split_merged | default False | code | ✅ | ☐ |
| refiner | MaskClosing | width | 3--9, default 5 | code | ✅ | ☐ |
| refiner | MaskClosing | n_iter | default 1 | code | ✅ | ☐ |
| refiner | MaskClosing | shape | auto\|square\|diamond\|disk\|ndarray\|None, default None | code | ✅ | ☐ |
| refiner | MaskDilation | width | 1--7, default 3 | code | ✅ | ☐ |
| refiner | MaskDilation | n_iter | default 1 | code | ✅ | ☐ |
| refiner | MaskDilation | shape | auto\|square\|diamond\|disk\|ndarray\|None, default None | code | ✅ | ☐ |
| refiner | MaskErosion | width | 1--7, default 3 | code | ✅ | ☐ |
| refiner | MaskErosion | n_iter | default 1 | code | ✅ | ☐ |
| refiner | MaskErosion | shape | auto\|square\|diamond\|disk\|ndarray\|None, default None | code | ✅ | ☐ |
| refiner | MaskFill | structure | ndarray\|None, default None | code | ✅ | ☐ |
| refiner | MaskFill | origin | default 0 | code | ✅ | ☐ |
| refiner | MaskGradient | width | 1--5, default 1 | code | ✅ | ☐ |
| refiner | MaskGradient | shape | auto\|square\|diamond\|disk\|ndarray\|None, default None | code | ✅ | ☐ |
| refiner | MaskOpening | width | 3--9; default 5 | code | ✅ | ☐ |
| refiner | MaskOpening | n_iter | 1--3; default 1 | code | ✅ | ☐ |
| refiner | MaskOpening | shape | auto/square/diamond/disk/ndarray/None; default None; auto=0.5% of smaller dim | code | ✅ | ☐ |
| refiner | MaskWhiteTophat | width | 3--10; default None; None=0.4% of smaller dim | code | ✅ | ☐ |
| refiner | MaskWhiteTophat | shape | disk/square/diamond/ndarray; default disk | code | ✅ | ☐ |
| refiner | MergeFragmentChains | distance_threshold | 10--50; default 20.0 | code | ✅ | ☐ |
| refiner | NearestNeighborMerger | distance_threshold | 10--50; default 20.0 | code | ✅ | ☐ |
| refiner | NearestNeighborMerger | min_size | 20--200; default 50; None=all eligible | code | ✅ | ☐ |
| refiner | RefineBySineFit | smoothing_sigma | 0.5--5.0; default 2.0 | code | ✅ | ☐ |
| refiner | RefineBySineFit | correlation_threshold | 0.1--0.6; default 0.3 | code | ✅ | ☐ |
| refiner | RefineBySineFit | peak_prominence | None auto = 10% of cross-correlation profile range | code | ✅ | ☐ |
| refiner | RefineBySineFit | min_peak_distance | None auto from expected spacing | code | ✅ | ☐ |
| refiner | RefineBySineFit | edge_refinement | bool; default True | code | ✅ | ☐ |
| refiner | RefineBySineFit | selection_mode | dominant/centered/regularized; default dominant | code | ✅ | ☐ |
| refiner | RefineBySineFit | split_merged | bool; default False | code | ✅ | ☐ |
| refiner | RemoveBorderObjects | border_size | 1--30 px; default 1; None=1% of smaller dim; float in (0,1)=fraction; >=1=absolute px | code | ✅ | ☐ |
| refiner | RemoveGridOutliers | cutoff_multiplier | 1.0--3.0; default 1.5 (alias stddev_multiplier) | code | ✅ | ☐ |
| refiner | RemoveGridOutliers | max_coeff_variance | 1--5; default 1 | code | ✅ | ☐ |
| refiner | RemoveGridOutliers | axis | None/0/1; default None | code | ✅ | ☐ |
| refiner | RemoveLowCircularity | cutoff | 0.5--0.9; default 0.785 (pi/4); validated [0,1] | code | ✅ | ☐ |
| refiner | SeparateObjects | min_distance | 5--50, default 10 | code | ✅ | ☐ |
| refiner | Skeletonize | method | {"zhang","lee",None}, default None | library | ✅ | ☐ |
| refiner | SmallObjectRemover | min_size | default 64; rough start 20--100 px near 300 dpi, 100--500 px near 1200 dpi | code | ✅ | ☐ |
| refiner | SmallToLargeMerger | distance_threshold | 10--50, default 30.0 (validated > 0) | code | ✅ | ☐ |
| refiner | SmallToLargeMerger | size_threshold | 50--200, default 100 (validated > 0) | code | ✅ | ☐ |
| refiner | Thinning | max_num_iter | None (run to convergence) or positive int; 1--5 gentle | code | ✅ | ☐ |
| enhancer | ContrastStretching | lower_percentile | 1--5, default 2 | code | ✅ | ☐ |
| enhancer | ContrastStretching | upper_percentile | 95--99, default 98 | code | ✅ | ☐ |
| enhancer | EnhanceLocalContrast | kernel_size | None auto-selects ~ height/15 | code | ✅ | ☐ |
| enhancer | EnhanceLocalContrast | clip_limit | 0.005--0.05, default 0.01 | library | ✅ | ☐ |
| enhancer | FlattenIllumination | sigma | 40--300, default 200.0, must be positive | code | ✅ | ☐ |
| enhancer | FlattenIllumination | gamma_low | 0.3--0.8, default 0.5 | code | ✅ | ☐ |
| enhancer | FlattenIllumination | gamma_high | 1.0--2.5, default 1.5 | code | ✅ | ☐ |
| enhancer | FlattenIllumination | eps | 1e-8--1e-4, default 1e-6 | code | ✅ | ☐ |
| enhancer | FocusBlobLoG | min_radius | 1.0--15.0 px, default 3.0, >0 and < max_radius | code | ✅ | ☐ |
| enhancer | FocusBlobLoG | max_radius | 8.0--60.0 px, default 12.0 | code | ✅ | ☐ |
| enhancer | FocusBlobLoG | num_scales | 4--20, default 12 (validator >=1) | code | ✅ | ☐ |
| enhancer | FocusEdgeLaplace | kernel_size | default 3; smaller fine edges, 5--7 smooth | library | ✅ | ☐ |
| enhancer | FocusEdgeLaplace | mask | None or boolean/0-1 array | code | ✅ | ☐ |
| enhancer | GaussianBlur | sigma | 0.5--5.0, default 2.0 | code | ✅ | ☐ |
| enhancer | GaussianBlur | mode | reflect/constant/nearest, default reflect | code | ✅ | ☐ |
| enhancer | GaussianBlur | cval | default 0.0 | code | ✅ | ☐ |
| enhancer | GaussianBlur | truncate | default 4.0; half-width = truncate*sigma | library | ✅ | ☐ |
| enhancer | GrayOpening | shape | square/diamond/disk, default square | code | ✅ | ☐ |
| enhancer | GrayOpening | width | 3--15, default 5 | code | ✅ | ☐ |
| enhancer | GrayOpening | n_iter | default 1 | code | ✅ | ☐ |
| enhancer | MedianFilter | mode | nearest/reflect/constant/mirror/wrap, default nearest | code | ✅ | ☐ |
| enhancer | MedianFilter | shape | disk/square/diamond or None, default None | code | ✅ | ☐ |
| enhancer | MedianFilter | width | 3--9, default 5 | code | ✅ | ☐ |
| enhancer | MedianFilter | cval | default 0.0 | code | ✅ | ☐ |
| enhancer | RankMedianEnhancer | shape | 'disk', 'square' (default 'square') | code | ✅ | ☐ |
| enhancer | RankMedianEnhancer | width | None default; auto ~0.2% of shorter dim (int(min(shape)*0.002)) | code | ✅ | ☐ |
| enhancer | RankMedianEnhancer | shift_x | default 0 | code | ✅ | ☐ |
| enhancer | RankMedianEnhancer | shift_y | default 0 | code | ✅ | ☐ |
| enhancer | SetDetectMode | mode | 'gray'(default),'red','green','blue','MinRGB','LabL','LabA','LabB','HsvS','HsvV','InvS' | code | ✅ | ☐ |
| enhancer | SharpenEdgeGauss | radius | 0.5--5.0 typical, up to 15; default 2.0; must be > 0 | code | ✅ | ☐ |
| enhancer | SharpenEdgeGauss | amount | default 1.0; <1.0 subtle, >2.0 halo risk | code | ✅ | ☐ |
| enhancer | SharpenEdgeGauss | preserve_range | default False | code | ✅ | ☐ |
| enhancer | SharpenEdgeGauss | n_iter | 1--3 typical; >=1; default 1 | code | ✅ | ☐ |
| enhancer | SubtractGaussian | sigma | 20--100 typical; default 50.0; set larger than colony diameter | code | ✅ | ☐ |
| enhancer | SubtractGaussian | mode | 'reflect'(default),'constant','nearest','mirror','wrap' | library | ✅ | ☐ |
| enhancer | SubtractGaussian | cval | default 0.0 | code | ✅ | ☐ |
| enhancer | SubtractGaussian | truncate | default 4.0 | code | ✅ | ☐ |
| enhancer | SubtractGaussian | preserve_range | default True | code | ✅ | ☐ |
| enhancer | SubtractGaussian | n_iter | 1--3 typical; >=1; default 1 | code | ✅ | ☐ |
| enhancer | SubtractOpening | shape | 'disk'(default),'square','diamond' | code | ✅ | ☐ |
| enhancer | SubtractOpening | width | 31--101 typical; default 51 | code | ✅ | ☐ |
| enhancer | SubtractOpening | n_iter | default 1 | code | ✅ | ☐ |
| enhancer | SubtractRollingBall | radius | 50--200 typical; default 100; > largest colony diameter | code | ✅ | ☐ |
| enhancer | SubtractRollingBall | kernel | default None; overrides radius when provided | library | ✅ | ☐ |
| enhancer | SubtractRollingBall | nansafe | default False | code | ✅ | ☐ |
| enhancer | SubtractWhiteTophat | shape | 'diamond'(default),'disk','square' | code | ✅ | ☐ |
| enhancer | SubtractWhiteTophat | width | None default; auto ~0.4% of shorter dim (int(min(shape)*0.004)) | code | ✅ | ☐ |
| enhancer | WhiteTophatEnhance | shape | 'disk'(default),'diamond','square' | code | ✅ | ☐ |
| enhancer | WhiteTophatEnhance | width | None default; auto ~0.4% of shorter dim (int(min(shape)*0.004)) | code | ✅ | ☐ |
| corrector | ColorCorrector | profile | fitted ColorCheckerProfile required (unfitted rejected) | code | ✅ | ☐ |
| corrector | ColorCorrector | output_illuminant | default "D65" | code | ✅ | ☐ |
| corrector | GridAligner | axis | 0 or 1 (default 0); ValueError otherwise | code | ✅ | ☐ |
| corrector | GridAligner | mode | 'edge' (default) or 'constant' | code | ✅ | ☐ |
| corrector | ImageCropper | left | >= 0 or None (negative rejected by validator) | code | ✅ | ☐ |
| corrector | ImageCropper | right | >= 0 or None (negative rejected by validator) | code | ✅ | ☐ |
| corrector | ImageCropper | top | >= 0 or None (negative rejected by validator) | code | ✅ | ☐ |
| corrector | ImageCropper | bottom | >= 0 or None (negative rejected by validator) | code | ✅ | ☐ |
| corrector | ImagePadder | left | >= 0 or None (negative rejected by validator) | code | ✅ | ☐ |
| corrector | ImagePadder | right | >= 0 or None (negative rejected by validator) | code | ✅ | ☐ |
| corrector | ImagePadder | top | >= 0 or None (negative rejected by validator) | code | ✅ | ☐ |
| corrector | ImagePadder | bottom | >= 0 or None (negative rejected by validator) | code | ✅ | ☐ |
| corrector | ImagePadder | mode | 11 np.pad modes (PadMode Literal), default 'constant' | code | ✅ | ☐ |
| corrector | ImagePadder | constant_value | int\|float, default 0 | code | ✅ | ☐ |
| threshold-detector | HysteresisDetector | low | method name or float; 0-255 (8-bit), 0-65535 (16-bit); default 'mean' | code | ✅ | ☐ |
| threshold-detector | HysteresisDetector | high | method name or float; must resolve >= low; default 'otsu' | code | ✅ | ☐ |
| threshold-detector | HysteresisDetector | ignore_zeros | bool; default False | code | ✅ | ☐ |
| threshold-detector | HysteresisDetector | ignore_borders | bool; default True | code | ✅ | ☐ |
| threshold-detector | MadHysteresisDetector | k_high | 3.0-8.0; default 5.0; must be > k_low | code | ✅ | ☐ |
| threshold-detector | MadHysteresisDetector | k_low | 1.5-4.0; default 2.5; must be < k_high | code | ✅ | ☐ |
| threshold-detector | MadHysteresisDetector | min_size | px area floor; default 20 | code | ✅ | ☐ |
| threshold-detector | MadHysteresisDetector | connectivity | 1 or 2; default 2 | code | ✅ | ☐ |
| threshold-detector | MadHysteresisDetector | ignore_zeros | bool; default False | code | ✅ | ☐ |
| threshold-detector | MadHysteresisDetector | ignore_borders | bool; default True | code | ✅ | ☐ |
| detector | ManualGridPointDetector | coord1 | (y, x) pixel tuple; default (0, 0) | code | ✅ | ☐ |
| detector | ManualGridPointDetector | coord2 | (y, x) pixel tuple or None; default None | code | ✅ | ☐ |
| detector | ManualGridPointDetector | shape | 'disk'\|'square'\|'diamond'; default 'disk' | code | ✅ | ☐ |
| detector | ManualGridPointDetector | width | 5-50 px; default 15 | code | ✅ | ☐ |
| detector | ManualPointDetector | centers | N x 2 (y, x) array-like or None; default None | code | ✅ | ☐ |
| detector | ManualPointDetector | shape | 'disk'\|'square'\|'diamond'; default 'disk' | code | ✅ | ☐ |
| detector | ManualPointDetector | width | 5-50 px; default 15 | code | ✅ | ☐ |
| threshold-detector | UserThreshold | threshold | 0-255 (8-bit) / 0-65535 (16-bit) / 0.0-1.0 (float); must be non-negative; default 0.5 | code | ✅ | ☐ |
| threshold-detector | UserThreshold | ignore_zeros | bool; default False | code | ✅ | ☐ |
| threshold-detector | UserThreshold | ignore_borders | bool; default True | code | ✅ | ☐ |
| refiner | ManualRefine | centers | N x 2 (y, x) array-like or None; default None | code | ✅ | ☐ |
| refiner | ManualRefine | shape | 'disk'\|'square'\|'diamond'; default 'disk' | code | ✅ | ☐ |
| refiner | ManualRefine | width | 5-50 px; default 15 | code | ✅ | ☐ |
