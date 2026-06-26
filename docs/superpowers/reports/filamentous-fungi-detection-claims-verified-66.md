# Adversarially-verified claims (66) — filamentous-fungi detection survey

Claims that survived 3-vote adversarial verification in the `deep-research-litmax` run.


## Astronomy: streaks, cosmic-web & ISM filaments
- **LSD is parameter-free and training-free: it works on any digital image without parameter tuning, requiring no labeled data or per-image threshold tuning.**
  > “It is designed to work on any digital image without parameter tuning.”  
  [https://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf]
- **LSD controls its own false-detection rate via an a-contrario (Desolneux-Moisan-Morel) validation, allowing on average one false alarm per image — an automatic statistical threshold rather than a hand-tuned one, directly analogous to the a-contrario/FDR low-SNR thresholds called for in the brief.**
  > “It controls its own number of false detections: On average, one false alarm is allowed per image.”  
  [https://www.ipol.im/pub/art/2012/gjmr-lsd/article.pdf]
- **The method uses a matched filter modeling coronal loops as a Gaussian cross-section profile (h(x,y) = 1/(sqrt(2pi)sigma) exp(-x^2/(2sigma^2)) for |y| <= L/2), rotated across 0-180 degrees, selecting the maximum response per pixel — i.e. it integrates evidence along a hypothesized oriented line and maximizes SNR (matched-filter / oriented directivity principle).**
  > “Io(x,y) = max{Iin(x,y) * h̃ᵢ(u,v)} i = 1,2,...,N ... the match filter requires that its output SNR should be maximal at time t₀”  
  [https://academic.oup.com/mnras/article/490/4/5567/5625861]
- **The method is unsupervised and training-free, relying on prior knowledge of loop characteristics (Gaussian profile, curvature, width range) rather than learned parameters — directly relevant to the no-large-labeled-corpus constraint.**
  > “Most coronal loops can be considered as curved lines with a certain width”  
  [https://academic.oup.com/mnras/article/490/4/5567/5625861]

## Biomedical analogs & curvilinear tracking
- **A Hessian-based enhancement embedding module (using second-order-gradient eigenvalues lambda_1, lambda_2) improves the contrast of low-contrast thin guidewires before segmentation, integrating classical Hessian ridge enhancement into a deep network.**
  > “Hessian-based enhancement embedding module ... designed to improve contrast of the guidewire ... Uses eigenvalue features (λ₁, λ₂) computed from second-order gradients for enhancement”  
  [https://arxiv.org/html/2404.08805v1]
- **A detector outputs a bounding box first to crop the structure, which reduces the foreground/background class imbalance of the thin-structure segmentation task.**
  > “a detector is deployed first to output the bounding box...which could reduce the class imbalance of the segmentation task”  
  [https://arxiv.org/html/2404.08805v1]
- **The Hessian-enhanced network (HessianNet) outperforms a baseline U-Net on thin-structure segmentation, achieving Dice 0.8990 versus U-Net's 0.8890.**
  > “Achieves Dice score of 0.8990 vs. baseline U-Net's 0.8890”  
  [https://arxiv.org/html/2404.08805v1]
- **The catheter tip tracking pipeline frames the thin-structure detection task as deep-learning semantic SEGMENTATION (producing a per-pixel mask) rather than direct point/bbox tracking, comparing U-Net, U-Net+Transformer, and SegFormer architectures in both two-class and three-class formulations — transferable as a learned, mask-producing alternative to the classical phase-congruency baseline for curvilinear structures.**
  > “The team trained and compared three deep learning segmentation architectures (U-Net, U-Net+Transformer, and SegFormer) using both two-class and three-class formulations, followed by component filtering and contour-based path tracking.”  
  [https://link.springer.com/article/10.1007/s11548-026-03647-7]

## Cryo-EM low-SNR filament picking
- **STRIPER is a training-free classical line detector: it enhances filaments with oriented Gaussian smoothing kernels (taking the max response over rotated kernels) and then extracts lines with Steger's (1998) ridge-detection algorithm using hysteresis thresholds — directly transferable as a Steger + oriented-filter alternative to phase congruency.**
  > “Instead of using a trained deep neural network to identify the boxes along the filaments, the STRIPER filament-picking procedure is based on a classical line-detection approach... the filaments are enhanced using the same oriented Gaussian ”  
  [https://www.biorxiv.org/content/10.1101/2020.02.28.969196v1.full]
- **Steger's ridge detector explicitly handles junctions/crossings: it detects line crossings, marks the meeting point as a junction and splits the line in two, and STRIPER then splits detected lines at crossing points — a junction-preserving behavior contrasting with Frangi/Sato vesselness which suppresses junctions.**
  > “Repeat (b) until (i) no valid line point is found, thus indicating the end of the line, or (ii) the selected line point is already part of a different line. Mark this point as a junction and split the line in two... After extracting the lin”  
  [https://www.biorxiv.org/content/10.1101/2020.02.28.969196v1.full]
- **Low SNR breaks naive line detection, so STRIPER adds an oriented-Gaussian line-enhancing preprocessing step specifically to integrate evidence along oriented lines before ridge detection — confirming the brief's claim that low-SNR favors orientation-integrating filters over per-pixel decisions.**
  > “Cryo-EM images typically have a very low signal-to-noise ratio, which is problematic for line-detection algorithms. We therefore included a line-enhancing pre-processing step in STRIPER.”  
  [https://www.biorxiv.org/content/10.1101/2020.02.28.969196v1.full]
- **The recall/precision operating point is controllable through a small, tunable parameter set: STRIPER exposes four parameters (filament width, mask width, upper and lower hysteresis thresholds) and provides a grid-search optimization to estimate the thresholds from 2-3 manually annotated micrographs — a concrete single-knob-with-default design analog.**
  > “To run the STRIPER filament procedure, four parameters need to be provided: (i) the filament width in pixels... (ii) the mask width in pixels, which is set to 100 by default... and (iii) the upper and (iv) the lower threshold used as hyster”  
  [https://www.biorxiv.org/content/10.1101/2020.02.28.969196v1.full]

## Cytoskeleton filament extraction (closest twin)
- **SOAX's core method is multiple Stretching Open Active Contours (SOACs) — parametric active-contour curves that are seeded on intensity ridges, evolve toward filament centerlines under combined image and stretching forces, and stop at filament tips. This is the canonical 'stretching open snake' technique for curvilinear-structure extraction, directly transferable to fungal hyphae as a centerline-tracing alternative to threshold/Hessian methods.**
  > “The underlying method of SOAX is the multiple Stretching Open Active Contours (SOACs) method that was proposed to extract the 3D meshwork of actin filaments imaged by confocal microscopy... A SOAC is a parametric curve that “evolves”: it is”  
  [https://www.nature.com/articles/srep09081]
- **SOAX explicitly forms and configures junctions: SOAC tips that collide with another SOAC body form T-junctions, nearby T-junctions are clustered into single junctions, and SOACs are cut/spliced so they do not end or bend sharply at junctions — i.e. junctions/crossings are first-class outputs that are preserved, the opposite of Frangi/Sato vesselness which suppresses junctions.**
  > “SOACs stop stretching at a filament end or when their tip collides with the body of another SOAC to form a T-junction... The final stage... is clustering nearby T-junctions into a single junction followed by configuring the local connectivi”  
  [https://www.nature.com/articles/srep09081]
- **SOAX is positioned specifically for low/inhomogeneous-SNR images where thresholding+thinning of a tubularity map fails; integrating evidence along an evolving contour is robust to noise, but the trade-off is that its two key parameters (ridge threshold tau and stretch factor) must be tuned to the image SNR — a controllable but not parameter-free sensitivity knob.**
  > “Thresholding results, however, can suffer from inhomogeneous signal-to-noise ratio (SNR)... While the SOAX method is robust against noise, its parameters need to be adjusted depending on the type of biopolymer and the image SNR... network e”  
  [https://www.nature.com/articles/srep09081]

## Diffusion preprocessing & chromatic-aberration rejection
- **Coherence-enhancing diffusion uses the structure tensor's eigenvectors/eigenvalues to build a diffusion tensor that preferentially smooths ALONG the dominant local orientation (the coherence direction), so it diffuses along curvilinear/flow-like structures rather than across them. This is the core orientation-aware preprocessing principle directly relevant to enhancing thin oriented hyphae before thresholding.**
  > “This coherence descriptor enables us to construct a diffusion tensor which steers the diffusion process in each channel in such a way that diffusion is encouraged along the preferred structure orientation.”  
  [https://www.sciencedirect.com/science/article/abs/pii/S0262885698001024]
- **The method is explicitly designed to enhance one-dimensional (line-like) structures and to close interrupted lines in poor-quality flow-like images, i.e. it performs gap-bridging along oriented structures, the same connectivity problem central to reconnecting fragmented hyphae.**
  > “Some images containing flow-like patterns are of poor quality, such that it becomes necessary to enhance them by closing interrupted lines”  
  [https://www.sciencedirect.com/science/article/abs/pii/S0262885698001024]
- **Coherence-enhancing diffusion is an enhancement/restoration filter for flow-like, line-like structures that smooths anisotropically mainly ALONG the dominant local orientation (eigenvector of the structure tensor with the smallest eigenvalue), not a binary segmenter. This makes it a preprocessing step that strengthens hyphal ridges before thresholding, exactly the role the brief asks of coherence-enhancing diffusion as a confounder-robust preprocessor.**
  > “This scale-space and image restoration technique has been introduced in [40] for the enhancement of flow-like textures with line-like structures... apply a nonlinear diffusion process whose diffusion tensor allows anisotropic smoothing by a”  
  [https://link.springer.com/article/10.1023/A:1008009714131]
- **A primary, explicitly stated design goal of coherence-enhancing diffusion is closing gaps in interrupted line-like structures by smoothing along the line direction — directly relevant to the dense-overlap/intensity-gap regime and as an alternative gap-bridging mechanism to Dijkstra reconnection.**
  > “When the goal consists e.g. of closing gaps in an interrupted line-like structure, it is clear that slight deviations from the correct smoothing direction will destroy any desired filter effect and result in a deterioration of the line by i”  
  [https://link.springer.com/article/10.1023/A:1008009714131]

## Foundation segmenters (SAM/SAM2) & small-data transfer
- **Medical SAM3 is built by fully fine-tuning the SAM3 foundation model on large-scale heterogeneous 2D and 3D medical imaging datasets to enable text-prompt-driven (semantic) segmentation without requiring spatial prompts such as points or boxes.**
  > “a foundation model for universal prompt-driven medical image segmentation, obtained by fully fine-tuning SAM3 on large-scale, heterogeneous 2D and 3D medical imaging datasets”  
  [https://arxiv.org/html/2601.10880v1]
- **Fine-tuning SAM2 by adapting only the mask decoder (no added architectural layers) achieves high segmentation accuracy with small training datasets, supporting a small-data transfer-learning workflow rather than training from scratch.**
  > “By coupling mask-decoder fine-tuning with biologically informed post-processing, our framework achieves robust segmentation across diverse imaging modalities... High segmentation accuracy (Dice/Jaccard scores) achieved with small datasets.”  
  [https://www.biorxiv.org/content/10.1101/2025.11.08.687405.full.pdf]
- **The fine-tuning pipeline runs in a single Google Colab notebook without specialized hardware, lowering the compute barrier (no large GPU cluster required) for domain-specific SAM2 adaptation.**
  > “we introduce a lightweight, open-source Google Colab pipeline that enables efficient fine-tuning of SAM2 on domain-specific datasets without additional architectural layers or specialized hardware”  
  [https://www.biorxiv.org/content/10.1101/2025.11.08.687405.full.pdf]
- **Fine-tuned SAM2 substantially improves accuracy over zero-shot/basic SAM2 and matches leading purpose-built segmentation tools.**
  > “fine-tuned SAM2 demonstrates substantial gains of accuracy relative to basic SAM2 and matches leading tools”  
  [https://www.biorxiv.org/content/10.1101/2025.11.08.687405.full.pdf]

## Hessian vesselness & junction suppression
- **Hessian/Frangi eigenvalue vesselness loses vessel-like structures at junctions because at intersections both eigenvalues become similarly large, driving the vesselness measure toward zero — the named structural reason naive vesselness suppresses X/T crossings.**
  > “both eigenvalues have similarly large values leading to a vesselness measure close to zero. Thus, vessel-like structures can be lost at junctions and therefore”  
  [https://arxiv.org/pdf/1709.05495]
- **The bowler-hat transform keeps the network connected at junctions: because longer line structuring elements still fit within a junction area, junctions are enhanced as brightly as the vessels joining them, unlike many other enhancement methods.**
  > “a junction should appear bright like those vessels joining that junction, something that many other vessel enhancement methods fail to do. This is due to the ability to fit longer line-based structural elements within the junction area. As ”  
  [https://arxiv.org/pdf/1709.05495]
- **Frangi vesselness derives a continuous tubular-structure response from the eigenvalues of the multiscale second-order local image structure (the Hessian), i.e. it is a ridge/tubularity filter driven by local second-derivative geometry rather than an intensity threshold — exactly the curvilinear-detection family the brief seeks as an out-of-family alternative to phase congruency.**
  > “The authors examined "the multiscale second order local structure of an image (Hessian)" to develop a vessel enhancement filter. A vesselness metric was derived from the Hessian's eigenvalues”  
  [https://link.springer.com/chapter/10.1007/bfb0056195]
- **Frangi's vesselness filter analyzes the multiscale second-order local structure of an image (the Hessian matrix) specifically to develop a vessel/tubular-structure enhancement filter, rather than relying on intensity thresholding.**
  > “The multiscale second order local structure of an image (Hessian) is examined with the purpose of developing a vessel enhancement filter.”  
  [https://link.springer.com/chapter/10.1007/bfb0056195]
- **The vesselness measure is derived from all eigenvalues of the Hessian matrix, meaning the discrimination of tubular structures is based on local ridge geometry (eigenvalue relationships) rather than a per-pixel intensity decision.**
  > “A vesselness measure is obtained on the basis of all eigenvalues of the Hessian.”  
  [https://link.springer.com/chapter/10.1007/bfb0056195]

## Morphology, path-openings, tensor voting, curvelets, active contours
- **Path openings are morphological filters that explore all paths from a defined class and filter them with a length criterion, designed specifically to preserve long, thin, and tortuous structures in gray-level images — directly applicable to thin curvilinear hyphae detection.**
  > “preserve long, thin, and tortuous structures in gray level images... These operators explore all paths from a defined class, and filter them with a length criterion.”  
  [https://pubmed.ncbi.nlm.nih.gov/24569442/]
- **RORPO is a non-linear, non-local mathematical-morphology operator (built on path operators) that explicitly avoids the local-neighborhood analysis used by Hessian-style filters, which it says is poorly adapted to curvilinear anisotropy and causes false detections — positioning it as a genuinely out-of-family alternative to the Hessian/phase-congruency baselines.**
  > “This local analysis is not well adapted to the anisotropy of such structures, and often results in false detections. By using path operators, our method called RORPO tends to avoid these false detections, and better detects curvilinear stru”  
  [https://www.researchgate.net/publication/315838176_RORPO_A_morphological_framework_for_curvilinear_structure_analysis_Application_to_the_filtering_and_segmentation_of_blood_vessels]
- **Path openings and closings are morphological operators specifically designed to preserve long, thin, and tortuous (curvilinear) structures in grayscale images, matching the ridge/filament-like geometry of hyphae.**
  > “Path openings and closings are morphological tools used to preserve long, thin, and tortuous structures in gray level images.”  
  [https://pubmed.ncbi.nlm.nih.gov/24569442/]
- **The parsimonious path-opening algorithm has computational complexity that is linear in the number of pixels and independent of the structuring-element/opening length, and runs in streaming mode for both integer and floating-point input.**
  > “Its complexity is linear with respect to the number of pixels, independent of the size of the opening. Furthermore, it is fast for any input data accuracy (integer or floating point) and works in stream.”  
  [https://pubmed.ncbi.nlm.nih.gov/24569442/]
- **The method is extended to incomplete paths containing gaps, so that noise-corrupted curvilinear structures with signal dropout can be processed with the same approach and complexity — directly relevant to gap-bridging across intensity gaps in dense/overlapping hyphae.**
  > “Parsimonious path openings are also extended to incomplete paths, i.e., paths containing gaps. Noise-corrupted paths can thus be processed with the same approach and complexity.”  
  [https://pubmed.ncbi.nlm.nih.gov/24569442/]

## Oriented voting, RHT & Radon (low-SNR line integration)
- **The RHT integrates evidence of linear structure along orientations through each pixel: a circular window of diameter D_W rolls across the image, and for each pixel it performs a Hough transform restricted to rho=0, reducing the (rho,theta) space to a 1-D function R(theta,x,y) that quantifies how much coherent linear structure passes through that pixel at each angle theta. This per-pixel oriented accumulation is exactly the 'integrate evidence along a hypothesized line' strategy the brief favors for low-SNR curvilinear detection.**
  > “The RHT mapping is performed on a circular domain, di-ameter DW , centered on each image-space pixel ( x0,y 0) in turn (Figure 2, step 4). Then a Hough transform is performed on this area, limited to rho = 0 (Figure 2, step 5). Thus the rho”  
  [https://arxiv.org/pdf/1312.1338]
- **The RHT exposes a single tunable sensitivity knob Z (a percentage threshold for how much of a candidate line must contain signal), where Z*D_W pixels must contain signal along a direction for that direction to be recorded; setting Z below 100% deliberately accepts structures that are physically coherent even when not visibly connected — i.e. a controllable recall/precision knob with explicit gap-tolerance.**
  > “All intensity over a set inten-sity threshold Z is stored as R(theta,x 0,y 0)... Z is a percentage. In every direction theta, Z×DW pixels must contain signal in order for the transform to record the data in that direction.”  
  [https://arxiv.org/pdf/1312.1338]
- **RHT preprocessing converts the grayscale image into a binary bitmask before the Hough accumulation, via unsharp masking: convolve with a top-hat smoothing kernel of diameter D_K, subtract the smoothed map from the original (a high-pass / large-scale-structure suppression), then threshold at 0. This D_K-controlled background subtraction is directly analogous to the PhenoTypic baseline's large-sigma Gaussian background subtraction, and yields a binary mask intermediate.**
  > “The image is convolved with a two-dimensional top-hat smoothing kernel of a user-deﬁned diameter, DK (Figure 2, step 1). The smoothed data is then subtracted from the original data (Figure 2, step 2), and the resulting map is thresholded at”  
  [https://arxiv.org/pdf/1312.1338]
- **The RHT is explicitly designed to detect linear structure independent of overall brightness, making it suited to faint structures near the noise floor whose signal is geometric (coherent linearity) rather than intensity-based; it encodes a per-pixel probability of belonging to a coherent linear structure rather than thresholding intensity.**
  > “The RHT operates on two-dimensional data and is de-signed to be sensitive to linear structure irrespective of the overall brightness of the region... The RHT does not merely identify ﬁbers; it encodes the probability that any given image pi”  
  [https://arxiv.org/pdf/1312.1338]
- **The Rolling Hough Transform integrates evidence along hypothesized lines by performing a Hough transform on a rolling circular window (diameter DW) centered on each pixel, limited to rho=0, reducing parameter space to a 1D function of orientation R(theta, x0, y0) per pixel; this line-integration approach is what makes it suited to faint, low-contrast linear structure.**
  > “The RHT mapping is performed on a circular domain, diameter DW, centered on each image-space pixel (x0,y0) in turn... Then a Hough transform is performed on this area, limited to rho = 0... Thus the rho-theta space is reduced to a one-dimen”  
  [https://arxiv.org/pdf/1312.1338]

## Particle-physics & seismic tracking (gap-bridging)
- **Kalman Filter-based track finding is the dominant, proven approach for connecting detector hits into continuous tracks at the LHC, making it a battle-tested gap-bridging / sequential-hypothesis-extension analog for reconnecting fragmented curvilinear structures.**
  > “The most common track finding techniques in use today...are those based on the Kalman Filter, and are known to provide high physics performance, are robust, and are in use today at the LHC.”  
  [https://arxiv.org/abs/1601.08245]
- **Track finding/fitting is among the most computationally challenging problems, which motivates re-engineering the Kalman approach for modern parallel hardware -- relevant to whether a Kalman-style reconnection step is affordable in an image pipeline.**
  > “Track finding and fitting is one of the most computationally challenging problems”  
  [https://arxiv.org/abs/1601.08245]
- **Ant colony optimization can be applied to seismic coherency data to automatically track faults, improving fault continuity (i.e. bridging discontinuities/gaps in the curvilinear fault network) — directly analogous to gap-bridging reconnection of fragmented thin structures.**
  > “can effectively reduce the noise level and improve the continuity of faults on seismic coherency cube”  
  [https://www.sciencedirect.com/science/article/abs/pii/S0098300412002804]
- **The same ant-colony pass that reconnects fault continuity also suppresses noise, i.e. it performs joint denoising and connectivity enhancement on a noisy attribute volume — relevant to the low-SNR + connectivity dual regime in the brief.**
  > “can effectively reduce the noise level”  
  [https://www.sciencedirect.com/science/article/abs/pii/S0098300412002804]

## Phase congruency & optimally oriented flux
- **OOF (Optimally Oriented Flux) localizes its detection at the boundary of a local spherical region and therefore does NOT include the region in the vicinity of the structure, making it robust against disturbance from closely-located adjacent structures — unlike the Hessian, whose second-derivative-of-Gaussian response averages over a neighborhood that can include nearby objects.**
  > “The major advantage of the proposed method is that the OOF based detection is localized at the boundary of the local spherical region. Distinct from the Hessian matrix, OOF does not consider the region in the vicinity of the structure where”  
  [https://cse.hkust.edu.hk/~achung/eccv08_law_chung.pdf]
- **Hessian-based detection (the Frangi/Sato lineage) degrades specifically when intensity around a structure is non-homogeneous due to closely-located adjacent structures, because the second-derivative-of-Gaussian differential effect is corrupted by neighboring objects — the precise failure mode OOF is designed to avoid (directly relevant to dense/overlapping hyphae causing gaps).**
  > “if the intensity around the objects is not homogeneous due to the presence of closely located adjacent structures, the diffe rential effect given by the second derivatives of Gaussian is adversely affected.”  
  [https://cse.hkust.edu.hk/~achung/eccv08_law_chung.pdf]
- **OOF requires no large-scale Gaussian smoothing (sigma fixed at 1 in all implementations), so it preserves edge sharpness of structure boundaries and is more robust to image noise than the Hessian, whose scale factor must grow for large structures and thereby blurs boundaries that noise then corrupts.**
  > “For OOF, the detection does not require Gaussian smoothing using a large scale factor (σ =1 for OOF). It re-tains the edge sharpness of the structure boundaries. Therefore, the OOF detection has higher robustness against image noise than th”  
  [https://cse.hkust.edu.hk/~achung/eccv08_law_chung.pdf]
- **On a real angiographic volume, OOF extracted several weak-intensity, low-SNR vessels (arrows 1,2,3,4,7) that the Hessian-based method missed, and OOF resolved the small separation between closely-located vessels that the Hessian merged — empirical evidence for the low-SNR-recovery and gap/junction-preservation advantage over the Hessian/Frangi baseline.**
  > “several vessels with weak intensity (arrows 1, 2, 3, 4 and 7) are missed by the Hessian based method where the OOF based method has no problem to extract them ... the Hessian based method misidentiﬁes closely located vessels as me rged stru”  
  [https://cse.hkust.edu.hk/~achung/eccv08_law_chung.pdf]
- **OOF localizes its detection at the boundary of a local spherical region rather than integrating over the structure's neighborhood, making it robust to disturbance from closely located adjacent structures (the regime analogous to dense/overlapping hyphae) where Hessian-based methods degrade.**
  > “The major advantage of the proposed method is that the OOF based detection is localized at the boundary of the local spherical region. Distinct from the Hessian matrix, OOF does not consider the region in the vicinity of the structure where”  
  [https://cse.hkust.edu.hk/~achung/eccv08_law_chung.pdf]

## Remote sensing & materials networks
- **clDice is a training-time loss function (soft-clDice) that is architecture-agnostic — it can be added to any deep-learning segmentation network (U-Net 2D/3D, FCN 2D/3D were tested) to improve topology of thin/tubular structure masks.**
  > “Our soft-clDice loss can be applied to any arbitrary segmentation network ... can be readily deployed to any other deep learning-based segmentation.”  
  [https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf]
- **The soft-skeletonization that makes clDice differentiable is built from iterative min- and max-pooling (the grey-scale analogues of morphological erosion and dilation), so it is fully differentiable and optimizable by gradient descent.**
  > “Min- and max filters are commonly used as the grey-scale alternative of morphological dilation and erosion ... [soft-skeleton uses] iterative min- and max-pooling ... fully differentiable, real-valued, optimizable.”  
  [https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf]
- **Training with clDice recovers true connections that the standard soft-Dice loss drops as false negatives, and avoids the over-segmentation/false-positive connections that soft-Dice produces — directly improving network connectivity. On Massachusetts Roads, Betti-0 error fell from 1.474 to 0.920 and clDice score rose from 70.79% to 76.25%.**
  > “Our networks trained on the proposed loss term recover connections, which were false negatives when trained with the soft-Dice loss.”  
  [https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf]

## SOTA curvilinear benchmarks
- **Betti matching is a topology-aware training loss that matches topological features between predicted and ground-truth segmentations via induced matchings between their persistence barcodes, implicitly accounting for the spatial relationship of features (unlike lifetime-only matching in TopoNet).**
  > “The Betti matching loss builds on techniques from topological data analysis, specifically persistent homology.”  
  [https://arxiv.org/html/2407.04683]
- **The method computes the supervision target (a 'tubed skeleton' of the ground truth) using simple CPU-based image-processing operations (e.g. scikit-image) during data loading or precomputed, rather than computing a differentiable skeleton on predictions on the GPU, so it adds only minimal GPU memory and training time.**
  > “can be computed with simple CPU-based operations using common image processing frameworks (e.g. scikit-image) as part of data-loading or even be precomputed”  
  [https://arxiv.org/html/2404.03010v1]
- **It is the first multi-class-capable topology-aware loss for thin-structure segmentation; clDice ran out of memory on the 13-class TopCoW task on an A100 40GB GPU while Skeleton Recall Loss scaled to all classes.**
  > “clDice Loss rendered it infeasible on all 13 classes as it exceeded the memory capacity of an A100 40GB GPU”  
  [https://arxiv.org/html/2404.03010v1]

## Steger, steerable & matched filters
- **Steger's detector locates line points via the Hessian: it computes Gaussian-smoothed first/second partial derivatives (rx, ry, rxx, rxy, ryy), takes the eigenvector of the Hessian whose eigenvalue has maximum absolute value as the local line direction n(t), and declares a line point where the second directional derivative perpendicular to the line is large — i.e. detection is governed by oriented ridge second-derivative geometry, not an intensity threshold.**
  > “The direction in which the second directional derivative of z(x, y) takes on its maximum absolute value is used as the direction n(t). This direction can be determined by calculating the eigenvalues and eigenvectors of the Hessian matrix”  
  [http://howardzzh.com/research/papers/vision/1998.PAMI.Steger.UnbiasedDetector.pdf]
- **Steger's detector models a line not as an intensity threshold but as a ridge defined by the first and second directional derivatives, extracting line points where the second directional derivative across the line is maximal; salient lines are selected by the magnitude of this second derivative, giving the ridge-geometry signature the brief calls for rather than a per-pixel intensity test.**
  > “The approach only uses the first and second directional derivatives of an image for the extraction of the line points. No specialized directional filters are needed.”  
  [http://howardzzh.com/research/papers/vision/1998.PAMI.Steger.UnbiasedDetector.pdf]
- **Junctions are detected and preserved as explicit network structure: linking follows one branch, and when a candidate point already belongs to another line it is marked as a junction and the line is split there, producing a connected lines-and-junctions graph rather than a fragmented mask.**
  > “If this happens, the point is marked as a junction, and the line that contains the point is split into two lines at the junction point, unless it is the first point of the currently processed line, in which case a line describing a closed l”  
  [http://howardzzh.com/research/papers/vision/1998.PAMI.Steger.UnbiasedDetector.pdf]
- **Line linking uses a hysteresis-style double threshold on the second directional derivative: new lines start only where the response exceeds a user-selectable upper threshold, while points are appended while above a lower threshold, giving a single tunable sensitivity knob to trade recall against precision and to follow faint lines into intersections.**
  > “New lines are created as long as the starting point has a second directional derivative that lies above a certain, user-selectable upper threshold. Points are added to the current line as long as their second directional derivative is great”  
  [http://howardzzh.com/research/papers/vision/1998.PAMI.Steger.UnbiasedDetector.pdf]
- **B-COSFIRE filters achieve orientation selectivity for curvilinear/vessel-like structures by computing the weighted geometric mean of a pool of Difference-of-Gaussians filter responses aligned collinearly, i.e. it integrates oriented evidence along a hypothesized line rather than making per-pixel threshold decisions.**
  > “A B-COSFIRE filter achieves orientation selectivity by computing the weighted geometric mean of the output of a pool of Difference-of-Gaussians filters, whose supports are aligned in a collinear manner.”  
  [https://pubmed.ncbi.nlm.nih.gov/25240643/]

## Topology / centerline-aware losses & tracing nets
- **Skeleton Recall Loss preserves connectivity/topology of thin tubular structures while reducing computational overhead by more than 90% relative to existing topology-aware losses, by replacing intensive GPU-based calculations with inexpensive CPU operations.**
  > “reducing computational overheads by more than 90%... circumventing intensive GPU-based calculations with inexpensive CPU operations”  
  [https://arxiv.org/pdf/2404.03010]
- **In direct comparison to clDice (a differentiable-skeletonization topology loss), clDice adds ~88% training time and ~52% VRAM over the plain nnU-Net backbone averaged across 5 datasets, whereas Skeleton Recall Loss adds only ~8% training time and ~2% VRAM.**
  > “For our differentiable skeleton baseline clDice Loss, this leads to approximately 88% additional training time and 52% more VRAM consumption compared to the plain nnUNet backbone when averaged across our 5 datasets (excluding multi-class To”  
  [https://arxiv.org/pdf/2404.03010]
- **The loss computes a skeleton of the ground-truth mask, dilates it with a radius-2 diamond kernel to make it tubular, and applies a soft recall loss over that tubular skeleton region to emphasize thin centerline structures.**
  > “Subsequently, we dilate the skeleton with a diamond kernel of radius 2 to make it tubular, thereby enlarging the effective area for loss computation around the otherwise thin, single-pixel-wide skeleton.”  
  [https://arxiv.org/pdf/2404.03010]
- **Embedding persistence images (a vectorized persistent-homology representation) as a topological feature into a U-Net segmentation network preserves curvilinear/vessel connectivity and reduces fragmentation (vessel breaks), cutting the beta-0 connected-component error on the DRIVE retinal vessel benchmark by ~41% (from 217.2 to 126.8) versus a baseline U-Net trained with cross-entropy.**
  > “incorporation of topological features not only refines the pixel-level accuracy but also robustly preserves vessel connectivity, thereby mitigating issues such as vessel breaks”  
  [https://arxiv.org/html/2601.18045]

## U-Net thin-structure segmentation & class imbalance
- **soft-clDice is a computationally efficient, differentiable loss function that can be used to train arbitrary neural segmentation networks (e.g. any U-Net), directly addressing the brief's topology-aware DL loss for thin tubular structures.**
  > “Extending this, we propose a computationally efficient, differentiable loss function (soft-clDice) for training arbitrary neural segmentation networks.”  
  [https://arxiv.org/abs/2003.07311]
- **clDice is computed on the intersection of the segmentation masks and their morphological skeleta (centerlines), making it a centerline/skeleton-aware similarity measure rather than a per-pixel overlap measure.**
  > “We introduce a novel similarity measure termed centerlineDice (short clDice), which is calculated on the intersection of the segmentation masks and their (morphological) skeleta.”  
  [https://arxiv.org/abs/2003.07311]
- **clDice is theoretically proven to guarantee topology preservation up to homotopy equivalence for binary 2D and 3D segmentation, which directly targets connectivity preservation (the junction/gap-bridging requirement in the brief).**
  > “We theoretically prove that clDice guarantees topology preservation up to homotopy equivalence for binary 2D and 3D segmentation.”  
  [https://arxiv.org/abs/2003.07311]
- **Training with soft-clDice empirically improves connectivity and graph-level network quality (not just pixel overlap), supporting the brief's preference for reconnected, junction-preserving networks for downstream length/branching metrics.**
  > “Training on soft-clDice leads to segmentation with more accurate connectivity information, higher graph similarity, and better volumetric scores.”  
  [https://arxiv.org/abs/2003.07311]