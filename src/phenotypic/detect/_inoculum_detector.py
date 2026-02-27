from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import gc
import logging
from typing import Literal

import cv2
import numpy as np
import scipy.ndimage as ndimage
import skimage.filters as filters
import skimage.morphology as morphology

from phenotypic.abc_ import ObjectDetector
from phenotypic.enhance._homomorphic_filter import HomomorphicFilter
from phenotypic.enhance._multiscale_log_enhancer import MultiscaleLoGEnhancer
from phenotypic.refine._gmm_core_extractor import GMMCoreExtractor

logger = logging.getLogger(__name__)


class InoculumDetector(ObjectDetector):
    """Detect inoculation sites and small spots on agar plates via multi-scale blob enhancement.

    InoculumDetector identifies inoculation spots (e.g., from pin-deposition tools,
    liquid spotting, or serial dilutions) using a purpose-built pipeline: homomorphic
    illumination correction, morphological opening, multi-scale Laplacian-of-Gaussian
    blob enhancement, adaptive/global thresholding, and optional GMM-based core
    extraction. All processing occurs on local copies; ``image.detect_mat`` is never
    modified (enforced by ObjectDetector integrity validation).

    Args:
        thresh_method: Thresholding method for binary segmentation of the LoG-enhanced
            image. Options: ``'otsu'`` (global, histogram-based; recommended for
            balanced intensity), ``'mean'`` (simple global), ``'local'`` (adaptive,
            spatial window-based), ``'triangle'``, ``'minimum'``, ``'isodata'``,
            ``'li'``. Default: ``'otsu'``.
        subtract_background: If True, apply white top-hat (closing - original) before
            homomorphic filtering to suppress large-scale (plate-level) intensity
            variation. Useful for uneven illumination or agar color gradients.
            Default: False.
        background_tophat_width: Side length of the square structuring element for
            top-hat (pixels). Larger values remove broader intensity variations but
            may suppress inoculum edges if too large. Typical range: 200–400 pixels.
            Default: 300.
        homomorphic_sigma: Gaussian sigma (pixels) that separates illumination from
            reflectance in homomorphic filtering. Larger values suppress slower
            illumination gradients (vignetting, uneven LED lighting). Typical range:
            100–500 pixels. Default: 300.0.
        homomorphic_gamma_low: Gain (multiplier) for low-frequency (illumination)
            component post-separation. Values < 1.0 suppress illumination variation.
            Typical range: 0.3–0.8. Default: 0.5.
        homomorphic_gamma_high: Gain for high-frequency (reflectance/colony) component.
            Values > 1.0 enhance inoculum contrast. Typical range: 1.0–2.5.
            Default: 1.5.
        opening_shape: Footprint shape for morphological opening: ``'square'``
            (conservative, all directions), ``'diamond'`` (diagonal-reduced),
            or ``'disk'`` (circular, naturalistic). Default: ``'disk'``.
        opening_width: Footprint size (pixels) for morphological opening. Removes
            noise/salt-and-pepper smaller than this footprint. Typical range:
            20–100 pixels (scales with resolution). Default: 50.
        log_min_radius: Minimum blob radius (pixels) for multi-scale Laplacian-of-Gaussian
            enhancement. Set based on smallest inoculum diameter at your resolution.
            Typical range: 5–30 pixels. Default: 25.0.
        log_max_radius: Maximum blob radius (pixels). Typical range: 20–80 pixels.
            Default: 50.0.
        log_num_scales: Number of logarithmically spaced LoG scales between
            ``log_min_radius`` and ``log_max_radius``. More scales catch size variation
            but increase computation. Typical range: 3–8. Default: 5.
        enable_gmm: If True (default), apply Gaussian Mixture Model core extraction
            to refine detected regions down to bright, compact cores. Useful for
            inocula with diffuse/scattered fluorescence or age-related diffusion.
        gmm_n_components: Gaussian mixture components per region. 2 = core vs. surround
            (typical). 3+ for multi-scale intensity structure. Default: 2.
        gmm_separation_threshold: Normalized Euclidean distance between GMM component
            means. Below this threshold, region is left unmodified (no clear core).
            Typical range: 0.8–1.2. Default: 0.9.
        gmm_min_core_area: Minimum acceptable GMM-extracted core area (pixels).
            Prevents spurious tiny cores. Typical range: 10–50 pixels. Default: 30.
        gmm_morph_open_radius: Morphological opening radius (pixels) applied to the
            GMM binary core mask. Removes noise from core extraction. Typical range:
            2–15 pixels. Default: 10.
        gmm_morph_close_radius: Morphological closing radius (pixels) applied to the
            core mask. Fills small holes in the core. Typical range: 1–5 pixels.
            Default: 2.

    Attributes:
        thresh_method, subtract_background, background_tophat_width,
        homomorphic_sigma, homomorphic_gamma_low, homomorphic_gamma_high,
        opening_shape, opening_width, log_min_radius, log_max_radius,
        log_num_scales, enable_gmm, gmm_n_components,
        gmm_separation_threshold, gmm_min_core_area, gmm_morph_open_radius,
        gmm_morph_close_radius

    Returns:
        Image: Input image with ``objmask`` (binary inoculum mask) and
        ``objmap`` (labeled object map with unique integer IDs per spot) populated.

    Raises:
        ValueError: If ``opening_shape`` is not one of ``'square'``, ``'diamond'``,
            or ``'disk'``.

    **Intuition**

    Inoculation spots are characteristically small, blob-like objects (5–200 pixels
    diameter) deposited on agar with distinct intensity relative to background. They
    differ from colonies in being deposited *before* growth, so they are compact and
    morphologically simple. Detecting them requires specialized tools:

    - **Homomorphic filtering** corrects plate-level illumination gradients (DSLR
      sensor vignetting, uneven LED ring lights, shadows from well walls) that would
      otherwise skew global thresholds.
    - **Morphological opening** suppresses fine noise (dust, agar texture, salt-and-pepper
      artifacts from JPEG compression) without destroying compact inocula.
    - **Multi-scale Laplacian-of-Gaussian (LoG)** enhancement detects blob-like structures
      across a range of radii simultaneously, making it invariant to inoculum size
      variation (serial dilutions, different pin tools, etc.).
    - **Thresholding** produces a binary decision; adaptive (local) methods work better
      for spatially varying background; global (Otsu) works when illumination is
      relatively uniform.
    - **GMM core extraction** optionally refines boundaries, useful when inocula have
      diffuse edges or age-related intensity gradients.

    **Use Cases**

    1. **Pin-tool inoculation (laboratory standard):** High-density plates (96-well,
       384-well format or 8×12 manually pinned) where inocula are 30–80 pixels diameter
       at typical DSLR resolution. Homomorphic filtering + LoG capture geometric
       variation while thresholding adapts to media color (YPD yellow, LB tan, minimal
       white).

    2. **Spot-dilution assays:** Serial 10-fold dilutions produce inocula ranging
       10–150 pixels. The multi-scale LoG radius window ``[log_min_radius, log_max_radius]``
       automatically detects all spot sizes without manual adjustment.

    3. **Pre-growth phenotyping baseline:** Image plates at inoculation (T=0) and again
       after growth (T=24h, etc.). InoculumDetector on T=0 images provides reference
       coordinates for downstream growth measurements and colony tracking.

    4. **Liquid spotting assays:** Automated liquid handlers deposit small volumes;
       resulting spots are compact and well-separated. GMM core extraction refines
       boundaries for accurate area/circularity measurements.

    5. **High-resolution scanning (e.g., raw photography):** Scanner vignetting and
       uneven backlighting create severe illumination gradients. Homomorphic filtering
       + large ``homomorphic_sigma`` stabilizes detection across the plate.

    **Parameter Effects**

    **Pixel-based parameters** (scale with image resolution):

    *background_tophat_width, opening_width, log_min_radius, log_max_radius,
    gmm_morph_open_radius, gmm_morph_close_radius*

    All pixel-based parameters should be scaled by the formula:

        adjusted_param = default_param × (image_width / 512)

    At reference resolution (512×768 DSLR images):
      - typical inoculum diameter: 30–80 pixels
      - small noise artifacts: 1–5 pixels
      - agar texture: 2–10 pixel scale

    Examples for common resolutions (scaling default values):

    | Parameter | 512×768 (ref) | 640×960 (1.25×) | 1024×1536 (2×) | 2000×3000 (3.9×) |
    |-----------|---------------|-----------------|----------------|------------------|
    | background_tophat_width | 300 | 375 | 600 | 1170 |
    | opening_width | 50 | 63 | 100 | 195 |
    | log_min_radius | 25.0 | 31.25 | 50.0 | 97.5 |
    | log_max_radius | 50.0 | 62.5 | 100.0 | 195.0 |
    | gmm_morph_open_radius | 10 | 13 | 20 | 39 |
    | gmm_morph_close_radius | 2 | 3 | 4 | 8 |

    **homomorphic_sigma (pixel-based, but special handling):**

    Controls Gaussian blur scale in illumination separation. Larger sigma catches broader
    vignetting; smaller sigma preserves fine inoculum details. Recommend scaling similarly
    to other pixel parameters, but with some manual tuning for your lighting rig:

    - Well-lit plates (even LED ring): sigma ≈ 150–200 pixels (smaller → preserve detail)
    - Uneven illumination (vignetting, shadows): sigma ≈ 250–400 pixels (larger → smooth gradients)
    - Scanning with severe vignetting: sigma ≈ 400–600 pixels

    **Intensity-based parameters** (DO NOT scale with resolution):

    *homomorphic_gamma_low, homomorphic_gamma_high, gmm_separation_threshold, thresh_method*

    These depend on camera response, agar color, and inoculum pigmentation—not image size.

    - **homomorphic_gamma_low:** Lower values (0.3–0.5) suppress illumination variation
      more aggressively. Use for severe vignetting. Higher values (0.6–0.8) preserve
      subtle features.

    - **homomorphic_gamma_high:** Higher values (1.5–2.5) boost inoculum contrast.
      Useful if inocula are dim or high-density (touching). Lower values (1.0–1.3) for
      bright, isolated inocula.

    - **gmm_separation_threshold:** Controls GMM core-finding sensitivity. Lower values
      (0.7–0.8) only refine regions with clear core-surround separation. Higher values
      (1.0–1.2) refine more permissively, risking spurious shrinking.

    - **thresh_method:** Depends on illumination and background:
      - ``'otsu'``: Bimodal histogram (good for balanced lighting, standardized media).
      - ``'local'``: Spatially varying background (use if illumination gradient persists
        after homomorphic filtering).
      - ``'mean'``: Simple, fast; works when inocula are distinctly brighter than background.
      - ``'triangle'``, ``'minimum'``, ``'isodata'``, ``'li'``: Specialized; try if Otsu fails.

    **Colony morphology context:**

    InoculumDetector assumes blob-like inocula (roughly circular, compact). Effect varies:

    - **Round, compact inocula (standard pin-tool, liquid spots):** All parameters work well.
      LoG enhancement and GMM extraction shine.

    - **Elongated or irregular inocula (manual spotting, dried droplets):** LoG still detects
      overall feature, but GMM core extraction may under-estimate true extent. Consider
      disabling GMM (``enable_gmm=False``) or increasing ``gmm_separation_threshold`` to
      relax refinement.

    - **Very small inocula (<5 pixels at reference resolution):** Reduce ``log_min_radius`` and
      ``opening_width`` proportionally. May need ``enable_gmm=False`` if cores are too small.

    - **Diffuse inocula (aged, pigmented, fluorescent):** Increase ``gmm_separation_threshold``
      or disable GMM. Consider post-processing with dilation to restore original extent.

    **Caveats and Limitations**

    1. **LoG blob model assumption:** LoG detection assumes roughly circular, isotropic blobs.
       Highly filamentous, elongated, or fractal-like inocula (e.g., fungal mycelium) may
       be under-detected or fragmented. Workaround: manually adjust ``log_min_radius`` /
       ``log_max_radius`` to match inoculum extent.

    2. **Homomorphic filter edge artifacts:** Homomorphic filtering can create bright/dark
       halos at image edges and near saturated regions. Mitigation: crop image before
       detection or use ``subtract_background=True`` for pre-flattening.

    3. **GMM core extraction over-refinement:** If ``gmm_n_components=2`` and inocula have
       distributed intensity (not two-mode), GMM may shrink detection to a subset of true
       extent. Check ``gmm_separation_threshold``; if separation < threshold, region is
       left unmodified (good). If separation ≥ threshold but result seems spurious, try
       increasing threshold or disabling GMM.

    4. **Thresholding failure on unimodal histograms:** If all pixels (background + inocula)
       have overlapping intensities, even Otsu thresholding may fail. Solution: apply
       preprocessing enhancer (e.g., ``CLAHE`` for contrast) before InoculumDetector.

    5. **GridImage handling:** If input is a GridImage, InoculumDetector automatically keeps
       the largest inoculum per grid cell (Step 10). Useful for organized array formats but
       may discard secondary inocula in overcrowded wells. Mitigate by increasing ``log_max_radius``
       or post-processing to recover removed objects.

    6. **Memory overhead on high-resolution images:** Homomorphic filtering and LoG
       enhancement allocate temporary arrays proportional to image size. On 4000×6000 DSLR
       raw scans, expect ~150–300 MB peak. LoG computation is O(n_scales × H × W); set
       ``log_num_scales`` conservatively (~3–5) for very large images.

    7. **No multi-modal separation:** All detected regions are treated independently. If two
       touching inocula of similar intensity exist, they may merge into one object. To
       separate touching inocula, apply ``ObjectRefiner`` (e.g., ``WatershedRefiner``) after
       InoculumDetector.

    **Mathematical/Technical Background**

    - **Homomorphic filtering:** Decomposes image I(x,y) = i(x,y) × r(x,y) (illumination ×
      reflectance). Applies Gaussian low-pass to log(I) to separate, then reconstructs.
      See ``HomomorphicFilter`` for details.

    - **Laplacian-of-Gaussian:** LoG = ∇² G_σ, a rotationally symmetric operator sensitive
      to blob-like structures. Multi-scale LoG computes LoG at scales σ ∈ [σ_min, σ_max]
      and takes the maximum response. Detects blobs regardless of exact scale in that range.

    - **GMM core extraction:** Fits mixture of ``gmm_n_components`` Gaussians to grayscale
      intensity within each detected region. Assumes core (bright) and surround (dim) follow
      separate Gaussian distributions. If separation (Euclidean distance between means in
      normalized feature space) exceeds threshold, binarizes at the component means to isolate
      the bright core. See ``GMMCoreExtractor`` for implementation.

    Examples:
        Basic inoculum detection on a pinned plate::

            from phenotypic import Image
            from phenotypic.detect import InoculumDetector

            plate = Image.imread("inoculated_plate.jpg")
            detector = InoculumDetector()
            detected = detector.apply(plate)
            print(f"Detected {detected.num_objects} inoculation sites")

        Parameter tuning for a high-resolution scan with noisy background::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur, CLAHE
            from phenotypic.detect import InoculumDetector

            # For 2000×3000 image: scale pixel params by ~3.9×
            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.0),
                CLAHE(clip_limit=2.0),  # boost local contrast first
                InoculumDetector(
                    subtract_background=True,
                    background_tophat_width=600,  # scaled from 300
                    homomorphic_sigma=400.0,  # larger for broader gradients
                    opening_width=100,  # scaled from 50
                    log_min_radius=50.0,  # scaled from 25
                    log_max_radius=100.0,  # scaled from 50
                    thresh_method='local',  # spatial adaptation
                    enable_gmm=True,
                    gmm_separation_threshold=0.95,
                ),
            ])

            image = Image.imread("hires_scan.jpg")
            result = pipeline.apply(image)
            print(f"Inocula detected: {result.num_objects}")
    """

    def __init__(
        self,
        thresh_method: Literal[
            "otsu", "mean", "local", "triangle", "minimum", "isodata", "li"
        ] = "otsu",
        subtract_background: bool = False,
        background_tophat_width: int = 300,
        homomorphic_sigma: float = 300.0,
        homomorphic_gamma_low: float = 0.5,
        homomorphic_gamma_high: float = 1.5,
        opening_shape: Literal["square", "diamond", "disk"] = "disk",
        opening_width: int = 50,
        log_min_radius: float = 25.0,
        log_max_radius: float = 50.0,
        log_num_scales: int = 5,
        enable_gmm: bool = True,
        gmm_n_components: int = 2,
        gmm_separation_threshold: float = 0.9,
        gmm_min_core_area: int = 30,
        gmm_morph_open_radius: int = 10,
        gmm_morph_close_radius: int = 2,
    ):
        """Initialize InoculumDetector with multi-stage blob enhancement parameters.

        See class docstring for detailed parameter effects, resolution scaling guidance,
        and use case guidance.

        Args:
            thresh_method: Thresholding method for binary segmentation. One of
                ``'otsu'`` (global histogram-based; recommended for balanced lighting),
                ``'mean'``, ``'local'`` (adaptive/spatial), ``'triangle'``, ``'minimum'``,
                ``'isodata'``, ``'li'``. Default: ``'otsu'``.
            subtract_background: If True, apply white top-hat background subtraction
                before homomorphic filtering. Useful for large-scale illumination
                gradients. Default: False.
            background_tophat_width: Side length (pixels) of the square structuring
                element for white top-hat. Pixel-based parameter; scale with resolution.
                Default: 300.
            homomorphic_sigma: Gaussian sigma (pixels) for illumination-reflectance
                separation. Larger values suppress broader vignetting; smaller values
                preserve detail. Pixel-based but tuned by lighting rig. Default: 300.0.
            homomorphic_gamma_low: Gain for low-frequency (illumination) component.
                Values < 1.0 suppress variation. Default: 0.5.
            homomorphic_gamma_high: Gain for high-frequency (reflectance/inoculum)
                component. Values > 1.0 enhance contrast. Default: 1.5.
            opening_shape: Footprint shape for morphological opening: ``'square'``
                (all directions), ``'diamond'`` (diagonal-reduced), or ``'disk'``
                (circular). Default: ``'disk'``.
            opening_width: Footprint size (pixels) for morphological opening.
                Pixel-based parameter; scale with resolution. Default: 50.
            log_min_radius: Minimum blob radius (pixels) for multi-scale LoG.
                Pixel-based parameter; scale with resolution. Default: 25.0.
            log_max_radius: Maximum blob radius (pixels). Pixel-based parameter.
                Default: 50.0.
            log_num_scales: Number of logarithmically spaced LoG scales. Typical
                range: 3–8. Default: 5.
            enable_gmm: If True, apply GMM core extraction to refine detected regions.
                Default: True.
            gmm_n_components: Number of Gaussian mixture components. 2 = core vs.
                surround. Default: 2.
            gmm_separation_threshold: Normalized mean separation threshold for GMM
                refinement. Regions below threshold are left unmodified. Default: 0.9.
            gmm_min_core_area: Minimum acceptable core area (pixels) after GMM
                extraction. Default: 30.
            gmm_morph_open_radius: Morphological opening radius (pixels) applied to
                GMM core mask. Pixel-based parameter. Default: 10.
            gmm_morph_close_radius: Morphological closing radius (pixels) applied to
                GMM core mask. Pixel-based parameter. Default: 2.
        """
        super().__init__()

        self.thresh_method = thresh_method
        self.subtract_background = subtract_background
        self.background_tophat_width = background_tophat_width
        self.homomorphic_sigma = homomorphic_sigma
        self.homomorphic_gamma_low = homomorphic_gamma_low
        self.homomorphic_gamma_high = homomorphic_gamma_high
        self.opening_shape = opening_shape
        self.opening_width = opening_width
        self.log_min_radius = log_min_radius
        self.log_max_radius = log_max_radius
        self.log_num_scales = log_num_scales
        self.enable_gmm = enable_gmm
        self.gmm_n_components = gmm_n_components
        self.gmm_separation_threshold = gmm_separation_threshold
        self.gmm_min_core_area = gmm_min_core_area
        self.gmm_morph_open_radius = gmm_morph_open_radius
        self.gmm_morph_close_radius = gmm_morph_close_radius

    def _operate(self, image: Image) -> Image:
        """Detect inoculation sites via multi-step blob enhancement pipeline.

        Executes the complete inoculum detection workflow entirely on local arrays
        (``image.detect_mat`` is never modified in-place, preserving detector integrity).

        Pipeline steps:

        1. Read ``detect_mat[:]`` into a local working copy.
        2. If ``subtract_background``: apply white top-hat to suppress plate-level
           illumination variation.
        3. Apply homomorphic illumination correction (``HomomorphicFilter``) with
           configured ``gamma_low`` and ``gamma_high``.
        4. Save a contrast-stretched [0, 1] copy as GMM intensity reference (used in
           step 11 if GMM is enabled).
        5. Morphological opening with configured ``opening_shape`` and ``opening_width``
           to suppress sub-footprint noise.
        6. Multi-scale Laplacian-of-Gaussian (``MultiscaleLoGEnhancer``) blob detection
           spanning ``[log_min_radius, log_max_radius]`` with ``log_num_scales`` scales.
        7. Contrast stretch the LoG output to [0, 1].
        8. Threshold to binary mask using the configured ``thresh_method``. If
           thresholding fails, fall back to Otsu with a logged warning.
        9. Connected-component labeling (8-connectivity) to produce integer label map.
        10. If input is a ``GridImage``: keep only the largest object per grid cell.
            This enforces one inoculum per well in 96-well, 384-well, etc. arrays.
        11. If ``enable_gmm``: apply ``GMMCoreExtractor`` to refine each detected
            region down to its bright core using the saved intensity reference.
        12. Write final ``objmask`` (binary) and ``objmap`` (labeled integers) to
            ``image``. Relabel connected components for consecutive labeling (1, 2, 3, ...).
        13. Clean up temporary arrays and memory.

        Args:
            image: Image to process. May be a plain ``Image`` or a ``GridImage``.
                ``image.detect_mat[:]`` is read but never modified in-place.

        Returns:
            Image: The same image object with ``objmask`` and ``objmap`` populated.

        Notes:
            - All processing is on local numpy arrays; ``image.rgb``, ``image.gray``,
              and ``image.detect_mat`` remain unchanged.
            - Memory usage scales with image dimensions. Temporary arrays totaling
              ~2–3× the image size may be allocated.
            - For GridImage inputs with overcrowded wells, step 10 may remove secondary
              inocula. Use post-processing (e.g., ``ObjectRefiner``) if this is undesired.
        """
        from phenotypic import GridImage

        # Step 1 -- read detect_mat into a local working copy
        enh = image.detect_mat[:].copy()
        self._log_memory_usage("read detect_mat")

        # Step 2 -- optional background subtraction
        if self.subtract_background:
            enh = self._apply_tophat(enh)
            self._log_memory_usage("background subtraction")

        # Step 3 -- homomorphic illumination correction
        enh = HomomorphicFilter._filter(
            enh,
            sigma=self.homomorphic_sigma,
            gamma_low=self.homomorphic_gamma_low,
            gamma_high=self.homomorphic_gamma_high,
        )
        self._log_memory_usage("homomorphic filter")

        # Step 4 -- contrast-stretched copy as GMM intensity reference
        gmm_intensity_ref = self._apply_contrast_stretch(enh.copy())
        self._log_memory_usage("GMM intensity reference saved")

        # Step 5 -- morphological opening
        enh = self._apply_opening(enh)
        self._log_memory_usage("morphological opening")

        # Step 6 -- multi-scale LoG blob enhancement
        enh = MultiscaleLoGEnhancer._enhance(
            enh,
            min_radius=self.log_min_radius,
            max_radius=self.log_max_radius,
            num_scales=self.log_num_scales,
        )
        self._log_memory_usage("multiscale LoG")

        # Step 7 -- contrast stretch to [0, 1]
        enh = self._apply_contrast_stretch(enh)
        self._log_memory_usage("contrast stretch")

        # Step 8 -- thresholding
        binary = self._apply_threshold(enh)
        del enh
        self._log_memory_usage("thresholding")

        # Step 9 -- connected-component labelling
        labeled, num_features = ndimage.label(
            binary,
            structure=ndimage.generate_binary_structure(rank=2, connectivity=2),
        )
        del binary
        self._log_memory_usage(f"labelling ({num_features} features)")

        # Step 10 -- GridImage: keep largest object per grid cell
        if isinstance(image, GridImage):
            labeled = self._apply_grid_section_largest(labeled, image)
            self._log_memory_usage("grid section largest")

        # Step 11 -- optional GMM core extraction
        if self.enable_gmm:
            labeled = GMMCoreExtractor._extract_cores(
                intensity_array=gmm_intensity_ref,
                label_map=labeled,
                n_components=self.gmm_n_components,
                separation_threshold=self.gmm_separation_threshold,
                min_core_area=self.gmm_min_core_area,
                morph_open_radius=self.gmm_morph_open_radius,
                morph_close_radius=self.gmm_morph_close_radius,
            )
            self._log_memory_usage("GMM core extraction")

        del gmm_intensity_ref

        # Step 12 -- write results
        objmask = labeled > 0
        image.objmask[:] = objmask
        image.objmap[:] = labeled.astype(image._OBJMAP_DTYPE, copy=False)
        image.objmap.relabel(connectivity=1)
        del labeled, objmask

        gc.collect()
        self._log_memory_usage(
            "final cleanup", include_process=True, include_tracemalloc=True,
        )
        return image

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------

    def _apply_tophat(self, enh: np.ndarray) -> np.ndarray:
        """Apply white top-hat background subtraction to flatten large-scale illumination.

        White top-hat (closing - original) removes large-scale features (vignetting,
        illumination gradients) while preserving small, high-frequency details
        (inocula, noise). Useful before homomorphic filtering to suppress plate-level
        illumination drift.

        Args:
            enh: 2-D array (detection matrix copy), any dtype.

        Returns:
            Array with white top-hat applied (preserves input dtype).
        """
        ksize = self.background_tophat_width
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (ksize, ksize),
        )
        tophat = cv2.morphologyEx(
            enh.astype(np.float32), cv2.MORPH_TOPHAT, kernel,
        )
        return tophat.astype(enh.dtype)

    def _apply_opening(self, enh: np.ndarray) -> np.ndarray:
        """Apply morphological opening to suppress sub-footprint noise.

        Morphological opening (erosion followed by dilation) removes small objects
        and noise smaller than the footprint while preserving large structures like
        inocula. Footprint shape (square, diamond, disk) determines directionality
        sensitivity: disk is rotationally symmetric; diamond is diagonal-reduced;
        square treats all directions equally.

        Args:
            enh: 2-D array (detection matrix or enhanced copy).

        Returns:
            Opened array (same dtype and shape as input).

        Raises:
            ValueError: If ``self.opening_shape`` is not 'square', 'diamond', or 'disk'.
        """
        match self.opening_shape:
            case "square":
                footprint = morphology.footprint_rectangle(
                    (self.opening_width, self.opening_width),
                )
            case "diamond":
                footprint = morphology.diamond(self.opening_width // 2)
            case "disk":
                footprint = morphology.disk(self.opening_width // 2)
            case _:
                raise ValueError(
                    f"Unknown opening_shape {self.opening_shape!r}; "
                    "expected 'square', 'diamond', or 'disk'"
                )
        return morphology.opening(enh, footprint=footprint)

    @staticmethod
    def _apply_contrast_stretch(enh: np.ndarray) -> np.ndarray:
        """Rescale array to [0, 1] to normalize dynamic range.

        Linear contrast stretch: (x - min) / (max - min). If the input range is
        negligible (< 1e-12), returns zeros. Used after enhancing operations
        (homomorphic filtering, LoG) to normalize before thresholding and to prepare
        the GMM intensity reference.

        Args:
            enh: Input array (any dtype).

        Returns:
            Contrast-stretched array (float64, range [0, 1]). Returns all-zeros
            if input range is too small.
        """
        enh = enh.astype(np.float64, copy=False)
        lo, hi = enh.min(), enh.max()
        rng = hi - lo
        if rng < 1e-12:
            return np.zeros_like(enh)
        return (enh - lo) / rng

    def _apply_threshold(self, enh: np.ndarray) -> np.ndarray:
        """Convert enhanced array to binary mask using the configured threshold method.

        Applies the thresholding strategy specified by ``self.thresh_method``. Some
        methods (e.g., ``'minimum'``, ``'isodata'``) may fail on certain histogram
        shapes (e.g., unimodal, monotonic); in such cases, the method falls back to
        Otsu thresholding with a logged warning.

        Thresholding method choice:
          - ``'otsu'``: Global, histogram-based; assumes bimodal distribution. Robust
            for balanced illumination and standardized media.
          - ``'mean'``: Global; threshold = mean intensity. Simple and fast.
          - ``'local'``: Adaptive, spatial window-based (block_size derived from
            ``opening_width``). Better for uneven illumination post-homomorphic filtering.
          - ``'triangle'``, ``'minimum'``, ``'isodata'``, ``'li'``: Specialized methods;
            use if Otsu fails or if specific histogram structure is expected.

        Args:
            enh: 2-D float array normalized to [0, 1].

        Returns:
            Boolean (numpy uint8, 0/255 or True/False) binary mask.
        """
        try:
            match self.thresh_method:
                case "otsu":
                    thresh = filters.threshold_otsu(enh)
                case "mean":
                    thresh = filters.threshold_mean(enh)
                case "local":
                    block_size = max(self.opening_width * 2 + 1, 3)
                    # Ensure odd
                    if block_size % 2 == 0:
                        block_size += 1
                    thresh = filters.threshold_local(enh, block_size=block_size)
                case "triangle":
                    thresh = filters.threshold_triangle(enh)
                case "minimum":
                    thresh = filters.threshold_minimum(enh)
                case "isodata":
                    thresh = filters.threshold_isodata(enh)
                case "li":
                    thresh = filters.threshold_li(enh)
                case _:
                    thresh = filters.threshold_otsu(enh)
        except (RuntimeError, IndexError):
            logger.warning(
                "threshold_method '%s' failed; falling back to Otsu",
                self.thresh_method,
            )
            thresh = filters.threshold_otsu(enh)
        return enh >= thresh

    @staticmethod
    def _apply_grid_section_largest(
        labeled: np.ndarray, image: Image,
    ) -> np.ndarray:
        """Keep only the largest inoculum per grid cell (well) for organized arrays.

        For GridImage inputs (96-well, 384-well, manual 8×12 pinned arrays, etc.),
        this method iterates over grid cells and retains only the largest object
        per cell. Useful for enforcing one-inoculum-per-well in high-density plate
        formats. If a well contains no objects, that well is left empty (zero).

        This operation is automatically applied in the pipeline (step 10) but may
        discard valid secondary inocula if wells are overcrowded. If secondary
        inocula are important, disable this behavior via post-processing or
        manual detection adjustment.

        Args:
            labeled: Integer label map from ``scipy.ndimage.label`` with
                consecutive labels 1, 2, 3, ...
            image: A ``GridImage`` instance with ``image.grid`` metadata
                (row/column edge coordinates).

        Returns:
            New label map (same shape and dtype as input) with at most one object
            per grid cell, relabeled consecutively (1, 2, 3, ...).
        """
        row_edges = np.round(image.grid.get_row_edges()).astype(int)
        col_edges = np.round(image.grid.get_col_edges()).astype(int)

        new_labeled = np.zeros_like(labeled, dtype=np.int32)
        label_counter = 1
        for r in range(len(row_edges) - 1):
            r0, r1 = row_edges[r], row_edges[r + 1]
            for c in range(len(col_edges) - 1):
                c0, c1 = col_edges[c], col_edges[c + 1]
                region = labeled[r0:r1, c0:c1]
                if region.size == 0:
                    continue
                uniq, counts = np.unique(region, return_counts=True)
                valid = uniq != 0
                uniq, counts = uniq[valid], counts[valid]
                if uniq.size == 0:
                    continue
                dominant_label = uniq[np.argmax(counts)]
                mask = region == dominant_label
                if np.any(mask):
                    new_labeled[r0:r1, c0:c1][mask] = label_counter
                    label_counter += 1
        return new_labeled
