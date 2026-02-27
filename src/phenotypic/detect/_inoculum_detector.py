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
from phenotypic.enhance._homomorphic_filter import homomorphic_filter
from phenotypic.enhance._multiscale_log_enhancer import multiscale_log_enhance
from phenotypic.refine._gmm_core_extractor import extract_gmm_cores

logger = logging.getLogger(__name__)


class InoculumDetector(ObjectDetector):
    """Detect inoculation sites on agar plates via multi-step blob enhancement.

    InoculumDetector identifies inoculation spots using a pipeline of
    homomorphic illumination correction, morphological opening, multi-scale
    Laplacian-of-Gaussian blob enhancement, thresholding, and optional
    GMM-based core extraction.  All intermediate processing is performed on
    local numpy arrays -- ``image.detect_mat`` is never modified (enforced
    by ObjectDetector integrity validation).

    Args:
        thresh_method: Thresholding method applied to the LoG-enhanced image.
            Options: ``'otsu'``, ``'mean'``, ``'local'``, ``'triangle'``,
            ``'minimum'``, ``'isodata'``, ``'li'``.  Default ``'otsu'``.
        subtract_background: If True, apply a white top-hat transform to the
            detection matrix before homomorphic filtering to flatten large-
            scale background intensity variation.  Default False.
        background_tophat_width: Side length of the square kernel used for
            the top-hat transform (pixels).  Only used when
            ``subtract_background=True``.  Default 300.
        homomorphic_sigma: Gaussian sigma for the homomorphic filter that
            separates illumination from reflectance.  Larger values capture
            broader illumination gradients.  Default 300.0.
        homomorphic_gamma_low: Gain for low-frequency (illumination) component.
            Values < 1 suppress illumination variation.  Default 0.5.
        homomorphic_gamma_high: Gain for high-frequency (reflectance) component.
            Values > 1 enhance colony contrast.  Default 1.5.
        opening_shape: Shape of the footprint used for morphological opening.
            ``'square'``, ``'diamond'``, or ``'disk'``.  Default ``'disk'``.
        opening_width: Width of the morphological opening footprint (pixels).
            Larger values remove more noise but may erode small spots.
            Default 50.
        log_min_radius: Minimum blob radius (pixels) for multi-scale LoG
            enhancement.  Default 25.0.
        log_max_radius: Maximum blob radius (pixels).  Default 50.0.
        log_num_scales: Number of logarithmically spaced LoG scales.
            Default 5.
        enable_gmm: If True (default), apply GMM core extraction to refine
            each detected region down to its compact bright core.
        gmm_n_components: Number of Gaussian mixture components per region.
            Default 2 (core vs. surround).
        gmm_separation_threshold: Normalised GMM mean separation below
            which a region is left unchanged.  Default 0.9.
        gmm_min_core_area: Minimum acceptable core area (pixels) for GMM
            extraction.  Default 30.
        gmm_morph_open_radius: Morphological opening radius used inside GMM
            core extraction.  Default 10.
        gmm_morph_close_radius: Morphological closing radius used inside GMM
            core extraction.  Default 2.

    Attributes:
        thresh_method, subtract_background, background_tophat_width,
        homomorphic_sigma, homomorphic_gamma_low, homomorphic_gamma_high,
        opening_shape, opening_width, log_min_radius, log_max_radius,
        log_num_scales, enable_gmm, gmm_n_components,
        gmm_separation_threshold, gmm_min_core_area, gmm_morph_open_radius,
        gmm_morph_close_radius

    Returns:
        Image: Input image with ``objmask`` (binary inoculum mask) and
        ``objmap`` (labelled inoculum map) set.

    Raises:
        ValueError: If an unrecognised ``opening_shape`` is passed.

    **Use cases**

    - **Pinned inoculation plates:** Detect small inoculum spots deposited by
      a pin tool onto solid media.  The multi-scale LoG captures spots across
      a range of sizes while homomorphic correction handles scanner vignetting.
    - **Spot-dilution assays:** Identify serial-dilution droplets of varying
      diameter, where the LoG radius window brackets the expected spot sizes.
    - **Pre-growth baseline:** Detect inocula before incubation as a reference
      for growth measurements taken at later time-points.

    **Limitations**

    - Not suited for filamentous or highly irregular colony morphologies; the
      LoG blob model assumes roughly circular features.
    - Large ``homomorphic_sigma`` and ``opening_width`` values increase
      computation time on high-resolution images.
    - GMM core extraction adds per-object overhead; disable (``enable_gmm=False``)
      for speed on large plate arrays where inocula are already compact.

    **Parameter effects on detection**

    - **homomorphic_sigma / gamma_low / gamma_high:** Control illumination
      correction strength.  Reduce ``gamma_low`` or increase ``gamma_high``
      for plates with severe vignetting.
    - **opening_width / opening_shape:** Morphological opening removes noise
      smaller than the footprint.  Increase for noisy backgrounds; decrease
      to retain small inocula.
    - **log_min_radius / log_max_radius:** Define the blob size window.
      Narrow the range if inoculum size is known in advance.
    - **enable_gmm / gmm_separation_threshold:** GMM tightens masks around
      bright cores.  Increase separation threshold to only refine regions
      with strong core-surround contrast.

    Examples:
        Basic inoculum detection::

            from phenotypic import Image
            from phenotypic.detect import InoculumDetector

            plate = Image.imread("inoculated_plate.jpg")
            detector = InoculumDetector()
            detected = detector.apply(plate)
            print(f"Detected {detected.num_objects} inoculation sites")

        Pipeline with preprocessing for noisy images::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur
            from phenotypic.detect import InoculumDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.0),
                InoculumDetector(
                    homomorphic_sigma=200.0,
                    log_min_radius=10.0,
                    log_max_radius=30.0,
                    enable_gmm=False,
                ),
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
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
        """Initialize the InoculumDetector with the given parameters.

        Args:
            thresh_method: Thresholding method.  One of ``'otsu'``,
                ``'mean'``, ``'local'``, ``'triangle'``, ``'minimum'``,
                ``'isodata'``, ``'li'``.
            subtract_background: Apply white top-hat background subtraction
                before homomorphic filtering.
            background_tophat_width: Square kernel side for top-hat.
            homomorphic_sigma: Gaussian sigma for homomorphic filter.
            homomorphic_gamma_low: Low-frequency gain.
            homomorphic_gamma_high: High-frequency gain.
            opening_shape: Footprint shape for morphological opening
                (``'square'``, ``'diamond'``, ``'disk'``).
            opening_width: Footprint width for morphological opening.
            log_min_radius: Minimum LoG blob radius.
            log_max_radius: Maximum LoG blob radius.
            log_num_scales: Number of LoG scales.
            enable_gmm: Apply GMM core extraction after labelling.
            gmm_n_components: Gaussian mixture components per region.
            gmm_separation_threshold: Normalised separation below which
                a region is not refined.
            gmm_min_core_area: Minimum core area (pixels).
            gmm_morph_open_radius: Opening radius inside GMM extraction.
            gmm_morph_close_radius: Closing radius inside GMM extraction.
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
        """Detect inoculation sites using a multi-step enhancement pipeline.

        Pipeline steps (all on local arrays -- ``image.detect_mat`` is not
        modified):

        1. Read ``detect_mat`` into a local array.
        2. Optional white top-hat background subtraction.
        3. Homomorphic illumination correction.
        4. Save a contrast-stretched copy as GMM intensity reference.
        5. Morphological opening.
        6. Multi-scale LoG blob enhancement.
        7. Contrast stretch to [0, 1].
        8. Thresholding to binary mask.
        9. Connected-component labelling.
        10. If GridImage: keep only the largest object per grid cell.
        11. If ``enable_gmm``: GMM core extraction.
        12. Set ``objmask`` and ``objmap``; relabel.

        Args:
            image: Image to process.  May be a plain Image or GridImage.

        Returns:
            Image with ``objmask`` and ``objmap`` populated.
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
        enh = homomorphic_filter(
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
        enh = multiscale_log_enhance(
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
            labeled = extract_gmm_cores(
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
        """Apply white top-hat background subtraction.

        Args:
            enh: 2-D float array (detection matrix copy).

        Returns:
            Background-subtracted array.
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
        """Apply morphological opening with the configured footprint.

        Args:
            enh: 2-D array to open.

        Returns:
            Opened array.
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
        """Rescale array to [0, 1].

        Args:
            enh: Input array.

        Returns:
            Contrast-stretched array (float64).
        """
        enh = enh.astype(np.float64, copy=False)
        lo, hi = enh.min(), enh.max()
        rng = hi - lo
        if rng < 1e-12:
            return np.zeros_like(enh)
        return (enh - lo) / rng

    def _apply_threshold(self, enh: np.ndarray) -> np.ndarray:
        """Threshold the enhanced array to produce a binary mask.

        Some skimage threshold methods (``minimum``, ``isodata``) can fail on
        certain histogram shapes.  When that happens the method falls back to
        Otsu thresholding with a logged warning.

        Args:
            enh: 2-D float array normalised to [0, 1].

        Returns:
            Boolean binary mask.
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
        """Keep only the largest object per grid cell for GridImage inputs.

        Args:
            labeled: Integer label map from ``scipy.ndimage.label``.
            image: A ``GridImage`` instance with grid metadata.

        Returns:
            New label map with one object per grid cell.
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
