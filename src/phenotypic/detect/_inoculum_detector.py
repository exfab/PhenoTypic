from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import gc
import logging
from typing import Literal

import numpy as np

from phenotypic.abc_ import ObjectDetector
from phenotypic.detect._round_peaks_detector import RoundPeaksDetector
from phenotypic.enhance._contrast_streching import ContrastStretching
from phenotypic.enhance._gray_opening import GrayOpening
from phenotypic.enhance._median_filter import MedianFilter
from phenotypic.enhance._multiscale_log_enhancer import MultiscaleLoGEnhancer
from phenotypic.enhance._subtract_gaussian import SubtractGaussian
from phenotypic.refine._gmm_core_extractor import GMMCoreExtractor
from phenotypic.refine._grid_section_largest import GridSectionLargest

logger = logging.getLogger(__name__)


class InoculumDetector(ObjectDetector):
    """Detect inoculation sites on agar plates via Gaussian background subtraction and multi-scale blob enhancement.

    InoculumDetector identifies inoculation spots (e.g., from pin-deposition
    tools, liquid spotting, or serial dilutions) using a composable pipeline of
    existing PhenoTypic operations: Gaussian background subtraction, median
    filtering, multi-scale Laplacian-of-Gaussian blob enhancement, contrast
    stretching, morphological opening, round-peaks detection, and optional
    GMM-based core extraction. All processing occurs on a working copy;
    ``image.detect_mat`` is never modified (enforced by ObjectDetector
    integrity validation).

    The API is organised around **biologically meaningful parameters** — primarily
    the expected inoculum diameter range in pixels — from which all internal
    algorithm parameters are derived. This replaces the previous low-level
    18-parameter interface with 7 intuitive controls.

    Args:
        min_diameter: Smallest expected inoculum diameter in pixels. Used to
            derive LoG minimum radius and GMM morphological parameters. Set
            based on the smallest spot visible in your images. Default: 30.0.
        max_diameter: Largest expected inoculum diameter in pixels. Used to
            derive Gaussian background subtraction sigma and LoG maximum
            radius. Default: 100.0.
        thresh_method: Thresholding method for binary segmentation within the
            RoundPeaksDetector step. Options: ``'otsu'`` (global,
            histogram-based), ``'mean'``, ``'local'`` (adaptive), ``'triangle'``,
            ``'minimum'``, ``'isodata'``, ``'li'``. Default: ``'otsu'``.
        enable_gmm: If True (default), apply Gaussian Mixture Model core
            extraction to refine detected regions down to bright, compact
            cores. Useful for inocula with diffuse edges or age-related
            intensity gradients.
        gmm_n_components: Gaussian mixture components per region. 2 = core
            vs. surround (typical). Default: 2.
        gmm_separation_threshold: Normalised Euclidean distance between GMM
            component means. Below this threshold, a region is left unmodified
            (no clear core). Typical range: 0.8-1.2. Default: 0.9.
        validate_obj_count: If True and the input is a ``GridImage``, raise
            ``ValueError`` when the final detected object count exceeds
            ``nrows * ncols``. Catches over-segmentation. Default: True.

    Returns:
        Image: Input image with ``objmask`` (binary inoculum mask) and
        ``objmap`` (labelled object map) populated.

    Raises:
        ValueError: If detected object count exceeds grid capacity (when
            ``validate_obj_count=True`` and input is ``GridImage``).

    **Derived Internal Parameters**

    From ``min_diameter`` and ``max_diameter``, the pipeline automatically
    computes algorithm settings:

    - SubtractGaussian sigma = ``max_diameter * 2``
    - LoG min_radius = ``min_diameter / 2``, max_radius = ``max_diameter / 2``
    - GMM morph_open_radius = ``max(1, round(min_diameter / 30))``
    - GMM min_core_area = ``max(5, round(min_diameter * 0.8))``

    **Pipeline Steps**

    1. Copy input image as working copy (preserves ``detect_mat``)
    2. SubtractGaussian — remove large-scale illumination gradients
    3. MedianFilter — suppress salt-and-pepper noise
    4. MultiscaleLoGEnhancer — blob enhancement across radius range
    5. ContrastStretching — normalise dynamic range
    6. GrayOpening — smooth small noise artefacts
    7. RoundPeaksDetector — threshold and detect round peaks
    8. GridSectionLargest — keep one object per grid cell (GridImage only)
    9. GMMCoreExtractor — refine to bright cores (optional)
    10. Copy results back to original image

    **Use Cases**

    1. **Pin-tool inoculation:** High-density plates (96-well, 384-well) where
       inocula are 30-80 pixels diameter. Default parameters work well.

    2. **Spot-dilution assays:** Serial dilutions produce 10-150 pixel inocula.
       Set ``min_diameter=10, max_diameter=150`` to capture the full range.

    3. **Pre-growth phenotyping baseline:** Image at T=0 for reference
       coordinates before growth measurements.

    4. **Liquid spotting assays:** GMM core extraction refines boundaries for
       accurate area/circularity measurements.

    **Caveats and Limitations**

    1. **LoG blob model assumption:** Assumes roughly circular inocula. Highly
       filamentous structures may be under-detected.

    2. **GMM over-refinement:** If inocula lack clear core-surround structure,
       try increasing ``gmm_separation_threshold`` or ``enable_gmm=False``.

    3. **GridImage handling:** Automatically keeps the largest inoculum per grid
       cell. May discard secondary inocula in overcrowded wells.

    Examples:
        Basic inoculum detection on a pinned plate::

            from phenotypic import Image
            from phenotypic.detect import InoculumDetector

            plate = Image.imread("inoculated_plate.jpg")
            detector = InoculumDetector()
            detected = detector.apply(plate)
            print(f"Detected {detected.num_objects} inoculation sites")

        Adjusting for large, well-separated inocula::

            detector = InoculumDetector(
                min_diameter=50.0,
                max_diameter=200.0,
                enable_gmm=False,
            )
    """

    def __init__(
        self,
        min_diameter: float = 30.0,
        max_diameter: float = 100.0,
        thresh_method: Literal[
            "otsu", "mean", "local", "triangle", "minimum", "isodata", "li"
        ] = "otsu",
        enable_gmm: bool = True,
        gmm_n_components: int = 2,
        gmm_separation_threshold: float = 0.9,
        validate_obj_count: bool = True,
    ):
        """Initialise InoculumDetector with biology-driven parameters.

        Args:
            min_diameter: Smallest expected inoculum diameter (pixels).
                Default: 30.0.
            max_diameter: Largest expected inoculum diameter (pixels).
                Default: 100.0.
            thresh_method: Thresholding method. Default: ``'otsu'``.
            enable_gmm: Apply GMM core extraction. Default: True.
            gmm_n_components: GMM components per region. Default: 2.
            gmm_separation_threshold: GMM mean separation threshold.
                Default: 0.9.
            validate_obj_count: Validate object count for GridImage.
                Default: True.
        """
        super().__init__()

        if min_diameter <= 0:
            raise ValueError(f"min_diameter must be positive, got {min_diameter}")
        if max_diameter <= 0:
            raise ValueError(f"max_diameter must be positive, got {max_diameter}")
        if min_diameter >= max_diameter:
            raise ValueError(
                f"min_diameter ({min_diameter}) must be less than "
                f"max_diameter ({max_diameter})"
            )

        self.min_diameter = min_diameter
        self.max_diameter = max_diameter
        self.thresh_method = thresh_method
        self.enable_gmm = enable_gmm
        self.gmm_n_components = gmm_n_components
        self.gmm_separation_threshold = gmm_separation_threshold
        self.validate_obj_count = validate_obj_count

    def _operate(self, image: Image) -> Image:
        """Detect inoculation sites via composable Gaussian pipeline.

        All enhancement and detection happens on a working copy; the returned
        image has its ``objmask`` and ``objmap`` populated but ``detect_mat``
        unchanged.

        Args:
            image: Image to process. May be ``Image`` or ``GridImage``.

        Returns:
            Image with ``objmask`` and ``objmap`` populated.
        """
        from phenotypic import GridImage

        # --- Derive internal parameters from diameter range ---
        subtract_sigma = self.max_diameter * 2
        log_min_radius = self.min_diameter / 2
        log_max_radius = self.max_diameter / 2
        gmm_morph_open = max(1, round(self.min_diameter / 30))
        gmm_min_area = max(5, round(self.min_diameter * 0.8))

        # --- Step 1: working copy with float32 detect_mat ---
        work = image.copy()
        # Direct _data access: the accessor (detect_mat[:] =) writes into the
        # existing backing array, which would truncate float32 values if the
        # backing is uint8. We must replace the entire array object.
        dm = work._data.detect_mat
        if dm.dtype.kind != "f":
            work._data.detect_mat = dm.astype(np.float32) / np.iinfo(dm.dtype).max
        elif dm.dtype != np.float32:
            work._data.detect_mat = dm.astype(np.float32)
        self._log_memory_usage("working copy created")

        # --- Step 2: Gaussian background subtraction ---
        SubtractGaussian(sigma=subtract_sigma, n_iter=2).apply(
            work, inplace=True,
        )
        self._log_memory_usage("SubtractGaussian")

        # --- Step 3: Median filter ---
        MedianFilter(width=5, shape="square").apply(work, inplace=True)
        self._log_memory_usage("MedianFilter")

        # --- Step 4: Multi-scale LoG blob enhancement ---
        MultiscaleLoGEnhancer(
            min_radius=log_min_radius,
            max_radius=log_max_radius,
            num_scales=15,
        ).apply(work, inplace=True)
        self._log_memory_usage("MultiscaleLoGEnhancer")

        # --- Step 5: Contrast stretching ---
        ContrastStretching().apply(work, inplace=True)
        self._log_memory_usage("ContrastStretching")

        # --- Step 6: Gray opening ---
        GrayOpening(width=5, shape="disk", n_iter=2).apply(
            work, inplace=True,
        )
        self._log_memory_usage("GrayOpening")

        # --- Step 7: Round peaks detection ---
        RoundPeaksDetector(
            thresh_method=self.thresh_method,
            noise_radius=2,
            smoothing_sigma=0.0,
            subtract_background=False,
            edge_refinement=True,
        ).apply(work, inplace=True)
        self._log_memory_usage("RoundPeaksDetector")

        # --- Step 8: GridImage → keep largest per cell ---
        if isinstance(work, GridImage):
            GridSectionLargest().apply(work, inplace=True)
            self._log_memory_usage("GridSectionLargest")

        # --- Step 9: Optional GMM core extraction ---
        if self.enable_gmm:
            GMMCoreExtractor(
                n_components=self.gmm_n_components,
                separation_threshold=self.gmm_separation_threshold,
                min_core_area=gmm_min_area,
                morph_open_radius=gmm_morph_open,
                morph_close_radius=2,
            ).apply(work, inplace=True)
            self._log_memory_usage("GMMCoreExtractor")

        # --- Step 10: Copy results back ---
        image.objmask[:] = work.objmask[:]
        image.objmap[:] = work.objmap[:]
        image.objmap.relabel(connectivity=1)

        del work
        gc.collect()

        # --- Step 11: Validate object count for GridImage ---
        if self.validate_obj_count and isinstance(image, GridImage):
            max_objects = image.nrows * image.ncols
            num_objects = int(image.objmap[:].max())
            if num_objects > max_objects:
                raise ValueError(
                    f"Detected {num_objects} objects but GridImage has only "
                    f"{image.nrows}x{image.ncols} = {max_objects} cells. "
                    f"Set validate_obj_count=False to skip this check."
                )

        self._log_memory_usage(
            "final cleanup", include_process=True, include_tracemalloc=True,
        )
        return image
