from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import gc
import logging
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator
from typing_extensions import Self

from phenotypic.abc_ import ObjectDetector
from phenotypic.sdk_.typing_ import TuneSpec
from phenotypic.detect._round_peaks_detector import RoundPeaksDetector
from phenotypic.enhance._contrast_streching import ContrastStretching
from phenotypic.enhance._gray_opening import GrayOpening
from phenotypic.enhance._median_filter import MedianFilter
from phenotypic.enhance._focus_blob_log import FocusBlobLoG
from phenotypic.enhance._subtract_gaussian import SubtractGaussian
from phenotypic.refine._extract_colony_core import ExtractColonyCore
from phenotypic.refine._keep_section_largest import KeepSectionLargest

logger = logging.getLogger(__name__)


class InoculumDetector(ObjectDetector):
    """Detect inoculation sites on agar plates via Gaussian background subtraction and multi-scale blob enhancement.

    Identify inoculation spots (from pin-deposition tools, liquid spotting,
    or serial dilutions) by composing Gaussian background subtraction, median
    filtering, multi-scale Laplacian-of-Gaussian blob enhancement, contrast
    stretching, morphological opening, round-peaks detection, and optional
    GMM-based core extraction. All processing occurs on a working copy so
    that ``image.detect_mat`` is never modified. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        - Pin-tool inoculation on high-density plates (96-well, 384-well)
          where inocula are 30--80 pixels in diameter.
        - Spot-dilution assays producing 10--150 pixel inocula across
          serial dilution series.
        - Pre-growth phenotyping baselines (T=0 imaging) for establishing
          reference coordinates before growth measurements.
        - Liquid spotting assays where GMM core extraction refines
          boundaries for accurate area and circularity measurements.

    Consider Also:
        - :class:`RoundPeaksDetector` when colonies (not inocula) must be
          detected on a regular grid without multi-scale blob enhancement.
        - :class:`OtsuDetector` when inocula are high-contrast and a simple
          global threshold is sufficient.
        - :class:`FilamentousFungiDetector` when inoculation sites have
          developed into filamentous fungal growth.

    Args:
        min_diameter: Smallest expected inoculum diameter in pixels. Used
            to derive LoG minimum radius and GMM morphological parameters.
            Set based on the smallest visible spot in your images. Typical
            range: 5--80. Default: 30.0.
        max_diameter: Largest expected inoculum diameter in pixels. Used
            to derive Gaussian background subtraction sigma and LoG maximum
            radius. Must be greater than ``min_diameter``. Typical range:
            50--300. Default: 100.0.
        thresh_method: Thresholding method for binary segmentation within
            the RoundPeaksDetector step. Accepted values: ``'otsu'``
            (default), ``'mean'``, ``'local'``, ``'triangle'``,
            ``'minimum'``, ``'isodata'``, ``'li'``. ``'otsu'`` works well
            for most standardised setups; ``'local'`` adapts to spatial
            illumination gradients. Default: ``'otsu'``.
        enable_gmm: Apply Gaussian Mixture Model core extraction to refine
            detected regions to bright, compact cores. Disable for inocula
            that lack clear core-surround structure. Default: True.
        gmm_n_components: Number of GMM components per region. Two
            components separate core from surround; increase only for
            complex multi-layered spots. Default: 2.
        gmm_separation_threshold: Normalised Euclidean distance between
            GMM component means below which a region is left unmodified
            (no clear core). Increase to make refinement less aggressive.
            Typical range: 0.8--1.2. Default: 0.9.
        validate_obj_count: When True and the input is a ``GridImage``,
            raise ``ValueError`` when the detected object count exceeds
            ``nrows * ncols``. Catches over-segmentation early. Default:
            True.

    Returns:
        Image: Input image with ``objmask`` (binary inoculum mask) and
        ``objmap`` (labelled inoculum map) populated.

    Raises:
        ValueError: If ``min_diameter`` >= ``max_diameter``, or if
            detected object count exceeds grid capacity when
            ``validate_obj_count`` is True and input is a GridImage.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    # TODO: review bound (unverified vs literature)
    min_diameter: Annotated[float, TuneSpec(5.0, 80.0, log=True)] = Field(30.0, gt=0.0)
    # TODO: review bound (unverified vs literature)
    max_diameter: Annotated[float, TuneSpec(50.0, 300.0, log=True)] = Field(100.0, gt=0.0)
    thresh_method: Literal[
        "otsu", "mean", "local", "triangle", "minimum", "isodata", "li"
    ] = "otsu"
    enable_gmm: bool = True
    gmm_n_components: Annotated[int, TuneSpec(2, 4)] = 2
    # TODO: review bound (unverified vs literature)
    gmm_separation_threshold: Annotated[float, TuneSpec(0.8, 1.2)] = 0.9
    validate_obj_count: bool = True

    @model_validator(mode="after")
    def _check_diameter_order(self) -> Self:
        """Require ``min_diameter`` to be strictly less than ``max_diameter``.

        Reproduces the cross-field guard from the pre-migration
        ``__init__``; the per-field positivity checks are enforced by the
        ``Field(gt=0.0)`` constraints on ``min_diameter`` / ``max_diameter``
        above.
        """
        if self.min_diameter >= self.max_diameter:
            raise ValueError(
                    f"min_diameter ({self.min_diameter}) must be less than "
                    f"max_diameter ({self.max_diameter})"
            )
        return self

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
        FocusBlobLoG(
                min_radius=log_min_radius,
                max_radius=log_max_radius,
                num_scales=15,
        ).apply(work, inplace=True)
        self._log_memory_usage("FocusBlobLoG")

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
                edge_refinement=False,
        ).apply(work, inplace=True)
        self._log_memory_usage("RoundPeaksDetector")

        # --- Step 8: GridImage → keep largest per cell ---
        if isinstance(work, GridImage):
            KeepSectionLargest().apply(work, inplace=True)
            self._log_memory_usage("KeepSectionLargest")

        # --- Step 9: Optional GMM core extraction ---
        if self.enable_gmm:
            ExtractColonyCore(
                    n_components=self.gmm_n_components,
                    separation_threshold=self.gmm_separation_threshold,
                    min_core_area=gmm_min_area,
                    morph_open_radius=gmm_morph_open,
                    morph_close_radius=2,
            ).apply(work, inplace=True)
            self._log_memory_usage("ExtractColonyCore")

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
