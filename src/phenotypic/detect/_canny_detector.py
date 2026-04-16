from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image

from skimage import feature, morphology
from scipy import ndimage

from phenotypic.abc_ import ThresholdDetector


class CannyDetector(ThresholdDetector):
    """Detect colonies by Canny edge detection and enclosed-region labelling.

    Apply multi-stage Canny edge detection (Gaussian smoothing, gradient
    magnitude, non-maximum suppression, hysteresis thresholding) to produce
    thin, connected boundary contours, then label the enclosed regions as
    individual colonies. This edge-based approach segments colonies by
    boundary contrast rather than absolute intensity, making it effective on
    plates with uneven illumination or translucent colonies. For a full
    comparison see :doc:`/explanation/detection_strategies_compared`.

    Best For:
        * Well-separated colonies on solid media where colony edges are
          sharper than intensity differences relative to background.
        * Translucent or lightly pigmented colonies that lack sufficient
          intensity contrast for threshold-based methods.
        * Plates with heterogeneous colony texture or pigmentation that
          fragments under watershed or simple thresholding.
        * Images with moderate vignetting where edge contrast is preserved
          even though absolute intensity varies spatially.

    Consider Also:
        * :class:`OtsuDetector` when colonies differ from background
          primarily in brightness rather than edge contrast.
        * :class:`WatershedDetector` when touching colonies must be split
          by region-growing from interior seeds.
        * :class:`HysteresisDetector` when dual-threshold intensity
          segmentation is preferred over edge-based detection.

    Args:
        sigma: Gaussian smoothing standard deviation before edge detection
            (default 1.0). Higher values suppress noise and spurious edges
            but may blur fine boundaries or merge nearby colonies. Typical
            range: 0.5--3.0. Start with 1.0 for clean images; increase to
            2.0--3.0 for noisy scans.

        low_threshold: Lower hysteresis threshold (default 0.1). When
            *use_quantiles* is True, this is a fraction (0.1 = retain edges
            stronger than 10 % of gradient magnitudes). When False, an
            absolute gradient value. Increase to suppress weak noise edges.
            Typical quantile range: 0.05--0.2.

        high_threshold: Upper hysteresis threshold seeding edge traces
            (default 0.2). Same interpretation as *low_threshold*. Must
            exceed *low_threshold*. Typical quantile range: 0.1--0.4.

        use_quantiles: If True (default), interpret thresholds as gradient-
            magnitude quantiles, adapting automatically to image contrast.
            Set to False for absolute gradient values when imaging
            conditions are tightly controlled.

        min_size: Minimum object area in pixels (default 50). Increase to
            filter dust, debris, and plate-edge artefacts; decrease to
            retain tiny colonies. Typical range: 20--500.

        invert_edges: If True (default), label enclosed regions as colony
            objects. If False, label edge pixels themselves (useful for
            debugging edge quality).

        connectivity: Connectivity for region labelling (1 = 4-connected,
            2 = 8-connected; default 2). Higher connectivity bridges
            diagonal gaps in edge contours.

    Returns:
        Image: Input image with ``objmap`` set to a labelled colony map
        where each enclosed region receives a unique integer label.
        ``objmask`` is derived from the non-zero region of the label map.

    Raises:
        ValueError: If *high_threshold* is less than *low_threshold* or if
            threshold values are outside the valid range.

    References:
        [1] J. Canny, "A computational approach to edge detection," *IEEE
        Trans. Pattern Anal. Mach. Intell.*, vol. 8, no. 6, pp. 679--698,
        1986.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    def __init__(
            self,
            sigma: float = 1.0,
            low_threshold: float = 0.1,
            high_threshold: float = 0.2,
            use_quantiles: bool = True,
            min_size: int = 50,
            invert_edges: bool = True,
            connectivity: int = 2,
    ):
        """
        Parameters:
            sigma (float): Gaussian smoothing strength before edge detection. Start
                with 1-2 for clean images; increase for noisy scans to suppress
                spurious edges. Keep below typical colony width to avoid merging.
            low_threshold (float): Lower hysteresis threshold. If use_quantiles=True,
                a fraction (e.g., 0.1 = retain edges stronger than 10% of gradients).
                If False, an absolute gradient magnitude. Increase to suppress weak
                edges from noise; decrease to recover faint colony boundaries.
            high_threshold (float): Upper hysteresis threshold. Seeds edge traces.
                If use_quantiles=True, a fraction (e.g., 0.2 = top 80% gradients);
                if False, an absolute magnitude. Raise to focus on strong boundaries;
                lower to include fainter edges. Must exceed low_threshold.
            use_quantiles (bool): Interpret thresholds as quantiles (True, default)
                or absolute values (False). Quantiles adapt to image contrast
                automatically, reducing manual tuning.
            min_size (int): Minimum object area in pixels. Increase to filter out
                dust, debris, and small artifacts; decrease to retain tiny colonies.
            invert_edges (bool): If True (default), label enclosed regions as
                objects (colonies). If False, label edge pixels (for atypical cases
                like ring colonies or edge quality checks).
            connectivity (int): Connectivity for labeling regions (1 or 2 in 2D).
                Higher values merge diagonally touching pixels, useful for bridging
                fragmented boundaries but may merge touching colonies.
        """
        super().__init__()
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.use_quantiles = use_quantiles
        self.min_size = min_size
        self.invert_edges = invert_edges
        self.connectivity = connectivity

    def _operate(self, image: Image | GridImage) -> Image:

        enhanced_matrix = image.detect_mat[:]

        # Apply Canny edge detection
        edges = feature.canny(
                image=enhanced_matrix,
                sigma=self.sigma,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                use_quantiles=self.use_quantiles,
        )

        # Invert edges to get regions (colonies) if requested
        if self.invert_edges:
            regions = ~edges
        else:
            regions = edges

        # Label connected components
        objmap, _ = ndimage.label(
                regions,
                structure=ndimage.generate_binary_structure(2, self.connectivity)
        )

        # Remove small objects
        objmap = morphology.remove_small_objects(objmap, max_size=self.min_size)

        # Ensure correct dtype
        if objmap.dtype != image._OBJMAP_DTYPE:
            objmap = objmap.astype(image._OBJMAP_DTYPE)

        # Relabel to ensure consecutive labels
        image.objmap[:] = objmap
        image.objmap.relabel(connectivity=self.connectivity)

        return image


# Set the docstring so that it appears in the sphinx documentation
CannyDetector.apply.__doc__ = CannyDetector._operate.__doc__
