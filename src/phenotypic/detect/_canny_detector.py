from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image, GridImage

from skimage import feature, morphology
from scipy import ndimage

from phenotypic.abc_ import ThresholdDetector


class CannyDetector(ThresholdDetector):
    """Edge-based colony detection using Canny edge detection and region labeling.

    CannyDetector identifies colony boundaries using the Canny edge detector,
    then labels enclosed regions as individual objects. This multi-stage algorithm
    (Gaussian smoothing, gradient calculation, non-maximum suppression, hysteresis)
    produces thin, connected edges that robustly delineate colony perimeters even
    in noisy or unevenly illuminated images.

    Args:
        sigma: Gaussian smoothing standard deviation before edge detection (default 1.0).
            Higher values suppress noise and spurious edges but may blur fine
            boundaries or merge nearby colonies. Start with 1-2 for clean images.

        low_threshold: Lower hysteresis threshold (default 0.1). If use_quantiles=True,
            a fraction (0.1 = retain edges stronger than 10% of gradients); if False,
            absolute gradient magnitude. Increase to suppress weak edges from noise.

        high_threshold: Upper hysteresis threshold (default 0.2). Seeds edge traces.
            If use_quantiles=True, a fraction; if False, absolute magnitude. Must
            exceed low_threshold.

        use_quantiles: If True (default), thresholds interpreted as quantiles,
            adapting to image contrast automatically. If False, thresholds are
            absolute values, requiring manual tuning per imaging setup.

        min_size: Minimum object area in pixels (default 50). Increase to filter
            dust/debris; decrease to retain tiny colonies.

        invert_edges: If True (default), label enclosed regions as objects (colonies).
            If False, label edge pixels (for atypical cases or debugging).

        connectivity: Connectivity for labeling regions (1=4-connected, 2=8-connected,
            default 2). Higher values merge diagonally adjacent pixels.

    Attributes:
        sigma, low_threshold, high_threshold, use_quantiles, min_size,
        invert_edges, connectivity

    Returns:
        Image: Input image with objmap set to labeled colonies identified via edges.

    Raises:
        ValueError: If thresholds invalid or high_threshold < low_threshold.

    **Use cases**

    - **Well-separated colonies:** Clear boundaries on solid media where edge sharpness
      dominates over intensity differences.
    - **Variable illumination:** Plates with low contrast or uneven lighting that
      challenge intensity-based methods. Works well with translucent colonies.
    - **Textured colonies:** Heterogeneous internal texture or pigmentation that
      fragments under watershed or simple thresholding.

    **Limitations**

    - Diffuse/gradual colony boundaries (fuzzy, mucoid colonies) yield fragmented
      edges and under-segmentation.
    - Overlapping or touching colonies may merge into single edge contour. Increase
      sigma or use post-detection refinement to split merged regions.
    - Threshold tuning critical. Too aggressive = noise dominates; too conservative =
      colony boundaries vanish. Use use_quantiles=True for safer defaults.
    - Not intensity-based. If colonies differ mainly in brightness (not edges),
      consider Otsu or watershed instead.
    - May detect plate edges, dust, scratches as spurious boundaries. Use min_size
      filtering and ensure clean agar surfaces.

    **Parameter effects on colony detection**

    - **sigma:** Controls pre-smoothing. Higher values → fewer spurious edges but risk
      merging nearby colonies.
    - **low/high_threshold:** Balance edge detection sensitivity. Quantile mode adapts
      to image contrast; absolute mode requires manual calibration.
    - **min_size:** Filters small noise artifacts while preserving genuine small
      colonies.

    Examples:
        Basic edge-based detection::

            from phenotypic import Image
            from phenotypic.detect import CannyDetector

            plate = Image.imread("plate.jpg")
            detector = CannyDetector(sigma=1.0, use_quantiles=True)
            detected = detector.apply(plate)
            num_colonies = detected.objects.count
            print(f"Detected {num_colonies} colonies via edges")

        Pipeline with preprocessing and refinement::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur, CLAHE
            from phenotypic.detect import CannyDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                CLAHE(clip_limit=2.0),
                CannyDetector(sigma=1.0, low_threshold=0.1, high_threshold=0.2)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
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
        objmap = morphology.remove_small_objects(objmap, min_size=self.min_size)

        # Ensure correct dtype
        if objmap.dtype != image._OBJMAP_DTYPE:
            objmap = objmap.astype(image._OBJMAP_DTYPE)

        # Relabel to ensure consecutive labels
        image.objmap[:] = objmap
        image.objmap.relabel(connectivity=self.connectivity)

        return image


# Set the docstring so that it appears in the sphinx documentation
CannyDetector.apply.__doc__ = CannyDetector._operate.__doc__
