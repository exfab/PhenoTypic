from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image, GridImage

import numpy as np
from skimage import feature, morphology
from scipy import ndimage

from phenotypic.abc_ import ThresholdDetector


class CannyDetector(ThresholdDetector):
    """
    Canny edge-based object detection for microbial colonies.

    Applies the Canny edge detector to identify colony boundaries, then labels
    the enclosed regions as individual objects. The Canny algorithm uses a
    multi-stage process: Gaussian smoothing, gradient calculation, non-maximum
    suppression, and hysteresis thresholding to produce thin, connected edges
    that robustly delineate colony perimeters even in noisy or unevenly
    illuminated images.

    Use cases (agar plates):
    - Detect well-separated colonies with clear boundaries on solid media where
      edge sharpness dominates over intensity differences.
    - Handle plates with variable illumination or low contrast that challenge
      intensity-based thresholding (e.g., translucent colonies on light agar).
    - Segment colonies with heterogeneous internal texture or pigmentation that
      might fragment under watershed or simple thresholding.
    - Robustly trace colony perimeters when background subtraction is imperfect
      or when agar texture is pronounced.

    Tuning and effects:
    - sigma: Controls pre-smoothing intensity. Higher sigma reduces noise
      sensitivity and suppresses spurious edges from agar granularity or scanner
      artifacts, but may blur fine colony boundaries or merge nearby colonies
      if set too high. Start with 1-2 for high-resolution images; increase for
      noisier scans.
    - low_threshold: Lower bound in hysteresis thresholding. Raising this
      suppresses weak edges from noise or faint texture but may fragment colony
      boundaries if edges are dim. Lowering it recovers more boundary detail but
      risks false edges. If use_quantiles=True, this is a fraction (0-1) of
      gradient values; if False, an absolute gradient magnitude.
    - high_threshold: Upper bound in hysteresis thresholding. Strong edges above
      this seed the edge traces; too high and faint colonies lose boundaries, too
      low and noise creates spurious edges. Adjust relative to low_threshold to
      control edge connectivity.
    - use_quantiles: When True, thresholds are interpreted as quantiles of the
      gradient distribution (e.g., 0.1 = 10th percentile), making behavior more
      robust to image-specific intensity ranges. When False, thresholds are
      absolute gradient magnitudes, requiring manual tuning per imaging setup.
    - min_size: Minimum pixel area to retain as an object after labeling regions
      enclosed by edges. Increase to remove dust, debris, or imaging artifacts;
      decrease to capture very small colonies. Setting too high discards genuine
      small colonies.
    - invert_edges: When True (default), regions *between* edges (i.e., enclosed
      areas) are labeled as objects, suitable for detecting solid colonies. When
      False, edges themselves are labeled (useful for atypical cases like ring-
      shaped colonies or debugging edge quality).
    - connectivity: Connectivity level for labeling regions (1 for 4-connected,
      2 for 8-connected in 2D). Higher connectivity merges diagonally adjacent
      pixels into the same object, which can join fragmented colony regions but
      may also merge nearby colonies touching at corners.

    Caveats:
    - Canny assumes objects are defined by edges. Colonies with very diffuse or
      gradual boundaries (e.g., fuzzy/mucoid colonies) may yield incomplete or
      fragmented edges, resulting in under-segmentation or missed objects.
    - Overlapping or touching colonies may be outlined as a single contiguous
      edge, causing multiple colonies to merge into one object. Pre-blur or
      increase sigma to regularize boundaries, or use watershed refinement post-
      detection to split merged regions.
    - Threshold tuning is critical: too aggressive and noise dominates, too
      conservative and colony boundaries vanish. use_quantiles=True often
      provides a safer starting point.
    - Does not inherently handle intensity-based segmentation; if colonies differ
      mainly in brightness (not edges), consider Otsu or watershed instead.
    - May detect plate edges, dust, or scratches as spurious boundaries. Use
      min_size filtering and ensure clean agar surfaces or pre-mask the plate
      region if needed.

    Attributes:
        sigma (float): Standard deviation for Gaussian smoothing applied before
            edge detection. Higher values reduce noise but blur fine boundaries.
        low_threshold (float): Lower threshold for hysteresis. Controls weak edge
            retention; interpretation depends on use_quantiles.
        high_threshold (float): Upper threshold for hysteresis. Seeds edge traces;
            interpretation depends on use_quantiles.
        use_quantiles (bool): If True, thresholds are quantiles (0-1); if False,
            absolute gradient magnitudes.
        min_size (int): Minimum object size in pixels. Objects smaller than this
            are removed to filter noise and artifacts.
        invert_edges (bool): If True, label regions between edges as objects
            (default for colonies); if False, label edge pixels themselves.
        connectivity (int): Connectivity for labeling (1 = 4-connected, 2 =
            8-connected). Higher connectivity merges diagonal neighbors.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        low_threshold: float = 0.1,
        high_threshold: float = 0.2,
        use_quantiles: bool = True,
        min_size: int = 50,
        invert_edges: bool = True,
        connectivity: int = 1,
    ):
        """
        Parameters:
            sigma (float): Gaussian smoothing strength before edge detection. Start
                with 1-2 for clean images; increase for noisy scans to suppress
                spurious edges. Keep below typical colony radius to avoid merging.
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
        from phenotypic import Image, GridImage

        enhanced_matrix = image.enh_gray[:]

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
        objmap, _ = ndimage.label(regions, structure=ndimage.generate_binary_structure(2, self.connectivity))

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

