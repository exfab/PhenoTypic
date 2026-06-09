from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from scipy.ndimage import labeled_comprehension
from skimage.filters import threshold_otsu
from skimage.measure import label
from phenotypic.abc_ import ThresholdDetector


def _safe_otsu(values: np.ndarray) -> float:
    """Compute Otsu threshold, returning -inf if not possible."""
    if len(values) < 2 or values.min() == values.max():
        return -np.inf
    try:
        return threshold_otsu(values)
    except ValueError:
        return -np.inf


class SecondaryOtsuDetector(ThresholdDetector):
    """Detect colonies by two-stage Otsu thresholding with per-object boundary refinement.

    Apply an initial global Otsu threshold (or reuse an existing ``objmask``),
    then re-threshold each detected object independently using its own intensity
    distribution. This sharpens colony boundaries on plates where colonies vary
    in brightness, removing soft halos while preserving colony cores. For a
    full comparison of detection strategies see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        - Refining boundaries after an initial global threshold that leaves
          soft or blurry colony edges.
        - Heterogeneous plates where colonies differ in pigmentation or optical
          density across the plate surface.
        - Suppressing preprocessing halos that expand colony outlines beyond
          their true boundaries.

    Consider Also:
        - :class:`OtsuDetector` when a single global threshold already
          produces clean colony boundaries.
        - :class:`HysteresisDetector` when colony intensity varies smoothly
          and dual-threshold expansion is more appropriate than per-object
          refinement.
        - :class:`RankOtsuDetector` when spatially varying illumination is the
          primary cause of boundary inaccuracy.

    Returns:
        Image: Input image with ``objmask`` set to the refined binary colony
        mask and ``objmap`` set to labeled connected components.

    References:
        [1] N. Otsu, "A threshold selection method from gray-level
        histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1,
        pp. 62--66, 1979.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies` for a step-by-step
        tutorial on basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm` for guidance on
        selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared` for an in-depth
        comparison of all thresholding strategies.
    """

    def _operate(self, image: Image) -> Image:
        """Apply Otsu thresholding independently to each object in the mask.

        If no object map exists, performs initial Otsu on the full image first,
        then applies per-object Otsu refinement to each detected region.
        """
        detect_mat = image.detect_mat[:]

        # If there are no objects, perform an initial global Otsu
        if image.num_objects == 0:
            initial_mask = detect_mat >= threshold_otsu(detect_mat)
        else:
            initial_mask = image.objmask[:]

        # Label connected components in the initial mask
        labeled_mask = label(initial_mask)
        num_objects = labeled_mask.max()

        if num_objects == 0:
            image.objmask = initial_mask
            return image

        # Compute Otsu threshold for each object (vectorized across all objects)
        # Returns array of thresholds indexed by object id (1 to num_objects)
        thresholds = labeled_comprehension(
            detect_mat, labeled_mask, range(1, num_objects + 1),
            _safe_otsu, float, -np.inf
        )

        # Build threshold lookup: index 0 = background (inf), indices 1..n = object thresholds
        # Using inf for background ensures those pixels stay False
        threshold_lookup = np.concatenate([[np.inf], thresholds])

        # Create per-pixel threshold map via label indexing (vectorized)
        threshold_map = threshold_lookup[labeled_mask]

        # Vectorized comparison: pixels above their object's threshold
        image.objmask = detect_mat >= threshold_map
        return image
