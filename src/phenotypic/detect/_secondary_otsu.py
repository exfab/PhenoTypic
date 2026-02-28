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
    """Two-stage Otsu refinement detector with per-object thresholding.

    SecondaryOtsuDetector applies Otsu's threshold in two stages: (1) initial Otsu
    on full image (or use existing objmask), (2) re-apply Otsu independently to each
    detected object. This refines boundaries by computing a local threshold for each
    colony based on its own intensity distribution, improving edge accuracy when
    colonies have varying intensities or backgrounds.

    Args:
        None

    Attributes:
        None

    Returns:
        Image: Input image with objmask refined via per-object Otsu thresholding.

    Raises:
        ValueError: If threshold computation fails (e.g., all pixels same value).

    **Use cases**

    - **Boundary refinement:** Initial detection is correct but blurry at edges.
      Per-object Otsu sharpens colony boundaries using local intensity distribution.
    - **Heterogeneous plates:** Colonies vary in intensity across the plate. Each
      object gets its own threshold adapted to local conditions.
    - **Halo suppression:** Soft halos around colonies from preprocessing can blur
      edges. Per-object thresholding removes halo while keeping colony centers.

    **Limitations**

    - Two-stage approach loses small colonies at edges. Initial Otsu may miss faint
      or small objects; secondary stage can't recover them.
    - Requires prior detection. If no objects initially detected (objmask all zeros),
      falls back to applying Otsu twice on full image, which may not improve results.
    - Small objects may fail thresholding. Objects with too few pixels or uniform
      intensity cannot compute a meaningful Otsu threshold and are preserved as-is.
    - Per-object processing adds overhead. For plates with many objects (>1000),
      consider whether global secondary thresholding might suffice.

    **Parameter effects on colony detection**

    - No user-tunable parameters. Behavior is deterministic: apply Otsu to each
      object independently. Results depend on input image intensity distribution
      and initial objmask quality.

    Examples:
        Refine initial detection with secondary Otsu::

            from phenotypic import Image
            from phenotypic.detect import OtsuDetector, SecondaryOtsuDetector

            plate = Image.imread("plate.jpg")

            # Initial Otsu detection
            detector1 = OtsuDetector()
            intermediate = detector1.apply(plate)

            # Refine boundaries with per-object secondary Otsu
            detector2 = SecondaryOtsuDetector()
            refined = detector2.apply(intermediate)
            mask = refined.objmask[:]
            print(f"Refined mask: {mask.sum()} colony pixels")

        Pipeline combining threshold refinement::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur
            from phenotypic.detect import OtsuDetector, SecondaryOtsuDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                OtsuDetector(),
                SecondaryOtsuDetector()
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
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
