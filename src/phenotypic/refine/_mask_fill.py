from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from scipy.ndimage import binary_fill_holes
from typing import Optional

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.funcs_ import is_binary_mask


class MaskFill(ObjectRefiner):
    """Fill holes inside colony masks to produce solid, contiguous regions.

    Uses binary flood fill to close voids left by illumination gradients,
    pigment heterogeneity, or glare within colonies. Produces masks that
    better match the true colony footprint for area and shape measurements.

    Best For:
        - Donut-like masks from global thresholding on colonies with dark centers.
        - Colonies with radial pigment texture that creates interior gaps.
        - Pre-measurement cleanup to ensure simply connected shapes.

    Consider Also:
        - :class:`MaskCloser` for bridging narrow gaps *between* fragments
          rather than filling holes *within* objects.
        - :class:`MaskOpener` for the opposite effect — removing thin
          connections between objects.

    Args:
        structure: Binary structuring element defining the fill neighborhood.
            ``None`` uses the default cross-shaped element. Default: ``None``.
        origin: Center offset for the structuring element. Default: 0.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` updated with
        filled holes.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a walkthrough
        of refinement operations.
    """

    def __init__(self, structure: Optional[np.ndarray] = None, origin: int = 0):
        """Initialize the filler and validate inputs.

        Args:
            structure (Optional[np.ndarray]): Binary structuring element. Larger
                or more connected structures fill bigger holes and may reduce
                small-scale texture within colony masks. If provided, must be a
                binary array; otherwise a ValueError is raised.
            origin (int): Origin offset for the structuring element. Typically
                left at 0; changing it slightly alters how neighborhoods are
                centered, which may affect edge sharpness at boundaries.

        Raises:
            ValueError: If ``structure`` is provided and is not a binary mask.
        """
        if structure is not None:
            if not is_binary_mask(structure):
                raise ValueError("arr object array must be a binary array")
        self.structure = structure
        self.origin = origin

    def _operate(self, image: Image) -> Image:
        image.objmask[:] = binary_fill_holes(
                input=image.objmask[:], structure=self.structure, origin=self.origin
        )
        return image
