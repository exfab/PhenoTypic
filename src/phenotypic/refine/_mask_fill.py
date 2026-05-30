from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import field_validator
from scipy.ndimage import binary_fill_holes

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.funcs_ import is_binary_mask
from phenotypic.tools_.typing_ import NdArrayField


class MaskFill(ObjectRefiner):
    """Fill holes inside colony masks to produce solid, contiguous regions.

    Uses binary flood fill to close voids left by illumination gradients,
    pigment heterogeneity, or glare within colonies. Produces masks that
    better match the true colony footprint for area and shape measurements.

    Args:
        structure: Binary structuring element defining the fill neighborhood.
            ``None`` uses the default cross-shaped element. Default: ``None``.
        origin: Center offset for the structuring element. Default: 0.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` updated with
        filled holes.

    Best For:
        - Donut-like masks from global thresholding on colonies with dark centers.
        - Colonies with radial pigment texture that creates interior gaps.
        - Pre-measurement cleanup to ensure simply connected shapes.

    Consider Also:
        - :class:`MaskClosing` for bridging narrow gaps *between* fragments
          rather than filling holes *within* objects.
        - :class:`MaskOpening` for the opposite effect — removing thin
          connections between objects.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a walkthrough
        of refinement operations.
    """

    structure: NdArrayField | None = None
    origin: int = 0

    @field_validator("structure")
    @classmethod
    def _validate_structure(cls, structure: np.ndarray | None) -> np.ndarray | None:
        """Require ``structure``, when provided, to be a binary mask.

        Reproduces the pre-migration ``__init__`` guard verbatim. The
        ``NdArrayField`` ``BeforeValidator`` has already coerced any
        list input to an ``np.ndarray`` by the time this runs.
        """
        if structure is not None:
            if not is_binary_mask(structure):
                raise ValueError("arr object array must be a binary array")
        return structure

    def _operate(self, image: Image) -> Image:
        image.objmask[:] = binary_fill_holes(
                input=image.objmask[:], structure=self.structure, origin=self.origin
        )
        return image
