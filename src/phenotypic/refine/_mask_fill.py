from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import field_validator
from scipy.ndimage import binary_fill_holes

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.funcs_ import is_binary_mask
from phenotypic.tools_.typing_ import NdArrayField, TuneSpec


class MaskFill(ObjectRefiner):
    """Fill enclosed holes within colony masks using binary flood fill.

    Applies ``scipy.ndimage.binary_fill_holes`` to close voids produced by
    illumination gradients, pigment heterogeneity, or specular glare within
    colonies. Only holes that are completely enclosed by the mask are filled;
    gaps that connect to the image boundary are left unchanged. The operation
    produces simply connected colony masks better suited for area and shape
    measurements.

    For an overview of morphological refinement methods, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Donut-shaped masks produced by global thresholding on colonies with
          dark or depigmented centers.
        - Colonies with radial pigment patterns or sector-level texture that
          create interior gaps in the binary mask.
        - Pre-measurement cleanup to ensure colony area counts every interior
          pixel and circularity is not distorted by enclosed voids.

    Consider Also:
        - :class:`MaskClosing` for bridging narrow gaps between separate
          fragments rather than filling holes within a single object.
        - :class:`MaskOpening` for removing thin protrusions and external
          connections rather than filling internal holes.

    Args:
        structure: Binary ndarray defining the connectivity neighbourhood for
            flood fill. ``None`` uses the default cross-shaped (4-connected)
            element from ``scipy.ndimage``. Default: ``None``.
        origin: Integer offset applied to the structuring element centre.
            Rarely needs adjustment from the default. Default: 0.

    Returns:
        Image: Input image with ``objmask`` updated so all enclosed holes are
        filled; ``objmap`` is updated to reflect the filled mask.

    Raises:
        ValueError: If ``structure`` is provided but is not a binary array.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a walkthrough
        of refinement operations including hole filling.
    """

    structure: NdArrayField | None = None
    origin: Annotated[int, TuneSpec(tunable=False)] = 0

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
