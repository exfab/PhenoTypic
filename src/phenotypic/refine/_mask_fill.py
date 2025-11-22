from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from scipy.ndimage import binary_fill_holes
from typing import Optional

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools.funcs_ import is_binary_mask


class MaskFill(ObjectRefiner):
    """Fill holes inside binary object masks to produce solid colonies.

    Intuition:
        Thresholding on agar-plate images can leave small voids inside colony
        masks due to illumination gradients, pigment heterogeneity, or glare.
        Filling these holes produces contiguous masks that better match the
        true colony footprint and improve area-based phenotypes.

    Why this is useful for agar plates:
        Many colonies exhibit darker centers or radial texture. Hole filling
        mitigates these within-colony gaps so that downstream measurements
        (area, perimeter, intensity summaries) are less biased.

    Use cases:
        - After global or adaptive thresholding where donut-like masks appear.
        - Prior to morphological measurements that assume simply connected
          shapes.

    Caveats:
        - Over-aggressive filling with a large structuring element can bridge
          adjacent colonies through narrow gaps, hurting separation.
        - If masks contain genuine cavities that should remain (e.g., hollow
          artifacts), filling may misrepresent structure.

    Attributes:
        structure (Optional[np.ndarray]): Structuring element used to define the
            neighborhood for filling. A larger or denser structure tends to
            close larger holes but can also smooth away fine boundaries.
        origin (int): Center offset for the structuring element. Adjusting the
            origin subtly shifts how neighborhoods are evaluated, which can
            influence edge behavior at colony boundaries.

    Examples:
        >>> from phenotypic.refine import MaskFill
        >>> op = MaskFill()
        >>> image = op.apply(image, inplace=True)  # doctest: +SKIP
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
            if not is_binary_mask(structure): raise ValueError('arr object array must be a binary array')
        self.structure = structure
        self.origin = origin

    def _operate(self, image: Image) -> Image:
        image.objmask[:] = binary_fill_holes(
                input=image.objmask[:],
                structure=self.structure,
                origin=self.origin
        )
        return image
