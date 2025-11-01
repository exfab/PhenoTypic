from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from scipy.ndimage import binary_fill_holes
from typing import Optional

from phenotypic.ABC_ import MapModifier
from phenotypic.tools.funcs_ import is_binary_mask


class MaskFill(MapModifier):
    """A class for filling holes in binary masks within an image.

    This class provides functionality to process an input binary mask
    in an image, filling any holes using the specified structuring element
    and origin.

    Attributes:
        structure (Optional[np.ndarray]): Structuring element to define the neighborhood
            for filling holes. If None, a default cross-shaped structure is used.
        origin (int): Origin point to define the center of the structuring element,
            influencing how the neighborhood is considered.
    """

    def __init__(self, structure: Optional[np.ndarray] = None, origin: int = 0):
        """
        Initializes an instance of the class and validates input parameters.

        Args:
            structure (Optional[np.ndarray]): A binary mask array. This parameter must
                be a binary array. If provided and not valid, a ValueError is raised.
                Defaults to None.
            origin (int): An integer value representing the origin or offset. Defaults
                to 0.

        Raises:
            ValueError: If the `structure` parameter is provided and is not a binary
                mask array.
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
