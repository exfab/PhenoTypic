from __future__ import annotations
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np

from ._image_operation import ImageOperation
from phenotypic.tools.exceptions_ import OperationFailedError, InterfaceError, DataIntegrityError
from phenotypic.tools.funcs_ import validate_operation_integrity
from abc import ABC
from skimage.morphology import disk, square, diamond


# <<Interface>>
class ObjectRefiner(ImageOperation, ABC):
    """`ObjectRefiner`s edit the object mask and object map.
    They are used for removing, combining, and re-ordering objects."""

    @validate_operation_integrity('image.rgb', 'image.gray', 'image.enh_gray')
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)

    @staticmethod
    def _make_footprint(shape: Literal["square", "diamond", "disk"], radius: int) -> np.ndarray:
        """
        Creates a morphological footprint for image processing.

        This static utility method generates a structuring element (footprint) useful
        for morphological operations like dilation and erosion. It supports different
        shapes such as square, diamond, and disk, which are often used in image analysis
        tasks. These morphological tools are particularly helpful in analyzing colonies
        of microbes on solid media agar.

        Args:
            shape (Literal["square", "diamond", "disk"]): The shape of the footprint to create.
                Adjusting the shape changes the way the morphological operations interact
                with the image. For example:
                - "square" creates a square footprint, which may emphasize features with
                  sharp edges.
                - "diamond" creates a diamond-shaped footprint, which may enhance diagonal
                  connections while being less sensitive to orthogonal edges.
                - "disk" generates a circular footprint, which may better preserve rounded
                  microbial colony shapes.

            radius (int): The radius of the footprint. This defines the size of the
                structuring element. Larger radii will lead to broader morphological
                effects, which could impact the resolution of small colonies but can help
                to merge fragmented edges or clean noise.

        Returns:
            np.ndarray: A binary numpy array representing the generated footprint. The
            footprint will be used for convolutional operations over the microbial colony
            image. The specific shape and radius passed as arguments dictate the size
            and morphology of this array.

        Raises:
            ValueError: If an unsupported shape type is passed to the function.
        """
        radius = int(radius)
        match shape:
            case "square":
                return square(width=radius*2)
            case "diamond":
                return diamond(radius=radius)
            case "disk":
                return disk(radius=radius)
            case _:
                raise ValueError(f"Unknown shape: {shape}")
