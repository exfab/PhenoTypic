from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np

from ._image_operation import ImageOperation
from phenotypic.util.exceptions_ import OperationFailedError, InterfaceError, DataIntegrityError
from phenotypic.util.funcs_ import validate_operation_integrity


# <<Interface>>
class MapModifier(ImageOperation):
    """Map modifiers edit the object map and are used for removing, combining, and re-ordering objects."""

    @validate_operation_integrity('image.array', 'image.matrix', 'image.enh_matrix')
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)