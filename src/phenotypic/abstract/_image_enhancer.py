from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from ._image_operation import ImageOperation
from phenotypic.util.exceptions_ import InterfaceError, DataIntegrityError, OperationFailedError
from phenotypic.util.funcs_ import validate_operation_integrity


class ImageEnhancer(ImageOperation):
    """ImageEnhancers impact the enh_matrix of the Image object and are used for improving detection quality."""

    @validate_operation_integrity('image.array', 'image.matrix')
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)