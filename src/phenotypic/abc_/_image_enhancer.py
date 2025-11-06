from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from ._image_operation import ImageOperation
from phenotypic.tools.exceptions_ import InterfaceError, DataIntegrityError, OperationFailedError
from phenotypic.tools.funcs_ import validate_operation_integrity
from abc import ABC


class ImageEnhancer(ImageOperation, ABC):
    """ImageEnhancers impact the enh_gray of the Image object and are used for improving detection quality."""

    @validate_operation_integrity('image.rgb', 'image.gray')
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)
