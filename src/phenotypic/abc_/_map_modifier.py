from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np

from ._image_operation import ImageOperation
from phenotypic.tools.exceptions_ import OperationFailedError, InterfaceError, DataIntegrityError
from phenotypic.tools.funcs_ import validate_operation_integrity
from abc import ABC


# <<Interface>>
class ObjectRefiner(ImageOperation, ABC):
    """`ObjectRefiner`s edit the object mask and object map.
    They are used for removing, combining, and re-ordering objects."""

    @validate_operation_integrity('image.rgb', 'image.gray', 'image.enh_gray')
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)
