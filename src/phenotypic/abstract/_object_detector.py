from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from ._image_operation import ImageOperation
from phenotypic.util.exceptions_ import OperationFailedError, DataIntegrityError, InterfaceError
from phenotypic.util.funcs_ import validate_array_integrity, validate_matrix_integrity, validate_enh_matrix_integrity

# <<Interface>>
class ObjectDetector(ImageOperation):
    """ObjectDetectors are for detecting objects in an image. They change the image object mask and map."""
    _VALIDATION_OPERATIONS = (validate_array_integrity, validate_matrix_integrity, validate_enh_matrix_integrity,)