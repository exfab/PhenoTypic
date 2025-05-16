from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from ._image_operation import ImageOperation
from phenotypic.util.exceptions_ import InterfaceError, DataIntegrityError, OperationFailedError
from phenotypic.util.constants_ import IMAGE_FORMATS
from phenotypic.util.funcs_ import validate_array_integrity, validate_matrix_integrity, validate_enh_matrix_integrity, validate_objmask_integrity


class ImageEnhancer(ImageOperation):
    """ImageEnhancers impact the enh_matrix of the Image object and are used for improving detection quality."""
    _VALIDATION_OPERATIONS = (validate_array_integrity, validate_matrix_integrity)
