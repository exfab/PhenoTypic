from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np

from ._image_operation import ImageOperation
from ..util.constants_ import IMAGE_FORMATS
from phenotypic.util.exceptions_ import OperationFailedError, InterfaceError, DataIntegrityError
from phenotypic.util.funcs_ import validate_array_integrity, validate_matrix_integrity, validate_enh_matrix_integrity


# <<Interface>>
class MapModifier(ImageOperation):
    """Map modifiers edit the object map and are used for removing, combining, and re-ordering objects."""
    _VALIDATION_OPERATIONS = (validate_array_integrity, validate_matrix_integrity, validate_enh_matrix_integrity,)
