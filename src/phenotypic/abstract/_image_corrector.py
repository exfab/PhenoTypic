from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from typing import Union, Dict

from ._image_operation import ImageOperation
from phenotypic.util.exceptions_ import InterfaceError, OperationFailedError
from abc import ABC


class ImageCorrector(ImageOperation, ABC):
    """ImageCorrectors are for general operations that alter every image component such as rotating.
    These have no integrity checks due to every component being altered by the operation..

    """
    pass
