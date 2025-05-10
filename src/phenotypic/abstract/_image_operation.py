from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np

from ..util.exceptions_ import InterfaceError, OperationIntegrityError

class ImageOperation:
    """
    Represents an abstract base class for image operations.

    This class provides a common abstract for applying transformations or
    operations to images. It defines a method to apply the operation and
    enforces the implementation of the specific operation in a subclass.
    Users can apply operations either in-place or on a copy of the image.

    """
    def apply(self, image, inplace=False) -> Image:
        """
        Applies the operation to an image, either in-place or on a copy.

        Args:
            image (Image): The input_image image to apply the operation on.
            inplace (bool): If True, modifies the image in place; otherwise,
                operates on a copy of the image.

        Returns:
            Image: The modified image after applying the operation.
        """
        if inplace:
            return self._operate(image)
        else:
            return self._operate(image.copy())

    def _operate(self, image: Image) -> Image:
        """
        A placeholder for the subfunction for an image operator for processing image objects.

        This method is called from ImageOperation.apply() and must be implemented in a subclass. This allows for checks for data integrity to be made.

        Args:
            image (Image): The image object to be processed by internal operations.

        Raises:
            InterfaceError: Raised if the method is not implemented in a subclass.
        """
        raise InterfaceError

    @staticmethod
    def _validate_integrity(pre_op_imcopy, post_op_image) -> None:
        pass

    @staticmethod
    def _validate_array_integrity(pre_op_imcopy, post_op_image) -> None:
        if post_op_image.imformat.isarray():
            if not np.array_equal(pre_op_imcopy.array[:], post_op_image.array[:]):
                raise OperationIntegrityError('array', post_op_image.name)

    @staticmethod
    def _validate_matrix_integrity(pre_op_imcopy, post_op_image) -> None:
        if not np.array_equal(pre_op_imcopy.matrix[:], post_op_image.matrix[:]):
            raise OperationIntegrityError('matrix', post_op_image.name)

    @staticmethod
    def _validate_enh_matrix_integrity(pre_op_imcopy, post_op_image) -> None:
        if not np.array_equal(pre_op_imcopy.enh_matrix[:], post_op_image.enh_matrix[:]):
            raise OperationIntegrityError('enh_matrix', post_op_image.name)

    @staticmethod
    def _validate_objmask_integrity(pre_op_imcopy, post_op_image) -> None:
        if not np.array_equal(pre_op_imcopy.objmask[:], post_op_image.objmask[:]):
            raise OperationIntegrityError('objmask', post_op_image.name)