from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from ._base_operation import BaseOperation
from ..util.exceptions_ import InterfaceError, OperationIntegrityError


class ImageOperation(BaseOperation):
    """
    Represents an abstract base class for image operations.

    This class provides a common abstract for applying transformations or
    operations to images. It defines a method to apply the operation and
    enforces the implementation of the specific operation in a subclass.
    Users can apply operations either in-place or on a copy of the image.

    """

    # Which integrity validation checks to perform
    # Can be set to validate_array_integrity, validate_matrix_integrity, validate_enh_matrix_integrity, validate_objmap_integrity, validate_objmap_integrity_consistency, validate_objmap_integrity_consistency_with_matrix
    # or a custom function that takes two images and returns None if the integrity is valid, otherwise raises OperationIntegrityError
    # If not set, no integrity validation checks are performed.

    def apply(self, image: Image, inplace=False) -> Image:
        """
        Applies the operation to an image, either in-place or on a copy.

        Args:
            image (Image): The input_image image to apply the operation on.
            inplace (bool): If True, modifies the image in place; otherwise,
                operates on a copy of the image.

        Returns:
            Image: The modified image after applying the operation.
        """
        try:
            matched_args = self._get_matched_operation_args()
            image = self._apply_to_single_image(
                cls_name=self.__class__.__name__,
                image=image,
                operation=self._operate,
                inplace=inplace,
                matched_args=matched_args,
            )
            return image
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            raise RuntimeError(f'{self.__class__.__name__} failed on image {image.name}: {e}') from e

    @staticmethod
    def _operate(image: Image) -> Image:
        """
        A placeholder for the main subfunction for an image operator for processing image objects.

        This method is called from ImageOperation.apply() and must be implemented in a subclass. This allows for checks for data integrity to be made.

        Args:
            image (Image): The image object to be processed by internal operations.

        Raises:
            InterfaceError: Raised if the method is not implemented in a subclass.
        """
        return image

    @staticmethod
    def _apply_to_single_image(cls_name, image, operation, inplace, matched_args):
        """Applies the operation to a single image. this intermediate function is needed for parallel execution."""
        try:
            return operation(image=image if inplace else image.copy(), **matched_args)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            raise Exception(f'{cls_name} failed on image {image.name}: {e}') from e
