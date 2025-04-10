from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from ..util.exceptions_ import InterfaceError

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
        Applies a certain operation to an image, either in-place or on a copy.

        Args:
            image (Image): The input image to apply the operation on.
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

        This method is called from apply and must be implemented in a subclass. This allows for checks for data integrity to be made.

        Args:
            image (Image): The image object to be processed by internal operations.

        Raises:
            InterfaceError: Raised if the method is not implemented in a subclass.
        """
        raise InterfaceError
