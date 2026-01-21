from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from PIL import ImageEnhance
from skimage import morphology

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic.abc_ import ImageEnhancer
from phenotypic.tools_ import FootprintMixin


class GrayOpening(ImageEnhancer, FootprintMixin):
    """Performs morphological opening on grayscale image arrays as a preprocessing step.

    This class applies morphological opening, which is an erosion followed by a dilation,
    to grayscale image arrays. It is particularly useful for removing small artifacts
    and noise from the image, such as dust particles or small microbe colonies, while
    preserving the overall shape of larger structures. The operation is applied based
    on a specified shape shape and size.

    Attributes:
        shape (Literal["square", "diamond", "disk"]): Specifies the shape of the
            shape used for the morphological opening operation. Changing this
            affects how structures in the image are eroded and dilated. For example,
            a "square" may preserve edges better, "diamond" is more rounded at
            diagonals, while "disk" provides uniform circular operations. Selecting
            an improper shape might remove desired features or alter key microbe
            colony structures.
        width (int): Determines the size of the shape for the operation. Larger
            values increase the area affected during erosion and dilation. This can
            lead to the complete removal of smaller colonies or features, but a
            sufficiently large width is required to remove noise. A width that is too
            small may leave artifacts and undesired details, compromising image
            preprocessing for larger features.
    """

    def __init__(self, shape: Literal["square", "diamond", "disk"] = "square",
                 width: int = 5):
        """
        A kernel configuration class for image processing tasks, particularly suited for applications
        such as analyzing and processing images of microbe colonies on solid media agar. This class
        enables the definition of a kernel shape and size, which significantly impacts the morphological
        operations applied to the image (e.g., filtering, dilation, erosion). Adjusting these parameters
        can enhance or hinder the detection and analysis of colony boundaries, shapes, and distribution.

        Attributes:
            shape (Literal["square", "diamond", "disk"]): The geometric shape of the kernel. This attribute
                governs the pattern and extent of neighboring pixels involved in the processing operation.
                Choosing "square" results in a uniform rectangular influence, which may be suitable for
                isotropic features but could introduce angular artifacts in circular features like microbe
                colonies. The "diamond" shape provides a more angular neighborhood pattern that helps
                preserve diagonal structures. On the other hand, "disk" introduces a circular pattern
                that can align well with colony boundaries and reduce distortions in rounded features.

            width (int): The size (diameter) of the kernel in pixels. A larger width increases the
                area of influence during image processing, which can smooth out smaller features like
                noise but potentially merge closely spaced microbe colonies into larger regions. Smaller
                values offer finer detail and greater distinction between colonies but may leave noise
                unprocessed or small artifacts unchanged.
        """
        self.shape = shape
        self.width = width

    def _operate(self, image: Image) -> Image:
        image.enh_gray[:] = morphology.opening(
                image=image.enh_gray[:],
                footprint=self._make_footprint(
                        shape=self.shape,
                        width=self.width,
                ),
        )
        return image
