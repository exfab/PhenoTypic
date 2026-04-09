from typing import Literal

import numpy as np
from skimage import morphology
from skimage.morphology import diamond, disk


class FootprintMixin:
    """
    Provides a mixin for creating morphological footprints for image processing.

    The FootprintMixin class contains a static utility method to generate structuring elements
    (footprints) used in various image processing tasks. This functionality is particularly
    helpful in the context of analyzing microbial colonies on solid media agar plates.
    Morphological footprints are used to highlight specific features in images, such as
    colony edges, shapes, or connectivity, and can assist in segmentation, noise reduction,
    and feature extraction.

    Attributes:
        None
    """

    @staticmethod
    def _make_footprint(
            shape: Literal["square", "diamond", "disk"], width: int
    ) -> np.ndarray:
        """
        Creates a morphological shape for image processing.

        This static utility method generates a structuring element (shape) useful
        for morphological operations like dilation and erosion. It supports different
        shapes such as square, diamond, and disk, which are often used in image analysis
        tasks. These morphological tools are particularly helpful in analyzing colonies
        of microbes on solid media agar.

        Args:
            shape (Literal["square", "diamond", "disk"]): The shape of the shape to create.
                Adjusting the shape changes the way the morphological operations interact
                with the image. For example:
                - "square" creates a square shape, which may emphasize features with
                  sharp edges.
                - "diamond" creates a diamond-shaped shape, which may enhance diagonal
                  connections while being less sensitive to orthogonal edges.
                - "disk" generates a circular shape, which may better preserve rounded
                  microbial colony shapes.

            width (int): The width of the shape. This defines the size of the
                structuring element. Larger width will lead to broader morphological
                effects, which could impact the resolution of small colonies but can help
                to merge fragmented edges or clean noise.

        Returns:
            np.ndarray: A binary numpy array representing the generated shape. The
            shape will be used for convolutional operations over the microbial colony
            image. The specific shape and width passed as arguments dictate the size
            and morphology of this array.

        Raises:
            ValueError: If an unsupported shape type is passed to the function.
        """
        width = int(width)
        match shape:
            case "square":
                return morphology.footprint_rectangle(shape=(width, width))
            case "diamond":
                return diamond(radius=width // 2)
            case "disk":
                return disk(radius=width // 2)
            case _:
                raise ValueError(f"Unknown shape: {shape}")

    @staticmethod
    def _stamp_footprint(
        mask: np.ndarray,
        labeled: np.ndarray,
        fp_mask: np.ndarray,
        cy: int,
        cx: int,
        label_id: int,
    ) -> None:
        """Stamp a boolean footprint at ``(cy, cx)`` into *mask* and *labeled*.

        Handles boundary clipping so the footprint can extend beyond image
        edges without raising an ``IndexError``.

        Args:
            mask: Boolean mask array to update in place (logical OR).
            labeled: Integer label array to update in place.
            fp_mask: Boolean footprint array.
            cy: Row centre of the stamp.
            cx: Column centre of the stamp.
            label_id: Integer label written into *labeled* where the
                footprint is ``True``.
        """
        h, w = mask.shape
        fp_h, fp_w = fp_mask.shape

        img_y0 = max(0, cy - fp_h // 2)
        img_y1 = min(h, cy - fp_h // 2 + fp_h)
        img_x0 = max(0, cx - fp_w // 2)
        img_x1 = min(w, cx - fp_w // 2 + fp_w)

        fp_y0 = img_y0 - (cy - fp_h // 2)
        fp_x0 = img_x0 - (cx - fp_w // 2)
        fp_slice = fp_mask[
            fp_y0 : fp_y0 + (img_y1 - img_y0),
            fp_x0 : fp_x0 + (img_x1 - img_x0),
        ]

        mask[img_y0:img_y1, img_x0:img_x1] |= fp_slice
        region = labeled[img_y0:img_y1, img_x0:img_x1]
        labeled[img_y0:img_y1, img_x0:img_x1] = np.where(
            fp_slice, label_id, region
        )