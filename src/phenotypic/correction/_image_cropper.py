from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image

from phenotypic.abc_ import ImageCorrector


class ImageCropper(ImageCorrector):
    """Remove pixels from image edges by specifying crop margins.

    Crops all image components (rgb, gray, detect_mat, objmask, objmap)
    together. When applied to a GridImage, the grid structure is preserved
    and grid positions are recalculated for the cropped dimensions.

    Best For:
        - Removing scanner margins or borders outside the agar plate.
        - Eliminating edge artifacts (bent agar, labeling, moisture).
        - Standardizing image dimensions across a batch of plates.

    Consider Also:
        - :class:`ImagePadder` for adding pixels instead of removing them.
        - :class:`BorderObjectRemover` for removing edge-touching colonies
          without changing image dimensions.

    Args:
        left: Pixels to remove from the left edge. ``None`` means no
            cropping. Default: ``None``.
        right: Pixels to remove from the right edge. Default: ``None``.
        top: Pixels to remove from the top edge. Default: ``None``.
        bottom: Pixels to remove from the bottom edge. Default: ``None``.

    Returns:
        Image: Input image with all components cropped to the specified
        margins.

    See Also:
        :doc:`/how_to/notebooks/crop_and_pad` for a visual walkthrough of
        cropping and padding operations.
    """

    def __init__(self,
                 left: int | None = None,
                 right: int | None = None,
                 top: int | None = None,
                 bottom: int | None = None
                 ):
        """Initialize an ImageCropper with pixel margins to remove from each edge.

        Creates a cropper that removes the specified number of pixels from each edge of
        the image. All parameters are optional and default to None (no cropping from that
        edge).

        Args:
            left (int | None, optional): Number of pixels to remove from the left edge.
                Must be non-negative. If None, the left edge is not cropped (equivalent
                to 0). Defaults to None.
            right (int | None, optional): Number of pixels to remove from the right edge.
                Must be non-negative. If None, the right edge is not cropped (equivalent
                to 0). Defaults to None.
            top (int | None, optional): Number of pixels to remove from the top edge.
                Must be non-negative. If None, the top edge is not cropped (equivalent
                to 0). Defaults to None.
            bottom (int | None, optional): Number of pixels to remove from the bottom edge.
                Must be non-negative. If None, the bottom edge is not cropped (equivalent
                to 0). Defaults to None.

        Raises:
            ValueError: If any parameter is negative. All crop margins must be
                non-negative integers (or None).
        """
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.__prescreen_idxes()

    def __prescreen_idxes(self):
        if (self.left is not None) and (self.left < 0):
            raise ValueError("left cannot be negative")

        if (self.right is not None) and (self.right < 0):
            raise ValueError("right cannot be negative")

        if (self.top is not None) and (self.top < 0):
            raise ValueError("top cannot be negative")

        if (self.bottom is not None) and (self.bottom < 0):
            raise ValueError("bottom cannot be negative")

    def _operate(self, image: Image) -> Image:
        """Crop the image by removing pixels from edges specified in __init__.

        Extracts the crop indices for each edge and returns a new Image with the cropped
        region. The entire image structure (rgb, gray, detect_mat, objmask, objmap) is
        cropped identically, ensuring all components remain synchronized.

        For GridImage instances, preserves the grid structure (grid_finder, nrows, ncols)
        while recalculating grid positions for the cropped region.

        Args:
            image (Image): The image to crop. The image is not modified; a new cropped
                Image (or GridImage if input was GridImage) is returned.

        Returns:
            Image: A new Image instance (or GridImage if input was GridImage) containing
                only the central cropped region, with all image components (rgb, gray,
                detect_mat, objmask, objmap) reduced to the same rectangular region.

        Raises:
            ValueError: If the crop parameters would result in an invalid slice (e.g.,
                top edge >= bottom edge or left edge >= right edge after cropping).

        Examples:
            Basic cropping of a loaded image:

            >>> from phenotypic import Image
            >>> from phenotypic.correction import ImageCropper
            >>> image = Image.imread('plate.jpg')  # doctest: +SKIP
            >>> cropper = ImageCropper(top=50, bottom=50, left=40, right=40)
            >>> # Returns new cropped Image; original is unchanged
            >>> cropped = cropper.apply(image)  # doctest: +SKIP
            >>> print(f"Original shape: {image.shape}")  # doctest: +SKIP
            >>> print(f"Cropped shape: {cropped.shape}")  # doctest: +SKIP

            Cropping a GridImage preserves grid settings:

            >>> from phenotypic import GridImage
            >>> from phenotypic.correction import ImageCropper
            >>> # Load plate image with scanner border
            >>> grid_img = GridImage('plate_with_border.tiff', nrows=8, ncols=12)  # doctest: +SKIP
            >>> # Remove scanner border
            >>> cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
            >>> cropped = cropper.apply(grid_img)  # doctest: +SKIP
            >>> # GridImage type and settings preserved
            >>> assert isinstance(cropped, GridImage)  # doctest: +SKIP
            >>> assert cropped.nrows == 8  # doctest: +SKIP
            >>> assert cropped.ncols == 12  # doctest: +SKIP
            >>> # Grid positions are recalculated for cropped image
            >>> cropped.show(overlay=True, show_gridlines=True)  # doctest: +SKIP
        """
        top, bottom_idx, left, right_idx = self._get_idxes(image)

        # Use existing slicing (__getitem__) - returns Image for both Image and GridImage
        cropped = image[top:bottom_idx, left:right_idx]

        # Import GridImage locally to avoid potential circular imports
        from phenotypic import GridImage

        # Preserve original image's name before modifications
        original_name = image.name

        # If input was GridImage, upgrade the cropped Image back to GridImage
        if isinstance(image, GridImage):
            # Convert the cropped Image to GridImage with preserved settings
            cropped = GridImage(
                    arr=cropped,  # Pass the Image instance
                    name=image.name,  # Preserve original image's name
                    grid_finder=image.grid_finder,  # Preserve grid_finder instance
                    nrows=image.nrows,
                    ncols=image.ncols,
                    bit_depth=cropped.bit_depth,
                    illuminant=cropped.illuminant,
                    gamma=cropped.gamma,
            )

        result = self._cpy_new_arrs(image, cropped)
        # Restore original name (may be overwritten by set_image).
        # set_image() calls _set_from_class_instance() which deep copies all metadata
        # from the source image, including the name. We preserve the original image's
        # name to maintain its identity through the cropping operation.
        result.name = original_name

        return result

    @staticmethod
    def _cpy_new_arrs(img: Image | GridImage,
                      new_img: Image | GridImage) -> Image | GridImage:
        """Copies the array data over. This is needed due to needing to make changes to
        the image in-place."""
        img.set_image(new_img)
        return img

    def _get_idxes(self, image: Image) -> Tuple[int, int, int, int]:
        """Calculate array slice indices for the crop region based on image dimensions.

        Given the image dimensions and the configured crop margins (left, right, top,
        bottom), computes the array slice indices for the cropped region. Validates that
        the resulting crop region is valid (i.e., top edge is above bottom edge and left
        edge is left of right edge after cropping).

        Args:
            image (Image): The image to be cropped. The image's height and width are
                used to determine the final slice indices.

        Returns:
            Tuple[int, int, int, int]: A tuple of (top_idx, bottom_idx, left_idx,
                right_idx) representing the array slice bounds for the crop region:
                - top_idx: The starting row index (pixels removed from top edge)
                - bottom_idx: The ending row index (total height minus pixels removed
                  from bottom edge)
                - left_idx: The starting column index (pixels removed from left edge)
                - right_idx: The ending column index (total width minus pixels removed
                  from right edge)

                These indices are intended to be used in NumPy slicing as
                image_array[top_idx:bottom_idx, left_idx:right_idx].

        Raises:
            ValueError: If the crop parameters would result in invalid dimensions:
                - If top_idx >= bottom_idx (no valid rows remain after cropping)
                - If left_idx >= right_idx (no valid columns remain after cropping)

                This can occur if the sum of crop margins from opposite edges exceeds
                the image dimension (e.g., cropping 100 pixels top and 100 pixels bottom
                from a 150-pixel tall image).

        Examples:
            Understanding slice indices from crop parameters:

            >>> from phenotypic import Image
            >>> from phenotypic.correction import ImageCropper
            >>> # Example: 1000x1200 image, crop 50 from all edges
            >>> image = Image.imread('large_plate.tiff')  # doctest: +SKIP
            >>> cropper = ImageCropper(top=50, bottom=50, left=50, right=50)
            >>> # Internal calculation:
            >>> # height=1000, width=1200
            >>> # top_idx = 0 + 50 = 50
            >>> # bottom_idx = 1000 - 50 - 1 = 949
            >>> # left_idx = 0 + 50 = 50
            >>> # right_idx = 1200 - 50 - 1 = 1149
            >>> # Result: image[50:949, 50:1149] extracts central 899x1099 region
            >>> cropped = cropper.apply(image)  # doctest: +SKIP
            >>> print(cropped.shape)  # doctest: +SKIP
        """
        height, width = image.shape[:2]

        # Set dynamic defaults (None means 0 pixels cropped from that edge)
        top = 0 if self.top is None else self.top
        bottom = 0 if self.bottom is None else self.bottom
        bottom_idx = height - bottom
        if top > bottom_idx:
            raise ValueError(
                    f"top index ({top}) cannot be > bottom index ({bottom_idx}).")

        left = 0 if self.left is None else self.left
        right = 0 if self.right is None else self.right
        right_idx = width - right
        if left > right_idx:
            raise ValueError(
                    f"left index ({left}) cannot be > right index ({right_idx}).")

        return top, bottom_idx, left, right_idx
