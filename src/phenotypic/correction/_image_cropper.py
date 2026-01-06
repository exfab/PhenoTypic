from __future__ import annotations

from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import GridImage, Image

from phenotypic.abc_ import ImageCorrector


class ImageCropper(ImageCorrector):
    """Remove pixels from image edges by specifying crop margins.

    ImageCropper crops an image by removing a specified number of pixels from each edge
    (left, right, top, bottom). This is useful for eliminating edge artifacts, removing
    scanner borders, or cropping to a region of interest before colony detection and
    analysis.

    **Use cases for colony phenotyping:**

    - **Remove scanner margins:** Scanners often capture a border region outside the agar
      plate. Crop these margins to isolate the plate area containing colonies.
    - **Eliminate edge artifacts:** The edges of agar plates often have artifacts
      (bent agar, labeling, moisture) that interfere with detection. Cropping the outer
      wells/rows removes these problems.
    - **Focus on region of interest:** For large plates with sparse colonies, crop to the
      area where colonies are expected, improving detection efficiency and reducing noise
      from empty regions.
    - **Standardize image size:** For batch processing of plates with slightly different
      captured boundaries, cropping to consistent margins ensures uniform image dimensions.

    **Important caveat:** The cropping is applied to the entire image (rgb, gray, enh_gray,
    objmask, objmap) together, making this an ImageCorrector. If detection has already been
    performed, the detection results (objmask, objmap) are cropped along with the image
    data. Re-detection after cropping may be necessary if the crop affects detection quality.

    **Grid-aware cropping:** When applied to a GridImage, the cropper preserves the grid
    structure (nrows, ncols, grid_finder) while adapting grid positions to the cropped
    dimensions. Grid positions are automatically recalculated by the grid_finder to align
    with the new image boundaries.

    Attributes:
        left (int | None): Number of pixels to crop from the left edge. If None, no left
            cropping is performed (equivalent to 0).
        right (int | None): Number of pixels to crop from the right edge. If None, no right
            cropping is performed (equivalent to 0).
        top (int | None): Number of pixels to crop from the top edge. If None, no top
            cropping is performed (equivalent to 0).
        bottom (int | None): Number of pixels to crop from the bottom edge. If None, no
            bottom cropping is performed (equivalent to 0).

    Examples:
        .. dropdown:: Basic usage: crop scanner border from all edges

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.correction import ImageCropper

                # Load a scanned plate image (may include scanner margins)
                image = Image.imread('plate_with_border.tiff')
                print(f"Original size: {image.shape}")  # (1200, 1600, 3)

                # Remove 50 pixels from all edges
                cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
                cropped = cropper.apply(image)

                print(f"Cropped size: {cropped.shape}")  # (1100, 1500, 3)

        .. dropdown:: Asymmetric cropping: remove only top/right margins

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.correction import ImageCropper

                image = Image.imread('plate_image.jpg')

                # Remove top margin (label text) and right margin (edge artifact)
                # Keep left and bottom edges intact
                cropper = ImageCropper(top=80, right=60, left=None, bottom=None)
                cropped = cropper.apply(image)

        .. dropdown:: Crop after detection to isolate plate region

            .. code-block:: python

                from phenotypic import Image, ImagePipeline
                from phenotypic.enhance import GaussianBlur
                from phenotypic.detect import OtsuDetector
                from phenotypic.correction import ImageCropper

                # Load and process a plate image
                image = Image.imread('raw_plate.tiff')

                # Build preprocessing pipeline
                pipeline = ImagePipeline([
                    GaussianBlur(sigma=1.5),
                    OtsuDetector()
                ])

                # Apply processing
                detected = pipeline.operate(image)

                # Now crop to remove edge noise
                cropper = ImageCropper(left=40, right=40, top=40, bottom=40)
                final = cropper.apply(detected)

                # Detection results are preserved in cropped image
                colonies = final.objects
                print(f"Detected {len(colonies)} colonies in cropped region")
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

        Examples:
            .. dropdown:: Create a cropper for symmetric margins

                .. code-block:: python

                    from phenotypic.correction import ImageCropper

                    # Remove 50 pixels from all four edges
                    cropper = ImageCropper(left=50, right=50, top=50, bottom=50)

            .. dropdown:: Create a cropper for asymmetric margins

                .. code-block:: python

                    from phenotypic.correction import ImageCropper

                    # Remove top (label) and right (artifact), keep left and bottom
                    cropper = ImageCropper(top=100, right=75, left=None, bottom=None)

            .. dropdown:: Create a cropper that only removes top margin

                .. code-block:: python

                    from phenotypic.correction import ImageCropper

                    cropper = ImageCropper(top=80)
                    # Equivalent to ImageCropper(left=None, right=None, top=80, bottom=None)
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
        region. The entire image structure (rgb, gray, enh_gray, objmask, objmap) is
        cropped identically, ensuring all components remain synchronized.

        For GridImage instances, preserves the grid structure (grid_finder, nrows, ncols)
        while recalculating grid positions for the cropped region.

        Args:
            image (Image): The image to crop. The image is not modified; a new cropped
                Image (or GridImage if input was GridImage) is returned.

        Returns:
            Image: A new Image instance (or GridImage if input was GridImage) containing
                only the central cropped region, with all image components (rgb, gray,
                enh_gray, objmask, objmap) reduced to the same rectangular region.

        Raises:
            ValueError: If the crop parameters would result in an invalid slice (e.g.,
                top edge >= bottom edge or left edge >= right edge after cropping).

        Examples:
            .. dropdown:: Basic cropping of a loaded image

                .. code-block:: python

                    from phenotypic import Image
                    from phenotypic.correction import ImageCropper

                    image = Image.imread('plate.jpg')
                    cropper = ImageCropper(top=50, bottom=50, left=40, right=40)

                    # Returns new cropped Image; original is unchanged
                    cropped = cropper.apply(image)

                    print(f"Original shape: {image.shape}")
                    print(f"Cropped shape: {cropped.shape}")

            .. dropdown:: Cropping a GridImage preserves grid settings

                .. code-block:: python

                    from phenotypic import GridImage
                    from phenotypic.correction import ImageCropper

                    # Load plate image with scanner border
                    grid_img = GridImage('plate_with_border.tiff', nrows=8, ncols=12)

                    # Remove scanner border
                    cropper = ImageCropper(left=50, right=50, top=50, bottom=50)
                    cropped = cropper.apply(grid_img)

                    # GridImage type and settings preserved
                    assert isinstance(cropped, GridImage)
                    assert cropped.nrows == 8
                    assert cropped.ncols == 12
                    # Grid positions are recalculated for cropped image
                    cropped.show_overlay(show_gridlines=True)
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
            .. dropdown:: Understanding slice indices from crop parameters

                .. code-block:: python

                    from phenotypic import Image
                    from phenotypic.correction import ImageCropper

                    # Example: 1000x1200 image, crop 50 from all edges
                    image = Image.imread('large_plate.tiff')
                    cropper = ImageCropper(top=50, bottom=50, left=50, right=50)

                    # Internal calculation:
                    # height=1000, width=1200
                    # top_idx = 0 + 50 = 50
                    # bottom_idx = 1000 - 50 - 1 = 949
                    # left_idx = 0 + 50 = 50
                    # right_idx = 1200 - 50 - 1 = 1149
                    # Result: image[50:949, 50:1149] extracts central 899x1099 region

                    cropped = cropper.apply(image)
                    print(cropped.shape)  # (899, 1099, 3) for 3-channel image
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
