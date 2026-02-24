from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from scipy.sparse import csr_matrix
from phenotypic.abc_ import ImageCorrector


class ImagePadder(ImageCorrector):
    """Add pixels to image edges using configurable padding modes.

    ImagePadder extends images by adding pixels to any combination of edges (left, right,
    top, bottom). This is useful for creating consistent image dimensions, adding safety
    margins before rotation, or providing border space for edge-preserving operations.

    **Use cases for colony phenotyping:**

    - **Prevent clipping during rotation:** Add safety margin before rotating plate images
      to ensure no colonies are cropped out at corners.
    - **Standardize batch dimensions:** Pad images to consistent size for batch processing
      pipelines that require uniform dimensions.
    - **Create detection margin:** Add border space when colonies grow near plate edges,
      improving grid detection accuracy.
    - **Edge-preserving operations:** Pad with 'reflect' or 'edge' mode before convolution
      operations to reduce boundary artifacts.
    - **Align multi-timepoint images:** Pad images from different timepoints to align
      coordinate systems when plate position varies.

    **Important caveats:**

    - Padding increases memory usage proportional to added pixels.
    - Padded regions contain no real data (affects measurements near new edges).
    - Constant padding creates artificial background that may affect detection thresholds.
    - Reflect/edge modes can create false colony-like features if padding is large.
    - GridImage grid positions are recalculated after padding (old positions invalid).

    **Grid-aware padding:** When applied to GridImage, the padder preserves grid structure
    (nrows, ncols, grid_finder) while adapting grid positions to the padded dimensions.
    Grid positions are automatically recalculated by grid_finder to align with new boundaries.

    Attributes:
        left (int | None): Pixels to add on left edge. If None, no left padding (equiv to 0).
        right (int | None): Pixels to add on right edge. If None, no right padding (equiv to 0).
        top (int | None): Pixels to add on top edge. If None, no top padding (equiv to 0).
        bottom (int | None): Pixels to add on bottom edge. If None, no bottom padding
            (equiv to 0).
        mode (str): Padding mode for np.pad. Options: 'constant' (default), 'reflect', 'edge',
            'symmetric', 'wrap', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum',
            'empty'. Default: 'constant'.
        constant_value (int | float): Value for constant mode padding. Only used when
            mode='constant'. Default: 0 (black).

    **Parameter effects:**

    - left/right/top/bottom: Control padding amount. Larger values add more edge space but
      increase memory. Typical values: 50-200 pixels depending on resolution.
    - mode='constant': Creates uniform background (good for dark/light borders). Use
      constant_value=0 for black, 255 for white background.
    - mode='reflect': Mirrors image at boundaries (reduces artifacts but may create false
      colonies if padding is large). Best for edge-preserving convolutions.
    - mode='edge': Extends edge pixels (safest for colony analysis, minimal artifacts).
    - mode='symmetric': Like reflect but includes edge pixel in mirror. Similar artifact
      risk to reflect.

    Examples:
        Basic usage: add 50px black border to all edges:

        >>> from phenotypic import Image
        >>> from phenotypic.correction import ImagePadder
        >>> image = Image.imread('plate_image.tiff')  # doctest: +SKIP
        >>> print(f"Original size: {image.shape}")  # doctest: +SKIP
        >>> padder = ImagePadder(left=50, right=50, top=50, bottom=50)
        >>> padded = padder.apply(image)  # doctest: +SKIP
        >>> print(f"Padded size: {padded.shape}")  # doctest: +SKIP

        Pipeline: pad before rotation to prevent clipping:

        >>> from phenotypic import Image, ImagePipeline
        >>> from phenotypic.correction import ImagePadder
        >>> from phenotypic.detect import OtsuDetector
        >>> image = Image.imread('rotated_plate.jpg')  # doctest: +SKIP
        >>> # Add 100px safety margin with reflection to avoid artifacts
        >>> pipeline = ImagePipeline([
        ...     ImagePadder(left=100, right=100, top=100, bottom=100, mode='reflect'),
        ...     OtsuDetector()
        ... ])
        >>> result = pipeline.operate(image)  # doctest: +SKIP
        >>> print(f"Detected {len(result.objects)} colonies")  # doctest: +SKIP
    """

    def __init__(self,
                 left: int | None = None,
                 right: int | None = None,
                 top: int | None = None,
                 bottom: int | None = None,
                 mode: str = "constant",
                 constant_value: int | float = 0
                 ):
        """Initialize an ImagePadder with pixel margins to add on each edge.

        Creates a padder that adds the specified number of pixels to each edge of the image.
        All margin parameters are optional and default to None (no padding from that edge).

        Args:
            left (int | None, optional): Pixels to add on left edge. Must be non-negative.
                If None, no left padding (equivalent to 0). Defaults to None.
            right (int | None, optional): Pixels to add on right edge. Must be non-negative.
                If None, no right padding (equivalent to 0). Defaults to None.
            top (int | None, optional): Pixels to add on top edge. Must be non-negative.
                If None, no top padding (equivalent to 0). Defaults to None.
            bottom (int | None, optional): Pixels to add on bottom edge. Must be non-negative.
                If None, no bottom padding (equivalent to 0). Defaults to None.
            mode (str, optional): Padding mode for np.pad. Options include 'constant'
                (uniform value), 'reflect' (mirror at boundary), 'edge' (replicate edge pixels),
                'symmetric' (symmetric reflection), 'wrap' (periodic), and others. Defaults
                to 'constant'.
            constant_value (int | float, optional): Value for constant mode padding. Only
                used when mode='constant'. Typical values: 0 for black (default), 255 for white.
                Defaults to 0.

        Raises:
            ValueError: If any padding parameter is negative. All padding margins must be
                non-negative integers (or None).
            ValueError: If mode is not a valid np.pad mode.

        Examples:
            Create a padder for symmetric margins:

            >>> from phenotypic.correction import ImagePadder
            >>> # Add 50 pixels to all four edges
            >>> padder = ImagePadder(left=50, right=50, top=50, bottom=50)

            Create a padder for asymmetric margins:

            >>> from phenotypic.correction import ImagePadder
            >>> # Add padding on top and right, keep left and bottom minimal
            >>> padder = ImagePadder(top=100, right=75, left=0, bottom=0)

            Create a padder with reflection to avoid artifacts:

            >>> from phenotypic.correction import ImagePadder
            >>> padder = ImagePadder(
            ...     left=80, right=80, top=80, bottom=80,
            ...     mode='reflect'
            ... )
            >>> # Reflection preserves edge patterns, good for convolutions
        """
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.mode = mode
        self.constant_value = constant_value
        self.__prescreen_params()

    def __prescreen_params(self):
        """Validate padding parameters before use.

        Raises:
            ValueError: If any padding parameter is negative.
            ValueError: If mode is not a valid np.pad mode.
        """
        if (self.left is not None) and (self.left < 0):
            raise ValueError("left cannot be negative")

        if (self.right is not None) and (self.right < 0):
            raise ValueError("right cannot be negative")

        if (self.top is not None) and (self.top < 0):
            raise ValueError("top cannot be negative")

        if (self.bottom is not None) and (self.bottom < 0):
            raise ValueError("bottom cannot be negative")

        valid_modes = [
            'constant', 'edge', 'reflect', 'symmetric', 'wrap',
            'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'empty'
        ]
        if self.mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes}, got '{self.mode}'"
            )

    def _get_pad_width_2d(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate pad_width tuple for 2D arrays (gray, detect_mat, objmap).

        Converts None → 0 for each parameter and returns the pad_width format
        expected by np.pad for 2D arrays.

        Returns:
            Tuple of ((top, bottom), (left, right)) for np.pad.

        Examples:
            >>> padder = ImagePadder(left=10, right=20, top=30, bottom=40)
            >>> padder._get_pad_width_2d()
            ((30, 40), (10, 20))
        """
        top = 0 if self.top is None else self.top
        bottom = 0 if self.bottom is None else self.bottom
        left = 0 if self.left is None else self.left
        right = 0 if self.right is None else self.right

        return ((top, bottom), (left, right))

    def _get_pad_width_3d(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """Calculate pad_width tuple for 3D arrays (RGB).

        Returns pad_width for RGB arrays where spatial dimensions (height, width) are
        padded but channel dimension is NOT padded. This preserves the number of channels.

        Returns:
            Tuple of ((top, bottom), (left, right), (0, 0)) for np.pad.
            The third dimension (channels) has (0, 0) padding.

        Examples:
            >>> padder = ImagePadder(left=10, right=20, top=30, bottom=40)
            >>> padder._get_pad_width_3d()
            ((30, 40), (10, 20), (0, 0))
        """
        top = 0 if self.top is None else self.top
        bottom = 0 if self.bottom is None else self.bottom
        left = 0 if self.left is None else self.left
        right = 0 if self.right is None else self.right

        return ((top, bottom), (left, right), (0, 0))

    def _operate(self, image: Image) -> Image:
        """Pad the image by adding pixels to edges specified in __init__.

        Pads all image components (rgb, gray, detect_mat, objmask, objmap) together,
        maintaining synchronization. Object map is ALWAYS padded with constant mode
        and value 0 to preserve integer label integrity, regardless of user mode.

        For RGB arrays, padding is applied ONLY to spatial dimensions (height, width),
        not to the channel dimension. This preserves the number of channels.

        Args:
            image (Image): The image to pad. The image is modified with padding applied.

        Returns:
            Image: A new Image instance (or GridImage if input was GridImage) with all
                components padded by the specified amounts. Original image is unchanged
                unless inplace=True is used with apply().

        Raises:
            ValueError: If mode is not supported by np.pad.

        Examples:
            Basic padding of a loaded image:

            >>> from phenotypic import Image
            >>> from phenotypic.correction import ImagePadder
            >>> image = Image.imread('plate.jpg')  # doctest: +SKIP
            >>> padder = ImagePadder(left=50, right=50, top=50, bottom=50)
            >>> # Returns new padded Image; original is unchanged
            >>> padded = padder.apply(image)  # doctest: +SKIP
            >>> print(f"Original shape: {image.shape}")  # doctest: +SKIP
            >>> print(f"Padded shape: {padded.shape}")  # doctest: +SKIP

            Padding a GridImage preserves grid settings:

            >>> from phenotypic import GridImage
            >>> from phenotypic.correction import ImagePadder
            >>> # Load plate image
            >>> grid_img = GridImage('plate.tiff', nrows=8, ncols=12)  # doctest: +SKIP
            >>> # Add safety margin
            >>> padder = ImagePadder(left=50, right=50, top=50, bottom=50)
            >>> padded = padder.apply(grid_img)  # doctest: +SKIP
            >>> # GridImage type and settings preserved
            >>> assert isinstance(padded, GridImage)  # doctest: +SKIP
            >>> assert padded.nrows == 8  # doctest: +SKIP
            >>> assert padded.ncols == 12  # doctest: +SKIP
        """
        # Get padding widths
        pad_width_2d = self._get_pad_width_2d()
        pad_width_3d = self._get_pad_width_3d()

        # Prepare kwargs for np.pad
        pad_kwargs: dict[str, Any] = {}
        if self.mode == 'constant':
            pad_kwargs['constant_values'] = self.constant_value

        # Pad RGB if it exists (3D array, spatial dims only)
        if not image.rgb.isempty():
            image._data.rgb = np.pad(  # type: ignore
                image._data.rgb,
                pad_width=pad_width_3d,
                mode=self.mode,
                **pad_kwargs
            )

        # Pad gray (2D array)
        image._data.gray = np.pad(  # type: ignore
            image._data.gray,
            pad_width=pad_width_2d,
            mode=self.mode,
            **pad_kwargs
        )

        # Pad detect_mat (2D array)
        image._data.detect_mat = np.pad(  # type: ignore
            image._data.detect_mat,
            pad_width=pad_width_2d,
            mode=self.mode,
            **pad_kwargs
        )

        # CRITICAL: Pad objmap with constant mode and value 0 ALWAYS
        # This preserves integer object labels regardless of user mode.
        # Must modify _data directly to handle shape change.
        padded_objmap = np.pad(
            image._data.sparse_object_map.toarray(),
            pad_width=pad_width_2d,
            mode='constant',
            constant_values=0
        )
        # Convert back to sparse and update
        image._data.sparse_object_map = csr_matrix(padded_objmap)

        # Handle GridImage type preservation
        from phenotypic import GridImage

        original_name = image.name

        if isinstance(image, GridImage):
            # Grid positions will be recalculated by grid_finder automatically
            padded = GridImage(
                arr=image,
                name=image.name,
                grid_finder=image.grid_finder,
                nrows=image.nrows,
                ncols=image.ncols,
                bit_depth=image.bit_depth,
                illuminant=image.illuminant,
                gamma=image.gamma,
            )
            image = padded

        # Restore original name
        image.name = original_name

        return image
