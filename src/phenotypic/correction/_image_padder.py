from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Tuple

from pydantic import ValidationInfo, field_validator

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from scipy.sparse import csr_matrix
from phenotypic.abc_ import ImageCorrector

#: The eleven fill strategies accepted by ``np.pad``. Declared as a
#: ``Literal`` so the pre-migration ``__prescreen_params`` mode-membership
#: check becomes a type-level constraint that also enumerates the allowed
#: values in ``model_json_schema()``.
PadMode = Literal[
    "constant",
    "edge",
    "reflect",
    "symmetric",
    "wrap",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "empty",
]


class ImagePadder(ImageCorrector):
    """Extend image dimensions by adding pixels to any combination of edges.

    Pad the image on the left, right, top, and/or bottom using a
    configurable fill mode. All image components (RGB, gray, detect_mat,
    objmap) are padded in sync; the object map is always zero-padded to
    preserve label integrity. When applied to a GridImage, grid structure
    is preserved and positions are recalculated automatically.

    For usage context, see :doc:`/how_to/notebooks/crop_and_pad`.

    Args:
        left: Pixels to add on the left edge. ``None`` means no padding.
            Typical range: 50--200. Default: ``None``.
        right: Pixels to add on the right edge. ``None`` means no padding.
            Typical range: 50--200. Default: ``None``.
        top: Pixels to add on the top edge. ``None`` means no padding.
            Typical range: 50--200. Default: ``None``.
        bottom: Pixels to add on the bottom edge. ``None`` means no
            padding. Typical range: 50--200. Default: ``None``.
        mode: Fill strategy passed to ``np.pad``. Accepted values:
            ``'constant'``, ``'reflect'``, ``'edge'``, ``'symmetric'``,
            ``'wrap'``, ``'linear_ramp'``, ``'maximum'``, ``'mean'``,
            ``'median'``, ``'minimum'``, ``'empty'``. ``'edge'`` is safest
            for colony analysis; ``'reflect'`` reduces convolution boundary
            artifacts. Default: ``'constant'``.
        constant_value: Fill value when ``mode='constant'``. ``0`` for
            black borders, ``255`` for white. Default: ``0``.

    Returns:
        Image: Input image with all components padded by the specified
        amounts. GridImage grid positions are recalculated.

    Raises:
        ValueError: If any padding value is negative.
        ValueError: If ``mode`` is not a valid ``np.pad`` mode.

    Best For:
        - Adding safety margins before rotation so corner colonies are
          not clipped.
        - Standardizing image dimensions across a batch for pipelines
          that require uniform size.
        - Creating border space when colonies grow near plate edges,
          improving grid detection accuracy.

    Consider Also:
        - :class:`ImageCropper` when the image needs to be reduced rather
          than extended.
        - :class:`GridAligner` for correcting plate rotation after
          padding.

    See Also:
        :doc:`/how_to/notebooks/crop_and_pad` for a visual walkthrough
        of padding and cropping plate images.
        :doc:`/how_to/notebooks/correct_grid_rotation` for combining
        padding with rotation correction.
    """

    left: int | None = None
    right: int | None = None
    top: int | None = None
    bottom: int | None = None
    mode: PadMode = "constant"
    constant_value: int | float = 0

    @field_validator("left", "right", "top", "bottom")
    @classmethod
    def _reject_negative_margin(
            cls, value: int | None, info: ValidationInfo
    ) -> int | None:
        """Reject a negative padding margin, preserving the legacy message.

        Reproduces the pre-migration ``__prescreen_params`` guard: ``None``
        is accepted (no padding on that edge), any negative value raises
        ``"<edge> cannot be negative"``.
        """
        if value is not None and value < 0:
            raise ValueError(f"{info.field_name} cannot be negative")
        return value

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
