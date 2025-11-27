from __future__ import annotations

import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal, TYPE_CHECKING, Union

import numpy as np
from scipy.sparse import csc_matrix
from skimage.color import rgb2gray, rgba2rgb

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic.tools.constants_ import IMAGE_MODE, METADATA, IMAGE_TYPES


@dataclass
class ImageData:
    """Container for core image data representations."""
    rgb: np.ndarray | None = None
    gray: np.ndarray | None = None
    enh_gray: np.ndarray | None = None
    sparse_object_map: csc_matrix = None

    def clear(self):
        self.rgb = np.empty((0, 3), dtype=np.uint8)
        self.gray = np.empty((0, 2), dtype=np.float32)
        self.enh_gray = np.empty((0, 2), dtype=np.float32)
        self.sparse_object_map = csc_matrix((0, 0), dtype=np.uint16)


@dataclass
class ImageMetadata:
    """
    Represents metadata associated with an image.

    This class is used to store and organize metadata related to an image. It
    categorizes metadata into private, protected, public, and imported categories
    for better organization and control over data visibility or accessibility.

    Attributes:
        private (dict[str, Any]): Metadata intended for internal use only and is
            not shared externally. It may contain any type of data.
        protected (dict[str, Union[int, str, float, bool]]): Metadata that is
            partially restricted and can be accessed but not deleted.
            Limited to primitive data types like integers, strings, floats, and
            booleans.
        public (dict[str, Union[int, str, float, bool]]): Metadata that is available
            for public access. Structured as key-value pairs containing integers,
            strings, floats, or booleans.
        imported (dict[str, Union[int, str, float, bool]]): Metadata from the original image if imported from a file
            instead of from an array.
    """
    private: dict[str, Any] = field(default_factory=dict)
    protected: dict[str, Union[int, str, float, bool, np.nan]] = field(default_factory=dict)
    public: dict[str, Union[int, str, float, bool, np.nan]] = field(default_factory=dict)
    imported: dict[str, Union[int, str, float, bool, np.nan]] = field(default_factory=dict)

    def clear(self) -> None:
        self.protected[METADATA.IMAGE_NAME] = np.nan
        self.protected[METADATA.IMAGE_TYPE] = IMAGE_TYPES.BASE.value
        self.public.clear()


class ImageDataManager:
    """Manages image data initialization, storage, and format handling.

    This class is the foundational layer for image data management, responsible for:
    - Initializing and managing core data structures (RGB, grayscale, enhanced grayscale, object maps)
    - Detecting and converting between different image formats (grayscale, RGB, RGBA)
    - Handling metadata storage with private, protected, public, and imported categories
    - Setting images from various input types (NumPy arrays, Image class instances)
    - Managing bit depth inference and conversion

    The class provides a separation of concerns by focusing purely on data management,
    leaving presentation and format-specific operations to subclasses.

    Attributes:
        _data (ImageData): Container for image arrays in different representations
            (rgb, gray, enh_gray, and sparse object map).
        _metadata (ImageMetadata): Container for categorized metadata (private, protected,
            public, and imported).
    """

    _ARRAY8_DTYPE = np.uint8
    _ARRAY16_DTYPE = np.uint16
    _OBJMAP_DTYPE = np.uint16

    def __init__(self,
                 name: str | None = None,
                 bit_depth: Literal[8, 16] | None = None):
        """
        Initializes a class instance to manage the image data and metadata.

        Args:
            name (str | None): The name of the image. If None, the name is unspecified.
            bit_depth (Literal[8, 16] | None): The bit depth of the image, can be either 8 or 16.
                If None, the bit depth is unspecified.

        Raises:
            ValueError: If any invalid values for `bit_depth` are assigned during initialization.
        """

        # Initialize image data
        self._data = ImageData()
        self._data.clear()

        # Initialize metadata structure
        self._metadata = ImageMetadata(
                private={
                    METADATA.UUID: uuid.uuid4()
                },
                protected={
                    METADATA.IMAGE_NAME: name,
                    METADATA.IMAGE_TYPE: IMAGE_TYPES.BASE.value,
                    METADATA.BIT_DEPTH : bit_depth,
                },
                public={},
                imported={},
        )

    @property
    def bit_depth(self) -> int:
        """Get the bit depth of the image.

        The bit depth determines the number of bits used to represent each pixel
        value. Common values are 8 (0-255 range) and 16 (0-65535 range).

        Returns:
            int | None: The bit depth value (8 or 16) stored in protected metadata,
                or None if not yet set.
        """
        return self._metadata.protected.get(METADATA.BIT_DEPTH)

    def clear(self) -> None:
        """Reset all image data to empty state.

        Note:
            - bit_depth is retained. To change the bit depth, make a new Image object.
        """
        self._data.clear()
        self._metadata.clear()
        return

    def set_image(self, input_image: Image | np.ndarray) -> None:
        """
        Sets the image for the object by processing the provided input, which can be either
        a NumPy array or an instance of the Image class. If the input type is unsupported,
        an exception is raised to notify the user.

        Args:
            input_image: A NumPy array or an instance of the Image class representing
                the image to be set.

        Raises:
            ValueError: If the input is not a NumPy array or an Image instance.
        """
        match input_image:
            case x if isinstance(x, np.ndarray):
                self._handle_array_input(x)

            case x if self._is_image_handler(x):
                self._set_from_class_instance(x)
            case _:
                raise ValueError(
                        f'Input must be a NumPy array, Image instance. Got {type(input_image)}'
                )

    def _handle_array_input(self, arr: np.ndarray):
        """Handle array input and set bit depth if needed."""
        if self.bit_depth is None:
            bit_depth = self._infer_bit_depth(arr)
            self._metadata.protected[METADATA.BIT_DEPTH] = bit_depth

        if np.issubdtype(arr.dtype, np.floating) and arr.ndim == 3:
            arr = self._convert_float_array_to_int(arr, bit_depth=self.bit_depth)
        self._set_from_array(arr)

    @staticmethod
    def _infer_bit_depth(arr: np.ndarray) -> int:
        """Infer bit depth from array dtype.
        
        Args:
            arr (np.ndarray): Input array.
            
        Returns:
            int: Inferred bit depth (8 or 16).
        """
        match arr.dtype:
            case np.uint8:
                return 8
            case np.uint16:
                return 16
            case y if np.issubdtype(y, np.floating):
                return 16
            case _:
                warnings.warn(
                        'Input image has unknown dtype, bit_depth could not be guessed. '
                        'Defaulting to 16'
                )
                return 16

    @staticmethod
    def _is_image_handler(obj) -> bool:
        """Check if object is an ImageHandler instance."""
        try:
            from phenotypic.core._image_parts._image_handler import ImageHandler

            return isinstance(obj, ImageHandler) or issubclass(type(obj), ImageHandler)
        except ImportError:
            return False

    def _set_from_class_instance(self, input_cls):
        """Copy data from another Image instance.
        
        Args:
            input_cls: Source Image instance to copy from.
        """
        if not self._is_image_handler(input_cls):
            raise ValueError('Input is not an Image object')

        # Determine format from whether RGB data exists
        if not input_cls.rgb.isempty():
            self._set_from_array(input_cls.rgb[:])
        else:
            self._set_from_array(input_cls.gray[:])

        # Deep copy all data attributes
        for key, value in input_cls._data.__dict__.items():
            self._data.__dict__[key] = value.copy() if value is not None else None

        self._metadata.protected = deepcopy(input_cls._metadata.protected)
        self._metadata.public = deepcopy(input_cls._metadata.public)

    def _set_from_matrix(self, matrix: np.ndarray):
        """Initialize 2-D image components from a matrix.
        
        Args:
            matrix (np.ndarray): A 2-D array form of an image.
        """
        self._data.gray = matrix
        self._data.enh_gray = matrix.copy()
        self._data.sparse_object_map = csc_matrix(
                np.zeros(matrix.shape, dtype=self._OBJMAP_DTYPE)
        )

    def _set_from_rgb(self, rgb_array: np.ndarray):
        """Initialize all components from an RGB array.
        
        Args:
            rgb_array (np.ndarray): RGB image array.
        """
        self._data.rgb = rgb_array.copy()
        self._set_from_matrix(rgb2gray(rgb_array))

    def _set_from_array(self, imarr: np.ndarray) -> None:
        """Initialize all components from an array.

        Args:
            imarr (np.ndarray): Input image array.
        """
        # Guess format from array shape
        format_enum = self._guess_image_format(imarr)

        # Process based on detected format
        match format_enum:
            case IMAGE_MODE.GRAYSCALE | IMAGE_MODE.GRAYSCALE_SINGLE_CHANNEL:
                self._set_from_matrix(
                        imarr if imarr.ndim == 2 else imarr[:, :, 0]
                )

            case IMAGE_MODE.RGB | IMAGE_MODE.RGB_OR_BGR:
                self._set_from_rgb(imarr)

            case IMAGE_MODE.LINEAR_RGB:
                self._set_from_rgb(imarr)

            case IMAGE_MODE.RGBA | IMAGE_MODE.RGBA_OR_BGRA:
                self._set_from_rgb(rgba2rgb(imarr))

            case _:
                raise ValueError(f'Unsupported image format: {format_enum}')

    @staticmethod
    def _guess_image_format(img: np.ndarray) -> IMAGE_MODE:
        """Determine image format from array dimensions and channels.
        
        Args:
            img (np.ndarray): Input image array.
            
        Returns:
            IMAGE_MODE: Detected format of the image.
            
        Raises:
            TypeError: If input is not a numpy array.
            ValueError: If image has unsupported dimensions or channels.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        if img.ndim == 2:
            return IMAGE_MODE.GRAYSCALE

        if img.ndim == 3:
            h, w, c = img.shape
            if c == 1:
                return IMAGE_MODE.GRAYSCALE_SINGLE_CHANNEL
            elif c == 3:
                return IMAGE_MODE.RGB
            elif c == 4:
                return IMAGE_MODE.RGBA
            else:
                raise ValueError(f"Image with {c} channels (unknown format)")

        raise ValueError("Unknown format (unsupported number of dimensions)")

    @staticmethod
    def _convert_float_array_to_int(float_array: np.ndarray,
                                    bit_depth: Literal[8, 16]) -> np.ndarray:
        """Convert normalized float array to integer array.
        
        Args:
            float_array (np.ndarray): Array with float values in range [0, 1].
            bit_depth (Literal[8, 16]): Target bit depth (8 or 16).
            
        Returns:
            np.ndarray: Converted integer array (uint8 or uint16).
            
        Raises:
            ValueError: If bit_depth is not 8 or 16, or if array values 
                       are outside [0, 1] range.
        """
        if bit_depth not in (8, 16):
            raise ValueError(f"bit_depth must be 8 or 16, got {bit_depth}")

        if np.any(float_array < 0) or np.any(float_array > 1):
            raise ValueError(
                    f"Float array contains values outside [0, 1] range. "
                    f"Min: {float_array.min()}, Max: {float_array.max()}"
            )

        if bit_depth == 8:
            target_dtype = np.uint8
            max_value = 255
        else:
            target_dtype = np.uint16
            max_value = 65535

        return (float_array*max_value).astype(target_dtype)
