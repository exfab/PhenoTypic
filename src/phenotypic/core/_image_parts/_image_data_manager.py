from __future__ import annotations

import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, TYPE_CHECKING

import numpy as np
from scipy.sparse import csc_matrix
from skimage.color import rgb2gray, rgba2rgb

if TYPE_CHECKING:
    from phenotypic import Image

from phenotypic.tools.constants_ import IMAGE_FORMATS, METADATA, IMAGE_TYPES


@dataclass
class ImageData:
    """Container for core image data representations."""
    rgb: np.ndarray | None = None
    gray: np.ndarray | None = None
    enh_gray: np.ndarray | None = None
    sparse_object_map: csc_matrix = None

    def reset(self):
        self.rgb = np.empty((0, 3), dtype=np.uint8)
        self.gray = np.empty((0, 2), dtype=np.float32)
        self.enh_gray = np.empty((0, 2), dtype=np.float32)
        self.sparse_object_map = csc_matrix((0, 0), dtype=np.uint16)


class ImageDataManager:
    """Manages image data initialization, storage, and format handling.
    
    This class is responsible for:
    - Initializing core data structures (rgb, gray, enh_gray, object maps)
    - Managing image format detection and conversion
    - Handling metadata storage
    - Setting images from various input types (arrays, class instances)
    
    Attributes:
        _data (ImageData): Container for image data in different representations.
        _metadata (SimpleNamespace): Container for private, protected, and public metadata.
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

        # Initialize metadata structure
        self._metadata = SimpleNamespace(
                private={
                    METADATA.UUID: uuid.uuid4()
                },
                protected={
                    METADATA.IMAGE_NAME       : name,
                    METADATA.PARENT_IMAGE_NAME: b'',
                    METADATA.IMAGE_TYPE       : IMAGE_TYPES.BASE.value,
                    METADATA.BIT_DEPTH        : bit_depth,
                },
                public={},
        )

        # Set data to empty arrays first
        self._clear_data()

    @property
    def bit_depth(self) -> int:
        """Get the bit depth from metadata.
        
        Returns:
            int: The bit depth value stored in metadata.
        """
        return self._metadata.protected.get(METADATA.BIT_DEPTH)

    def _clear_data(self):
        """Reset all image data to empty state."""
        self._data.reset()

    def set_image(self, input_image: Image | np.ndarray | None = None) -> None:
        """Set the image data.
        
        Args:
            input_image (Image | np.ndarray | None): Image data to set.
        """
        match input_image:
            case x if isinstance(x, np.ndarray):
                self._handle_array_input(x)

            case x if self._is_image_handler(x):
                self._set_from_class_instance(x)
            case None:
                self._clear_data()
            case _:
                raise ValueError(
                        f'Input must be a NumPy array, Image instance, or None. Got {type(input_image)}'
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
            case IMAGE_FORMATS.GRAYSCALE | IMAGE_FORMATS.GRAYSCALE_SINGLE_CHANNEL:
                self._set_from_matrix(
                        imarr if imarr.ndim == 2 else imarr[:, :, 0]
                )

            case IMAGE_FORMATS.RGB | IMAGE_FORMATS.RGB_OR_BGR:
                self._set_from_rgb(imarr)

            case IMAGE_FORMATS.LINEAR_RGB:
                self._set_from_rgb(imarr)

            case IMAGE_FORMATS.RGBA | IMAGE_FORMATS.RGBA_OR_BGRA:
                self._set_from_rgb(rgba2rgb(imarr))

            case _:
                raise ValueError(f'Unsupported image format: {format_enum}')

    @staticmethod
    def _guess_image_format(img: np.ndarray) -> IMAGE_FORMATS:
        """Determine image format from array dimensions and channels.
        
        Args:
            img (np.ndarray): Input image array.
            
        Returns:
            IMAGE_FORMATS: Detected format of the image.
            
        Raises:
            TypeError: If input is not a numpy array.
            ValueError: If image has unsupported dimensions or channels.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        if img.ndim == 2:
            return IMAGE_FORMATS.GRAYSCALE

        if img.ndim == 3:
            h, w, c = img.shape
            if c == 1:
                return IMAGE_FORMATS.GRAYSCALE_SINGLE_CHANNEL
            elif c == 3:
                return IMAGE_FORMATS.RGB
            elif c == 4:
                return IMAGE_FORMATS.RGBA
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
