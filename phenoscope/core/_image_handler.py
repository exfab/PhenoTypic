from typing import Optional, Union
import warnings
from pathlib import Path
import numpy as np
import skimage as ski
from skimage.color import rgb2gray, rgba2rgb, hsv2rgb
import uuid

from ._image_components import (
    ImageArray,
    ImageMatrix,
    ImageDetectionMatrix,
    ObjectMask,
    ObjectMap
)

from ..util.constants import C_ImageHandler, C_ImageFormats

ACCEPTED_FORMATS = ['RGB', 'RGBA', 'Grayscale', 'BGR', 'BGRA', 'HSV']


class ImageHandler:
    """An ImageHandler class to sync and automate different operations between image processing components

    The ImageHandler class is responsible for automating syncing of different image processing components during processing. This can help
    maintain data integrity through complex pipelines. It employs a tree hierarchy to help manage components.

    Protected Attributes:


    """

    def __init__(self, input_image: Optional[np.ndarray] = None, input_format=None):  # TODO
        """Initializes the ImageHandler class

        Note:
            If the input_image is 2-D, the ImageHandler leave the array form as None
            If the input_image is 3-D, the ImageHandler will automatically set the matrix component to the grayscale representation.

        Args:
            input_image: (optional, np.ndarray) Can be a 2-D, 3-D, or 4-D numpy array representing an image
        """

        # Assign Image UUID (Used for HDF5 IO save integrity)
        self.__uuid = uuid.uuid4()

        # Initialize core image attributes
        self.__array = self.__matrix = self.__det_matrix = self.__object_mask = self.__object_map = None

        # Initialize input tracking
        self.__input_format = None

        # If input was a numpy array
        if isinstance(input_image, np.ndarray):
            self._set_from_array(input_image, input_format=input_format)

    @property
    def shape(self):
        """Returns the shape of the image array or matrix depending on input format or none if no image is set.

        Returns:
            Optional[Tuple(int,int,...)]: Returns the shape of the array or matrix depending on input format or none if no image is set.
        """
        if self.__array is not None:
            return self.__array.shape
        elif self.__matrix is not None:
            return self.__matrix.shape
        else:
            return None

    @property
    def input_format(self):
        """Returns the input format of the image array or matrix depending on input format"""
        return self.__input_format

    def isempty(self):
        """Returns True if the image array is empty"""
        return True if self.__array is None and self.__matrix is None else False

    @property
    def array(self) -> Optional[np.ndarray]:
        """The image's array representation. An image array can represent multi-channels, thus the array can be 3-D or 4-D

        Note:
            - array elements are not directly mutable in order to preserve image information integrity
            - change array elements by changing the image being represented with Image.set_image()
            - Raises an error if input image has no array form

        Returns:
            Optional[ImageArray]: A class that can be accessed like a numpy array, but has extra methods to streamline development, or None if not set
        """
        if self.__array is None:
            if self.__matrix is None:
                raise C_ImageHandler.EmptyImageError()
            else:
                raise C_ImageHandler.NoArrayError()
        else:
            return self.__array

    @array.setter
    def array(self, array):
        raise C_ImageHandler.IllegalAssignmentError('array')

    @property
    def matrix(self):
        """The image's matrix representation. The array form is converted into a matrix form since some algorithm's only handle 2-D

        Note:
            - matrix elements are not directly mutable in order to preserve image information integrity
            - Change matrix elements by changing the image being represented with Image.set_image()

        Returns:
            ImageMatrix: An immutable container for the image matrix that can be accessed like a numpy array, but has extra methods to streamline development.
        """
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix):
        raise C_ImageHandler.IllegalAssignmentError('matrix')

    @property
    def det_matrix(self):
        """A mutable copy of the image's matrix representation. Preprocessing steps can be applied to this component to improve detection performance.

        Returns:
            ImageDetectionMatrix: A mutable container that stores a copy of the image's matrix form
        """
        return self.__det_matrix

    @det_matrix.setter
    def det_matrix(self, det_matrix):
        raise C_ImageHandler.IllegalAssignmentError('det_matrix')

    @property
    def object_mask(self):
        """A mutable binary representation of the objects in an image to be analyzed. Changing elements of the mask will reset object_map labeling.

        Note:
            - If the image has not been processed by a detector, the target for analysis is the entire image itself. Accessing the object_mask in this case
                will return a 2-D array entirely with value 1 that is the same shape as the matrix
            - Changing elements of the mask will relabel of objects in the object_map (A workaround to this issue may or may not come in future versions)

        Returns:
            ObjectMaskErrors: A mutable binary representation of the objects in an image to be analyzed.
        """
        return self.__object_mask

    @object_mask.setter
    def object_mask(self, object_mask):
        raise C_ImageHandler.IllegalAssignmentError('object_mask')

    @property
    def object_map(self):
        """A mutable integer matrix that identifies the different objects in an image to be analyzed. Changes to elements of the object_map sync to the object_mask.

        The object_map is stored as a compressed sparse column matrix in the backend. This is to save on memory consumption at the cost of adding
        increased computational overhead between converting between sparse and dense matrices.

        Note:
              - Has accessor methods to get sparse representations of the object map that can streamline measurement calculations.

        Returns:
            ObjectMap: A mutable integer matrix that identifies the different objects in an image to be analyzed.
        """
        return self.__object_map

    @object_map.setter
    def object_map(self, object_map):
        raise C_ImageHandler.IllegalAssignmentError('object_map')

    def imread(self, filepath):
        """Imports an image from a filepath and initilizes class components

        Args:
            filepath (str): Path to the image file
        """
        # Convert to Path object
        filepath = Path(filepath)

        self.set_image(
            ski.io.imread(filepath),
        )

    def copy(self):
        """Creates a copy of the current Image instance, excluding the UUID.
        Note:
            - The new instance is only informationally a copy. The UUID of the new instance is different.

        Returns:
            Image: A copy of the current Image instance.
        """
        # Create a new instance of ImageHandler
        if self.isempty():
            return self.__class__()
        else:
            if self.__array is not None:
                new_instance = self.__class__(self.array[:], self.input_format)
            elif self.__matrix is not None:
                new_instance = self.__class__(self.__matrix[:], self.input_format)
            else:
                raise C_ImageHandler.UnknownError

            # Copy remaining information
            new_instance.det_matrix[:] = self.det_matrix[:]
            new_instance.object_map[:] = self.object_map[:]

            return new_instance

    def set_image(self, input_image, input_format: str = None) -> None:
        """Sets the image to the inputted array

        Args:
            input_image: (np.ndarray, phenoscope.Image) The image data to be set
            input_format: (str, optional) If input image is a np.ndarray and format is None, the image format will be guessed.
                If the image format is ambiguous between RGB/BGR it will assume RGB unless otherwise specified.
                Accepted formats are ['RGB', 'RGBA','Grayscale','BGR','BGRA','HSV']
        """
        if isinstance(input_image, np.ndarray):
            self._set_from_array(input_image, input_format)
        elif isinstance(input_image, self.__class__):
            self._set_from_class_instance(input_image)

    def _set_from_class_instance(self, class_instance):
        if class_instance.array is not None:
            self._set_from_array(class_instance.array[:], class_instance.input_format)
        else:
            self._set_from_array(class_instance.matrix[:], class_instance.input_format)

        self.object_map[:] = class_instance.object_map[:]

    def _set_from_array(self, image: np.ndarray, input_format: str) -> None:
        """Initializes all the components of an image from an array

        Note:
            The format of the input should already have been set or guessed
        Args:
            image: the input image array
            input_format: (str, optional) The format of the input image
        """
        if input_format is not None:
            # Valid Format Check
            if input_format not in C_ImageFormats.SUPPORTED_FORMATS:
                raise C_ImageFormats.UnsupportedFormatError(input_format)
            else:
                self.__input_format = input_format
        else:
            self.__input_format = self._guess_image_format(image)

        # Grayscale Formats
        if (self.__input_format == C_ImageFormats.GRAYSCALE
                or self.__input_format == C_ImageFormats.GRAYSCALE_SINGLE_CHANNEL
        ): self._from_grayscale(image)

        # RGB Formats
        if (self.__input_format == C_ImageFormats.RGB
                or self.__input_format == C_ImageFormats.RGBA
        ): self._from_rgb_or_rgba(image)

        # BGR Formats
        if (self.__input_format == C_ImageFormats.BGR
                or self.__input_format == C_ImageFormats.BGRA
        ): self._from_bgr_or_bgra(image)

        # HSV Formats
        if self.__input_format == C_ImageFormats.HSV:
            self._from_hsv(image)

    def _from_grayscale(self, grayscale_array: np.ndarray):
        """Initialize class components from a grayscale image array"""
        if self.__input_format == C_ImageFormats.GRAYSCALE:
            self._initialize_matrix_and_objects(grayscale_array)

        elif self.__input_format == C_ImageFormats.GRAYSCALE_SINGLE_CHANNEL:
            self._initialize_matrix_and_objects(grayscale_array[:, :, 0])

    def _from_rgb_or_rgba(self, rgb_or_rgba_array: np.ndarray):
        """Initialize class components from an RGB or RGBA image array"""
        if self.__input_format == C_ImageFormats.RGB:
            self.__array = ImageArray(self, rgb_or_rgba_array)
            self._initialize_matrix_and_objects(
                matrix=rgb2gray(rgb_or_rgba_array)
            )
        elif self.__input_format == C_ImageFormats.RGBA:
            self.__array = ImageArray(self, rgb_or_rgba_array)
            self._initialize_matrix_and_objects(
                matrix=rgb2gray(
                    rgba2rgb(rgb_or_rgba_array)
                )
            )

    def _from_bgr_or_bgra(self, bgr_or_bgra_array: np.ndarray):
        """Initialize class components from a BGR or BGRA image array"""
        if self.__input_format == C_ImageFormats.BGR:
            self.__array = ImageArray(handler=self, image_array=bgr_or_bgra_array)
            self._initialize_matrix_and_objects(
                matrix=rgb2gray(
                    self._convert_bgr_to_rgb(bgr_or_bgra_array)
                )
            )
        elif self.__input_format == C_ImageFormats.BGRA:
            self.__array = ImageArray(handler=self, image_array=bgr_or_bgra_array)
            self._initialize_matrix_and_objects(
                matrix=rgb2gray(
                    rgba2rgb(
                        self._convert_bgra_to_rgba(bgr_or_bgra_array)
                    )
                )
            )

    def _from_hsv(self, hsv_array: np.ndarray):
        """Initialize class components from an HSV image array"""
        if self.__input_format == C_ImageFormats.HSV:
            self.__array = ImageArray(handler=self, image_array=hsv_array)
            self._initialize_matrix_and_objects(
                matrix=rgb2gray(
                    hsv2rgb(
                        hsv_array
                    )
                )
            )

    def _initialize_matrix_and_objects(self, matrix):
        """Initializes all the 2-D components of an image

        Args:
            matrix: A 2-D array form of an image
        """
        self.__matrix = ImageMatrix(self, matrix)
        self.__det_matrix = ImageDetectionMatrix(self)
        self.__object_mask = ObjectMask(self)
        self.__object_map = ObjectMap(self)

    @staticmethod
    def _convert_bgr_to_rgb(bgr_image: np.ndarray) -> np.ndarray:
        """Rearranges the channels of a BGR array to an RGB array"""
        return bgr_image[:, :, ::-1]

    @staticmethod
    def _convert_bgra_to_rgba(bgra_image: np.ndarray) -> np.ndarray:
        """Rearranges the channels of a BGRA array to an RGBA array"""
        return bgra_image[:, :, [2, 1, 0, 3]]

    @staticmethod
    def _guess_image_format(img: np.ndarray) -> str:
        """
        Attempts to determine the color format of an image represented as a NumPy array.

        The function examines:
          - The number of dimensions (ndim) and channels (shape)
          - Basic statistics (min, max) for each channel in a 3-channel image

        Returns a string that describes the likely image format.

        Args:
            img (np.ndarray): The input image as a NumPy array.

        Returns:
            str: A string describing the guessed image format.

        Notes:
            - A 2D array is assumed to be a grayscale image.
            - A 3D array with one channel (shape: (H, W, 1)) is also treated as grayscale.
            - A 3-channel image (shape: (H, W, 3)) may be RGB, BGR, or even HSV.
              The heuristic here checks if the first channel’s maximum is within the typical
              range for Hue in an HSV image (0–179 for OpenCV) while one of the other channels
              exceeds that range.
            - A 4-channel image is assumed to be some variant of RGBA (or BGRA), but the ordering
              remains ambiguous without further metadata.
            - In many cases (especially when values are normalized or images have been post-processed),
              these heuristics may not be conclusive.
        """
        # Ensure input is a numpy array
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        # Handle grayscale images: 2D arrays or 3D with a single channel.
        if img.ndim == 2:
            return C_ImageFormats.GRAYSCALE
        if img.ndim == 3:
            h, w, c = img.shape
            if c == 1:
                return C_ImageFormats.GRAYSCALE_SINGLE_CHANNEL

            # If there are 3 channels, we need to differentiate between several possibilities.
            if c == 3:
                # Compute basic statistics for each channel.
                # (These are used in heuristics; formulas: ch_i_min = min(img[..., i]),
                # ch_i_max = max(img[..., i]) for i in {0,1,2})
                ch0_min, ch0_max = np.min(img[..., 0]), np.max(img[..., 0])
                ch1_min, ch1_max = np.min(img[..., 1]), np.max(img[..., 1])
                ch2_min, ch2_max = np.min(img[..., 2]), np.max(img[..., 2])

                # Heuristic for detecting an HSV image (using OpenCV’s convention):
                # In an 8-bit image, Hue is in the range [0, 179] while Saturation and Value are in [0, 255].
                # Here, if channel 0 (possible Hue) never exceeds 180 but at least one of the other channels does,
                # it might be an HSV image.
                if ch0_max <= 180 and (ch1_max > 180 or ch2_max > 180):
                    return C_ImageFormats.HSV

                # Without further metadata, we cannot distinguish between RGB and BGR.
                # Both are 3-channel images with similar ranges. This is left as ambiguous.
                warnings.warn(C_ImageHandler.AMBIGUOUS_IMAGE_FORMAT, UserWarning)
                return C_ImageFormats.RGB_OR_BGR

            # Handle 4-channel images.
            if c == 4:
                # In many cases a 4-channel image is either RGBA or BGRA.
                # Without further context, we report it as ambiguous.
                warnings.warn(C_ImageHandler.AMBIGUOUS_IMAGE_FORMAT, UserWarning)
                return C_ImageFormats.RGBA_OR_BGRA

            # For any other number of channels, we note it as an unknown format.
            raise ValueError(f"Image with {c} channels (unknown format)")

        # If the array has more than 3 dimensions, we don't have a standard interpretation.
        raise ValueError("Unknown format (unsupported number of dimensions)")
