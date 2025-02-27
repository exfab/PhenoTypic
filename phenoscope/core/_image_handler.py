import uuid
from typing import Optional, Union, Dict, Tuple
from pathlib import Path
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt

from skimage.color import rgb2gray, rgba2rgb, rgb2hsv
from skimage.transform import rotate as skimage_rotate
from scipy.ndimage import rotate as scipy_rotate
from copy import deepcopy
from typing import TypeVar, Type

from scipy.sparse import csc_matrix

from ._image_components import (
    ImageArraySubhandler,
    ImageMatrixSubhandler,
    ImageDetectionMatrixSubhandler,
    ObjectMaskSubhandler,
    ObjectMapSubhandler,
    ImageObjectsSubhandler
)

from ..util.constants import C_ImageHandler, C_ImageFormats, C_Metadata

Image = TypeVar('Image', bound='ImageHandler')


class ImageHandler:
    """An ImageHandler class to sync and automate different operations between image processing components

    The ImageHandler class is responsible for automating syncing of different image processing components during processing. This can help
    maintain data integrity through complex pipelines. It employs a tree hierarchy to help manage components.

    Note:
        If the input_image is 2-D, the ImageHandler leave the array form as None
        If the input_image is 3-D, the ImageHandler will automatically set the matrix component to the grayscale representation.

    Args:
        input_image: (optional, np.ndarray) Can be a 2-D, 3-D, or 4-D numpy array representing an image
        input_schema: (str) The input image/array schema

    Protected Attributes:


    """

    def __init__(self, input_image: Optional[Union[np.ndarray, Type[Image]]] = None, input_schema: str = None):
        # Initialize core backend variables
        self.__image_schema = None

        self._array: Optional[np.ndarray] = None
        self._matrix: Optional[np.ndarray] = None
        self._det_matrix: Optional[np.ndarray] = None
        self._sparse_object_map: Optional[csc_matrix] = None

        # Private metadata cannot be edited and is not duplicated with copies of the class
        self._private_metadata = {
            C_Metadata.LABELS.UUID: uuid.uuid4()
        }

        # Protected Metadata can be edited, but not removed
        self._protected_metadata: Dict[
            str, Optional[Union[int, float, str, bool, np.integer, np.floating, np.bool_, np.complexfloating]]] = {
            C_Metadata.LABELS.IMAGE_NAME: None
        }

        # Public metadata can be edited or removed
        self._public_metadata: Dict[str, Union[int, float, str, bool, np.integer, np.floating, np.bool_, np.complexfloating]] = {}

        # Initialize image component handlers
        self.__array_subhandler: ImageArraySubhandler = ImageArraySubhandler(self)
        self.__matrix_subhandler: ImageMatrixSubhandler = ImageMatrixSubhandler(self)
        self.__det_matrix_subhandler: ImageDetectionMatrixSubhandler = ImageDetectionMatrixSubhandler(self)
        self.__object_mask_subhandler: ObjectMaskSubhandler = ObjectMaskSubhandler(self)
        self.__object_map_subhandler: ObjectMapSubhandler = ObjectMapSubhandler(self)
        self.__objects_subhandler: ImageObjectsSubhandler = ImageObjectsSubhandler(self)

        self.set_image(input_image=input_image, input_schema=input_schema)

    def __getitem__(self, slices) -> Type[Image]:
        """Returns a copy of the image at the slices specified

        Returns:
            Image: A copy of the image at the slices indicated
        """
        if self.schema not in C_ImageFormats.MATRIX_FORMATS:
            subimage = self.__class__(input_image=self.array[slices], input_schema=self.schema)
        else:
            subimage = self.__class__(input_image=self.matrix[slices], input_schema=self.schema)

        subimage.det_matrix[:] = self.det_matrix[slices]
        subimage.obj_map[:] = self.obj_map[slices]
        return subimage

    def __eq__(self, other) -> bool:
        return True if (
                self.schema == other.schema
                and np.array_equal(self.array[:], other.array[:])
                and np.array_equal(self.matrix[:], other.matrix[:])
                and np.array_equal(self.det_matrix[:], other.det_matrix[:])
                and np.array_equal(self.obj_map[:], other.obj_map[:])
                and self._protected_metadata == other._protected_metadata
                and self._public_metadata == other._public_metadata
        ) else False

    def __ne__(self, other):
        return not self == other

    @property
    def name(self):
        return self._protected_metadata[C_Metadata.LABELS.IMAGE_NAME]

    @name.setter
    def name(self, value):
        if type(value) != str:
            raise ValueError('Image name must be a string')
        self._protected_metadata[C_Metadata.LABELS.IMAGE_NAME] = value

    @property
    def shape(self):
        """Returns the shape of the image array or matrix depending on input format or none if no image is set.

        Returns:
            Optional[Tuple(int,int,...)]: Returns the shape of the array or matrix depending on input format or none if no image is set.
        """
        if self._array is not None:
            return self._array.shape
        elif self._matrix is not None:
            return self._matrix.shape
        else:
            raise C_ImageHandler.EmptyImageError

    def isempty(self) -> bool:
        """Returns True if there is no image data is empty"""
        return True if self._matrix is None else False

    @property
    def schema(self) -> str:
        """Returns the input format of the image array or matrix depending on input format"""
        if self.__image_schema is None:
            raise C_ImageHandler.EmptyImageError
        else:
            return self.__image_schema

    @property
    def array(self) -> ImageArraySubhandler:
        """The image's array representation. An image array can represent multi-channels, thus the array can be 3-D or 4-D

        Note:
            - array elements are not directly mutable in order to preserve image information integrity
            - change array elements by changing the image being represented with Image.set_image()
            - Raises an error if input image has no array form

        Returns:
            Optional[ImageArraySubhandler]: A class that can be accessed like a numpy array, but has extra methods to streamline development, or None if not set
        """
        if self._array is None:
            if self._matrix is None:
                raise C_ImageHandler.EmptyImageError
            else:
                raise C_ImageHandler.NoArrayError()
        else:
            return self.__array_subhandler

    @array.setter
    def array(self, array):
        if isinstance(array, np.ndarray):
            self.array[:] = array
        else:
            raise C_ImageHandler.IllegalAssignmentError('array')

    @property
    def matrix(self) -> ImageMatrixSubhandler:
        """The image's matrix representation. The array form is converted into a matrix form since some algorithm's only handle 2-D

        Note:
            - matrix elements are not directly mutable in order to preserve image information integrity
            - Change matrix elements by changing the image being represented with Image.set_image()

        Returns:
            ImageMatrixSubhandler: An immutable container for the image matrix that can be accessed like a numpy array, but has extra methods to streamline development.
        """
        if self._matrix is None:
            raise C_ImageHandler.EmptyImageError
        else:
            return self.__matrix_subhandler

    @matrix.setter
    def matrix(self, matrix):
        if isinstance(matrix, np.ndarray):
            self.matrix[:] = matrix
        else:
            raise C_ImageHandler.IllegalAssignmentError('matrix')

    @property
    def det_matrix(self) -> ImageDetectionMatrixSubhandler:
        """A mutable copy of the image's matrix representation. Preprocessing steps can be applied to this component to improve detection performance.

        Returns:
            ImageDetectionMatrixSubhandler: A mutable container that stores a copy of the image's matrix form
        """
        if self._det_matrix is None:
            raise C_ImageHandler.EmptyImageError
        else:
            return self.__det_matrix_subhandler

    @det_matrix.setter
    def det_matrix(self, det_matrix):
        if isinstance(det_matrix, np.ndarray):
            self._det_matrix[:] = det_matrix
        else:
            raise C_ImageHandler.IllegalAssignmentError('det_matrix')

    @property
    def obj_mask(self) -> ObjectMaskSubhandler:
        """A mutable binary representation of the objects in an image to be analyzed. Changing elements of the mask will reset object_map labeling.

        Note:
            - If the image has not been processed by a detector, the target for analysis is the entire image itself. Accessing the object_mask in this case
                will return a 2-D array entirely with value 1 that is the same shape as the matrix
            - Changing elements of the mask will relabel of objects in the object_map (A workaround to this issue may or may not come in future versions)

        Returns:
            ObjectMaskErrors: A mutable binary representation of the objects in an image to be analyzed.
        """
        if self._sparse_object_map is None:
            raise C_ImageHandler.EmptyImageError
        else:
            return self.__object_mask_subhandler

    @obj_mask.setter
    def obj_mask(self, object_mask):
        if isinstance(object_mask, np.ndarray):
            self.obj_mask[:] = object_mask
        else:
            raise C_ImageHandler.IllegalAssignmentError('object_mask')

    @property
    def obj_map(self) -> ObjectMapSubhandler:
        """A mutable integer matrix that identifies the different objects in an image to be analyzed. Changes to elements of the object_map sync to the object_mask.

        The object_map is stored as a compressed sparse column matrix in the backend. This is to save on memory consumption at the cost of adding
        increased computational overhead between converting between sparse and dense matrices.

        Note:
              - Has accessor methods to get sparse representations of the object map that can streamline measurement calculations.

        Returns:
            ObjectMapSubhandler: A mutable integer matrix that identifies the different objects in an image to be analyzed.
        """
        if self._sparse_object_map is None:
            raise C_ImageHandler.EmptyImageError
        else:
            return self.__object_map_subhandler

    @obj_map.setter
    def obj_map(self, object_map):
        if isinstance(object_map, np.ndarray):
            self.obj_map[:] = object_map
        else:
            raise C_ImageHandler.IllegalAssignmentError('object_map')

    @property
    def objects(self) -> ImageObjectsSubhandler:
        return self.__objects_subhandler

    @objects.setter
    def objects(self, objects):
        raise C_ImageHandler.IllegalAssignmentError('objects')

    def copy(self):
        """Creates a copy of the current Image instance, excluding the UUID.
        Note:
            - The new instance is only informationally a copy. The UUID of the new instance is different.

        Returns:
            Image: A copy of the current Image instance.
        """
        # Create a new instance of ImageHandler
        return self.__class__(self)

    def imread(self, filepath) -> Type[Image]:
        """Imports an image from a filepath and initilizes class components. The image name is automatically set to the file stem.


        Note:
            - Supported image formats: .png, .jpg, .jpeg, .tif, .tiff
        Args:
            filepath (str): Path to the image file
        """
        # Convert to Path object
        filepath = Path(filepath)
        if filepath.suffix in ['.png', '.jpg', '.jpeg']:
            self.set_image(
                input_image=ski.io.imread(filepath), input_schema=C_ImageFormats.RGB
            )
            self.name = filepath.stem
            return self
        elif filepath.suffix in ['.tif', '.tiff']:
            self.set_image(
                input_image=ski.io.imread(filepath), input_schema=C_ImageFormats.GRAYSCALE
            )
            self.name = filepath.stem
            return self
        else:
            raise C_ImageHandler.UnsupportedFileType(filepath.suffix)

    def set_image(self, input_image, input_schema: str = None) -> None:
        """Sets the image to the inputted array

        Args:
            input_image: (np.ndarray, phenoscope.Image, optional) The image data to be set
            input_schema: (str, optional) If input image is a np.ndarray and format is None, the image format will be guessed.
                If the image format is ambiguous between RGB/BGR it will assume RGB unless otherwise specified.
                Accepted formats are ['RGB', 'RGBA','Grayscale','BGR','BGRA','HSV']
        """
        if type(input_image) == np.ndarray:
            self._set_from_array(input_image, input_schema)
        elif type(input_image) == self.__class__ or isinstance(input_image, self.__class__) or issubclass(type(input_image), self.__class__
                                                                                                          ):
            self._set_from_class_instance(input_image)
        elif input_image is None:
            self.__image_schema = None
            self._array = None
            self._matrix = None
            self._det_matrix = None
            self._sparse_object_map = None

    def _set_from_class_instance(self, class_instance):
        self.__image_schema = class_instance.schema

        if class_instance.schema not in C_ImageFormats.MATRIX_FORMATS:
            self._set_from_array(class_instance.array[:].copy(), class_instance.schema)
        else:
            self._set_from_array(class_instance.matrix[:].copy(), class_instance.schema)
        self._det_matrix = class_instance._det_matrix.copy()
        self._sparse_object_map = class_instance._sparse_object_map.copy()
        self._protected_metadata = deepcopy(class_instance._protected_metadata)
        self._public_metadata = deepcopy(class_instance._public_metadata)

    def _set_from_matrix(self, matrix):
        """Initializes all the 2-D components of an image

        Args:
            matrix: A 2-D array form of an image
        """
        self._matrix = matrix.copy()
        self.__det_matrix_subhandler.reset()
        self.__object_map_subhandler.reset()

    def _set_from_array(self, image: np.ndarray, input_schema: str) -> None:
        """Initializes all the components of an image from an array

        Note:
            The format of the input should already have been set or guessed
        Args:
            image: the input image array
            input_schema: (str, optional) The format of the input image
        """

        if input_schema is None:
            self.__image_schema = self._guess_image_format(image)
            if self.__image_schema == C_ImageFormats.RGB_OR_BGR or self.__image_schema == C_ImageFormats.RGBA_OR_BGRA:
                # PhenoScope will assume in the event of rgb vs bgr that the input was rgb
                self.__image_schema = C_ImageFormats.RGB

        else:
            # Valid Format Check
            if input_schema not in C_ImageFormats.SUPPORTED_FORMATS:
                raise C_ImageFormats.UnsupportedFormatError(input_schema)
            else:
                self.__image_schema = input_schema

        # Grayscale Formats
        if (self.__image_schema == C_ImageFormats.GRAYSCALE
                or self.__image_schema == C_ImageFormats.GRAYSCALE_SINGLE_CHANNEL
        ): self._from_grayscale(image)

        # RGB Formats
        if (self.__image_schema == C_ImageFormats.RGB
                or self.__image_schema == C_ImageFormats.RGBA
        ): self._from_rgb_or_rgba(image)

        # BGR Formats
        if (self.__image_schema == C_ImageFormats.BGR
                or self.__image_schema == C_ImageFormats.BGRA
        ): self._from_bgr_or_bgra(image)

    def _from_grayscale(self, grayscale_array: np.ndarray):
        """Initialize class components from a grayscale image array"""
        if self.__image_schema == C_ImageFormats.GRAYSCALE:
            self._set_from_matrix(grayscale_array)

        elif self.__image_schema == C_ImageFormats.GRAYSCALE_SINGLE_CHANNEL:
            self._set_from_matrix(grayscale_array[:, :, 0])

    def _from_rgb_or_rgba(self, rgb_or_rgba_array: np.ndarray):
        """Initialize class components from an RGB or RGBA image array"""
        if self.__image_schema == C_ImageFormats.RGB:
            self._array = rgb_or_rgba_array
            self._set_from_matrix(
                matrix=rgb2gray(rgb_or_rgba_array)
            )

        elif self.__image_schema == C_ImageFormats.RGBA:
            self._array = rgb_or_rgba_array
            self._set_from_matrix(
                matrix=rgb2gray(
                    rgba2rgb(rgb_or_rgba_array)
                )
            )

    def _from_bgr_or_bgra(self, bgr_or_bgra_array: np.ndarray):
        """Initialize class components from a BGR or BGRA image array"""
        if self.__image_schema == C_ImageFormats.BGR:
            self._array = bgr_or_bgra_array
            self._set_from_matrix(
                matrix=rgb2gray(
                    self._convert_bgr_to_rgb(bgr_or_bgra_array)
                )
            )
        elif self.__image_schema == C_ImageFormats.BGRA:
            self._array = bgr_or_bgra_array
            self._set_from_matrix(
                matrix=rgb2gray(
                    rgba2rgb(
                        self._convert_bgra_to_rgba(bgr_or_bgra_array)
                    )
                )
            )

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
                return C_ImageFormats.RGB_OR_BGR

            # Handle 4-channel images.
            if c == 4:
                # In many cases a 4-channel image is either RGBA or BGRA.
                # Without further context, we report it as ambiguous.
                return C_ImageFormats.RGBA_OR_BGRA

            # For any other number of channels, we note it as an unknown format.
            raise ValueError(f"Image with {c} channels (unknown format)")

        # If the array has more than 3 dimensions, we don't have a standard interpretation.
        raise ValueError("Unknown format (unsupported number of dimensions)")

    def show(self,
             ax: plt.Axes = None,
             figsize: Tuple[int, int] = (9, 10)
             ) -> (plt.Figure, plt.Axes):
        """Returns a matplotlib figure showing the input image"""
        if self.schema not in C_ImageFormats.MATRIX_FORMATS:
            return self.array.show(ax=ax, figsize=figsize)
        else:
            return self.matrix.show(ax=ax, figsize=figsize)

    def show_overlay(self, object_label: Optional[int] = None, ax: plt.Axes = None,
                     figsize: Tuple[int, int] = (10, 5),
                     annotate: bool = False,
                     annotation_size: int = 12,
                     annotation_color: str = 'white',
                     annotation_facecolor: str = 'red',
                     ) -> (plt.Figure, plt.Axes):
        """Returns a matplotlib figure showing the overlay of the object map on the input image"""
        if self.schema not in C_ImageFormats.MATRIX_FORMATS:
            return self.array.show_overlay(object_label=object_label, ax=ax, figsize=figsize,
                                           annotate=annotate, annotation_size=annotation_size,
                                           annotation_color=annotation_color, annotation_facecolor=annotation_facecolor
                                           )
        else:
            return self.matrix.show_overlay(object_label=object_label, ax=ax, figsize=figsize,
                                            annotate=annotate, annotation_size=annotation_size,
                                            annotation_color=annotation_color, annotation_facecolor=annotation_facecolor
                                            )

    def rotate(self, angle_of_rotation: int, mode: str = 'edge', **kwargs) -> None:
        """Rotate's the image and all it's components"""
        if self.schema not in C_ImageFormats.MATRIX_FORMATS:
            self._array = skimage_rotate(image=self._array, angle=angle_of_rotation, mode=mode, clip=True, **kwargs)

        self._matrix = skimage_rotate(image=self._matrix, angle=angle_of_rotation, mode=mode, clip=True, **kwargs)
        self._det_matrix = skimage_rotate(image=self._det_matrix, angle=angle_of_rotation, mode=mode, clip=True, **kwargs)

        # Rotate the object map while preserving the details and using nearest-neighbor interpolation
        self.obj_map[:] = scipy_rotate(input=self.obj_map[:], angle=angle_of_rotation, mode='constant', cval=0, order=0, reshape=False)

    def reset(self) -> Type[Image]:
        """Resets the image detection matrix and object map"""
        self.det_matrix.reset()
        self.obj_map.reset()
        return self
