from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenoscope import Image

import uuid
from typing import Optional, Union, Dict, Tuple
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from os import PathLike
from pathlib import Path

from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import rotate as skimage_rotate
from scipy.ndimage import rotate as scipy_rotate
from copy import deepcopy
from typing import Type

from scipy.sparse import csc_matrix

from ..accessors import (
    ImageArray,
    ImageMatrix,
    ImageEnhancedMatrix,
    ObjectMask,
    ObjectMap,
    ImageObjects
)

from phenoscope.util.constants_ import IMAGE_FORMATS, METADATA_LABELS
from phenoscope.util.exceptions_ import (
    EmptyImageError, NoArrayError, NoObjectsError, IllegalAssignmentError,
    UnsupportedSchemaError, UnsupportedFileTypeError
)


class ImageHandler:
    """The core comprehensive class for handling image processing, including manipulation, information sync, metadata management, and format conversion.

    The `ImageHandler` class is designed to load, process, and manage image data using different
    representation formats (e.g., arrays and matrices). This class allows for metadata editing,
    schema definition, and subcomponent handling to streamline image processing tasks.

    Note:
        If the input_image is 2-D, the ImageHandler leave the array form as None
        If the input_image is 3-D, the ImageHandler will automatically set the matrix component to the grayscale representation.

    Attributes:
        _array (Optional[np.ndarray]): The numeric array representation of the image
            used for multichannel data processing.
        _matrix (Optional[np.ndarray]): The matrix representation of the image, primarily
            used for 2D image data or grayscale formats.
        _det_matrix (Optional[np.ndarray]): A detection matrix representation, useful for
            object detection tasks and similar applications.
        _sparse_object_map (Optional[csc_matrix]): A sparse object map for efficient
            representation of object detection overlays.
        __image_schema (Optional[str]): A schema defining the format of the image input,
            influencing how data is interpreted or manipulated.
        _private_metadata (Dict[str, Any]): Immutable metadata associated with the image,
            including a universally unique identifier (UUID).
        _protected_metadata (Dict[str, Optional[Union[int, float, str, bool, np.integer,
            np.floating, np.bool_, np.complexfloating]]]): Editable protected metadata,
            such as the image name, providing key image-related information.
        _public_metadata (Dict[str, Union[int, float, str, bool, np.integer, np.floating,
            np.bool_, np.complexfloating]]): Public metadata that can be fully edited or
            removed without restrictions.
        __array_subhandler (ImageArray): Subhandler responsible for array-based operations.
            Extends functionality for working with image arrays.
        __matrix_subhandler (ImageMatrix): Subhandler for performing operations on the image
            matrix representation.
        __det_matrix_subhandler (ImageDetectionMatrix): Subhandler for handling detection
            matrix operations.
        __object_mask_subhandler (ObjectMask): Subhandler for managing object masks in the
            image.
        __object_map_subhandler (ObjectMap): Subhandler for working with object maps.
        __objects_subhandler (ImageObjects): Subhandler for managing objects detected in
            the image.
    """

    def __init__(self, input_image: Optional[Union[np.ndarray, Image, PathLike]] = None,
                 input_schema: Optional[str] = None,
                 name: Optional[str] = None):
        """
        Initializes an image processing object with provided input image, schema, and name.

        The class constructor initializes core backend variables, including the image schema,
        protected and public metadata dictionaries, and various image component handlers
        to manage subcomponents and data operations for image manipulations. It also sets up
        private and protected metadata with specific handling for input names or defaults.

        Args:
            input_image: An optional input image represented as either a NumPy array or an image
                object. Defaults to None.
            input_schema: An optional string defining the schema for the input image to specify
                how data should be interpreted or processed. Defaults to None.
            name: An optional string to assign a name to the image, used as metadata. If not
                provided, a universally unique identifier (UUID) will be generated and assigned.
        """
        self._array: Optional[np.ndarray] = None
        self._matrix: Optional[np.ndarray] = None
        self._det_matrix: Optional[np.ndarray] = None
        self._sparse_object_map: Optional[csc_matrix] = None

        # Initialize core backend variables
        self.__image_schema = None

        # Private metadata cannot be edited and is not duplicated with copies of the class
        self._private_metadata = {
            METADATA_LABELS.UUID: uuid.uuid4()
        }

        # Protected Metadata can be edited, but not removed
        self._protected_metadata: Dict[
            str, Optional[Union[int, float, str, bool, np.integer, np.floating, np.bool_, np.complexfloating]]
        ] = {
            METADATA_LABELS.IMAGE_NAME: name if name is not None else self._private_metadata[METADATA_LABELS.UUID],
        }

        # Public metadata can be edited or removed
        self._public_metadata: Dict[str, Union[int, float, str, bool, np.integer, np.floating, np.bool_, np.complexfloating]] = {}

        # Initialize image component handlers
        self.__array_subhandler: ImageArray = ImageArray(self)
        self.__matrix_subhandler: ImageMatrix = ImageMatrix(self)
        self.__det_matrix_subhandler: ImageEnhancedMatrix = ImageEnhancedMatrix(self)
        self.__object_mask_subhandler: ObjectMask = ObjectMask(self)
        self.__object_map_subhandler: ObjectMap = ObjectMap(self)
        self.__objects_subhandler: ImageObjects = ImageObjects(self)

        if isinstance(input_image, (PathLike, str, Path)):
            self.imread(input_image)
        else:
            self.set_image(input_image=input_image, input_schema=input_schema)

    def __getitem__(self, key) -> Image:
        """Returns a subimage from the current object based on the provided key. The subimage is initialized
        as a new instance of the same class, maintaining the schema and format consistency as the original
        image object. This method supports 2-dimensional slicing and indexing.

        Args:
            key: A slicing key or index used to extract a subset or part of the image object.

        Returns:
            Image: An instance of the Image representing the subimage corresponding to the provided key.

        Raises:
            KeyError: If the provided key does not match the expected slicing format or dimensions.
        """
        if self.schema not in IMAGE_FORMATS.MATRIX_FORMATS:
            subimage = self.__class__(input_image=self.array[key], input_schema=self.schema)
        else:
            subimage = self.__class__(input_image=self.matrix[key], input_schema=self.schema)

        subimage.enh_matrix[:] = self.enh_matrix[key]
        subimage.omap[:] = self.omap[key]
        return subimage

    def __setitem__(self, key, value):
        """Sets an item in the object with a given key and Image object. Ensures that the Image being set matches the expected shape and type, and updates internal properties accordingly.

        Args:
            key (Any): The array slices for accesssing the elements of the image.
            value (ImageHandler): The value to be set, which must match the shape of the
                existing elements and conform to the expected schema.

        Raises:
            ValueError: If the shape of the `value` does not match the shape of the existing
                elements being accessed.
        """
        if np.array_equal(self.shape, value.shape) is False: raise ValueError(
            'The image being set must be of the same shape as the image elements being accessed.'
        )
        if isinstance(value, ImageHandler) or issubclass(type(value), ImageHandler):
            if value.schema not in IMAGE_FORMATS.MATRIX_FORMATS and self.schema not in IMAGE_FORMATS.MATRIX_FORMATS:
                self._array[key] = value.array[:]
            self._matrix[key] = value.matrix[:]
            self._det_matrix[key] = value.enh_matrix[:]
            self.omask[key] = value.omask[:]

    def __eq__(self, other) -> bool:
        return True if (
                self.schema == other.schema
                and np.array_equal(self.array[:], other.array[:])
                and np.array_equal(self.matrix[:], other.matrix[:])
                and np.array_equal(self.enh_matrix[:], other.enh_matrix[:])
                and np.array_equal(self.omap[:], other.omap[:])
                and self._protected_metadata == other._protected_metadata
                and self._public_metadata == other._public_metadata
        ) else False

    def __ne__(self, other):
        return not self == other

    def isempty(self) -> bool:
        """Returns True if there is no image data"""
        return True if self._matrix is None else False

    @property
    def name(self):
        """Returns the name of the image. If no name is set, the name will be the file stem of the image."""
        return self._protected_metadata[METADATA_LABELS.IMAGE_NAME]

    @name.setter
    def name(self, value):
        if type(value) != str:
            raise ValueError('Image name must be a string')
        self._protected_metadata[METADATA_LABELS.IMAGE_NAME] = value

    @property
    def uuid(self):
        """Returns the UUID of the image"""
        return self._private_metadata[METADATA_LABELS.UUID]

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
            raise EmptyImageError

    @property
    def schema(self) -> Optional[str]:
        """Returns the input format of the image array or matrix depending on input format"""
        if self.__image_schema is None:
            raise EmptyImageError
        else:
            return self.__image_schema

    @property
    def array(self) -> ImageArray:
        """The image's array representation. An image array can represent multi-channels, thus the array can be 3-D or 4-D

        Note:
            - array elements are not directly mutable in order to preserve image information integrity
            - change array elements by changing the image being represented with Image.set_image()
            - Raises an error if input image has no array form

        Returns:
            Optional[ImageArray]: A class that can be accessed like a numpy array, but has extra methods to streamline development, or None if not set

        See Also: :class:`ImageArray`
        """
        if self._array is None:
            if self._matrix is None:
                raise EmptyImageError
            else:
                raise NoArrayError()
        else:
            return self.__array_subhandler

    @array.setter
    def array(self, array):
        if isinstance(array, np.ndarray):
            self.array[:] = array
        else:
            raise IllegalAssignmentError('array')

    @property
    def matrix(self) -> ImageMatrix:
        """The image's matrix representation. The array form is converted into a matrix form since some algorithm's only handle 2-D

        Note:
            - matrix elements are not directly mutable in order to preserve image information integrity
            - Change matrix elements by changing the image being represented with Image.set_image()

        Returns:
            ImageMatrix: An immutable container for the image matrix that can be accessed like a numpy array, but has extra methods to streamline development.

        See Also: :class:`ImageMatrix`
        """
        if self._matrix is None:
            raise EmptyImageError
        else:
            return self.__matrix_subhandler

    @matrix.setter
    def matrix(self, matrix):
        if isinstance(matrix, np.ndarray):
            self.matrix[:] = matrix
        else:
            raise IllegalAssignmentError('matrix')

    @property
    def enh_matrix(self) -> ImageEnhancedMatrix:
        """Returns the image's enhanced matrix accessor (See: :class:`ImageEnhancedMatrix`. Preprocessing steps can be applied to this component to improve detection performance.

        The enhanceable matrix is a copy of the image's matrix form that can be modified and used to improve detection performance.
        The original matrix data should be left intact in order to preserve image information integrity for measurements.'

        Returns:
            ImageEnhancedMatrix: A mutable container that stores a copy of the image's matrix form

        See Also: :class:`ImageEnhancedMatrix`
        """
        if self._det_matrix is None:
            raise EmptyImageError
        else:
            return self.__det_matrix_subhandler

    @enh_matrix.setter
    def enh_matrix(self, det_matrix):
        if isinstance(det_matrix, np.ndarray):
            self._det_matrix[:] = det_matrix
        else:
            raise IllegalAssignmentError('enh_matrix')

    @property
    def omask(self) -> ObjectMask:
        """The Object Mask. A mutable binary representation of the objects in an image to be analyzed. Changing elements of the mask will reset object_map labeling.

        Note:
            - If the image has not been processed by a detector, the target for analysis is the entire image itself. Accessing the object_mask in this case
                will return a 2-D array entirely with value 1 that is the same shape as the matrix
            - Changing elements of the mask will relabel of objects in the object_map (A workaround to this issue may or may not come in future versions)

        Returns:
            ObjectMaskErrors: A mutable binary representation of the objects in an image to be analyzed.

        See Also: :class:`ObjectMask`
        """
        if self._sparse_object_map is None:
            raise EmptyImageError
        else:
            return self.__object_mask_subhandler

    @omask.setter
    def omask(self, object_mask):
        if isinstance(object_mask, np.ndarray):
            self.omask[:] = object_mask
        else:
            raise IllegalAssignmentError('object_mask')

    @property
    def omap(self) -> ObjectMap:
        """The Object Map. A mutable integer matrix that identifies the different objects in an image to be analyzed. Changes to elements of the object_map sync to the object_mask.

        The object_map is stored as a compressed sparse column matrix in the backend. This is to save on memory consumption at the cost of adding
        increased computational overhead between converting between sparse and dense matrices.

        Note:
            - Has accessor methods to get sparse representations of the object map that can streamline measurement calculations.

        Returns:
            ObjectMap: A mutable integer matrix that identifies the different objects in an image to be analyzed.

        See Also: :class:`ObjectMap`
        """
        if self._sparse_object_map is None:
            raise EmptyImageError
        else:
            return self.__object_map_subhandler

    @omap.setter
    def omap(self, object_map):
        if isinstance(object_map, np.ndarray):
            self.omap[:] = object_map
        else:
            raise IllegalAssignmentError('object_map')

    @property
    def props(self):
        """Fetches the properties of labeled regions in an image.

        Calculates region properties for the entire image using the matrix representation.
        The labeled image is generated as a full array with values of 1, and the
        intensity image corresponds to the `_matrix` attribute of the object.
        Cache is disabled in this configuration.

        Returns:
            list[skimage.measure._regionprops.RegionProperties]: A list of properties for the entire provided image.

        Notes:
            (Excerpt from skimage.measure.regionprops documentation on available properties. See more at :class:`skimage.measure.regionprops`.):

            area: float
                Area of the region i.e. number of pixels of the region scaled by pixel-area.

            area_bbox: float
                Area of the bounding box i.e. number of pixels of bounding box scaled by pixel-area.

            area_convex: float
                Area of the convex hull image, which is the smallest convex polygon that encloses the region.

            area_filled: float
                Area of the region with all the holes filled in.

            axis_major_length: float
                The length of the major axis of the ellipse that has the same normalized second central moments as the region.

            axis_minor_length: float
                The length of the minor axis of the ellipse that has the same normalized second central moments as the region.

            bbox: tuple
                Bounding box (min_row, min_col, max_row, max_col). Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col).

            centroid: array
                Centroid coordinate tuple (row, col).

            centroid_local: array
                Centroid coordinate tuple (row, col), relative to region bounding box.

            centroid_weighted: array
                Centroid coordinate tuple (row, col) weighted with intensity image.

            centroid_weighted_local: array
                Centroid coordinate tuple (row, col), relative to region bounding box, weighted with intensity image.

            coords_scaled(K, 2): ndarray
                Coordinate list (row, col) of the region scaled by spacing.

            coords(K, 2): ndarray
                Coordinate list (row, col) of the region.

            eccentricity: float
                Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.

            equivalent_diameter_area: float
                The diameter of a circle with the same area as the region.

            euler_number: int
                Euler characteristic of the set of non-zero pixels. Computed as number of connected components subtracted by number of holes (input.ndim connectivity). In 3D, number of connected components plus number of holes subtracted by number of tunnels.

            extent: float
                Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)

            feret_diameter_max: float
                Maximum Feret’s diameter computed as the longest distance between points around a region’s convex hull contour as determined by find_contours. [5]

            image(H, J): ndarray
                Sliced binary region image which has the same size as bounding box.

            image_convex(H, J): ndarray
                Binary convex hull image which has the same size as bounding box.

            image_filled(H, J): ndarray
                Binary region image with filled holes which has the same size as bounding box.

            image_intensity: ndarray
                Image inside region bounding box.

            inertia_tensor: ndarray
                Inertia tensor of the region for the rotation around its mass.

            inertia_tensor_eigvals: tuple
                The eigenvalues of the inertia tensor in decreasing order.

            intensity_max: float
                Value with the greatest intensity in the region.

            intensity_mean: float
                Value with the mean intensity in the region.

            intensity_min: float
                Value with the least intensity in the region.

            intensity_std: float
                Standard deviation of the intensity in the region.

            label: int
                The label in the labeled input image.

            moments(3, 3): ndarray
                Spatial moments up to 3rd order::

                    m_ij = sum{ array(row, col) * row^i * col^j }

            where the sum is over the row, col coordinates of the region.

            moments_central(3, 3): ndarray
                Central moments (translation invariant) up to 3rd order::

                    mu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

                where the sum is over the row, col coordinates of the region, and row_c and col_c are the coordinates of the region’s centroid.

            moments_hu: tuple
                Hu moments (translation, scale and rotation invariant).

            moments_normalized(3, 3): ndarray
                Normalized moments (translation and scale invariant) up to 3rd order::

                    nu_ij = mu_ij / m_00^[(i+j)/2 + 1]

                where m_00 is the zeroth spatial moment.

            moments_weighted(3, 3): ndarray
                Spatial moments of intensity image up to 3rd order::

                    wm_ij = sum{ array(row, col) * row^i * col^j }

                where the sum is over the row, col coordinates of the region.

            moments_weighted_central(3, 3): ndarray
                Central moments (translation invariant) of intensity image up to 3rd order::

                    wmu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

                where the sum is over the row, col coordinates of the region, and row_c and col_c are the coordinates of the region’s weighted centroid.

            moments_weighted_hu: tuple
                Hu moments (translation, scale and rotation invariant) of intensity image.

            moments_weighted_normalized(3, 3): ndarray
                Normalized moments (translation and scale invariant) of intensity image up to 3rd order::

                    wnu_ij = wmu_ij / wm_00^[(i+j)/2 + 1]

                where wm_00 is the zeroth spatial moment (intensity-weighted area).

            num_pixels: int
                Number of foreground pixels.

            orientation: float
                Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.

            perimeter: float
                Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.

            perimeter_crofton: float
                Perimeter of object approximated by the Crofton formula in 4 directions.

            slice: tuple of slices
                A slice to extract the object from the source image.

            solidity: float
                Ratio of pixels in the region to pixels of the convex hull image.

        References:
            https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops


        """
        return ski.measure.regionprops(label_image=np.full(shape=self.shape, fill_value=1), intensity_image=self._matrix, cache=False)

    @property
    def objects(self) -> ImageObjects:
        """Returns an abstract to access the objects in an image and perform operations on them, such as measurement calculations .

        This method provides access to `ImageObjects`.

        Returns:
            ImageObjects: The subhandler instance that manages image-related objects.
        """
        if self.num_objects==0:
            raise NoObjectsError(self.name)
        else:
            return self.__objects_subhandler

    @objects.setter
    def objects(self, objects):
        raise IllegalAssignmentError('objects')

    @property
    def num_objects(self) -> int:
        """Returns the number of objects in the image
        Note:
            If the number of objects is 0, the target for analysis is the entire image itself.
        """
        object_labels = np.unique(self._sparse_object_map.data)
        return len(object_labels[object_labels != 0])

    def copy(self):
        """Creates a copy of the current Image instance, excluding the UUID.
        Note:
            - The new instance is only informationally a copy. The UUID of the new instance is different.

        Returns:
            Image: A copy of the current Image instance.
        """
        # Create a new instance of ImageHandler
        return self.__class__(self)

    def imread(self, filepath: PathLike) -> Type[Image]:
        """
        Reads an image file from a given file path, processes it as per its format, and sets the image
        along with its schema in the current instance. Supports RGB formats (png, jpg, jpeg) and
        grayscale formats (tif, tiff). The name of the image processing instance is updated to match
        the file name without the extension. If the file format is unsupported, an exception is raised.

        Args:
            filepath (PathLike): Path to the image file to be read.

        Returns:
            Type[Image]: The current instance with the newly loaded image and schema.

        Raises:
            UnsupportedFileType: If the file format is not supported.
        """
        # Convert to Path object
        filepath = Path(filepath)
        if filepath.suffix in ['.png', '.jpg', '.jpeg']:
            self.set_image(
                input_image=ski.io.imread(filepath), input_schema=IMAGE_FORMATS.RGB
            )
            self.name = filepath.stem
            return self
        elif filepath.suffix in ['.tif', '.tiff']:
            self.set_image(
                input_image=ski.io.imread(filepath), input_schema=IMAGE_FORMATS.GRAYSCALE
            )
            self.name = filepath.stem
            return self
        else:
            raise UnsupportedFileTypeError(filepath.suffix)

    def set_image(self, input_image, input_schema: str = None) -> None:
        """Sets the image to the inputted array or Image

        Args:
            input_image: (np.ndarray, phenoscope.Image, optional) The image data to be set
            input_schema: (str, optional) If input image is a np.ndarray and format is None, the image format will be guessed.
                If the image format is ambiguous between RGB/BGR it will assume RGB unless otherwise specified.
                Accepted formats are ['RGB', 'RGBA','Grayscale','BGR','BGRA','HSV']
        """
        if type(input_image) == np.ndarray:
            self._set_from_array(input_image, input_schema)
        elif (type(input_image) == self.__class__
              or isinstance(input_image, self.__class__)
              or issubclass(type(input_image), ImageHandler)):
            self._set_from_class_instance(input_image)
        elif input_image is None:
            self.__image_schema = None
            self._array = None
            self._matrix = None
            self._det_matrix = None
            self._sparse_object_map = None

    def _set_from_class_instance(self, class_instance):
        self.__image_schema = class_instance.schema

        if class_instance.schema not in IMAGE_FORMATS.MATRIX_FORMATS:
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
            if self.__image_schema == IMAGE_FORMATS.RGB_OR_BGR or self.__image_schema == IMAGE_FORMATS.RGBA_OR_BGRA:
                # PhenoScope will assume in the event of rgb vs bgr that the input was rgb
                self.__image_schema = IMAGE_FORMATS.RGB

        else:
            # Valid Format Check
            if input_schema not in IMAGE_FORMATS.SUPPORTED_FORMATS:
                raise UnsupportedSchemaError(input_schema)
            else:
                self.__image_schema = input_schema

        # Grayscale Formats
        if (self.__image_schema == IMAGE_FORMATS.GRAYSCALE
                or self.__image_schema == IMAGE_FORMATS.GRAYSCALE_SINGLE_CHANNEL
        ): self._from_grayscale(image)

        # RGB Formats
        if (self.__image_schema == IMAGE_FORMATS.RGB
                or self.__image_schema == IMAGE_FORMATS.RGBA
        ): self._from_rgb_or_rgba(image)

        # BGR Formats
        if (self.__image_schema == IMAGE_FORMATS.BGR
                or self.__image_schema == IMAGE_FORMATS.BGRA
        ): self._from_bgr_or_bgra(image)

    def _from_grayscale(self, grayscale_array: np.ndarray):
        """Initialize class components from a grayscale image array"""
        if self.__image_schema == IMAGE_FORMATS.GRAYSCALE:
            self._set_from_matrix(grayscale_array)

        elif self.__image_schema == IMAGE_FORMATS.GRAYSCALE_SINGLE_CHANNEL:
            self._set_from_matrix(grayscale_array[:, :, 0])

    def _from_rgb_or_rgba(self, rgb_or_rgba_array: np.ndarray):
        """Initialize class components from an RGB or RGBA image array"""
        if self.__image_schema == IMAGE_FORMATS.RGB:
            self._array = rgb_or_rgba_array
            self._set_from_matrix(
                matrix=rgb2gray(rgb_or_rgba_array)
            )

        elif self.__image_schema == IMAGE_FORMATS.RGBA:
            self._array = rgb_or_rgba_array
            self._set_from_matrix(
                matrix=rgb2gray(
                    rgba2rgb(rgb_or_rgba_array)
                )
            )

    def _from_bgr_or_bgra(self, bgr_or_bgra_array: np.ndarray):
        """Initialize class components from a BGR or BGRA image array"""
        if self.__image_schema == IMAGE_FORMATS.BGR:
            self._array = bgr_or_bgra_array
            self._set_from_matrix(
                matrix=rgb2gray(
                    self._convert_bgr_to_rgb(bgr_or_bgra_array)
                )
            )
        elif self.__image_schema == IMAGE_FORMATS.BGRA:
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
            return IMAGE_FORMATS.GRAYSCALE
        if img.ndim == 3:
            h, w, c = img.shape
            if c == 1:
                return IMAGE_FORMATS.GRAYSCALE_SINGLE_CHANNEL

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
                    return IMAGE_FORMATS.HSV

                # Without further metadata, we cannot distinguish between RGB and BGR.
                # Both are 3-channel images with similar ranges. This is left as ambiguous.
                return IMAGE_FORMATS.RGB_OR_BGR

            # Handle 4-channel images.
            if c == 4:
                # In many cases a 4-channel image is either RGBA or BGRA.
                # Without further context, we report it as ambiguous.
                return IMAGE_FORMATS.RGBA_OR_BGRA

            # For any other number of channels, we note it as an unknown format.
            raise ValueError(f"Image with {c} channels (unknown format)")

        # If the array has more than 3 dimensions, we don't have a standard interpretation.
        raise ValueError("Unknown format (unsupported number of dimensions)")

    def show(self,
             ax: plt.Axes = None,
             figsize: Tuple[int, int] = (9, 10)
             ) -> (plt.Figure, plt.Axes):
        """Returns a matplotlib figure showing the input image"""
        if self.schema not in IMAGE_FORMATS.MATRIX_FORMATS:
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
        if self.schema not in IMAGE_FORMATS.MATRIX_FORMATS:
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
        """Rotate's the image and all its components"""
        if self.schema not in IMAGE_FORMATS.MATRIX_FORMATS:
            self._array = skimage_rotate(image=self._array, angle=angle_of_rotation, mode=mode, clip=True, **kwargs)

        self._matrix = skimage_rotate(image=self._matrix, angle=angle_of_rotation, mode=mode, clip=True, **kwargs)
        self._det_matrix = skimage_rotate(image=self._det_matrix, angle=angle_of_rotation, mode=mode, clip=True, **kwargs)

        # Rotate the object map while preserving the details and using nearest-neighbor interpolation
        self.omap[:] = scipy_rotate(input=self.omap[:], angle=angle_of_rotation, mode='constant', cval=0, order=0, reshape=False)

    def reset(self) -> Type[Image]:
        """
        Resets the internal state of the object and returns an updated instance.

        This method resets the state of DetectionMatrix and ObjectMap components maintained
        by the object. It ensures that the object is reset to its original state
        while maintaining its type integrity. Upon execution, the instance of the
        calling object itself is returned.

        Returns:
            Type[Image]: The instance of the object after resetting its internal
            state.
        """
        self.enh_matrix.reset()
        self.omap.reset()
        return self
