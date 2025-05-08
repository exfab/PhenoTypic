from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import uuid
from typing import Optional, Tuple, Literal
from types import SimpleNamespace
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from os import PathLike
from pathlib import Path
import warnings

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
    ObjectsAccessor,
    MetadataAccessor
)

from phenotypic.util.constants_ import IMAGE_FORMATS, METADATA_LABELS
from phenotypic.util.exceptions_ import (
    EmptyImageError, NoArrayError, NoObjectsError, IllegalAssignmentError,
    UnsupportedFileTypeError
)


class ImageHandler:
    """
    Handles image data and provides an abstraction for accessing and manipulating images
    through multiple formats like array, matrix, object maps, and metadata.

    The class offers streamlined access to image properties and supports operations like slicing,
    setting sub-images, and managing metadata. It is designed to handle images in various formats
    and ensures compatibility during transformations and data manipulations.

    Attributes:
        _data.array (Optional[np.ndarray]): Internal representation of image data in array form.
        _data.matrix (Optional[np.ndarray]): Internal representation of image data in matrix form.
        _data.enh_matrix (Optional[np.ndarray]): Enhanced matrix for extended manipulations.
        _data.sparse_object_map (Optional[csc_matrix]): Sparse object representation for mapping object labels.
        _image_format (Optional[str]): Tracks the format/schema of the input image.
        _metadata (SimpleNamespace): Container holding private, protected, and public metadata for
            the image.
        _accessors (SimpleNamespace): Provides property-based access"""

    def __init__(self,
                 input_image: np.ndarray | Image | PathLike | None = None,
                 imformat: str | None = None,
                 name: str | None = None):
        """
        Args:
            input_image: An optional input image represented as either a NumPy array or an image
                object. Defaults to None.
            imformat: An optional string defining the schema for the input image to specify
                how data should be interpreted or processed. Defaults to None.
            name: An optional string to assign a name to the image, used as metadata. If not
                provided, a universally unique identifier (UUID) will be generated and assigned.
        """
        # Initialize image data
        self._data = SimpleNamespace()
        self._data.array = None
        self._data.matrix = None
        self._data.enh_matrix = None
        self._data.sparse_object_map = None

        # Initialize core backend variables
        self._image_format = IMAGE_FORMATS.NONE

        # Public metadata can be edited or removed
        self._metadata = SimpleNamespace(
            private={
                METADATA_LABELS.UUID: uuid.uuid4()
            },
            protected={
                METADATA_LABELS.IMAGE_NAME: name
            },
            public={}
        )

        # Initialize image accessors
        self._accessors = SimpleNamespace()

        self._accessors.array = ImageArray(self)
        self._accessors.matrix = ImageMatrix(self)
        self._accessors.enh_matrix = ImageEnhancedMatrix(self)
        self._accessors.objmap = ObjectMap(self)
        self._accessors.objmask = ObjectMask(self)
        self._accessors.objects = ObjectsAccessor(self)
        self._accessors.metadata = MetadataAccessor(self)

        # Set data to empty arrays first
        self._reset_data_to_empty()

        # Handle non-empty inputs
        if isinstance(input_image, (PathLike, str, Path)):
            self.imread(input_image)
        else:
            self.set_image(input_image=input_image, imformat=imformat)

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
        if self._image_format.is_array():
            subimage = self.__class__(input_image=self.array[key], imformat=self.imformat)
        else:
            subimage = self.__class__(input_image=self.matrix[key], imformat=self.imformat)

        subimage.enh_matrix[:] = self.enh_matrix[key]
        subimage.objmap[:] = self.objmap[key]
        return subimage

    def __setitem__(self, key, value):
        """Sets an item in the object with a given key and Image object. Ensures that the Image being set matches the expected shape and type, and updates internal properties accordingly.

        Args:
            key (Any): The array slices for accesssing the elements of the image.
            value (ImageHandler): The other image to be set, which must match the shape of the
                existing elements accessed by the key and conform to the expected schema.

        Raises:
            ValueError: If the shape of the `value` does not match the shape of the existing
                elements being accessed.
        """

        # Sections can only be set to another Image class
        if isinstance(value, self.__class__) or issubclass(type(value), ImageHandler):
            # Handle in the array case
            if value.imformat.is_array() and self.imformat.is_array():
                if np.array_equal(self.array[key].shape, value.array.shape) is False: raise ValueError(
                    'The image being set must be of the same shape as the image elements being accessed.'
                )
                else:
                    self._data.array[key] = value.array[:]

            # handle other cases
            if np.array_equal(self.matrix[key].shape, value.matrix.shape) is False:
                raise ValueError(
                    'The image being set must be of the same shape as the image elements being accessed.'
                )
            else:
                self._data.matrix[key] = value.matrix[:]
                self._data.enh_matrix[key] = value.enh_matrix[:]
                self.objmask[key] = value.objmask[:]

    def __eq__(self, other) -> bool:
        """
        Compares the current object with another object for equality.

        This method checks if the current object's attributes are equal to another object's
        attributes. Equality is determined by comparing the `imformat` attribute and
        verifying that the numerical arrays (`array`, `matrix`, `enh_matrix`, `objmap`)
        are element-wise identical.

        Note:
            - Only checks core image data, and not any other attributes such as metadata.

        Args:
            other: The object to compare with the current instance.

        Returns:
            bool: True if all the attributes of the current object are identical to those
            of the `other` object. Returns False otherwise.
        """
        return True if (
                self.imformat == other.imformat
                and np.array_equal(self.array[:], other.array[:])
                and np.array_equal(self.matrix[:], other.matrix[:])
                and np.array_equal(self.enh_matrix[:], other.enh_matrix[:])
                and np.array_equal(self.objmap[:], other.objmap[:])
        ) else False

    def __ne__(self, other):
        return not self == other

    def isempty(self) -> bool:
        """Returns True if there is no image data"""
        if self.matrix.isempty() and self._image_format.is_none():
            return True
        else:
            return False

    @property
    def name(self) -> str:
        """Returns the name of the image. If no name is set, the name will be the uuid of the image."""
        name = self._metadata.protected.get(METADATA_LABELS.IMAGE_NAME, None)
        return name if name else str(self.uuid)

    @name.setter
    def name(self, value):
        if type(value) != str:
            raise ValueError('Image name must be a string')
        self.metadata[METADATA_LABELS.IMAGE_NAME] = value

    @property
    def uuid(self):
        """Returns the UUID of the image"""
        return self.metadata[METADATA_LABELS.UUID]

    @property
    def shape(self):
        """Returns the shape of the image array or matrix depending on input format or none if no image is set.

        Returns:
            Optional[Tuple(int,int,...)]: Returns the shape of the array or matrix depending on input format or none if no image is set.
        """
        if self._image_format.is_array():
            return self._data.array.shape
        elif self._image_format.is_matrix():
            return self._data.matrix.shape
        else:
            raise EmptyImageError

    @property
    def imformat(self) -> IMAGE_FORMATS:
        """Returns the input format of the image array or matrix depending on input format"""
        if not self._image_format.is_none():
            # if self._data.matrix is None or self._data.enh_matrix is None or self._data.sparse_object_map is None:
            #     raise AttributeError('Unknown error. Image format exists, but missing image data')
            return self._image_format
        else:
            raise EmptyImageError

    @property
    def metadata(self):
        return self._accessors.metadata

    @metadata.setter
    def metadata(self, value):
        raise IllegalAssignmentError('metadata')

    @property
    def array(self) -> ImageArray:
        """Returns the ImageArray accessor; An image array represents the multichannel information

        Note:
            - array/matrix element data is synced
            - change image shape by changing the image being represented with Image.set_image()
            - Raises an error if the input image has no array form

        Returns:
            ImageArray: A class that can be accessed like a numpy array, but has extra methods to streamline development, or None if not set

        Raises:
            NoArrayError: If no multichannel image data is set as input.
        See Also: :class:`ImageArray`
        """
        return self._accessors.array

    @array.setter
    def array(self, value):
        if isinstance(value, (np.ndarray, int, float)):
            self.array[:] = value
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
        if self._data.matrix is None:
            raise EmptyImageError
        else:
            return self._accessors.matrix

    @matrix.setter
    def matrix(self, value):
        if isinstance(value, (np.ndarray, int, float)):
            self.matrix[:] = value
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
        if self._data.enh_matrix is None:
            raise EmptyImageError
        else:
            return self._accessors.enh_matrix

    @enh_matrix.setter
    def enh_matrix(self, value):
        if isinstance(value, (np.ndarray, int, float)):
            self.enh_matrix[:] = value
        else:
            raise IllegalAssignmentError('enh_matrix')

    @property
    def objmask(self) -> ObjectMask:
        """Returns the ObjectMask Accessor; The object mask is a mutable binary representation of the objects in an image to be analyzed. Changing elements of the mask will reset object_map labeling.

        Note:
            - If the image has not been processed by a detector, the target for analysis is the entire image itself. Accessing the object_mask in this case
                will return a 2-D array entirely with value 1 that is the same shape as the matrix
            - Changing elements of the mask will relabel of objects in the object_map (A workaround to this issue may or may not come in future versions)

        Returns:
            ObjectMaskErrors: A mutable binary representation of the objects in an image to be analyzed.

        See Also: :class:`ObjectMask`
        """
        return self._accessors.objmask

    @objmask.setter
    def objmask(self, value):
        if isinstance(value, (np.ndarray, int, bool)):
            self.objmask[:] = value
        else:
            raise IllegalAssignmentError('objmask')

    @property
    def objmap(self) -> ObjectMap:
        """Returns the ObjectMap accessor; The object map is a mutable integer matrix that identifies the different objects in an image to be analyzed. Changes to elements of the object_map sync to the object_mask.

        The object_map is stored as a compressed sparse column matrix in the backend. This is to save on memory consumption at the cost of adding
        increased computational overhead between converting between sparse and dense matrices.

        Note:
            - Has accessor methods to get sparse representations of the object map that can streamline measurement calculations.

        Returns:
            ObjectMap: A mutable integer matrix that identifies the different objects in an image to be analyzed.

        See Also: :class:`ObjectMap`
        """
        return self._accessors.objmap

    @objmap.setter
    def objmap(self, value):
        if isinstance(value, (np.ndarray, int, float, bool)):
            self.objmap[:] = value
        else:
            raise IllegalAssignmentError('objmap')

    @property
    def props(self) -> list[ski.measure._regionprops.RegionProperties]:
        """Fetches the properties of the whole image.

        Calculates region properties for the entire image using the matrix representation.
        The labeled image is generated as a full array with values of 1, and the
        intensity image corresponds to the `_data.matrix` attribute of the object.
        Cache is disabled in this configuration.

        Returns:
            list[skimage.measure._regionprops.RegionProperties]: A list of properties for the entire provided image.

        Notes:
            (Excerpt from skimage.measure.regionprops documentation on available properties.):

            Read more at :class:`skimage.measure.regionprops`

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
        return ski.measure.regionprops(label_image=np.full(shape=self.shape, fill_value=1), intensity_image=self._data.matrix, cache=False)

    @property
    def objects(self) -> ObjectsAccessor:
        """Returns an acessor to the objects in an image and perform operations on them, such as measurement calculations.

        This method provides access to `ImageObjects`.

        Returns:
            ObjectsAccessor: The subhandler instance that manages image-related objects.

        Raises:
            NoObjectsError: If no objects are targeted in the image. Apply an ObjectDetector first.
        """
        if self.num_objects == 0:
            raise NoObjectsError(self.name)
        else:
            return self._accessors.objects

    @objects.setter
    def objects(self, objects):
        raise IllegalAssignmentError('objects')

    @property
    def num_objects(self) -> int:
        """Returns the number of objects in the image
        Note:
            If the number of objects is 0, the target for analysis is the entire image itself.
        """
        object_labels = np.unique(self._data.sparse_object_map.data)
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
        # Convert to a Path object
        filepath = Path(filepath)
        if filepath.suffix in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            self.set_image(
                input_image=ski.io.imread(filepath)
            )
            self.name = filepath.stem
            return self
        else:
            raise UnsupportedFileTypeError(filepath.suffix)

    def set_image(self, input_image: Image | np.ndarray | None = None, imformat: Literal['RGB', 'greyscale'] | None = None) -> None:
        """
        Sets the image data and format based on the provided input and parameters.

        This method accepts an image in the form of an array, another class instance,
        or a None value, and sets the internal image data accordingly. It determines
        how to process the input based on its type, and separates actions for arrays,
        instances of the class, and None input.

        Args:
            input_image (Image | np.ndarray): The image data input which can either
                be an instance of an Image, a NumPy array, or None. If None, the internal
                image-related attributes are reset.
            imformat (Literal['RGB', 'greyscale'] | None): Optional format specifier
                indicating the format of the input image. If None, it attempts to derive
                the format automatically based on the image data.
        """
        if type(input_image) == np.ndarray:
            self._set_from_array(input_image, imformat)
        elif (type(input_image) == self.__class__
              or isinstance(input_image, self.__class__)
              or issubclass(type(input_image), ImageHandler)):
            self._set_from_class_instance(input_image)
        elif input_image is None:
            self._reset_data_to_empty()
        else:
            raise ValueError(f'input_image must be a NumPy array, a class instance, or None. Got {type(input_image)}')

    def _reset_data_to_empty(self):
        self._data.array = np.empty((0, 3))  # Create an empty 3D array
        self._set_from_matrix(np.empty((0, 2)))
        self._image_format = IMAGE_FORMATS.NONE

    def _set_from_class_instance(self, class_instance):
        self._image_format = class_instance._image_format

        if class_instance._image_format.is_array():
            self._set_from_array(class_instance.array[:].copy(), class_instance._image_format.value)
        else:
            self._set_from_array(class_instance.matrix[:].copy(), class_instance._image_format.value)
        for key, value in class_instance._data.__dict__.items():
            self._data.__dict__[key] = value.copy() if value is not None else None

            self._metadata.protected = deepcopy(class_instance._metadata.protected)
            self._metadata.public = deepcopy(class_instance._metadata.public)

    def _set_from_matrix(self, matrix: np.ndarray):
        """Initializes all the 2-D components of an image

        Args:
            matrix: A 2-D array form of an image
        """
        self._data.matrix = matrix.copy()
        self._accessors.enh_matrix.reset()
        self._accessors.objmap.reset()

    def _set_from_rgb(self, rgb_array: np.ndarray):
        """Initializes all the components of an image from an RGB array

        """
        self._data.array = rgb_array.copy()
        self._set_from_matrix(rgb2gray(self._data.array.copy()))

    def _set_from_array(self, imarr: np.ndarray, imformat: Literal['RGB', 'greyscale'] | None) -> None:
        """Initializes all the components of an image from an array

        Note:
            The format of the input should already have been set or guessed
        Args:
            imarr: the input image array
            imformat: (str, optional) The format of the input image
        """

        # In the event of None for schema, PhenoTypic guesses the format
        if imformat is None:
            imformat = self._guess_image_format(imarr)

        if type(imformat) == IMAGE_FORMATS:
            if imformat.is_ambiguous():
                # PhenoTypic will assume in the event of rgb vs bgr that the input was rgb
                imformat = IMAGE_FORMATS.RGB.value
            else:
                imformat = imformat.value

        match imformat.upper():
            case 'GRAYSCALE' | IMAGE_FORMATS.GRAYSCALE | IMAGE_FORMATS.GRAYSCALE_SINGLE_CHANNEL:
                self._image_format = IMAGE_FORMATS.GRAYSCALE
                self._set_from_matrix(
                    imarr if imarr.ndim == 2 else imarr[:, :, 0]
                )

            case 'RGB' | IMAGE_FORMATS.RGB | IMAGE_FORMATS.RGB_OR_BGR:
                self._image_format = IMAGE_FORMATS.RGB
                self._set_from_rgb(imarr)

            case 'RGBA' | IMAGE_FORMATS.RGBA | IMAGE_FORMATS.RGBA_OR_BGRA:
                self._image_format = IMAGE_FORMATS.RGB
                self._set_from_rgb(rgba2rgb(imarr))

            # case 'BGR' | IMAGE_FORMATS.BGR:
            #     self._image_format = IMAGE_FORMATS.RGB
            #     warnings.warn('BGR Images are automatically converted to RGB')
            #     self._set_from_rgb(imarr[:, :, ::-1])
            #
            # case 'BGRA' | IMAGE_FORMATS.BGRA:
            #     self._image_format = IMAGE_FORMATS.RGB
            #     warnings.warn('BGRA Images are automatically converted to RGB')
            #     self._set_from_rgb(imarr[:, :, [2, 1, 0, 3]])

            case _:
                raise ValueError(f'Unsupported image format: {imformat}')

    @staticmethod
    def _guess_image_format(img: np.ndarray) -> IMAGE_FORMATS:
        """
        Determines the format of a given image based on its dimensions and number of color channels.

        Args:
            img (np.ndarray): Input image represented as a numpy array.

        Returns:
            IMAGE_FORMATS: Enum value indicating the detected format of the image.

        Raises:
            TypeError: If the input is not a numpy array.
            ValueError: If the image has an unsupported number of dimensions or channels.
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
                return IMAGE_FORMATS.RGB

            # Handle 4-channel images.
            if c == 4:
                # In many cases a 4-channel image is either RGBA or BGRA.
                # Without further context, we report it as ambiguous.
                return IMAGE_FORMATS.RGBA

            # For any other number of channels, we note it as an unknown format.
            raise ValueError(f"Image with {c} channels (unknown format)")

        # If the array has more than 3 dimensions, we don't have a standard interpretation.
        raise ValueError("Unknown format (unsupported number of dimensions)")

    def show(self,
             ax: plt.Axes = None,
             figsize: Tuple[int, int] = (9, 10)
             ) -> (plt.Figure, plt.Axes):
        """Returns a matplotlib figure and axes showing the input image"""
        if self.imformat not in IMAGE_FORMATS.MATRIX_FORMATS:
            return self.array.show(ax=ax, figsize=figsize)
        else:
            return self.matrix.show(ax=ax, figsize=figsize)

    def show_overlay(self, object_label: Optional[int] = None, ax: plt.Axes = None,
                     figsize: Tuple[int, int] = (10, 5),
                     annotate: bool = False,
                     annotation_params: None | dict = None,
                     ) -> (plt.Figure, plt.Axes):
        """
        Displays an overlay of the object specified by the given label on an image or
        matrix with optional annotations.

        This method checks the schema of the object to determine whether it belongs to
        matrix formats or image formats, and delegates the overlay rendering to the
        appropriate method accordingly. It optionally allows annotations to be added
        for the specified object label with customizable style settings.

        Args:
            object_label (Optional[int]): The label of the object to overlay. If None,
                the entire image or matrix is displayed without a specific object
                highlighted.
            ax (Optional[plt.Axes]): The matplotlib Axes instance to render the overlay
                on. If None, a new figure and axes are created for rendering.
            figsize (Tuple[int, int]): Tuple specifying the size (width, height) of the
                figure to create if no axes are provided.
            annotate (bool): Whether to annotate the image/matrix using the given
                annotation settings.
            annotation_params (None | dict): Additional parameters for customization of the
                object annotations. Defaults: size=12, color='white', facecolor='red

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure and
            Axes objects used to render the overlay.

        """
        if self._image_format.is_array():
            return self.array.show_overlay(object_label=object_label, ax=ax, figsize=figsize,
                                           annotate=annotate, annotation_params=annotation_params
                                           )
        else:
            return self.matrix.show_overlay(object_label=object_label, ax=ax, figsize=figsize,
                                            annotate=annotate, annotation_params=annotation_params
                                            )

    def rotate(self, angle_of_rotation: int, mode: str = 'edge', **kwargs) -> None:
        """Rotate the image and all its components"""
        if self._image_format.is_array():
            self._data.array = skimage_rotate(image=self._data.array, angle=angle_of_rotation, mode=mode, clip=True, **kwargs)

        self._data.matrix = skimage_rotate(image=self._data.matrix, angle=angle_of_rotation, mode=mode, clip=True, **kwargs)
        self._data.enh_matrix = skimage_rotate(image=self._data.enh_matrix, angle=angle_of_rotation, mode=mode, clip=True, **kwargs)

        # Rotate the object map while preserving the details and using nearest-neighbor interpolation
        self.objmap[:] = scipy_rotate(input=self.objmap[:], angle=angle_of_rotation, mode='constant', cval=0, order=0, reshape=False)

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
        self.objmap.reset()
        return self
