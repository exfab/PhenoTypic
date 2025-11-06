from __future__ import annotations

from typing import Literal, TYPE_CHECKING, Optional, Tuple, Type

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
from skimage.transform import rotate as skimage_rotate
import scipy.ndimage as ndimage
from types import SimpleNamespace

from phenotypic.core._image_parts._image_data_manager import ImageDataManager
from phenotypic.core._image_parts.accessors import (
    ImageRGB,
    ImageMatrix,
    ImageEnhancedMatrix,
    ObjectMask,
    ObjectMap,
    MetadataAccessor,
)

from phenotypic.tools.constants_ import IMAGE_FORMATS, METADATA, IMAGE_TYPES
from phenotypic.tools.exceptions_ import (
    EmptyImageError, IllegalAssignmentError
)


class ImageHandler(ImageDataManager):
    """Provides accessor functions and operations for image manipulation.
    
    This class extends ImageDataManager to add:
    - Property-based accessors for rgb, gray, enh_gray, objmask, objmap
    - Image manipulation operations (slicing, rotation, copying)
    - Visualization methods (show, show_overlay)
    - Comparison and equality operations
    
    The accessor pattern allows intuitive access to image data while maintaining
    internal consistency and supporting various image formats.
    
    Attributes:
        _accessors (SimpleNamespace): Property-based accessors for image components.
    """

    def __init__(self,
                 arr: np.ndarray | Image | None = None,
                 name: str | None = None,
                 bit_depth: Literal[8, 16] | None = None
                 ):
        """Initialize ImageHandler with accessors and optional image data.
        
        Args:
            arr (np.ndarray | Image | None): Optional initial image input.
            name (str | None): Optional name for the image.
        """
        # Initialize parent class (data management)
        super().__init__(name=name, bit_depth=bit_depth)

        # Initialize image accessors
        self._accessors = SimpleNamespace()
        self._accessors.rgb = ImageRGB(self)
        self._accessors.gray = ImageMatrix(self)
        self._accessors.enh_gray = ImageEnhancedMatrix(self)
        self._accessors.objmask = ObjectMask(self)
        self._accessors.objmap = ObjectMap(self)
        self._accessors.metadata = MetadataAccessor(self)

        # Handle non-empty inputs
        if arr is not None:
            self.set_image(input_image=arr)

    def __getitem__(self, key) -> Image:
        """Returns a new subimage from the current object based on the provided key. The subimage is initialized
        as a new instance of the same class, maintaining the schema and format consistency as the original
        image object. This method supports 2-dimensional slicing and indexing.

        Note:
            - The subimage arrays are copied from the original image object. This means that any changes made to the subimage will not affect the original image.
            - We may add this functionality in future updates if there is demand for it.
        Args:
            key: A slicing key or index used to extract a subset or part of the image object.

        Returns:
            Image: An instance of the Image representing the subimage corresponding to the provided key.

        Raises:
            KeyError: If the provided key does not match the expected slicing format or dimensions.
        """
        if not self.rgb.isempty():
            subimage = self.__class__(arr=self.rgb[key])
        else:
            subimage = self.__class__(arr=self.gray[key])

        subimage.enh_gray[:] = self.enh_gray[key].copy()
        subimage.objmap[:] = self.objmap[key].copy()
        subimage.metadata[METADATA.IMAGE_TYPE] = IMAGE_TYPES.CROP.value
        return subimage

    def __setitem__(self, key, other_image):
        """Sets an item in the object with a given key and Image object. Ensures that the Image being set matches the expected shape and type, and updates internal properties accordingly.

        Args:
            key (Any): The array slices for accesssing the elements of the image.
            other_image (ImageHandler): The other image to be set, which must match the shape of the
                existing elements accessed by the key and conform to the expected schema.

        Raises:
            ValueError: If the shape of the `value` does not match the shape of the existing
                elements being accessed.
        """

        # Sections can only be set to another Image class
        if isinstance(other_image, self.__class__) or issubclass(type(other_image), ImageHandler):
            # Handle the rgb case
            if not other_image.rgb.isempty() and not self.rgb.isempty():
                if np.array_equal(self.rgb[key].shape, other_image.rgb.shape) is False:
                    raise ValueError(
                            'The image being set must be of the same shape as the image elements being accessed.',
                    )
                else:
                    self._data.rgb[key] = other_image._data.rgb[:]

            # handle other cases
            if np.array_equal(self.gray[key].shape, other_image.gray.shape) is False:
                raise ValueError(
                        'The image being set must be of the same shape as the image elements being accessed.',
                )
            else:
                self._data.gray[key] = other_image._data.gray[:]
                self._data.enh_gray[key] = other_image._data.enh_gray[:]
                self.objmask[key] = other_image.objmask[:]

    def __eq__(self, other: Image) -> bool:
        """
        Compares the current object with another object for equality.

        This method checks if the current object's attributes are equal to another object's
        attributes. Equality is determined by verifying that the numerical arrays
        (`rgb`, `gray`, `enh_gray`, `objmap`) are element-wise identical.

        Note:
            - Only checks core image data, and not any other attributes such as metadata.

        Args:
            other: The object to compare with the current instance.

        Returns:
            bool: True if all the attributes of the current object are identical to those
            of the `other` object. Returns False otherwise.
        """
        # Check if both images have the same format (RGB vs grayscale)
        format_match = (self.rgb.isempty() == other.rgb.isempty())

        # Check RGB arrays: equal if both present and matching, or both absent
        self_has_rgb = not self.rgb.isempty()
        other_has_rgb = not other.rgb.isempty()

        rgb_check = (
            (self_has_rgb == other_has_rgb) and
            (not self_has_rgb or np.array_equal(self.rgb[:], other.rgb[:]))
        )

        return format_match and rgb_check and np.array_equal(self.gray[:], other.gray[:]) and np.array_equal(
                self.enh_gray[:], other.enh_gray[:]) and np.array_equal(self.objmap[:], other.objmap[:])

    def __ne__(self, other):
        return not self == other

    @property
    def name(self) -> str:
        """Returns the name of the image. If no name is set, the name will be the uuid of the image."""
        name = self._metadata.protected.get(METADATA.IMAGE_NAME, None)
        return name if name else str(self.uuid)

    @name.setter
    def name(self, value):
        self.metadata[METADATA.IMAGE_NAME] = str(value)

    @property
    def uuid(self):
        """Returns the UUID of the image"""
        return self.metadata[METADATA.UUID]

    @property
    def _image_type(self):
        return self.metadata[METADATA.IMAGE_TYPE]

    def isempty(self) -> bool:
        """Check if image data is empty.

        Returns:
            bool: True if no image data is set.
        """
        return self.gray.isempty()

    @property
    def shape(self):
        """Returns the shape of the image array or gray depending on arr format or none if no image is set.

        Returns:
            Optional[Tuple(int,int,...)]: Returns the shape of the array or gray depending on arr format or none if no image is set.
        """
        if not self.rgb.isempty():
            return self.rgb.shape
        elif not self.gray.isempty():
            return self.gray.shape
        else:
            raise EmptyImageError

    @property
    def metadata(self) -> MetadataAccessor:
        return self._accessors.metadata

    @metadata.setter
    def metadata(self, value):
        raise IllegalAssignmentError('metadata')

    @property
    def rgb(self) -> ImageRGB:
        """Returns the ImageArray accessor; An image rgb represents the multichannel information

        Note:
            - rgb/gray element data is synced
            - change image shape by changing the image being represented with Image.set_image()
            - Raises an error if the arr image has no rgb form

        Returns:
            ImageRGB: A class that can be accessed like a numpy rgb, but has extra methods to streamline development, or None if not set

        Raises:
            NoArrayError: If no multichannel image data is set as arr.

        .. code-block:: python
            from phenotypic import Image
            from phenotypic.data import load_colony

            image = Image(load_colony())

            # get the rgb data
            arr = image.rgb[:]
            print(type(arr))

            # set the rgb data
            # the shape of the new rgb must be the same shape as the original rgb
            image.rgb[:] = arr

            # without the bracket indexing the accessor is returned instead
            print(image.rgb[:])


        See Also: :class:`ImageArray`
        """
        return self._accessors.rgb

    @rgb.setter
    def rgb(self, value):
        if isinstance(value, (np.ndarray, int, float)):
            self.rgb[:] = value
        else:
            raise IllegalAssignmentError('rgb')

    @property
    def gray(self) -> ImageMatrix:
        """The image's gray representation. The array form is converted into a gray form since some algorithm's only handle 2-D

        Note:
            - gray elements are not directly mutable in order to preserve image information integrity
            - Change gray elements by changing the image being represented with Image.set_image()

        Returns:
            ImageMatrix: An immutable container for the image gray that can be accessed like a numpy array, but has extra methods to streamline development.

        .. code-block:: python
            from phenotypic import Image
            from phenotypic.data import load_colony

            image = Image(load_colony())

            # get the gray data
            arr = image.gray[:]
            print(type(arr))

            # set the gray data
            # the shape of the new gray must be the same shape as the original gray
            image.gray[:] = arr

            # without the bracket indexing the accessor is returned instead
            print(image.gray[:])

        See Also: :class:`ImageMatrix`
        """
        if self._data.gray is None:
            raise EmptyImageError
        else:
            return self._accessors.gray

    @gray.setter
    def gray(self, value):
        if isinstance(value, (np.ndarray, int, float)):
            self.gray[:] = value
        else:
            raise IllegalAssignmentError('gray')

    @property
    def enh_gray(self) -> ImageEnhancedMatrix:
        """Returns the image's enhanced gray accessor (See: :class:`ImageEnhancedMatrix`. Preprocessing steps can be applied to this component to improve detection performance.

        The enhanceable gray is a copy of the image's gray form that can be modified and used to improve detection performance.
        The original gray data should be left intact in order to preserve image information integrity for measurements.'

        Returns:
            ImageEnhancedMatrix: A mutable container that stores a copy of the image's gray form

        .. code-block:: python
            from phenotypic import Image
            from phenotypic.data import load_colony

            image = Image(load_colony())

            # get the enh_gray data
            arr = image.enh_gray[:]
            print(type(arr))

            # set the enh_gray data
            # the shape of the new enh_gray must be the same shape as the original enh_gray
            image.enh_gray[:] = arr

            # without the bracket indexing the accessor is returned instead
            print(image.enh_gray[:])

        See Also: :class:`ImageEnhancedMatrix`
        """
        if self._data.enh_gray is None:
            raise EmptyImageError
        else:
            return self._accessors.enh_gray

    @enh_gray.setter
    def enh_gray(self, value):
        if isinstance(value, (np.ndarray, int, float)):
            self.enh_gray[:] = value
        else:
            raise IllegalAssignmentError('enh_gray')

    @property
    def objmask(self) -> ObjectMask:
        """Returns the ObjectMask Accessor; The object mask is a mutable binary representation of the objects in an image to be analyzed. Changing elements of the mask will reset object_map labeling.

        Note:
            - If the image has not been processed by a detector, the target for analysis is the entire image itself. Accessing the object_mask in this case
                will return a 2-D array entirely with other_image 1 that is the same shape as the gray
            - Changing elements of the mask will relabel of objects in the object_map

        Returns:
            ObjectMaskErrors: A mutable binary representation of the objects in an image to be analyzed.

        .. code-block:: python
            from phenotypic import Image
            from phenotypic.data import load_colony

            image = Image(load_colony())

            # get the objmask data
            arr = image.objmask[:]
            print(type(arr))

            # set the objmask data
            # the shape of the new objmask must be the same shape as the original objmask
            image.objmask[:] = arr

            # without the bracket indexing the accessor is returned instead
            print(image.objmask[:])

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
        """Returns the ObjectMap accessor; The object map is a mutable integer gray that identifies the different objects in an image to be analyzed. Changes to elements of the object_map sync to the object_mask.

        The object_map is stored as a compressed sparse column gray in the backend. This is to save on memory consumption at the cost of adding
        increased computational overhead between converting between sparse and dense matrices.

        Note:
            - Has accessor methods to get sparse representations of the object map that can streamline measurement calculations.

        Returns:
            ObjectMap: A mutable integer gray that identifies the different objects in an image to be analyzed.

        .. code-block:: python
            from phenotypic import Image
            from phenotypic.data import load_colony

            image = Image(load_colony())

            # get the objmap data
            arr = image.objmap[:]
            print(type(arr))

            # set the objmap data
            # the shape of the new objmap must be the same shape as the original objmap
            image.objmap[:] = arr

            # without the bracket indexing the accessor is returned instead
            print(image.objmap[:])

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

        Calculates region properties for the entire image using the gray representation.
        The labeled image is generated as a full array with values of 1, and the
        intensity image corresponds to the `_data.gray` attribute of the object.
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
                Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The other_image is in the interval [0, 1). When it is 0, the ellipse becomes a circle.

            equivalent_diameter_area: float
                The diameter of a circle with the same area as the region.

            euler_number: int
                Euler characteristic of the set of non-zero pixels. Computed as number of connected components subtracted by number of holes (arr.ndim connectivity). In 3D, number of connected components plus number of holes subtracted by number of tunnels.

            extent: float
                Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (nrows * ncols)

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
                The label in the labeled arr image.

            moments(3, 3): ndarray
                Spatial moments up to 3rd order::

                    m_ij = sum{ array(row, col) * row^i * col^j }

            where the sum is over the row, col coordinates of the region.

            moments_central(3, 3): ndarray
                Central moments (translation invariant) up to 3rd order::

                    mu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

                where the sum is over the row, col coordinates of the region, and row_c and col_c are the coordinates of the region’s centroid.

            moments_hu: tuple
                Hu moments (translation, scale, and rotation invariant).

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
                Angle between the 0th axis (nrows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.

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
        return ski.measure.regionprops(label_image=np.full(shape=self.shape, fill_value=1),
                                       intensity_image=self._data.gray, cache=False)

    @property
    def num_objects(self) -> int:
        """Returns the number of objects in the image
        Note:
        """
        self._data.sparse_object_map.eliminate_zeros()
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

    def _set_from_matrix(self, matrix: np.ndarray):
        """Override parent to also reset accessors after setting matrix data.
        
        Args:
            matrix (np.ndarray): A 2-D array form of an image.
        """
        super()._set_from_matrix(matrix)
        self._accessors.enh_gray.reset()
        self._accessors.objmap.reset()

    def show(self,
             ax: plt.Axes = None,
             figsize: Tuple[int, int] | None = None,
             **kwargs
             ) -> (plt.Figure, plt.Axes):
        """
        Displays the image data using matplotlib.

        This method renders either the array or gray property of the instance
        depending on the image format. It either shows the content on the
        provided matplotlib axes (`ax`) or creates a
        new figure and axes for the visualization. Additional display-related
        customization can be passed using keyword arguments.

        Args:
            ax (plt.Axes, optional): The matplotlib Axes object where the image
                will be displayed. If None, a new Axes object is created.
            figsize (Tuple[int, int] | None, optional): The size of the resulting
                figure if no `ax` is provided. Defaults to None.
            **kwargs: Additional keyword arguments to customize the rendering
                behavior when showing the image.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple consisting of the matplotlib
                Figure and Axes that contain the rendered content.
        """
        if not self.rgb.isempty():
            return self.rgb.show(ax=ax, figsize=figsize, **kwargs)
        else:
            return self.gray.show(ax=ax, figsize=figsize, **kwargs)

    def show_overlay(self, object_label: Optional[int] = None,
                     figsize: Tuple[int, int] = (10, 5),
                     title: str | None = None,
                     show_labels: bool = False,
                     ax: plt.Axes = None,
                     *,
                     label_settings: None | dict = None,
                     overlay_settings: None | dict = None,
                     imshow_settings: None | dict = None,
                     ) -> (plt.Figure, plt.Axes):
        """
        Displays an overlay of the provided object label and image using the specified settings.

        This method combines an image and its segmentation or annotation mask overlay
        for visualization. The specific behavior is adjusted based on the instance's
        underlying image format (e.g., whether it operates on arrays or matrices).

        Args:
            object_label (Optional[int]): The label of the object to overlay. If None,
                overlays all available objects.
            figsize (Tuple[int, int]): A tuple specifying the figure size in inches.
            title (str | None): The title of the overlay figure. If None, no title will
                be displayed.
            show_labels (bool): Whether to display object labels on the overlay. Defaults
                to False.
            ax (plt.Axes): An optional Matplotlib axes object. If provided, the overlay
                will be plotted on this axes. If None, a new axes object will be created.
            label_settings (None | dict): A dictionary specifying configurations for
                displaying object labels. If None, default settings will be used.
            overlay_settings (None | dict): A dictionary specifying configurations for
                the overlay appearance. If None, default settings will be used.
            imshow_settings (None | dict): A dictionary specifying configurations for
                the image display (e.g., color map or interpolation). If None, default
                settings will be used.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib figure and
                axes used for the overlay. This allows further customization or saving
                of the visualization outside this method.
        """

        if not self.rgb.isempty():
            return self.rgb.show_overlay(object_label=object_label, figsize=figsize, title=title,
                                         show_labels=show_labels, ax=ax,
                                         label_settings=label_settings, overlay_settings=overlay_settings,
                                         imshow_settings=imshow_settings)
        else:
            return self.gray.show_overlay(object_label=object_label, figsize=figsize, title=title,
                                          show_labels=show_labels, ax=ax,
                                          label_settings=label_settings, overlay_settings=overlay_settings,
                                          imshow_settings=imshow_settings)

    def rotate(self, angle_of_rotation: int, mode: str = 'constant', cval=0, order=0, preserve_range=True) -> None:
        """
        Rotates various data attributes of the object by a specified angle.

        The method applies rotation transformations image data. It data that falls outside the border is clipped.

        Args:
            angle_of_rotation (int): The angle, in degrees, by which to rotate the data attributes.
                Positive values indicate counterclockwise rotation.
            mode (str): Mode parameter determining how borders are handled during the rotation.
                Default is 'constant'.
            cval: Constant value to fill edges in 'constant' mode. Default is 0.
            order (int): The order of the spline interpolation for rotating images. Must be an
                integer in the range [0, 5]. Default is 0 for nearest-neighbor interpolation.
            preserve_range (bool): Whether to keep the original input range of values after
                performing the rotation. Default is True.

        Returns:
            None
        """
        if not self.rgb.isempty():
            self._data.rgb = skimage_rotate(image=self._data.rgb, angle=angle_of_rotation, mode=mode, clip=True,
                                            cval=cval, order=order, preserve_range=preserve_range)

        self._data.gray = skimage_rotate(image=self._data.gray, angle=angle_of_rotation, mode=mode, clip=True,
                                         cval=cval, order=order, preserve_range=preserve_range)

        self._data.enh_gray = skimage_rotate(image=self._data.enh_gray, angle=angle_of_rotation, mode=mode,
                                             clip=True, cval=cval, order=order, preserve_range=preserve_range)

        # Rotate the object map while preserving the details and using nearest-neighbor interpolation
        # This one must be nearest-neighbor
        self.objmap[:] = ndimage.rotate(input=self.objmap[:], angle=angle_of_rotation, mode='constant', cval=0, order=0,
                                        reshape=False)

    def reset(self) -> Type[Image]:
        """
        Resets the internal state of the object and returns an updated instance.

        This method resets the state of enhanced gray and object map components maintained
        by the object. It ensures that the object is reset to its original state
        while maintaining its type integrity. Upon execution, the instance of the
        calling object itself is returned.

        Returns:
            Type[Image]: The instance of the object after resetting its internal
            state.
        """
        self.enh_gray.reset()
        self.objmap.reset()
        return self

    def _norm2dtypeMatrix(self, normalized_value: np.ndarray) -> np.ndarray:
        """
        Converts a normalized gray with values between 0 and 1 to a specified data type with the
        appropriate scaling. The method ensures that all values are clipped to the range [0, 1]
        before scaling them to the data type's maximum other_image.

        Args:
            normalized_value: A 2D NumPy array where all values are assumed to be in the range
                [0, 1]. These values will be converted using the specified data type scale.

        Returns:
            numpy.ndarray: A 2D NumPy array of the same shape as `normalized_matrix`, converted
            to the target data type with scaled values.
        """
        return normalized_value

    @staticmethod
    def _dtype2normMatrix(matrix: np.ndarray) -> np.ndarray:
        """
        Normalizes the given gray to have values between 0.0 and 1.0 based on its data type.

        The method checks the data type of the input gray against the expected data
        type. If the data type does not match, a warning is issued. The gray is
        then normalized by dividing its values by the maximum possible other_image for its
        data type, ensuring all elements remain within the range of [0.0, 1.0].

        Args:
            matrix (np.ndarray): The input gray to be normalized.

        Returns:
            np.ndarray: A normalized gray where all values are within [0.0, 1.0].
        """
        return matrix
