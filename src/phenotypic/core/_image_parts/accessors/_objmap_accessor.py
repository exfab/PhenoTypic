from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import numpy as np

from scipy.sparse import csc_matrix, coo_matrix
import matplotlib.pyplot as plt
from skimage.measure import label

from phenotypic.core._image_parts.accessor_abstracts import SingleChannelAccessor
from phenotypic.tools.exceptions_ import (
    ArrayKeyValueShapeMismatchError,
    InvalidMapValueError,
)


class ObjectMap(SingleChannelAccessor):
    """Manages an object map for labeled regions in an image.

    This class provides a mechanism to manipulate and access labeled object maps
    within a given image. It is tightly coupled with the parent image object and
    provides methods for accessing sparse and dense representations, relabeling,
    resetting, and visualization.

    The object map is stored internally as a compressed sparse column (CSC) matrix
    for memory efficiency. All public methods expose the data through dense array
    interfaces for ease of use while maintaining the sparse representation backend.
    Changes to the object map shapes are automatically reflected in the object mask.

    The class supports array-like indexing and slicing operations, sparse format
    conversion, and visualization with matplotlib.

    Attributes:
        _root_image: The parent Image object that this accessor is associated with.

    Note:
        Changes to the object map shapes will be automatically reflected in the
        object mask.
    """

    @property
    def _backend(self):
        """Return the current sparse backend reference.

        This property ensures we always access the live reference to the sparse
        object map, even if it's been replaced by another operation. This is
        critical for maintaining consistency when the sparse matrix is updated
        in-place by various operations.

        Returns:
            scipy.sparse.csc_matrix: The current sparse object map in compressed
                sparse column format.
        """
        return self._root_image._data.sparse_object_map

    @property
    def _num_objects(self):
        """Return the total number of distinct labeled objects in the map.

        Returns:
            int: The count of unique non-zero labels in the object map.
        """
        return len(self._labels)

    @property
    def _labels(self):
        """Return the unique object labels present in the image.

        Extracts all unique labels from the object map, excluding the background
        (label 0). This property internally converts the sparse representation to
        dense for label extraction.

        Returns:
            np.ndarray: A 1D array of unique non-zero labels, sorted in ascending
                order. An empty array is returned if no objects are present.
        """
        objmap = self._backend.toarray()
        labels = np.unique(objmap)
        return labels[labels != 0]

    def __array__(self, dtype=None, copy=None):
        """Implement the array interface for NumPy compatibility.

        This method enables NumPy functions to operate directly on ObjectMap
        instances by converting the sparse representation to a dense array.
        It allows usage patterns such as `np.sum(objmap)`, `np.max(objmap)`,
        and array unpacking operations.

        Args:
            dtype (type, optional): Optional NumPy dtype to cast the array to.
                If None, the array maintains its native dtype. Defaults to None.
            copy (bool, optional): Control whether the underlying data is copied.
                If True, a copy is guaranteed. If False (NumPy 2.0+), no copy
                is made. If None, a copy is made only if necessary for dtype
                conversion. Defaults to None.

        Returns:
            np.ndarray: A dense NumPy array representation of the object map
                with shape matching the image dimensions (height, width).

        Examples:
            .. dropdown:: Using ObjectMap with NumPy functions

                .. code-block:: python

                    import numpy as np
                    objmap = image.objmap
                    max_label = np.max(objmap)
                    unique_labels = np.unique(objmap)
                    arr = np.asarray(objmap, dtype=np.uint32)
        """
        arr = self._backend.toarray()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        elif copy:
            arr = arr.copy()
        return arr

    def __getitem__(self, key):
        """Return a slice of the object map as if it were a dense array.

        This method implements NumPy-style slicing and indexing for the ObjectMap.
        The sparse representation is converted to dense for the indexing operation,
        allowing all standard NumPy slicing patterns including integer indexing,
        boolean masking, and multi-dimensional slicing.

        Args:
            key: Index or slice specification. Can be an integer, tuple of integers,
                slice object, boolean array, or integer array for advanced indexing.
                All NumPy indexing patterns are supported.

        Returns:
            np.ndarray: The sliced portion of the object map as a dense array.
                The returned shape depends on the indexing pattern. Returns a scalar
                if a single element is indexed, otherwise returns an ndarray.

        Examples:
            .. dropdown:: NumPy-style indexing and slicing

                .. code-block:: python

                    objmap = image.objmap
                    row = objmap[5]  # Get row 5
                    region = objmap[10:20, 30:40]  # Get a rectangular region
                    pixel = objmap[5, 10]  # Get a single pixel
                    mask = objmap[objmap > 0]  # Boolean indexing
        """
        return self._backend.toarray()[key]

    def __setitem__(self, key, value):
        """Set values in the object map as if it were a dense array.

        This method implements NumPy-style assignment for the ObjectMap. It accepts
        both array and scalar values, validates shapes and types, and atomically
        updates the internal sparse representation. The operation is atomic with
        respect to the backend reference.

        Args:
            key: Index or slice specification for the location to update. Can be
                an integer, tuple of integers, slice object, boolean array, or
                integer array. All NumPy indexing patterns are supported.
            value: The value(s) to assign. Can be:
                - A NumPy array matching the shape of the key selection
                - A scalar integer, bool, or float (converted to int)

        Raises:
            ArrayKeyValueShapeMismatchError: If the shape or dtype of a supplied
                array does not match the shape of the key selection, or if the
                dtypes are incompatible.
            InvalidMapValueError: If value is not a supported type (not an array,
                int, bool, or float).

        Notes:
            - After assignment, the sparse matrix is compressed and zeros are
              eliminated to maintain memory efficiency.
            - All values are internally cast to uint16 to match the sparse matrix
              dtype and represent object labels.
            - The operation is atomic: if an exception occurs, the map is not
              modified.

        Examples:
            .. dropdown:: Assigning values to the object map

                .. code-block:: python

                    objmap = image.objmap

                    # Set a single pixel
                    objmap[5, 10] = 3

                    # Set a region with a scalar value
                    objmap[10:20, 30:40] = 0

                    # Set a region with an array
                    region_labels = np.array([[1, 1], [1, 1]])
                    objmap[10:12, 30:32] = region_labels
        """
        # Get current backend and convert to dense once
        dense = self._backend.toarray()
        backend_dtype = self._backend.dtype

        if isinstance(value, np.ndarray):  # Array case
            value = value.astype(backend_dtype)
            if dense[key].shape != value.shape:
                raise ArrayKeyValueShapeMismatchError
            elif dense.dtype != value.dtype:
                raise ArrayKeyValueShapeMismatchError

            dense[key] = value
        elif isinstance(value, (int, bool, float)):  # Scalar Case
            dense[key] = int(value)
        else:
            raise InvalidMapValueError

        # Protects against the case that the obj map is set on the filled mask that returns when no objects are in the _root_image
        # Note: removed due to confusing behavior
        # if 0 not in dense:
        #     dense = clear_border(dense, buffer_size=0, bgval=1)

        # Update backend atomically
        new_sparse = self._dense_to_sparse(dense)
        new_sparse.eliminate_zeros()  # Remove zero values to save space
        self._root_image._data.sparse_object_map = new_sparse

    @property
    def _subject_arr(self) -> np.ndarray:
        """Return the dense representation of the object map.

        This property provides access to the underlying array data by converting
        the sparse CSC matrix to a dense NumPy array. It is used internally by
        base class methods for operations requiring array-like access.

        Returns:
            np.ndarray: The dense object map with shape (height, width) and
                dtype uint16. Background pixels have value 0, object pixels
                have unique positive integer labels.
        """
        return self._backend.toarray()

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the object map.

        Returns the dimensions of the object map as a tuple of (height, width).
        This matches the shape of the parent image's grayscale representation.

        Returns:
            tuple[int, int]: A tuple (height, width) representing the spatial
                dimensions of the object map.

        Examples:
            .. dropdown:: Getting object map dimensions

                .. code-block:: python

                    height, width = image.objmap.shape
                    print(f"Object map dimensions: {width}x{height}")
        """
        return self._backend.shape

    def copy(self) -> np.ndarray:
        """Return a dense copy of the object map.

        Creates and returns a dense NumPy array copy of the object map. The
        returned array is independent and modifications to it will not affect
        the original object map.

        Returns:
            np.ndarray: A dense copy of the object map with shape (height, width)
                and dtype uint16.

        Examples:
            .. dropdown:: Creating an independent copy of the object map

                .. code-block:: python

                    objmap_copy = image.objmap.copy()
                    objmap_copy[0, 0] = 999  # Modifications don't affect the original
                    assert image.objmap[0, 0] != 999
        """
        return self._backend.toarray().copy()

    def as_csc(self) -> csc_matrix:
        """Return the object map as a compressed sparse column matrix.

        Converts the internal object map representation to CSC (Compressed Sparse
        Column) format. CSC format is optimized for column slicing and matrix
        operations.

        Returns:
            scipy.sparse.csc_matrix: A copy of the object map in CSC sparse format.

        Examples:
            .. dropdown:: Converting to CSC sparse format

                .. code-block:: python

                    sparse_csc = image.objmap.as_csc()
                    # Efficient column operations
                    first_column = sparse_csc[:, 0].toarray()
        """
        return self._backend.tocsc()

    def as_coo(self) -> coo_matrix:
        """Return the object map in COOrdinate (ijv) sparse format.

        Converts the internal object map representation to COO (Coordinate or ijv)
        format, which stores non-zero elements as (row, column, value) tuples.
        This format is useful for constructing sparse matrices or analyzing
        individual non-zero entries.

        Returns:
            scipy.sparse.coo_matrix: A copy of the object map in COO sparse format.

        Examples:
            .. dropdown:: Converting to COO sparse format and accessing entries

                .. code-block:: python

                    sparse_coo = image.objmap.as_coo()
                    # Access non-zero entries directly
                    for i, j, v in zip(sparse_coo.row, sparse_coo.col, sparse_coo.data):
                        print(f"Object label {v} at position ({i}, {j})")
        """
        return self._backend.tocoo()

    def show(
        self,
        figsize=None,
        title=None,
        cmap: str = "nipy_spectral",
        ax: None | plt.Axes = None,
        mpl_params: None | dict = None,
    ) -> (plt.Figure, plt.Axes):
        """Display the object map using matplotlib's imshow.

        This method visualizes the labeled object map using matplotlib. Each
        unique label is assigned a distinct color from the specified colormap,
        allowing visual distinction between labeled objects. The background
        (label 0) is typically shown in a neutral color.

        Args:
            figsize (tuple[int, int], optional): Tuple specifying the figure size
                in inches (width, height). If None, uses the default matplotlib
                figure size. Defaults to None.
            title (str, optional): Title text for the plot. If None, no title is
                displayed. Defaults to None.
            cmap (str, optional): The colormap name used for rendering the labeled
                object map. A diverse colormap like 'nipy_spectral' is recommended
                for clearly distinguishing between many objects. Defaults to
                'nipy_spectral'.
            ax (plt.Axes, optional): Existing Matplotlib Axes object where the
                object map will be plotted. If None, a new figure and axes are
                created. Defaults to None.
            mpl_params (dict, optional): Additional parameters passed to matplotlib's
                imshow function for plot customization (e.g., interpolation,
                normalization). If None, no extra parameters are applied.
                Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the matplotlib Figure
                and Axes objects where the object map is rendered. If an existing
                axes was provided, its parent figure is returned.

        Examples:
            .. dropdown:: Displaying the object map with various options

                .. code-block:: python

                    # Basic visualization
                    fig, ax = image.objmap.show()

                    # Custom figure size and title
                    fig, ax = image.objmap.show(figsize=(10, 8), title='Labeled Objects')

                    # Use a different colormap
                    fig, ax = image.objmap.show(cmap='tab20')

                    # Plot on an existing axes
                    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                    image.objmap.show(ax=ax[0])
                    image.gray.show(ax=ax[1])
        """
        return self._plot(
            arr=self._backend.toarray(),
            figsize=figsize,
            title=title,
            ax=ax,
            cmap=cmap,
            mpl_settings=mpl_params,
        )

    def reset(self) -> None:
        """Reset the object map to an empty state with no labeled objects.

        Clears all object labels from the object map, effectively removing all
        detected or manually-set objects. The resulting map contains only zeros
        (background). The shape of the map is preserved to match the parent
        image's dimensions.

        This method is useful when you want to clear the current segmentation
        and prepare for a new detection or labeling operation.

        Examples:
            .. dropdown:: Clearing all objects from the object map

                .. code-block:: python

                    image.objmap.reset()
                    # Now image.objmap contains no objects
                    assert image.objmap[:].max() == 0
        """
        self._root_image._data.sparse_object_map = self._dense_to_sparse(
            self._root_image.gray.shape
        )

    def relabel(self, connectivity: int = 1):
        """Relabel all connected components in the object map.

        This method reassigns labels to all connected components in the current
        object map based on the specified connectivity criterion. It uses
        scikit-image's `label` function to identify connected components and
        assigns them new sequential labels starting from 1. This is useful when:

        - Object labels are non-sequential or have gaps
        - The labeling needs to be reset to sequential labels
        - Connectivity needs to be changed (e.g., from 8-connectivity to 4-connectivity)

        The operation treats the current object map as a binary mask (foreground
        vs. background) and relabels the foreground regions.

        Args:
            connectivity (int, optional): Maximum number of orthogonal hops to
                consider a pixel as a neighbor of another. For 2D images:
                - 1: 4-connectivity (horizontal and vertical neighbors only)
                - 2: 8-connectivity (including diagonal neighbors)
                Higher values are supported for higher dimensional data.
                Defaults to 1.

        Returns:
            None: The operation modifies the object map in-place.

        Notes:
            - Background (label 0) is preserved in its original location.
            - After relabeling, object labels will be sequential (1, 2, 3, ...).
            - The relabeling process may change existing label values.

        Examples:
            .. dropdown:: Relabeling connected components

                .. code-block:: python

                    # Relabel with 4-connectivity
                    image.objmap.relabel(connectivity=1)

                    # Relabel with 8-connectivity for diagonal neighbors
                    image.objmap.relabel(connectivity=2)

        See Also:
            - `scikit-image.measure.label()`: The underlying function used for
              connected component labeling.
        """
        # Get the current mask and relabel it
        mask = self._backend.toarray() > 0
        relabeled = label(mask, connectivity=connectivity)
        self._root_image._data.sparse_object_map = self._dense_to_sparse(relabeled)

    @staticmethod
    def _dense_to_sparse(arg) -> csc_matrix:
        """Convert a dense array or shape to a compressed sparse column matrix.

        This static utility method constructs a CSC (Compressed Sparse Column)
        sparse matrix from either a dense NumPy array or a shape tuple. It is used
        internally to maintain the efficient sparse representation of the object
        map and provides a centralized point to change the underlying sparse
        format if needed.

        The resulting sparse matrix is optimized by:
        - Using uint16 dtype for label storage (supports up to 65,535 objects)
        - Eliminating zero entries to save memory

        Args:
            arg: Input for sparse matrix construction. Can be:
                - A NumPy array (dense object map): converted to sparse format
                - A tuple (height, width): creates an empty sparse matrix of this shape

        Returns:
            scipy.sparse.csc_matrix: A CSC sparse matrix with dtype uint16. Empty
                matrices have no non-zero entries, dense matrices are converted
                with zeros eliminated.

        Notes:
            - Zero values are eliminated after construction, so matrices are
              stored in a memory-efficient compressed format.
            - CSC format is efficient for column slicing and standard matrix
              operations.
            - The uint16 dtype allows labels from 0 (background) to 65,535.

        Examples:
            .. dropdown:: Converting between dense and sparse representations

                .. code-block:: python

                    # Create empty sparse matrix with specific shape
                    sparse = ObjectMap._dense_to_sparse((512, 512))

                    # Convert dense array to sparse
                    dense_map = np.array([[0, 1, 1], [0, 2, 2]])
                    sparse = ObjectMap._dense_to_sparse(dense_map)
        """
        sparse = csc_matrix(arg, dtype=np.uint16)
        sparse.eliminate_zeros()
        return sparse
