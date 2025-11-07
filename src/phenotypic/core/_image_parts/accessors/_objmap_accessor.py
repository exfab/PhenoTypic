from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: pass

import numpy as np

from scipy.sparse import csc_matrix, coo_matrix
import matplotlib.pyplot as plt
from skimage.measure import label

from phenotypic.core._image_parts.accessor_abstracts import SingleChannelAccessor
from phenotypic.tools.exceptions_ import ArrayKeyValueShapeMismatchError, InvalidMapValueError


class ObjectMap(SingleChannelAccessor):
    """Manages an object map for labeled regions in an image.

    This class provides a mechanism to manipulate and access labeled object maps
    within a given image. It is tightly coupled with the parent image object and
    provides methods for accessing sparse and dense representations, relabeling,
    resetting, and visualization.

    Note: changes to the object map shapes will be automatically reflected in the object mask

    """

    @property
    def _backend(self):
        """Returns the current sparse backend reference.
        
        This ensures we always access the live reference to the sparse object map,
        even if it's been replaced by another operation.
        """
        return self._root_image._data.sparse_object_map

    @property
    def _num_objects(self):
        return len(self._labels)

    @property
    def _labels(self):
        """Returns the labels in the image."""
        objmap = self._backend.toarray()
        labels = np.unique(objmap)
        return labels[labels != 0]

    def __array__(self, dtype=None, copy=None):
        """Implements the array interface for numpy compatibility.
        
        This allows numpy functions to operate on the dense form of the matrix.
        For example: np.sum(objmap), np.max(objmap), etc.
        
        Args:
            dtype: Optional dtype to cast the array to
            copy: Optional copy parameter for NumPy 2.0+ compatibility
            
        Returns:
            Dense numpy array representation of the object map
        """
        arr = self._backend.toarray()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False if copy is None else copy)
        elif copy:
            arr = arr.copy()
        return arr

    def __getitem__(self, key):
        """Returns a slice of the object_map as if it were a dense array.
        
        The slicing behavior matches numpy arrays, converting the sparse
        representation to dense for the operation.
        """
        return self._backend.toarray()[key]

    def __setitem__(self, key, value):
        """Sets values in the object map as if it were a dense array.
        
        Converts to dense, applies the update, then converts back to sparse.
        The operation is atomic with respect to the backend reference.
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
        return self._backend.toarray()

    @property
    def shape(self) -> tuple[int, int]:
        return self._backend.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the object_map."""
        return self._backend.toarray().copy()

    def as_csc(self) -> csc_matrix:
        """Returns a copy of the object map as a compressed sparse column matrix"""
        return self._backend.tocsc()

    def as_coo(self) -> coo_matrix:
        """Returns a copy of the object map in COOrdinate format or ijv matrix"""
        return self._backend.tocoo()

    def show(self, figsize=None, title=None, cmap: str = 'tab20', ax: None | plt.Axes = None,
             mpl_params: None | dict = None) -> (
            plt.Figure, plt.Axes
    ):
        """
        Displays the object map using matplotlib's imshow.

        This method visualizes the object map from the parent image instance.
        It offers customization options, including figure size, title, colormap, and matplotlib
        parameters, leveraging matplotlib's plotting capabilities.

        Args:
            figsize (tuple, optional): Tuple specifying the figure size in inches (width, height).
                If None, defaults to (6, 4).
            title (str, optional): Title text for the plot. If None, no title is displayed.
            cmap (str, optional): The colormap name used for rendering the sparse object map.
                Defaults to 'tab20'.
            ax (plt.Axes, optional): Existing Axes where the sparse object map will be plotted.
                If None, a new figure and axes are created.
            mpl_params (dict, optional): Additional parameters for matplotlib. If None, no extra
                parameters are applied.

        Returns:
            tuple: A tuple containing the matplotlib Figure and Axes objects, where the
                sparse object map is rendered.
        """
        return self._plot(arr=self._backend.toarray(),
                          figsize=figsize, title=title, ax=ax, cmap=cmap, mpl_settings=mpl_params,
                          )

    def reset(self) -> None:
        """Resets the object_map to an empty map array with no objects in it."""
        self._root_image._data.sparse_object_map = self._dense_to_sparse(self._root_image.gray.shape)

    def relabel(self, connectivity: int = 1):
        """Relabels all the objects based on their connectivity.
        
        This method relabels the object map using scikit-image's label function,
        ensuring all connected components get unique labels.
        
        Args:
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel 
                         as a neighbor. Accepted values are 1 or 2 for 2D.
        """
        # Get the current mask and relabel it
        mask = self._backend.toarray() > 0
        relabeled = label(mask, connectivity=connectivity)
        self._root_image._data.sparse_object_map = self._dense_to_sparse(relabeled)

    @staticmethod
    def _dense_to_sparse(arg) -> csc_matrix:
        """Constructs a sparse array from the arg parameter. Used so that the underlying sparse matrix can be changed

        Args:
            arg: either the dense object array or the shape

        Returns:

        """
        sparse = csc_matrix(arg, dtype=np.uint16)
        sparse.eliminate_zeros()
        return sparse
