from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: pass

import numpy as np

from scipy.sparse import csc_matrix, coo_matrix
import matplotlib.pyplot as plt
from skimage.measure import label

from phenotypic.core._image_parts.accessor_abstracts import ImageArrDataAccessor
from phenotypic.util.exceptions_ import ArrayKeyValueShapeMismatchError, InvalidMapValueError


class ObjectMap(ImageArrDataAccessor):
    """Manages an object map for labeled regions in an image.

    This class provides a mechanism to manipulate and access labeled object maps
    within a given image. It is tightly coupled with the parent image object and
    provides methods for accessing sparse and dense representations, relabeling,
    resetting, and visualization.

    Note: changes to the object map shapes will be automatically reflected in the object mask

    """

    @property
    def _num_objects(self):
        return len(self._labels)

    @property
    def _labels(self):
        """Returns the labels in the image."""
        objmap = self._root_image._data.sparse_object_map.toarray()
        labels = np.unique(objmap)
        return labels[labels != 0]

    def __getitem__(self, key):
        """Returns a copy of the object_map of the image."""
        return self._root_image._data.sparse_object_map.toarray()[key]

    def __setitem__(self, key, value):
        """Uncompresses the csc array & changes the values at the specified coordinates before recompressing the object map array."""
        dense = self._root_image._data.sparse_object_map.toarray()

        if isinstance(value, np.ndarray):  # Array case
            value = value.astype(self._root_image._data.sparse_object_map.dtype)
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

        self._root_image._data.sparse_object_map = self._dense_to_sparse(dense)
        self._root_image._data.sparse_object_map.eliminate_zeros()  # Remove zero values to save space

    @property
    def shape(self) -> tuple[int, int]:
        return self._root_image._data.sparse_object_map.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the object_map."""
        return self._root_image._data.sparse_object_map.toarray().copy()

    def as_csc(self) -> csc_matrix:
        """Returns a copy of the object map as a compressed sparse column matrix"""
        return self._root_image._data.sparse_object_map.tocsc()

    def as_coo(self) -> coo_matrix:
        """Returns a copy of the object map in COOrdinate format or ijv matrix"""
        return self._root_image._data.sparse_object_map.tocoo()

    def show(self, figsize=None, title=None, cmap: str = 'tab20', ax: None | plt.Axes = None, mpl_params: None | dict = None) -> (
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
        return self._plot(arr=self._root_image._data.sparse_object_map.toarray(),
                          figsize=figsize, title=title, ax=ax, cmap=cmap, mpl_kwargs=mpl_params,
                          )

    def reset(self) -> None:
        """Resets the object_map to an empty map array with no objects in it."""
        self._root_image._data.sparse_object_map = self._dense_to_sparse(self._root_image.matrix.shape)

    def relabel(self, connectivity: int = 1):
        """Relabels all the objects based on their connectivity"""
        self._root_image._data.sparse_object_map = self._dense_to_sparse(label(self._root_image.objmask[:], connectivity=connectivity))

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
