import numpy as np

from scipy.sparse import csc_matrix, coo_matrix
import matplotlib.pyplot as plt
from skimage.measure import regionprops_table, label

from ...util.constants import C_ObjectMap


class ObjectMap:
    """ObjectMap stores an array with the pixel coordinates of each object and syncs changes with the ObjectMap class.

    Note:
        Changes to the linked ObjectMask array will reset the labeling of the ObjectMap objects.

    """

    def __init__(self, handler):
        """

        Args:
            handler: (ImageHandler) The image handler that the ObjectMap belongs to.
            preset_map: (np.ndarray[np.uint32]) Allows a way to easily create new Map components by allowing preset_map
        """
        self._handler = handler

    @property
    def _num_objects(self):
        return len(self._labels)

    @property
    def _labels(self):
        """Returns the labels in the image.

               We considered using a simple numpy.unique() call on the object map, but wanted to guarantee that the labels will always be consistent
               with any skimage version outputs.

               """
        return regionprops_table(label_image=self._handler._sparse_object_map.toarray(), properties=['label'], cache=False)['label']

    def __getitem__(self, key):
        """Returns a copy of the object_map of the image. If there are no objects, this is a matrix with all values set to 1 and the same shape as the iamge matrix."""
        if self._num_objects > 0:
            return self._handler._sparse_object_map.toarray()[key]
        elif self._num_objects == 0:
            return np.full(self._handler._sparse_object_map.toarray()[key].shape, fill_value=1, dtype=np.uint32)
        else:
            raise RuntimeError(C_ObjectMap.UnknownError)

    def __setitem__(self, key, value):
        """Uncompresses the csc array & changes the values at the specified coordinates before recompressing the object map array."""
        dense = self._handler._sparse_object_map.toarray()

        if type(value) == np.ndarray:
            value = value.astype(self._handler._sparse_object_map.dtype)
            if dense[key].shape != value.shape:
                raise C_ObjectMap.ArrayKeyValueShapeMismatchError
            elif dense.dtype != value.dtype:
                raise C_ObjectMap.ArrayKeyValueShapeMismatchError

            dense[key] = value
        elif type(value) == int:
            dense[key] = value
        else:
            raise C_ObjectMap.InvalidValueTypeError

        self._handler._sparse_object_map = self._dense_to_sparse(dense)

    @property
    def shape(self) -> tuple[int, int]:
        return self._handler._sparse_object_map.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the object_map."""
        return self._handler._sparse_object_map.toarray().copy()

    def to_csc(self) -> csc_matrix:
        """Returns a copy of the object map as a compressed sparse column matrix"""
        return self._handler._sparse_object_map.tocsc()

    def to_coo(self) -> coo_matrix:
        """Returns a copy of the object map in COOrdinate format or ijv matrix"""
        return self._handler._sparse_object_map.tocoo()

    def show(self, ax=None, figsize=None, cmap='gray', title=None) -> (plt.Figure, plt.Axes):
        """Display the object_map with matplotlib.

        Args:
            ax: (plt.Axes) Axes object to use for plotting.
            figsize: (Tuple[int, int]): Figure size in inches.
            cmap: (str, optional) Colormap to use.
            title: (str) a title for the plot

        Returns:
            tuple(plt.Figure, plt.Axes): matplotlib figure and axes object
        """
        if figsize is None: figsize = (6, 4)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        ax.imshow(self._handler._sparse_object_map.toarray(), cmap=cmap)
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax

    def reset(self) -> None:
        """Resets the object_map to an empty map array with no objects in it."""
        if self._handler.isempty():
            self._handler._sparse_object_map = None
        else:
            self._handler._sparse_object_map = self._dense_to_sparse(self._handler.matrix.shape)

    def relabel(self):
        """Relables all the objects based on their connectivity"""
        self._dense_to_sparse(label(self._handler.omask[:]))


    @staticmethod
    def _dense_to_sparse(arg) -> csc_matrix:
        """Constructs a sparse array from the arg parameter. Used so that the underlying sparse matrix can be changed

        Args:
            arg: either the dense object array or the shape

        Returns:

        """
        sparse = csc_matrix(arg, dtype=np.uint32)
        sparse.eliminate_zeros()
        return sparse
