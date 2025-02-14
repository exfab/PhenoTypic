import numpy as np

from scipy.sparse import csc_matrix, coo_matrix
import matplotlib.pyplot as plt

from ...util.constants import ConstObjectMap


class ObjectMap:
    """ObjectMap stores an array with the pixel coordinates of each object and syncs changes with the ObjectMap class.

    Note:
        Changes to the linked ObjectMask array will reset the labeling of the ObjectMap objects.

    """

    def __init__(self, handler, preset_map: np.ndarray = None):
        """

        Args:
            handler: (ImageHandler) The image handler that the ObjectMap belongs to.
            preset_map: (np.ndarray[np.uint32]) Allows a way to easily create new Map components by allowing preset_map
        """
        self._handler = handler
        if preset_map is None:
            self._sparse = csc_matrix(arg1=self._handler.matrix.shape, dtype=np.uint32)
        else:
            if preset_map.shape != self._handler.matrix.shape: raise ConstObjectMap.InputShapeMismatchError('preset_map')
            else:
                self._sparse = csc_matrix(arg1=preset_map, dtype=np.uint32)

    def __getitem__(self, key):
        """Returns a copy of the object_map of the image. If there are no objects, this is a matrix with all values set to 1 and the same shape as the iamge matrix."""
        if self.num_objects > 0:
            return self._sparse.toarray()[key]
        elif self.num_objects == 0:
            return np.full(self._sparse.shape, fill_value=1, dtype=np.uint32)
        else:
            raise RuntimeError(ConstObjectMap.UnknownError)

    def __setitem__(self, key, value):
        """Uncompresses the csc array & changes the values at the specified coordinates before recompressing the object map array."""
        dense = self._sparse.toarray()

        if dense[key].shape != self._sparse.shape: raise ConstObjectMap.ArrayKeyValueShapeMismatchError()
        dense[key] = value
        self._sparse = self._dense_to_sparse(dense)

    @property
    def shape(self) -> tuple[int, int]:
        return self._sparse.shape

    def to_csc(self) -> csc_matrix:
        """Returns a copy of the object map as a compressed sparse column matrix"""
        return self._sparse.tocsc()

    def to_coo(self) -> coo_matrix:
        """Returns a copy of the object map in COOrdinate format or ijv matrix"""
        return self._sparse.tocoo()

    def reset(self) -> None:
        """Resets the object_map to an empty map array with no objects in it."""
        self._sparse = csc_matrix(arg1=self._handler.matrix.shape, dtype=np.uint32)

    @property
    def num_objects(self) -> int:
        """Returns the number of objects in the map."""
        return len(self.labels)

    @property
    def labels(self) -> np.ndarray:
        """Returns a sorted list of the different object labels in the map."""
        return np.unique(self._sparse.data)

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

        ax.imshow(self._sparse.toarray(), cmap=cmap)
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax

    @staticmethod
    def _dense_to_sparse(object_map: np.ndarray) -> csc_matrix:
        return csc_matrix(object_map, dtype=np.uint32)
