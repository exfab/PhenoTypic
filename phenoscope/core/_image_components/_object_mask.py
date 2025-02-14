from skimage.measure import label
import matplotlib.pyplot as plt
import numpy as np

from ...util.constants import ConstObjectMask


class ObjectMask:
    """Class provides access to the object map as a boolean array form and creates a way for binary operations to alter objects.


    Note:
        Changes to the ObjectMask array will reset the labeling of the ObjectMap objects.
    """

    def __init__(self, image_handler):
        """Initiallizes the ObjectMask object.

        Args:
            image_handler: (ImageHandler) The parent ImageHandler that the ObjectMask belongs to.
        """
        self._handler = image_handler

    def __getitem__(self, key):
        """Returns a copy of the binary object mask in array form"""
        return self._handler.object_map[key] > 0

    def __setitem__(self, key, value:np.ndarray):
        """Sets values of the object mask to value and resets the labeling in the map"""
        mask = self._handler.object_map[:] > 0

        # Check to make sure the section of the mask the key accesses is the same as the value
        if mask[key].shape != value.shape: raise ConstObjectMask.ARRAY_KEY_VALUE_SHAPE_MISMATCH

        # Sets the section of the binary mask to the shown value
        mask[key] = (value > 0).astype(np.bool_)

        # Relabel the mask and set the underlying csc matrix to the new mask
        # Where the reset of labeling occurs. May eventually add way to sync without label reset in future
        self._handler.object_map[:] = label(mask)

    @property
    def shape(self):
        return self._handler.object_map.shape

    def show(self, ax: plt.Axes = None,
             figsize: str = None,
             cmap: str = 'tab20',
             title: str = None
             ) -> (plt.Figure, plt.Axes):
        """Display the boolean object mask with matplotlib.

        Calls object_map linked by the image handler

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
            fig = ax.get_figure()

        ax.imshow(self._handler.object_map[:] > 0, cmap=cmap)
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax
