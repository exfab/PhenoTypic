from skimage.measure import label
import matplotlib.pyplot as plt
import numpy as np

from ...util.constants import C_ObjectMask


class ObjectMaskSubhandler:
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
        return (self._handler.obj_map[key] > 0).astype(np.bool_)

    def __setitem__(self, key, value: np.ndarray):
        """Sets values of the object mask to value and resets the labeling in the map"""
        mask = self._handler.obj_map[:] > 0

        # Check to make sure the section of the mask the key accesses is the same as the value
        if type(value) in [int, bool]:
            try:
                value = bool(value)
                mask[key] = value
            except TypeError:
                raise C_ObjectMask.InvalidScalarValueError
        elif type(value) == np.ndarray:
            # Check input and section have matching shape
            if mask[key].shape != value.shape:
                raise C_ObjectMask.ArrayKeyValueShapeMismatchError

            # Sets the section of the binary mask to the value array
            mask[key] = (value > 0)
        else:
            raise C_ObjectMask.InvalidValueTypeError(type(value))

        # Relabel the mask and set the underlying csc matrix to the new mask
        # Where the reset of labeling occurs. May eventually add way to sync without label reset in future
        self._handler.obj_map[:] = label(mask)

    @property
    def shape(self):
        return self._handler.obj_map.shape

    def copy(self) -> np.ndarray:
        """Returns a copy of the binary object mask"""
        return self._handler.obj_mask[:].copy()

    def reset(self):
        self._handler.obj_map.reset()

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

        ax.imshow(self._handler.obj_map[:] > 0, cmap=cmap)
        if title is not None: ax.set_title(title)
        ax.grid(False)

        return fig, ax

    def _extract_objects(self, array: np.ndarray, bg_color: int = 0):
        """Returns the array with every non-object pixel set to 0. Equivalent to np.ma.array.filled(bg_color)"""
        if array.ndim == 3:
            mask = np.dstack(
                [(self._handler.obj_map[:] > 0) for _ in range(array.shape[-1])]
            )
        else: mask = self._handler.obj_map[:] > 0
        new_arr = array.copy()
        new_arr[~mask] = bg_color
        return new_arr
