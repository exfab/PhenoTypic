import numpy as np

import warnings

import skimage.util
import matplotlib.pyplot as plt

from phenotypic.core.accessors import ImageAccessor
from phenotypic.util.exceptions_ import InterfaceError


class ImageArrDataAccessor(ImageAccessor):
    """
    Handles interaction with Image data by providing access to Image attributes and data.

    This class serves as a bridge for interacting with Image-related data structures.
    It is responsible for accessing and manipulating data associated with a parent
    Image. It includes methods to retrieve the shape of the data and to determine
    if the data is empty. The class extends the functionality of the base `ImageAccessor`.

    Attributes:
        _root_image (Any): Root Image object that this accessor is linked to.
        _main_arr (Any): Main array storing the Image-related data.
        _dtype (Any): Data type of the Image data stored in the target array.
    """

    def __init__(self, parent_image, target_array, dtype):
        self._root_image = parent_image
        self._main_arr = target_array
        self._dtype = dtype

    def shape(self) -> tuple[int, ...]:
        return self._main_arr.shape

    def isempty(self):
        return True if self.shape[0] == 0 else False

    def _norm2dtype(self, normalized_value: np.ndarray) -> np.ndarray:
        """
        Converts a normalized matrix with values between 0 and 1 to a specified data type with the
        appropriate scaling. The method ensures that all values are clipped to the range [0, 1]
        before scaling them to the data type's maximum other_image.

        Args:
            normalized_value: A 2D NumPy array where all values are assumed to be in the range
                [0, 1]. These values will be converted using the specified data type scale.

        Returns:
            numpy.ndarray: A 2D NumPy array of the same shape as `normalized_matrix`, converted
            to the target data type with scaled values.
        """
        match self._dtype:
            case np.uint8:
                return skimage.util.img_as_ubyte(normalized_value)
            case np.uint16:
                return skimage.util.img_as_uint(normalized_value)
            case _:
                raise AttributeError(f'Unsupported dtype {self._dtype} for matrix storage conversion')

    def _plot_overlay(self,
                      arr: np.ndarray,
                      objmap: np.ndarray,
                      figsize: (int, int) = (8, 6),
                      title: str = None,
                      cmap: str = 'gray',
                      ax: plt.Axes = None,
                      overlay_params: dict | None = None,
                      imshow_params: dict | None = None,
                      ) -> (plt.Figure, plt.Axes):
        """
        Plots an array with optional object map overlay and customization options.

        Note:
            - If ax is None, a new figure and axes are created.

        Args:
            arr (np.ndarray): The primary array to be displayed as an _root_image.
            objmap (np.ndarray, optional): An array containing labels for an object map to
                overlay on top of the _root_image. Defaults to None.
            figsize (tuple[int, int], optional): The size of the figure as a tuple of
                (width, height). Defaults to (8, 6).
            title (str, optional): Title of the plot to be displayed. If not provided,
                defaults to the name of the self._root_image.
            cmap (str, optional): Colormap to apply to the _root_image. Defaults to 'gray'. Only used if arr input_image is 2D.
            ax (plt.Axes, optional): An existing Matplotlib Axes instance for rendering
                the _root_image. If None, a new figure and axes are created. Defaults to None.
            overlay_params (dict | None, optional): Parameters passed to the
                `skimage.color.label2rgb` function for overlay customization.
                Defaults to None.
            imshow_params (dict | None, optional): Additional parameters for the
                `ax.imshow` Matplotlib function to control _root_image rendering.
                Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: The Matplotlib Figure and Axes objects used for
            the display. If an existing Axes is provided, its corresponding Figure is returned.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        overlay_params = overlay_params if overlay_params else {}

        imshow_params = imshow_params if imshow_params else {}
        cmap = imshow_params.get('cmap', cmap)

        overlay_alpha = overlay_params.get('alpha', 0.2)
        imarray = skimage.color.label2rgb(label=objmap, image=arr, bg_label=0, alpha=overlay_alpha, **overlay_params)
        ax.imshow(imarray, cmap=cmap, **imshow_params) if imarray.ndim == 2 else ax.imshow(imarray, **imshow_params)

        ax.grid(False)
        if title: ax.set_title(title)

        return fig, ax

    def _dtype2norm(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalizes the given matrix to have values between 0.0 and 1.0 based on its data type.

        The method checks the data type of the input matrix against the expected data
        type. If the data type does not match, a warning is issued. The matrix is
        then normalized by dividing its values by the maximum possible other_image for its
        data type, ensuring all elements remain within the range of [0.0, 1.0].

        Args:
            matrix (np.ndarray): The input matrix to be normalized.

        Returns:
            np.ndarray: A normalized matrix where all values are within [0.0, 1.0].
        """
        return skimage.util.img_as_float(matrix)

    def get_foreground(self):
        foreground = self[:].copy()
        foreground[self._root_image.objmask[:] == 0] = 0
        return foreground
