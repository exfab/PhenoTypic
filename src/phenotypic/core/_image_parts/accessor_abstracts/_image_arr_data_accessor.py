import numpy as np

import skimage.util
import matplotlib.pyplot as plt

from phenotypic.core._image_parts.accessor_abstracts import ImageAccessor


class ImageArrDataAccessor(ImageAccessor):
    """
    Handles interaction with Image data by providing access to Image attributes and data.

    This class serves as a bridge for interacting with Image-related data structures.
    It is responsible for accessing and manipulating data associated with a parent
    Image. It includes methods to retrieve the shape of the data and to determine
    if the data is empty. The class extends the functionality of the base `ImageAccessor`.

    Attributes:
        image (Any): Root Image object that this accessor is linked to.
        _main_arr (Any): Main array storing the Image-related data.
        _dtype (Any): Data type of the Image data stored in the target array.
    """

    def __init__(self, parent_image):
        self._root_image = parent_image

    # def shape(self) -> tuple[int, ...]:
    #     return self._main_arr.shape

    def isempty(self):
        return True if self.shape[0] == 0 else False

    def copy(self) -> np.ndarray:
        """
        Returns a copy of the array/matrix from the image.

        This method retrieves a copy of the image matrix, ensuring
        that modifications to the returned matrix do not affect the original
        data in the image's matrix.

        Returns:
            np.ndarray: A deep copy of the image matrix.
        """
        return self[:].copy()


    def _plot_overlay(self,
                      arr: np.ndarray,
                      objmap: np.ndarray,
                      figsize: (int, int) = (8, 6),
                      title: str | bool | None = None,
                      cmap: str = 'gray',
                      ax: plt.Axes = None,
                      overlay_params: dict | None = None,
                      mpl_kwargs: dict | None = None,
                      ) -> (plt.Figure, plt.Axes):
        """
        Plots an array with optional object map overlay and customization options.

        Note:
            - If ax is None, a new figure and axes are created.

        Args:
            arr (np.ndarray): The primary array to be displayed as an image.
            objmap (np.ndarray, optional): An array containing labels for an object map to
                overlay on top of the image. Defaults to None.
            figsize (tuple[int, int], optional): The size of the figure as a tuple of
                (width, height). Defaults to (8, 6).
            title (str, optional): Title of the plot to be displayed. If not provided,
                defaults to the name of the self.image.
            cmap (str, optional): Colormap to apply to the image. Defaults to 'gray'. Only used if arr input_image is 2D.
            ax (plt.Axes, optional): An existing Matplotlib Axes instance for rendering
                the image. If None, a new figure and axes are created. Defaults to None.
            overlay_params (dict | None, optional): Parameters passed to the
                `skimage.color.label2rgb` function for overlay customization.
                Defaults to None.
            mpl_kwargs (dict | None, optional): Additional parameters for the
                `ax.imshow` Matplotlib function to control image rendering.
                Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: The Matplotlib Figure and Axes objects used for
            the display. If an existing Axes is provided, its corresponding Figure is returned.
        """
        overlay_params = overlay_params if overlay_params else {}
        overlay_alpha = overlay_params.get('alpha', 0.2)
        overlay_arr = skimage.color.label2rgb(label=objmap, image=arr, bg_label=0, alpha=overlay_alpha, **overlay_params)

        fig, ax = self._plot(arr=overlay_arr, figsize=figsize, title=title, cmap=cmap, ax=ax, mpl_kwargs=mpl_kwargs)

        return fig, ax

    def get_foreground(self):
        foreground = self[:].copy()
        foreground[self._root_image.objmask[:] == 0] = 0
        return foreground