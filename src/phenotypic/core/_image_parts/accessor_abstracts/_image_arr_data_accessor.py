from abc import ABC
from pathlib import Path

import numpy
import numpy as np
import skimage

import skimage as ski
import skimage.util
import matplotlib.pyplot as plt

from phenotypic.core._image_parts.accessor_abstracts import ImageAccessorBase
from phenotypic.util.constants_ import METADATA_LABELS


class ImageArrDataAccessor(ImageAccessorBase, ABC):
    """
    Handles interaction with Image data by providing access to Image attributes and data.

    This class serves as a bridge for interacting with Image-related data structures.
    It is responsible for accessing and manipulating data associated with a parent
    Image. It includes methods to retrieve the shape of the data and to determine
    if the data is empty. The class extends the functionality of the base `ImageAccessorBase`.

    Attributes:
        image (Any): Root Image object that this accessor is linked to.
        _main_arr (Any): Main array storing the Image-related data.
        _dtype (Any): Data type of the Image data stored in the target array.
    """
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
                      *,
                      overlay_settings: dict | None = None,
                      mpl_settins: dict | None = None,
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
            overlay_settings (dict | None, optional): Parameters passed to the
                `skimage.color.label2rgb` function for overlay customization.
                Defaults to None.
            mpl_settins (dict | None, optional): Additional parameters for the
                `ax.imshow` Matplotlib function to control image rendering.
                Defaults to None.

        Returns:
            tuple[plt.Figure, plt.Axes]: The Matplotlib Figure and Axes objects used for
            the display. If an existing Axes is provided, its corresponding Figure is returned.
        """
        overlay_settings = overlay_settings if overlay_settings else {}
        overlay_alpha = overlay_settings.get('alpha', 0.2)
        overlay_arr = skimage.color.label2rgb(label=objmap, image=arr, bg_label=0, alpha=overlay_alpha, **overlay_settings)

        fig, ax = self._plot(arr=overlay_arr, figsize=figsize, title=title, cmap=cmap, ax=ax, mpl_kwargs=mpl_settins)

        return fig, ax

    def foreground(self):
        foreground = self[:].copy()
        foreground[self._root_image.objmask[:] == 0] = 0
        return foreground


    def imsave(self, fname: str | Path):
        fname = Path(fname)
        arr = self._subject_arr.copy()
        if (arr.dtype!=np.uint8) or (arr.dtype!=np.uint16):
            match self._root_image.metadata[METADATA_LABELS.BIT_DEPTH]:
                case 8:
                    arr = ski.util.img_as_ubyte(arr)
                case 16:
                    arr = ski.util.img_as_uint(arr)
                case _:
                    raise AttributeError(f"Unsupported bit depth: {self._root_image.metadata[METADATA_LABELS.BIT_DEPTH]}")
        ski.io.imsave(fname=fname, arr=arr, check_contrast=False)


    def show_overlay(self, object_label: None | int = None,
                     figsize: tuple[int, int] | None = None,
                     title: str | None = None,
                     show_labels: bool = False,
                     label_settings: None | dict = None,
                     ax: plt.Axes = None,
                     *,
                     overlay_settings: None | dict = None,
                     imshow_settings: None | dict = None,
                     ) -> tuple[plt.Figure, plt.Axes]:
        """
        Displays an overlay of the object map on the parent image with optional annotations.

        This method enables visualization by overlaying object regions on the parent image. It
        provides options for customization, including the ability to show_labels specific objects
        and adjust visual styles like figure size, colors, and annotation properties.

        Args:
            object_label (None | int): Specific object label to be highlighted. If None,
                all objects are displayed.
            figsize (tuple[int, int]): Size of the figure in inches (width, height).
            title (None | str): Title for the plot. If None, the parent image's name
                is used.
            show_labels (bool): If True, displays annotations for object labels on the
                object centroids.
            label_settings (None | dict): Additional parameters for customization of the
                object annotations. Defaults: size=12, color='white', facecolor='red'. Other kwargs
                are passed to the matplotlib.axes.text () method.
            ax (plt.Axes): Optional Matplotlib Axes object. If None, a new Axes is
                created.
            overlay_settings (None | dict): Additional parameters for customization of the
                overlay.
            imshow_settings (None|dict): Additional Matplotlib imshow configuration parameters
                for customization. If None, default Matplotlib settings will apply.

        Returns:
            tuple[plt.Figure, plt.Axes]: Matplotlib Figure and Axes objects containing
            the generated plot.

        """
        objmap = self._root_image.objmap[:]
        if object_label is not None: objmap[objmap != object_label] = 0
        if label_settings is None: label_settings = {}

        fig, ax = self._plot_overlay(
            arr=self._subject_arr,
            objmap=objmap,
            ax=ax,
            figsize=figsize,
            title=title,
            mpl_settins=imshow_settings,
            overlay_settings=overlay_settings,
        )

        if show_labels:
            ax = self._plot_obj_labels(
                ax=ax,
                color=label_settings.get('color', 'white'),
                size=label_settings.get('size', 12),
                facecolor=label_settings.get('facecolor', 'red'),
                object_label=object_label,
            )
        return fig, ax
