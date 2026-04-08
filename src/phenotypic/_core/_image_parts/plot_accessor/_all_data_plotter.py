"""All data visualization plotter for PhenoTypic images."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np

from phenotypic.tools_.register import register_plotter

from ._base_plotter import BasePlotter

if TYPE_CHECKING:
    pass


@register_plotter
class AllDataPlotter(BasePlotter):
    """Provides comprehensive multi-panel visualization of all image data layers.

    This class offers a single method to display multiple representations of image
    data in one figure, useful for inspecting the effects of preprocessing and
    detection operations on microbe colony images.

    Examples:
        View all data layers:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> fig, axes = image.plot.all()
        >>> plt.close(fig)  # Important: free memory

        View with overlay instead of object map:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> fig, axes = image.plot.all(mode="overlay")
        >>> plt.close(fig)
    """

    call_name = "all"

    def all(
            self,
            mode: Literal["objmap", "overlay"] = "objmap",
            figsize: Tuple[int, int] | None = (12, 8),
            **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """Display all data representations in a single figure.

        Creates a multi-panel figure showing RGB (if available), grayscale,
        detection matrix, and either object map or overlay visualization.
        This helps analyze microbe colonies cultured on solid media agar by
        visualizing various properties such as grayscale images, processed
        (enhanced) images, object maps, or overlays.

        Args:
            mode: Defines the type of data to display in the final subplot.
                - "objmap": Displays the object map, which highlights detected
                  regions of interest such as individual colonies.
                - "overlay": Adds an overlay of detected features on top of the
                  base image, helping to contextualize detected objects.
            figsize: Dimensions of the rendered figure (width, height) in inches.
                Larger values yield a more spread-out figure enhancing visibility
                of fine details. Defaults to (12, 8).
            **kwargs: Additional keyword arguments passed to the underlying
                visualization methods (e.g., colormap, scaling settings).

        Returns:
            Tuple containing:
                - The matplotlib Figure instance
                - The axes array (ndarray) for further customization

        Examples:
            >>> from phenotypic.data import load_synth_yeast_plate
            >>> image = load_synth_yeast_plate()
            >>> fig, axes = image.plot.all()
            >>> plt.close(fig)  # Free memory after use
        """
        if self._root_image.rgb.isempty():
            fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize)
            ax = axes.ravel()
            idxer_helper = 1
        else:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize)
            idxer_helper = 0
            ax = axes.ravel()
            self._root_image.rgb._mpl_plot(
                    arr=self._root_image.rgb[:], ax=ax[0], **kwargs)

        self._root_image.gray._mpl_plot(
                arr=self._root_image.gray[:], ax=ax[1 - idxer_helper], **kwargs)
        self._root_image.detect_mat._mpl_plot(
                arr=self._root_image.detect_mat[:], ax=ax[2 - idxer_helper], **kwargs)

        match mode:
            case "overlay":
                import skimage as ski
                base = (self._root_image.rgb[:]
                        if not self._root_image.rgb.isempty()
                        else self._root_image.gray[:])
                overlay_arr = ski.color.label2rgb(
                        label=self._root_image.objmap[:], image=base,
                        bg_label=0, alpha=0.15)
                self._root_image.rgb._mpl_plot(
                        arr=overlay_arr, ax=ax[3 - idxer_helper], **kwargs)
            case "objmap":
                self._root_image.objmap._mpl_plot(
                        arr=np.ma.masked_equal(self._root_image.objmap[:], value=0),
                        ax=ax[3 - idxer_helper], **kwargs)

        return fig, axes


__all__ = ["AllDataPlotter"]
