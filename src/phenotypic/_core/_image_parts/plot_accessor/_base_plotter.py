from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import ImageAccessorBase

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class BasePlotter(ImageAccessorBase):
    """Base class for all plotter accessors providing common plotting functionality.

    This abstract base class defines the interface and common methods for all
    plotting accessors in PhenoTypic. It provides utilities for memory management,
    figure handling, and common plotting patterns.
    """

    def __init__(self, root_image: Image) -> None:
        """Bind the call-time image without extending its lifetime."""
        self._root_image_ref = weakref.ref(root_image)

    @property
    def _root_image(self) -> Image:
        """Return the live image or fail after its caller releases it."""
        image = self._root_image_ref()
        if image is None:
            raise RuntimeError(
                "The image bound to this plotter has been released. Keep the "
                "image alive while interacting with its report."
            )
        return image

    @property
    def _accessor_property_name(self) -> str:
        """Name of the Image property that surfaces this accessor."""
        return "plot"

    @property
    def _subject_arr(self) -> np.ndarray:
        """Return the grayscale image for plotting."""
        return self._root_image.gray[:]

    def _cleanup_figure(self, fig: plt.Figure) -> None:
        """Clean up matplotlib figure to prevent memory leaks.

        Args:
            fig: Matplotlib figure to close
        """
        plt.close(fig)

    def _validate_objects_exist(self, use_binary: bool = False) -> None:
        """Validate that objects exist for plotting.

        Args:
            use_binary: If True, check objmask instead of objmap

        Raises:
            ValueError: If no objects are detected
        """
        if use_binary or self._root_image.num_objects == 0:
            mask = self._root_image.objmask[:].astype(bool)
            if not mask.any():
                raise ValueError("No objects detected. Apply an ObjectDetector first.")
        else:
            objmap = self._root_image.objmap[:]
            if objmap.max() == 0:
                raise ValueError("No labeled objects. Apply an ObjectDetector first.")

    def _get_mask_for_plotting(self, use_binary: bool = False) -> np.ndarray:
        """Get appropriate mask for plotting operations.

        Args:
            use_binary: If True, use objmask; otherwise use objmap converted to binary

        Returns:
            Binary mask array
        """
        if use_binary or self._root_image.num_objects == 0:
            return self._root_image.objmask[:].astype(bool)
        else:
            objmap = self._root_image.objmap[:]
            return (objmap > 0).astype(bool)

    def _create_colormap(self, n_colors: int, cmap_name: str = "tab10") -> np.ndarray:
        """Create a colormap array for overlay colors.

        Args:
            n_colors: Number of colors needed
            cmap_name: Name of matplotlib colormap

        Returns:
            Array of RGBA colors
        """
        cmap = plt.get_cmap(cmap_name)
        colors = []
        for i in range(n_colors):
            # Avoid division by zero
            if n_colors > 1:
                color = cmap(i / (n_colors - 1))
            else:
                color = cmap(0)
            colors.append(color)
        return np.array(colors)

    def _safe_divide(self, numerator: np.ndarray, denominator: float) -> np.ndarray:
        """Safely divide array by scalar, avoiding division by zero.

        Args:
            numerator: Array to divide
            denominator: Scalar divisor

        Returns:
            Result of division, with zeros where denominator would be zero
        """
        if denominator == 0:
            return np.zeros_like(numerator)
        return numerator / denominator

    def _validate_figsize(self, figsize: tuple | None) -> None:
        """Validate figure size parameter.

        Args:
            figsize: Tuple of (width, height) in inches or None

        Raises:
            ValueError: If figsize is invalid
        """
        if figsize is None:
            return
        if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
            raise ValueError("figsize must be a tuple/list of (width, height)")
        if not all(isinstance(x, (int, float)) for x in figsize):
            raise ValueError("figsize dimensions must be numeric")
        if figsize[0] <= 0 or figsize[1] <= 0:
            raise ValueError("figsize dimensions must be positive")

    def _validate_cmap(self, cmap_name: str) -> None:
        """Validate colormap name.

        Args:
            cmap_name: Name of matplotlib colormap

        Raises:
            ValueError: If colormap does not exist
        """
        if cmap_name not in plt.colormaps():
            raise ValueError(
                f"Unknown colormap: {cmap_name}. "
                f"Available colormaps: {', '.join(list(plt.colormaps())[:5])}..."
            )

    def _validate_alpha(self, alpha: float | None) -> None:
        """Validate transparency parameter.

        Args:
            alpha: Transparency value (0-1) or None

        Raises:
            ValueError: If alpha is out of valid range
        """
        if alpha is None:
            return
        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be numeric")
        if not (0 <= alpha <= 1):
            raise ValueError("alpha must be between 0 and 1")



