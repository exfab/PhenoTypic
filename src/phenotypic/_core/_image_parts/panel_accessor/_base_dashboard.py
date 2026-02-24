"""Base class for Panel dashboard accessors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import (
    ImageAccessorBase,
)

if TYPE_CHECKING:
    from phenotypic import Image

_PANEL_IMPORT_ERROR = (
    "Panel is required for interactive dashboards. "
    "Install it with: pip install panel"
)


class BaseDashboard(ImageAccessorBase):
    """Base class for all Panel dashboard accessors.

    Provides common helpers for rendering numpy arrays as Panel-compatible
    matplotlib panes and computing detection mode matrices from the registry.
    """

    @property
    def _accessor_property_name(self) -> str:
        return "panel"

    @property
    def _subject_arr(self) -> np.ndarray:
        return self._root_image.gray[:]

    def __init__(self, root_image: Image) -> None:
        super().__init__(root_image)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_array_as_pane(
        arr: np.ndarray,
        title: str | None = None,
        cmap: str = "gray",
        figsize: tuple[float, float] = (4, 4),
    ):
        """Wrap a numpy array in a ``pn.pane.Matplotlib`` figure pane.

        Args:
            arr: 2-D array to display.
            title: Optional title above the image.
            cmap: Matplotlib colormap name.
            figsize: Figure size in inches.

        Returns:
            A ``pn.pane.Matplotlib`` pane wrapping a tight-layout figure.
        """
        import panel as pn

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(arr, cmap=cmap, aspect="equal")
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=10)
        fig.tight_layout(pad=0.5)
        return pn.pane.Matplotlib(fig, tight=True, dpi=100)

    def _compute_all_mode_matrices(self) -> dict[str, np.ndarray]:
        """Compute detection matrices for every registered mode.

        Skips modes that require RGB data when the image has none.

        Returns:
            Dict mapping mode name to its 2-D float32 array.
        """
        from phenotypic._core._image_parts.detection_modes import (
            available_modes,
            get_detection_mode,
        )

        has_rgb = not self._root_image.rgb.isempty()
        result: dict[str, np.ndarray] = {}

        for name in available_modes():
            mode = get_detection_mode(name)
            if mode.requires_rgb and not has_rgb:
                continue
            result[name] = mode.compute(self._root_image)

        return result
