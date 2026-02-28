from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari
    from phenotypic import Image

import numpy as np

from .accessors import PlotAccessor
from ._image_objects_handler import ImageObjectsHandler


class ImagePlotHandler(ImageObjectsHandler):
    def __init__(
            self,
            arr: np.ndarray | Image | None = None,
            name: str | None = None,
            bit_depth: int | None = None,
    ):
        super().__init__(arr=arr, name=name, bit_depth=bit_depth)
        self._accessors.plot = PlotAccessor(self)

    @property
    def plot(self) -> PlotAccessor:
        return self._accessors.plot

    def napari(
            self, name: str | None = None, reset: bool = False,
            *, viewer: napari.Viewer | None = None,
    ) -> napari.Viewer:
        """Add all available image layers to a persistent global napari viewer.

        Args:
            name: Optional custom name for image layers. Each layer is named
                ``{accessor}_{name}``. Defaults to the image's name attribute.
            reset: If True, closes the current viewer and creates a fresh one
                before adding layers. Defaults to False.
            viewer: Optional external napari viewer instance to use instead of the
                global viewer. When provided, global viewer management is bypassed
                and all layers are added to this viewer. Defaults to None.

        Returns:
            The global napari viewer instance with all layers added.

        Raises:
            ImportError: If napari is not installed.

        Examples:
            Quickly inspect all layers after detection:

            >>> from phenotypic.data import load_synth_yeast_plate
            >>> from phenotypic.detect import OtsuDetector
            >>> image = OtsuDetector().apply(load_synth_yeast_plate())
            >>> viewer = image.napari()  # doctest: +SKIP
        """
        from .accessor_abstracts._image_accessor_base import (
            _HAS_NAPARI,
        )

        if not _HAS_NAPARI:
            raise ImportError(
                    "napari is required for interactive visualization. "
                    "Install with: pip install phenotypic[gui]"
            )

        first = True
        if not self.rgb.isempty():
            result = self.rgb.napari(name, reset=reset if first else False, viewer=viewer)
            first = False
        result = self.gray.napari(name, reset=reset if first else False, viewer=viewer)
        first = False
        result = self.detect_mat.napari(name, viewer=viewer)
        if self.num_objects > 0:
            result = self.objmap.napari(name, viewer=viewer)
        return result
