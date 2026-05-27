from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np

from phenotypic.abc_ import ObjectRefiner


class RemoveBorderObjects(ObjectRefiner):
    """Remove colonies that touch the image border within a configurable margin.

    Zeroes any labeled objects in ``objmap`` whose pixels fall within a
    border band, ensuring only fully contained colonies are analyzed.
    Partial edge colonies bias size and shape measurements.

    Args:
        border_size: Width of the exclusion border. ``None`` uses 1% of
            the smaller dimension. Float in (0, 1) is a fraction of the
            image size. Int or float >= 1 is absolute pixels. Default: 1.

    Returns:
        Image: Input image with ``objmask`` and ``objmap`` updated to
        exclude border-touching objects.

    Raises:
        TypeError: If ``border_size`` type is invalid.

    Best For:
        - Plates where the plate rim or image crop truncates edge colonies.
        - Grid assays where border wells are partially visible.
        - Automated crops that shift between frames.

    Consider Also:
        - :class:`SmallObjectRemover` for removing noise fragments that are
          not necessarily at the border.
        - :class:`GridOversizedObjectRemover` for removing abnormally large
          objects that span multiple grid sections.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a walkthrough
        of refinement operations.
        :doc:`/explanation/refinement_strategies` for choosing the right
        refinement sequence.
    """

    border_size: int | float | None = 1

    def _operate(self, image: Image) -> Image:
        if self.border_size is None:
            edge_size = int(np.min(image.shape[[1, 2]]) * 0.01)
        elif type(self.border_size) is float and 0.0 < self.border_size < 1.0:
            edge_size = int(np.min(image.shape) * self.border_size)
        elif isinstance(self.border_size, (int, float)):
            edge_size = self.border_size
        else:
            raise TypeError(
                    "Invalid edge size. Should be int, float, or None to use default edge size."
            )

        obj_map = image.objmap[:]
        edges = [
            obj_map[: edge_size - 1, :].ravel(),
            obj_map[-edge_size:, :].ravel(),
            obj_map[:, : edge_size - 1].ravel(),
            obj_map[:, -edge_size:].ravel(),
        ]
        edge_labels = np.unique(np.concatenate(edges))
        for label in edge_labels:
            obj_map[obj_map == label] = 0

        image.objmap = obj_map
        return image
