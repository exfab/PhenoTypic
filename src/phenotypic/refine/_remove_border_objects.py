from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np

from phenotypic.abc_ import ObjectRefiner


class RemoveBorderObjects(ObjectRefiner):
    """Remove colonies that touch or overlap the image border within a configurable margin.

    Identifies all labeled objects whose pixels fall within the exclusion
    band along any edge, then zeros those labels from ``objmap``. Colonies
    that are fully interior are unaffected. Partial border colonies bias
    area, perimeter, and shape measurements and should be excluded before
    phenotyping.

    For an overview of refinement strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Plates where the plate rim or scanner shadow truncates edge
          colonies, making size measurements unreliable.
        - Grid assays where border wells are only partially visible within
          the image crop.
        - Time-lapse or batch workflows where automated crops may shift
          between frames, intermittently clipping the same edge colonies.

    Consider Also:
        - :class:`SmallObjectRemover` when the target artefacts are small
          noise fragments scattered across the plate, not limited to the
          border region.
        - :class:`GridOversizedObjectRemover` when the problem is objects
          that span multiple grid sections rather than partial border
          colonies.

    Args:
        border_size: Width of the exclusion band along each image edge.
            ``None`` defaults to 1% of the smaller image dimension.
            A float in ``(0, 1)`` is interpreted as a fraction of the
            smaller image dimension. An integer or float ``>= 1`` is
            treated as an absolute pixel count. Typical range: 1--30 px.
            Default: 1.

    Returns:
        Image: Input image with ``objmap`` and ``objmask`` updated to
        exclude all labeled objects that touch the exclusion border.
        ``rgb``, ``gray``, and ``detect_mat`` are unchanged.

    Raises:
        TypeError: If ``border_size`` is not an integer, float, or ``None``.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a walkthrough
        of border and edge refinement on real plate images.
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
