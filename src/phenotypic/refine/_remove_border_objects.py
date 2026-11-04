from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import field_validator

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
        border_size: Width of the exclusion band along each image edge, in
            pixels: an object is removed when any of its pixels lies within
            this many pixels of an edge, so ``1`` removes the objects that
            touch the edge. ``None`` uses 1% of the smaller image dimension
            (height or width). A float in ``(0, 1)`` is a fraction of that
            dimension. A fractional or ``None`` band is at least 1 px wide; an
            integer or a float ``>= 1`` is a pixel count (truncated), and
            ``0`` removes nothing. Typical range: 1--30 px. Default: 1.

    Returns:
        Image: Input image with ``objmap`` and ``objmask`` updated to
        exclude all labeled objects that touch the exclusion border.
        ``rgb``, ``gray``, and ``detect_mat`` are unchanged.

    Raises:
        pydantic.ValidationError: If ``border_size`` is negative.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for a walkthrough
        of border and edge refinement on real plate images.
        :doc:`/explanation/refinement_strategies` for choosing the right
        refinement sequence.
    """

    border_size: int | float | None = 1

    @field_validator("border_size")
    @classmethod
    def _reject_negative_border(cls, value: int | float | None) -> int | float | None:
        if value is not None and value < 0:
            raise ValueError(f"border_size must be >= 0 or None, got {value}")
        return value

    def _band_width(self, height: int, width: int) -> int:
        """The band width in pixels for an image of *height* x *width*.

        Fractions are taken of the smaller *spatial* dimension: the object
        map's, never ``image.shape``, which on an RGB image also counts the
        three colour channels.
        """
        if self.border_size is None:
            fraction = 0.01
        elif isinstance(self.border_size, float) and 0.0 < self.border_size < 1.0:
            fraction = self.border_size
        else:
            return int(self.border_size)
        return max(1, int(min(height, width) * fraction))

    def _operate(self, image: Image) -> Image:
        obj_map = image.objmap[:]
        band = self._band_width(*obj_map.shape)
        if band == 0:
            return image

        in_band = np.zeros(obj_map.shape, dtype=bool)
        in_band[:band, :] = True
        in_band[-band:, :] = True
        in_band[:, :band] = True
        in_band[:, -band:] = True
        edge_labels = np.unique(obj_map[in_band])
        edge_labels = edge_labels[edge_labels != 0]
        obj_map[np.isin(obj_map, edge_labels)] = 0

        image.objmap = obj_map
        return image
