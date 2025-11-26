from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
from typing import Optional, Union

from phenotypic.abc_ import ObjectRefiner


class BorderObjectRemover(ObjectRefiner):
    """Remove objects that touch the image border within a configurable margin.

    Intuition:
        Colonies that intersect the plate boundary or image crop edge are often
        partial, poorly segmented, and bias size/shape measurements. This
        operation zeroes any labeled objects in ``image.objmap`` whose pixels
        fall within a user-defined border band, ensuring only fully contained
        colonies are analyzed.

    Why this is useful for agar-plate imaging:
        Plate crops or grid layouts frequently clip edge colonies. Removing
        border-touching objects stabilizes downstream phenotyping (area,
        circularity, intensity) and prevents partial colonies from contaminating
        statistics or training data.

    Use cases:
        - Single-plate captures where the plate rim truncates colonies.
        - Grid assays where wells or positions near the frame boundary are
          partially visible.
        - Automated crops that shift slightly between frames, cutting off
          colonies at the margins.

    Caveats:
        - A large border may remove valid edge colonies (lower recall). Too
          small a border may retain partial objects.
        - With very tight crops (little background), even modest margins can
          eliminate many colonies.

    Attributes:
        border_size (int): Width of the exclusion border around the image.

    Examples:
        >>> from phenotypic.refine import BorderObjectRemover
        >>> op = BorderObjectRemover(border_size=15)
        >>> image = op.apply(image, inplace=True)  # doctest: +SKIP
        >>> # All colonies intersecting a 15-pixel frame margin are removed

    Raises:
        TypeError: If an invalid ``border_size`` type is provided (raised during
            operation when parameters are validated).
    """

    def __init__(self, border_size: Optional[Union[int, float]] = 1):
        """Initialize the remover.

        Args:
            border_size: Width of the exclusion border around the image.
                - ``None``: Use a default margin equal to 1% of the smaller
                  image dimension.
                - ``float`` in (0, 1): Interpret as a fraction of the minimum
                  image dimension, producing a resolution-adaptive margin.
                - ``int`` or ``float`` ≥ 1: Interpret as an absolute number of
                  pixels.

        Notes:
            Larger margins remove more edge-touching colonies and are useful
            when crops are loose or the plate rim intrudes. Smaller margins
            preserve edge colonies but risk including partial objects.
        """
        self.border_size = border_size

    def _operate(self, image: Image) -> Image:
        if self.border_size is None:
            edge_size = int(np.min(image.shape[[1, 2]])*0.01)
        elif type(self.border_size) == float and 0.0 < self.border_size < 1.0:
            edge_size = int(np.min(image.shape)*self.border_size)
        elif isinstance(self.border_size, (int, float)):
            edge_size = self.border_size
        else:
            raise TypeError('Invalid edge size. Should be int, float, or None to use default edge size.')

        obj_map = image.objmap[:]
        edges = [obj_map[:edge_size - 1, :].ravel(),
                 obj_map[-edge_size:, :].ravel(),
                 obj_map[:, :edge_size - 1].ravel(),
                 obj_map[:, -edge_size:].ravel()
                 ]
        edge_labels = np.unique(np.concatenate(edges))
        for label in edge_labels:
            obj_map[obj_map == label] = 0

        image.objmap = obj_map
        return image
