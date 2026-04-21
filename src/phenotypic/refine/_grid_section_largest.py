from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

from phenotypic.abc_ import ObjectRefiner
from phenotypic.measure import MeasureSize
from phenotypic.tools_.measurement_info_ import SIZE, GRID
from phenotypic.tools_.constants_ import OBJECT


class GridSectionLargest(ObjectRefiner):
    """Retain only the largest object within each grid section by area.

    Measures the pixel area of every detected object, groups them by grid
    cell, and keeps only the largest per cell. Smaller fragments, debris,
    and secondary detections within each grid position are removed.

    Returns:
        Image: Input image with ``objmap`` reduced to the single largest
        object per grid section.

    Best For:
        - Grid plates where the true colony is the largest detection in
          each cell and smaller objects are noise or debris.
        - Quick post-detection cleanup on 96-well or 384-well formats.
        - Simplifying multi-detection grid cells to one object per position.

    Consider Also:
        - :class:`GridAlignmentRefiner` for full grid-aware dominant-object
          selection with configurable strategies.
        - :class:`CenterDeviationReducer` when the most centered object is
          more reliable than the largest.
        - :class:`ReduceMultipleGridObjects` for regression-based selection
          using expected grid positions.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for grid-based
        refinement workflows.
        :doc:`/explanation/refinement_strategies` for an overview of
        grid refinement approaches.
    """

    def _operate(self, image: GridImage) -> GridImage:
        size_table = MeasureSize().measure(image, include_meta=True)
        max_idx = size_table.groupby(
                by=GRID.ROW_MAJOR_IDX,
                observed=True
        )[SIZE.AREA].idxmax()
        max_size_labels = size_table.loc[max_idx, OBJECT.LABEL].to_numpy()

        # Drop objects not the largest
        nonmax_mask = ~np.isin(image.objmap[:], max_size_labels)
        image.objmap[nonmax_mask] = 0
        return image
