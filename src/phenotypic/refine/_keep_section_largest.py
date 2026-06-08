from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

from phenotypic.abc_ import GridObjectRefiner
from phenotypic.measure import MeasureSize
from phenotypic.schema import SIZE, GRID
from phenotypic.schema import OBJECT


class KeepSectionLargest(GridObjectRefiner):
    """Retain only the largest object by area within each grid section.

    Measures the pixel area of every detected object via :class:`MeasureSize`,
    groups objects by grid cell, and discards all but the largest per cell.
    Fragments, debris, and secondary detections within each grid position are
    removed, yielding at most one colony label per well. Requires a
    ``GridImage`` input so that grid section membership is known.

    For an overview of grid refinement approaches, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Pinned colony grids (96-well, 384-well) where the genuine colony is
          consistently the largest detection in each cell and smaller objects
          are condensation, debris, or satellite colonies.
        - Quick post-detection cleanup that does not require grid inference
          because the grid layout is already embedded in the ``GridImage``.
        - Pipelines that produce multi-detection grid cells and need a simple,
          parameter-free reduction to one object per position.

    Consider Also:
        - :class:`GridAlignmentRefiner` for grid-aware selection with
          configurable strategies (dominant, centered, regularized) and
          optional grid inference for plain ``Image`` inputs.
        - :class:`KeepNearestCenter` when the most spatially centered object
          is a more reliable proxy for the true colony than pixel area.
        - :class:`ReduceSectionsByLine` for regression-based selection using
          expected grid positions when area is not a reliable discriminant.

    Returns:
        Image: Input image with ``objmap`` reduced to the single largest
        object per grid section; all other objects are set to background.

    See Also:
        :doc:`/how_to/notebooks/refine_noisy_boundaries` for grid-based
        refinement workflows on real plate images.
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
