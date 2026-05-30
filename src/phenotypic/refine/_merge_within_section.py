from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

from phenotypic.abc_ import GridObjectRefiner
from phenotypic.schema import OBJECT
from phenotypic.schema import GRID


class MergeWithinSection(GridObjectRefiner):
    """Merge all objects within each grid section into a single labeled region.

    Reassigns object labels so that every detection in the same grid cell
    receives a common label, effectively consolidating fragmented detections
    into one object per grid position. Useful when multiple fragments from a
    single colony need to be treated as one unit for downstream measurement.

    Returns:
        Image: Input image with ``objmap`` relabeled so each grid section
        contains at most one object label.

    Best For:
        - Grid plates where a single colony fragments into multiple
          detections within the same cell.
        - Consolidating over-segmented grid cells before area or intensity
          measurements.
        - Ensuring exactly one label per grid position for well-based
          phenotyping.

    Consider Also:
        - :class:`KeepSectionLargest` when only the dominant fragment should
          be kept rather than merging all fragments.
        - :class:`MergeFragmentChains` for proximity-based merging on
          non-grid images.
        - :class:`SmallToLargeMerger` when small fragments should merge into
          the nearest large colony rather than all fragments merging equally.

    See Also:
        :doc:`/how_to/notebooks/merge_fragmented_detections` for merging
        workflows on grid plates.
        :doc:`/explanation/refinement_strategies` for an overview of
        merging strategies.
    """

    def _operate(self, image: GridImage) -> GridImage:
        # Cache the objmap into memory and make a copy for editing
        objmap = image.objmap[:]
        new_objmap = image.objmap[:].copy()

        grid_info: pd.DataFrame = image.grid.info()
        for i, gs in enumerate(grid_info[GRID.ROW_MAJOR_IDX].unique()):
            subtable = grid_info.loc[grid_info.loc[:, GRID.ROW_MAJOR_IDX] == gs, :]

            new_objmap[np.isin(objmap, subtable.loc[:, OBJECT.LABEL])] = i + 1
