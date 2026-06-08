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
    """Consolidate all detections within each grid cell into a single labeled region.

    Reassigns object labels so that every detection in the same grid section
    receives a common label equal to the section's row-major index. After
    merging, each grid cell contains at most one label, so downstream
    measurements treat fragmented per-cell detections as a single colony.

    Requires a ``GridImage`` with grid metadata already populated.

    For an overview of merging strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Grid plates where a single colony splits into several labeled
          regions within one well due to agar texture or thresholding.
        - Consolidating over-segmented grid cells before area, intensity,
          or size measurements.
        - Ensuring exactly one detection label per grid position for
          well-based phenotyping pipelines.

    Consider Also:
        - :class:`KeepSectionLargest` when only the dominant fragment per
          cell should be retained rather than all fragments merged into one
          object.
        - :class:`MergeFragmentChains` for proximity-based transitive
          merging on non-grid images where section membership is unknown.
        - :class:`SmallToLargeMerger` when small fragments should absorb
          into the nearest large colony regardless of grid section
          boundaries.

    Returns:
        Image: Input image with ``objmap`` relabeled so each grid section
        contains at most one object label. ``objmask``, ``rgb``, ``gray``,
        and ``detect_mat`` are unchanged.

    See Also:
        :doc:`/how_to/notebooks/merge_fragmented_detections` for merging
        workflows on grid plates.
        :doc:`/explanation/refinement_strategies` for an overview of
        grid-aware merging strategies.
    """

    def _operate(self, image: GridImage) -> GridImage:
        # Cache the objmap into memory and make a copy for editing
        objmap = image.objmap[:]
        new_objmap = image.objmap[:].copy()

        grid_info: pd.DataFrame = image.grid.info()
        for i, gs in enumerate(grid_info[GRID.ROW_MAJOR_IDX].unique()):
            subtable = grid_info.loc[grid_info.loc[:, GRID.ROW_MAJOR_IDX] == gs, :]

            new_objmap[np.isin(objmap, subtable.loc[:, OBJECT.LABEL])] = i + 1
