from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
from phenotypic.abc_ import GridMeasureFeatures
from phenotypic.schema import GRID_SPREAD

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from phenotypic.schema import BBOX, GRID, OBJECT


class MeasureGridSpread(GridMeasureFeatures):
    """Quantify within-well colony dispersion using pairwise centroid distances.

    Compute the sum of squared pairwise Euclidean distances between all
    colony centroids in each grid section. High values indicate multiple
    dispersed objects within a single well -- a sign of over-segmentation,
    fragmented growth, or invasive spreading.

    Returns:
        pd.DataFrame: One row per object, keyed by ``Object_Label``. Each
        object carries its grid section's spread and object count, so every
        object in a section shares the same values.

    Best For:
        - Detecting over-segmented wells where multiple objects were
          found instead of a single cohesive colony.
        - Identifying invasive or spreading growth that extends beyond
          the designated grid position.
        - Flagging wells with questionable data quality for manual
          review or exclusion from downstream analysis.

    Consider Also:
        - :class:`MeasureNeighborDist` for between-well neighbor
          distances rather than within-well dispersion.
        - :class:`MeasureGridLinRegStats` for positional accuracy
          metrics based on linear regression.
        - :class:`MeasureBounds` for raw centroid positions per colony.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of grid-level measurements.
    """

    _measurement_infoclass: ClassVar[type] = GRID_SPREAD

    def _operate(self, image: GridImage) -> pd.DataFrame:
        grid_info = image.grid.info(include_metadata=False)
        section = grid_info.loc[:, str(GRID.ROW_MAJOR_IDX)]

        section_spread = {}
        for section_idx, section_table in grid_info.groupby(section, observed=True):
            centers = section_table.loc[
                :, [str(BBOX.CENTER_CC), str(BBOX.CENTER_RR)]
            ].to_numpy(dtype=float)
            # pdist yields each unordered pair once, so distinct pairs at equal
            # distances are all counted.
            section_spread[section_idx] = float(np.sum(pdist(centers) ** 2))

        # Section-level values are broadcast to every object in the section so
        # the frame keys on Object_Label like every other measurer.
        return pd.DataFrame(
                {
                    OBJECT.LABEL: grid_info.loc[:, OBJECT.LABEL].to_numpy(),
                    str(GRID_SPREAD.OBJECT_SPREAD): section.map(section_spread).to_numpy(),
                    str(GRID_SPREAD.OBJECT_COUNT): section.map(section.value_counts()).to_numpy(),
                }
        )


MeasureGridSpread.__doc__ = GRID_SPREAD.append_rst_to_doc(MeasureGridSpread)
