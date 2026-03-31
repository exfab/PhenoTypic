from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage

import pandas as pd
from scipy.spatial.distance import euclidean

from phenotypic.abc_ import GridMeasureFeatures
from phenotypic.tools_.constants_ import OBJECT
from phenotypic.tools_.measurement_info_ import BBOX, GRID, GRID_LINREG_STATS


class MeasureGridLinRegStats(GridMeasureFeatures):
    """Evaluate grid alignment quality using row-wise and column-wise linear regression.

    Fit linear regressions to colony centroid positions along each row
    and column of the grid, then compute per-colony residual error
    (Euclidean distance between observed and predicted centroid). High
    residual errors flag off-grid growth, misdetections, or plate
    warping.

    Best For:
        - Identifying colonies that grew outside their designated grid
          position on arrayed plates.
        - Detecting systematic rotation or shear across the plate to
          validate grid detection quality.
        - Filtering or weighting colonies by positional confidence
          before downstream phenotypic analysis.

    Consider Also:
        - :class:`MeasureGridSpatial` for neighbor-distance metrics
          between adjacent grid cells.
        - :class:`MeasureGridSpread` for detecting over-segmentation
          and multi-object wells.
        - :class:`MeasureBounds` for raw centroid and bounding box
          coordinates without regression.

    Args:
        section_num: Grid section index to restrict measurements to.
            ``None`` measures across the entire grid. Default: ``None``.

    Returns:
        pd.DataFrame: Per-object metrics indexed by object label:

            - RowM, RowB: row regression slope and intercept.
            - ColM, ColB: column regression slope and intercept.
            - PredRR, PredCC: predicted centroid from regression.
            - ResidualError: Euclidean distance between actual and
              predicted centroid.

    See Also:
        :doc:`/tutorials/notebooks/07_measuring_and_exporting` for a
        walkthrough of grid-level measurements.
    """

    _measurement_info_class = GRID_LINREG_STATS

    def __init__(self, section_num: Optional[int] = None):
        super().__init__()
        self.section_num = section_num

    def _operate(self, image: GridImage) -> pd.DataFrame:
        # Collect the relevant section info. If no section was specified perform calculation on the entire grid info table.
        if self.section_num is None:
            section_info = image.grid.info().reset_index(drop=False)
        else:
            grid_info = image.grid.info().reset_index(drop=False)
            section_info = grid_info.loc[
                grid_info.loc[:, str(GRID.ROW_MAJOR_IDX)] == self.section_num, :
            ]

        # Get the current row-wise linreg info
        row_m, row_b = image.grid.get_centroid_alignment_info(axis=0)

        # Convert arrays to dataframe for join operation
        row_linreg_info = pd.DataFrame(
                data={
                    str(GRID_LINREG_STATS.ROW_LINREG_M): row_m,
                    str(GRID_LINREG_STATS.ROW_LINREG_B): row_b,
                },
                index=pd.Index(data=range(image.grid.nrows), name=str(GRID.ROW_NUM)),
        )

        section_info = pd.merge(
                left=section_info,
                right=row_linreg_info,
                left_on=str(GRID.ROW_NUM),
                right_on=str(GRID.ROW_NUM),
        )

        # NOTE: Row linear regression(CC) -> pred RR
        section_info.loc[:, str(GRID_LINREG_STATS.PRED_RR)] = (
                section_info.loc[:, str(BBOX.CENTER_CC)]
                * section_info.loc[:, str(GRID_LINREG_STATS.ROW_LINREG_M)]
                + section_info.loc[:, str(GRID_LINREG_STATS.ROW_LINREG_B)]
        )

        # Get the current column linreg info
        col_m, col_b = image.grid.get_centroid_alignment_info(axis=1)

        # convert array to dataframe for join operation
        col_linreg_info = pd.DataFrame(
                data={
                    str(GRID_LINREG_STATS.COL_LINREG_M): col_m,
                    str(GRID_LINREG_STATS.COL_LINREG_B): col_b,
                },
                index=pd.Index(data=range(image.grid.ncols), name=str(GRID.COL_NUM)),
        )

        section_info = pd.merge(
                left=section_info,
                right=col_linreg_info,
                left_on=str(GRID.COL_NUM),
                right_on=str(GRID.COL_NUM),
        )

        # NOTE: Col linear regression(RR) -> pred CC
        section_info.loc[:, str(GRID_LINREG_STATS.PRED_CC)] = (
                section_info.loc[:, str(BBOX.CENTER_RR)]
                * section_info.loc[:, str(GRID_LINREG_STATS.COL_LINREG_M)]
                + section_info.loc[:, str(GRID_LINREG_STATS.COL_LINREG_B)]
        )

        # Calculate the distance each object is from it's predicted center. This is the residual error
        section_info.loc[:, str(GRID_LINREG_STATS.RESIDUAL_ERR)] = (
            section_info.apply(
                    lambda row: euclidean(
                            u=[row[str(BBOX.CENTER_CC)], row[str(BBOX.CENTER_RR)]],
                            v=[
                                row[str(GRID_LINREG_STATS.PRED_CC)],
                                row[str(GRID_LINREG_STATS.PRED_RR)],
                            ],
                    ),
                    axis=1,
            )
        )

        return section_info.set_index(OBJECT.LABEL)


MeasureGridLinRegStats.__doc__ = GRID_LINREG_STATS.append_rst_to_doc(
        MeasureGridLinRegStats)
