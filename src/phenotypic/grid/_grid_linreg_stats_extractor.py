from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING: from phenotypic import GridImage

import pandas as pd
from scipy.spatial.distance import euclidean

from phenotypic.abstract import GridFeatureMeasure
from phenotypic.util.constants_ import GRID, GRID_LINREG_STATS_EXTRACTOR, OBJECT_INFO

class MeasureGridLinRegStats(GridFeatureMeasure):
    def __init__(self, section_num: Optional[int] = None):
        self.section_num = section_num

    def _operate(self, image: GridImage) -> pd.DataFrame:

        # Collect the relevant section info. If no section was specified perform calculation on the entire grid info table.
        if self.section_num is None:
            section_info = image.grid.info().reset_index(drop=False)
        else:
            grid_info = image.grid.info().reset_index(drop=False)
            section_info = grid_info.loc[grid_info.loc[:, GRID.GRID_SECTION_NUM] == self.section_num, :]

        # Get the current row-wise linreg info
        row_m, row_b = image.grid.get_linreg_info(axis=0)

        # Convert arrays to dataframe for join operation
        row_linreg_info = pd.DataFrame(data={
            GRID_LINREG_STATS_EXTRACTOR.ROW_LINREG_M: row_m,
            GRID_LINREG_STATS_EXTRACTOR.ROW_LINREG_B: row_b,
        }, index=pd.Index(data=range(image.grid.nrows), name=GRID.GRID_ROW_NUM))

        section_info = pd.merge(left=section_info,
                                right=row_linreg_info,
                                left_on=GRID.GRID_ROW_NUM,
                                right_on=GRID.GRID_ROW_NUM)

        # NOTE: Row linear regression(CC) -> pred RR
        section_info.loc[:, GRID_LINREG_STATS_EXTRACTOR.PRED_RR] = \
            section_info.loc[:, OBJECT_INFO.CENTER_CC] \
            * section_info.loc[:, GRID_LINREG_STATS_EXTRACTOR.ROW_LINREG_M] \
            + section_info.loc[:, GRID_LINREG_STATS_EXTRACTOR.ROW_LINREG_B]

        # Get the current column linreg info
        col_m, col_b = image.grid.get_linreg_info(axis=1)

        # convert array to dataframe for join operation
        col_linreg_info = pd.DataFrame(data={
            GRID_LINREG_STATS_EXTRACTOR.COL_LINREG_M: col_m,
            GRID_LINREG_STATS_EXTRACTOR.COL_LINREG_B: col_b,
        }, index=pd.Index(data=range(image.grid.ncols), name=GRID.GRID_COL_NUM))

        section_info = pd.merge(left=section_info,
                                right=col_linreg_info,
                                left_on=GRID.GRID_COL_NUM,
                                right_on=GRID.GRID_COL_NUM)

        # NOTE: Col linear regression(RR) -> pred CC
        section_info.loc[:, GRID_LINREG_STATS_EXTRACTOR.PRED_CC] = \
            section_info.loc[:, OBJECT_INFO.CENTER_RR] \
            * section_info.loc[:, GRID_LINREG_STATS_EXTRACTOR.COL_LINREG_M] \
            + section_info.loc[:, GRID_LINREG_STATS_EXTRACTOR.COL_LINREG_B]

        # Calculate the distance each object is from it's predicted center. This is the residual error
        section_info.loc[:, GRID_LINREG_STATS_EXTRACTOR.RESIDUAL_ERR] = section_info.apply(
                lambda row: euclidean(
                        u=[row[OBJECT_INFO.CENTER_CC], row[OBJECT_INFO.CENTER_RR]],
                        v=[row[GRID_LINREG_STATS_EXTRACTOR.PRED_CC], row[GRID_LINREG_STATS_EXTRACTOR.PRED_RR]],
                )
                , axis=1
        )

        return section_info.set_index(OBJECT_INFO.OBJECT_LABELS)