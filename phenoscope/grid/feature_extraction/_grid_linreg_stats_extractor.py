from phenoscope.grid import GriddedImage
from phenoscope.grid.interface import GridFeatureExtractor

from typing import Optional
import pandas as pd
from scipy.spatial.distance import euclidean


class GridLinRegStatsExtractor(GridFeatureExtractor):
    LABEL_ROW_LINREG_M, LABEL_ROW_LINREG_B = 'RowLinReg_M', 'RowLinReg_B'
    LABEL_COL_LINREG_M, LABEL_COL_LINREG_B = 'ColLinReg_M', 'LinReg_B'
    LABEL_PRED_RR, LABEL_PRED_CC = 'RowLinReg_PredRR', 'ColLinReg_PredCC'
    LABEL_RESIDUAL_ERR = 'LinReg_ResidualError'

    def __init__(self, section_num: Optional[int] = None):
        self.section_num = section_num

    def _operate(self, image: GriddedImage) -> pd.DataFrame:

        # Collect the relevant section info. If no sectionw was specified perform calculation on the entire grid info table.
        if self.section_num is None:
            section_info = image.grid_info.reset_index(drop=False)
        else:
            grid_info = image.grid_info.reset_index(drop=False)
            section_info = grid_info.loc[grid_info.loc[:, image.grid_extractor.LABEL_GRID_SECTION_NUM] == self.section_num, :]

        # Get the current linreg info
        row_m, row_b = image.get_linreg_info(axis=0)

        # Convert arrays to dataframe for join operation
        row_linreg_info = pd.DataFrame(data={
            self.LABEL_ROW_LINREG_M: row_m,
            self.LABEL_ROW_LINREG_B: row_b,
        }, index=pd.Index(data=range(image.n_rows), name=image.grid_extractor.LABEL_GRID_ROW_NUM))

        section_info = pd.merge(left=section_info,
                                right=row_linreg_info,
                                left_on=image.grid_extractor.LABEL_GRID_ROW_NUM,
                                right_on=image.grid_extractor.LABEL_GRID_ROW_NUM)

        # NOTE: Row linear regression(CC) -> pred RR
        section_info.loc[:, self.LABEL_PRED_RR] = \
            section_info.loc[:, image.bound_extractor.LABEL_CENTER_CC] \
            * section_info.loc[:, self.LABEL_ROW_LINREG_M] \
            + section_info.loc[:, self.LABEL_ROW_LINREG_B]

        # Get the current column linreg info
        col_m, col_b = image.get_linreg_info(axis=1)

        # convert array to dataframe for join operation
        col_linreg_info = pd.DataFrame(data={
            self.LABEL_COL_LINREG_M: col_m,
            self.LABEL_COL_LINREG_B: col_b,
        }, index=pd.Index(data=range(image.n_cols), name=image.grid_extractor.LABEL_GRID_COL_NUM))

        section_info = pd.merge(left=section_info,
                                right=col_linreg_info,
                                left_on=image.grid_extractor.LABEL_GRID_COL_NUM,
                                right_on=image.grid_extractor.LABEL_GRID_COL_NUM)

        # NOTE: Col linear regression(RR) -> pred CC
        section_info.loc[:, self.LABEL_PRED_CC] = \
            section_info.loc[:, image.bound_extractor.LABEL_CENTER_RR] \
            * section_info.loc[:, self.LABEL_COL_LINREG_M] \
            + section_info.loc[:, self.LABEL_COL_LINREG_B]

        # Calculate the distance each object is from it's predicted center. This is the residual error
        section_info.loc[:, self.LABEL_RESIDUAL_ERR] = section_info.apply(
                lambda row: euclidean(
                        u=[row[image.bound_extractor.LABEL_CENTER_CC], row[image.bound_extractor.LABEL_CENTER_RR]],
                        v=[row[self.LABEL_PRED_CC], row[self.LABEL_PRED_RR]],
                )
                , axis=1
        )

        return section_info.set_index(image.bound_extractor.LABEL_OBJ_MAP_ID)