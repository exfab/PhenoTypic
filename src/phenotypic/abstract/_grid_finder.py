from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
import numpy as np

from phenotypic.abstract import GridMeasureFeatures
from phenotypic.util.constants_ import OBJECT, GRID, BBOX


class GridFinder(GridMeasureFeatures):
    """
    GridFinder measures grid information from the objects in various ways. Using the names here allows for streamlined integration.
    Unlike other Grid series interfaces, GridExtractors can work on regular images and should not be dependent on the GridImage class.

    Note:
        - GridFinders should implement self.get_row_edges() and self.get_col_edges() methods to get the row and column edges for the grid.

    Parameters:
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.

    """

    def _operate(self, image: Image) -> pd.DataFrame:
        return pd.DataFrame()

    def _naive_grid_finding(self, image: Image) -> pd.DataFrame:
        pass

    @staticmethod
    def _clip_row_edges(row_edges, imshape: (int, int, ...)) -> np.ndarray:
        return np.clip(a=row_edges, a_min=0, a_max=imshape[0])

    def _add_row_number_info(self, table: pd.DataFrame, row_edges: np.array, imshape: (int, int)) -> pd.DataFrame:
        row_edges = self._clip_row_edges(row_edges=row_edges, imshape=imshape)
        table.loc[:, GRID.GRID_ROW_NUM] = pd.cut(
            table.loc[:, str(BBOX.CENTER_RR)],
            bins=row_edges,
            labels=range(self.nrows),
            include_lowest=True,
            right=True,
        )
        return table

    def _add_row_interval_info(self, table: pd.DataFrame, row_edges: np.array, imshape: (int, int)) -> pd.DataFrame:
        row_edges = self._clip_row_edges(row_edges=row_edges, imshape=imshape)
        table.loc[:, GRID.GRID_ROW_INTERVAL] = pd.cut(
            table.loc[:, str(BBOX.CENTER_RR)],
            bins=row_edges,
            labels=[(int(row_edges[i]), int(row_edges[i + 1])) for i in range(len(row_edges) - 1)],
            include_lowest=True,
            right=True,
        )
        return table

    @staticmethod
    def _clip_col_edges(col_edges, imshape: (int, int, ...)) -> np.ndarray:
        return np.clip(a=col_edges, a_min=0, a_max=imshape[1] - 1)

    def _add_col_number_info(self, table: pd.DataFrame, col_edges: np.array, imshape: (int, int)) -> pd.DataFrame:
        col_edges = self._clip_col_edges(col_edges=col_edges, imshape=imshape)
        table.loc[:, GRID.GRID_COL_NUM] = pd.cut(
            table.loc[:, str(BBOX.CENTER_CC)],
            bins=col_edges,
            labels=range(self.ncols),
            include_lowest=True,
            right=True,
        )
        return table

    def _add_col_interval_info(self, table: pd.DataFrame, col_edges: np.array, imshape: (int, int)) -> pd.DataFrame:
        col_edges = self._clip_col_edges(col_edges=col_edges, imshape=imshape)
        table.loc[:, GRID.GRID_COL_INTERVAL] = pd.cut(
            table.loc[:, str(BBOX.CENTER_CC)],
            bins=col_edges,
            labels=[(int(col_edges[i]), int(col_edges[i + 1])) for i in range(len(col_edges) - 1)],
            include_lowest=True,
            right=True,
        )
        return table

    def _add_section_interval_info(self, table: pd.DataFrame,
                                   row_edges: np.array, col_edges: np.array,
                                   imshape: (int, int)) -> pd.DataFrame:
        if GRID.GRID_ROW_NUM not in table.columns: self._add_row_number_info(table=table, row_edges=row_edges, imshape=imshape)
        if GRID.GRID_COL_NUM not in table.columns: self._add_col_number_info(table=table, col_edges=col_edges, imshape=imshape)
        table.loc[:, GRID.GRID_SECTION_IDX] = list(zip(table.loc[:, GRID.GRID_ROW_NUM], table.loc[:, GRID.GRID_COL_NUM]))
        table.loc[:, GRID.GRID_SECTION_IDX] = table.loc[:, GRID.GRID_SECTION_IDX].astype('category')
        return table

    def _add_section_number_info(self, table: pd.DataFrame,
                                 row_edges: np.array, col_edges: np.array,
                                 imshape: (int, int)) -> pd.DataFrame:
        if GRID.GRID_SECTION_IDX not in table.columns: self._add_section_interval_info(
            table=table, row_edges=row_edges, col_edges=col_edges, imshape=imshape
        )
        idx_map = np.reshape(np.arange(self.nrows * self.ncols), (self.nrows, self.ncols))
        for idx in np.sort(np.unique(table.loc[:, GRID.GRID_SECTION_IDX].values)):
            table.loc[table.loc[:, GRID.GRID_SECTION_IDX] == idx, GRID.GRID_SECTION_NUM] = idx_map[idx[0], idx[1]]

        table.loc[:, GRID.GRID_SECTION_NUM] = table.loc[:, GRID.GRID_SECTION_NUM].astype('category')
        return table
