from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
import numpy as np

from phenotypic.abstract import GridMeasureFeatures
from phenotypic.util.constants_ import BBOX, GRID
from abc import ABC


class GridFinder(GridMeasureFeatures, ABC):
    """
    GridFinder measures grid information from the objects in various ways. Using the names here allows for streamlined integration.
    Unlike other Grid series interfaces, GridExtractors can work on regular images and should not be dependent on the GridImage class.

    Note:
        - GridFinders should implement self.get_row_edges() and self.get_col_edges() methods to get the row and column edges for the grid.

    Parameters:
        nrows (int): Number of rows in the grid.
        ncols (int): Number of columns in the grid.

    """

    @abc.abstractmethod
    def _operate(self, image: Image) -> pd.DataFrame:
        return pd.DataFrame()

    @abc.abstractmethod
    def get_row_edges(self, image: Image) -> np.ndarray:
        """
        This method is to returns the row edges of the grid as numpy array.
        Args:
            image (Image): Image object.
        Returns:
            np.ndarray: Row-edges of the grid.
        """
        pass

    @abc.abstractmethod
    def get_col_edges(self, image: Image) -> np.ndarray:
        """
        This method is to returns the column edges of the grid as numpy array.
        Args:
            image:

        Returns:
            np.ndarray: Column-edges of the grid.

        """
        pass

    @staticmethod
    def _clip_row_edges(row_edges, imshape: (int, int, ...)) -> np.ndarray:
        return np.clip(a=row_edges, a_min=0, a_max=imshape[0])

    def _add_row_number_info(self, table: pd.DataFrame, row_edges: np.array, imshape: (int, int)) -> pd.DataFrame:
        row_edges = self._clip_row_edges(row_edges=row_edges, imshape=imshape)
        table.loc[:, str(GRID.ROW_NUM)] = pd.cut(
                table.loc[:, str(BBOX.CENTER_RR)],
                bins=row_edges,
                labels=range(self.nrows),
                include_lowest=True,
                right=True,
        )
        return table

    def _add_row_interval_info(self, table: pd.DataFrame, row_edges: np.array, imshape: (int, int)) -> pd.DataFrame:
        row_edges = self._clip_row_edges(row_edges=row_edges, imshape=imshape)

        # Get the bin indices
        bin_indices = pd.cut(
                table.loc[:, str(BBOX.CENTER_RR)],
                bins=row_edges,
                labels=False,
                include_lowest=True,
                right=True,
        )

        # Create start and end columns directly
        table.loc[:, str(GRID.ROW_INTERVAL_START)] = [int(row_edges[i]) if pd.notna(i) else None for i in bin_indices]
        table.loc[:, str(GRID.ROW_INTERVAL_END)] = [int(row_edges[i + 1]) if pd.notna(i) else None for i in bin_indices]

        return table

    @staticmethod
    def _clip_col_edges(col_edges, imshape: (int, int, ...)) -> np.ndarray:
        return np.clip(a=col_edges, a_min=0, a_max=imshape[1] - 1)

    def _add_col_number_info(self, table: pd.DataFrame, col_edges: np.array, imshape: (int, int)) -> pd.DataFrame:
        col_edges = self._clip_col_edges(col_edges=col_edges, imshape=imshape)
        table.loc[:, str(GRID.COL_NUM)] = pd.cut(
                table.loc[:, str(BBOX.CENTER_CC)],
                bins=col_edges,
                labels=range(self.ncols),
                include_lowest=True,
                right=True,
        )
        return table

    def _add_col_interval_info(self, table: pd.DataFrame, col_edges: np.array, imshape: (int, int)) -> pd.DataFrame:
        col_edges = self._clip_col_edges(col_edges=col_edges, imshape=imshape)

        # Get the bin indices
        bin_indices = pd.cut(
                table.loc[:, str(BBOX.CENTER_CC)],
                bins=col_edges,
                labels=False,
                include_lowest=True,
                right=True,
        )

        # Create start and end columns directly
        table.loc[:, str(GRID.COL_INTERVAL_START)] = [int(col_edges[i]) if pd.notna(i) else None for i in bin_indices]
        table.loc[:, str(GRID.COL_INTERVAL_END)] = [int(col_edges[i + 1]) if pd.notna(i) else None for i in bin_indices]

        return table

    def _add_section_interval_info(self, table: pd.DataFrame,
                                   row_edges: np.array, col_edges: np.array,
                                   imshape: (int, int)) -> pd.DataFrame:
        if str(GRID.ROW_NUM) not in table.columns: self._add_row_number_info(table=table, row_edges=row_edges,
                                                                             imshape=imshape)
        if str(GRID.COL_NUM) not in table.columns: self._add_col_number_info(table=table, col_edges=col_edges,
                                                                             imshape=imshape)
        table.loc[:, str(GRID.SECTION_IDX)] = list(
                zip(table.loc[:, str(GRID.ROW_NUM)], table.loc[:, str(GRID.COL_NUM)]))
        table.loc[:, str(GRID.SECTION_IDX)] = table.loc[:, str(GRID.SECTION_IDX)].astype('category')
        return table

    def _add_section_number_info(self, table: pd.DataFrame,
                                 row_edges: np.array, col_edges: np.array,
                                 imshape: (int, int)) -> pd.DataFrame:
        if str(GRID.SECTION_IDX) not in table.columns: self._add_section_interval_info(
                table=table, row_edges=row_edges, col_edges=col_edges, imshape=imshape
        )
        idx_map = np.reshape(np.arange(self.nrows*self.ncols), (self.nrows, self.ncols))
        for idx in np.sort(np.unique(table.loc[:, str(GRID.SECTION_IDX)].values)):
            table.loc[table.loc[:, str(GRID.SECTION_IDX)] == idx, str(GRID.SECTION_NUM)] = idx_map[idx[0], idx[1]]

        table.loc[:, str(GRID.SECTION_NUM)] = table.loc[:, str(GRID.SECTION_NUM)].astype('category')
        return table
