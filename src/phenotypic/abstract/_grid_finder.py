from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
import numpy as np

from phenotypic.abstract import GridMeasureFeatures
from phenotypic.tools.constants_ import BBOX, GRID
from abc import ABC


class GridFinder(GridMeasureFeatures, ABC):
    """
    GridFinder measures grid information from the objects in various ways. Using the names here allows for streamlined integration.
    Unlike other Grid series interfaces, GridExtractors can work on regular images and should not be dependent on the GridImage class.

    Note:
        - GridFinders should implement self.get_row_edges() and self.get_col_edges() methods to get the row and column edges for the grid.

    Parameters:
        nrows (int): Number of nrows in the grid.
        ncols (int): Number of columns in the grid.

    """

    def __init__(self, nrows: int, ncols: int) -> None:
        self.nrows = nrows
        self.ncols = ncols

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


    def _add_section_number_info(self, table: pd.DataFrame,
                                 row_edges: np.array, col_edges: np.array,
                                 imshape: (int, int)) -> pd.DataFrame:
        # Ensure ROW_NUM and COL_NUM exist
        if str(GRID.ROW_NUM) not in table.columns:
            self._add_row_number_info(table=table, row_edges=row_edges, imshape=imshape)
        if str(GRID.COL_NUM) not in table.columns:
            self._add_col_number_info(table=table, col_edges=col_edges, imshape=imshape)
        
        # Create section number directly from row and column indices
        idx_map = np.reshape(np.arange(self.nrows*self.ncols), (self.nrows, self.ncols))
        
        # Compute section number for each row using vectorized operations
        row_nums = table.loc[:, str(GRID.ROW_NUM)].values
        col_nums = table.loc[:, str(GRID.COL_NUM)].values
        
        # Handle NaN values by masking
        valid_mask = pd.notna(row_nums) & pd.notna(col_nums)
        section_nums = np.full(len(table), np.nan)
        
        if valid_mask.any():
            section_nums[valid_mask] = idx_map[
                row_nums[valid_mask].astype(int), 
                col_nums[valid_mask].astype(int)
            ]
        
        # Create a new column with proper dtype handling
        section_series = pd.Series(section_nums, index=table.index)
        # Convert to nullable integer type first to handle NaN, then to categorical
        table[str(GRID.SECTION_NUM)] = section_series.astype('Int64').astype(np.uint16).astype('category')
        return table

    def _get_grid_info(self, image: Image, row_edges: np.ndarray, col_edges: np.ndarray) -> pd.DataFrame:
        """
        Assembles complete grid information from row and column edges.
        
        This helper method takes pre-calculated edge coordinates and generates a complete
        DataFrame with all grid metadata including row/column numbers and section numbers.
        This eliminates code duplication across different GridFinder implementations.
        
        Args:
            image (Image): The image object containing objects to be gridded.
            row_edges (np.ndarray): Array of row edge coordinates (length = nrows + 1).
            col_edges (np.ndarray): Array of column edge coordinates (length = ncols + 1).
            
        Returns:
            pd.DataFrame: Complete grid information table with ROW_NUM, COL_NUM, and SECTION_NUM columns.
        """
        info_table = image.objects.info(include_metadata=False)

        # Add row information
        info_table = self._add_row_number_info(table=info_table, row_edges=row_edges, imshape=image.shape)

        # Add column information
        info_table = self._add_col_number_info(table=info_table, col_edges=col_edges, imshape=image.shape)

        # Add section information
        info_table = self._add_section_number_info(table=info_table, row_edges=row_edges,
                                                   col_edges=col_edges, imshape=image.shape)

        return info_table
