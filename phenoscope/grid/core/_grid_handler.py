import numpy as np
from typing import Tuple
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from skimage.color import label2rgb

from phenoscope import Image
from phenoscope.util.constants import C_ObjectInfo, C_Grid


class GridSubhandler:
    """An accessor object for the different information relating to an image's grid

    """

    def __init__(self, handler):
        self._handler = handler

    @property
    def nrows(self):
        return self._handler._grid_setter.nrows

    @nrows.setter
    def nrows(self, nrows):
        self._handler._grid_setter.nrows = nrows

    @property
    def ncols(self):
        return self._handler._grid_setter.ncols

    @ncols.setter
    def ncols(self, ncols):
        self._handler._grid_setter.ncols = ncols

    def info(self):
        return self._handler._grid_setter.extract(self._handler)

    @property
    def _idx_ref_matrix(self):
        """Returns a matrix of grid positions to help with indexing"""
        return np.reshape(np.arange(self.nrows * self.ncols), newshape=(self.nrows, self.ncols))

    def __getitem__(self, idx):
        """Returns a crop of the grid section based on it's flattened index. Ordered left to right, top to bottom."""
        row_edges, col_edges = self.get_row_edges(), self.get_col_edges()
        row_pos, col_pos = np.where(self._idx_ref_matrix == idx)
        min_cc = col_edges[col_pos]
        max_cc = col_edges[col_pos + 1]
        min_rr = row_edges[row_pos]
        max_rr = row_edges[row_pos + 1]
        return Image(self._handler[int(min_rr):int(max_rr), int(min_cc):int(max_cc)])

    def get_linreg_info(self, axis) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Returns the slope and intercept of a line of best fit across the objects of a certain axis.
        Args:
            axis: (int) 0=row-wise & 1=column-wise
        """
        if axis == 0:
            N = self.nrows
            x_group = C_Grid.GRID_ROW_NUM
            x_val = C_ObjectInfo.CENTER_CC
            y_val = C_ObjectInfo.CENTER_RR
        elif axis == 1:
            N = self.ncols
            x_group = C_Grid.GRID_COL_NUM
            x_val = C_ObjectInfo.CENTER_RR
            y_val = C_ObjectInfo.CENTER_CC
        else:
            raise ValueError('Axis should be 0 or 1.')

        # Generate & temporarilty cache grid_info to reduce runtime
        grid_info = self.info()

        # Create empty vectores to store m & b for all values
        m_slope = np.full(shape=N, fill_value=np.nan)
        b_intercept = np.full(shape=N, fill_value=np.nan)

        # Collect slope & intercept for the rows or columns
        for idx in range(N):
            warnings.simplefilter('ignore',
                                  np.RankWarning
                                  )  # TODO: When upgrading numpy version this will need to change
            m_slope[idx], b_intercept[idx] = np.polyfit(
                x=grid_info.loc[grid_info.loc[:, x_group] == idx, x_val],
                y=grid_info.loc[grid_info.loc[:, x_group] == idx, y_val],
                deg=1
            )
        return m_slope, np.round(b_intercept)

    """
    Grid Columns
    """

    def get_col_edges(self) -> np.ndarray:
        grid_info = self.info()
        left_edges = grid_info.loc[:, C_Grid.GRID_COL_INTERVAL].apply(
            lambda x: x[0]
        ).to_numpy()
        right_edges = grid_info.loc[:, C_Grid.GRID_COL_INTERVAL].apply(
            lambda x: x[1]
        ).to_numpy()
        edges = np.unique(np.concatenate([left_edges, right_edges]))
        return edges.astype(int)

    def get_col_map(self) -> np.ndarray:
        """Returns a version of the object map with each object numbered according to their grid column number"""
        grid_info = self.info()
        col_map = self._handler.obj_map[:]
        for n, col_bidx in enumerate(np.sort(grid_info.loc[:, C_Grid.GRID_COL_NUM].unique())):
            subtable = grid_info.loc[grid_info.loc[:, C_Grid.GRID_COL_NUM] == col_bidx, :]

            # Edit the new map's objects to equal the column number
            col_map[np.isin(
                element=self._handler.obj_map[:],
                test_elements=subtable.index.to_numpy()
            )] = n + 1
        return col_map

    def show_column_overlay(self, use_enhanced=False, show_gridlines=True, ax=None,
                            figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(label2rgb(label=self.get_col_map(), image=self._handler.det_matrix[:]))
        else:
            func_ax.imshow(label2rgb(label=self.get_col_map(), image=self._handler.matrix[:]))

        if show_gridlines:
            col_edges = self.get_col_edges()
            row_edges = self.get_row_edges()
            func_ax.vlines(x=col_edges, ymin=row_edges.min(), ymax=row_edges.max(), colors='c', linestyles='--')

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    """
    Grid Rows
    """

    def get_row_edges(self) -> np.ndarray:
        """Returns the row edges of the grid"""
        grid_info = self.info()
        left_edges = grid_info.loc[:, C_Grid.GRID_ROW_INTERVAL].apply(
            lambda x: x[0]
        ).to_numpy()
        right_edges = grid_info.loc[:, C_Grid.GRID_ROW_INTERVAL].apply(
            lambda x: x[1]
        ).to_numpy()
        edges = np.unique(np.concatenate([left_edges, right_edges]))
        return edges.astype(int)

    def get_row_map(self) -> np.ndarray:
        """Returns a version of the object map with each object numbered according to their grid row number"""
        grid_info = self.info()
        row_map = self._handler.obj_map[:]
        for n, col_bidx in enumerate(np.sort(grid_info.loc[:, C_Grid.GRID_ROW_NUM].unique())):
            subtable = grid_info.loc[grid_info.loc[:, C_Grid.GRID_ROW_NUM] == col_bidx, :]

            # Edit the new map's objects to equal the column number
            row_map[np.isin(
                element=self._handler.obj_map[:],
                test_elements=subtable.index.to_numpy()
            )] = n + 1
        return row_map

    def show_row_overlay(self, use_enhanced=False, show_gridlines=True, ax=None,
                         figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(label2rgb(label=self.get_row_map(), image=self._handler.det_matrix[:]))
        else:
            func_ax.imshow(label2rgb(label=self.get_row_map(), image=self._handler.matrix[:]))

        if show_gridlines:
            col_edges = self.get_col_edges()
            row_edges = self.get_row_edges()
            func_ax.vlines(x=col_edges, ymin=row_edges.min(), ymax=row_edges.max(), colors='c', linestyles='--')

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    """
    Grid Sections
    """

    def get_section_map(self) -> np.ndarray:
        """Returns a version of the object map with each object numbered according to their section number"""
        grid_info = self.info()

        section_map = self._handler.obj_map[:]
        for n, bidx in enumerate(np.sort(grid_info.loc[:, C_Grid.GRID_SECTION_NUM].unique())):
            subtable = grid_info.loc[grid_info.loc[:, C_Grid.GRID_SECTION_NUM] == bidx, :]
            section_map[np.isin(
                element=self._handler.obj_map[:],
                test_elements=subtable.index.to_numpy()
            )] = n + 1

        return section_map

    def get_section_counts(self, ascending=False) -> pd.DataFrame:
        """Returns a sorted dataframe with the number of objects within each section"""
        return self.info().loc[:, C_Grid.GRID_SECTION_NUM].value_counts().sort_values(ascending=ascending)

    def get_info_by_section(self, idx):
        """ Get the grid info based on the section. Can be accessed by section number or row/column indexes

        Args:
            idx:

        Returns:

        """
        if isinstance(idx, int):  # Access by section number
            grid_info = self.info()
            return grid_info.loc[grid_info.loc[:, C_Grid.GRID_SECTION_NUM] == idx, :]
        elif isinstance(idx, tuple) and len(idx) == 2:  # Access by row and col number
            grid_info = self.info()
            grid_info = grid_info.loc[grid_info.loc[:, C_Grid.GRID_ROW_NUM] == idx[0], :]
            return grid_info.loc[grid_info.loc[:, C_Grid.GRID_ROW_NUM] == idx[1], :]
        else:
            raise ValueError('Section index should be int or a tuple of indexes')
