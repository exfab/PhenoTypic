from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage

import numpy as np
from typing import Tuple
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from skimage.color import label2rgb
from packaging import version
import math

import phenotypic
from phenotypic.core.accessors import ImageAccessor
from phenotypic.util.constants_ import OBJECT_INFO, GRID
from phenotypic.util.exceptions_ import NoObjectsError


class GridAccessor(ImageAccessor):
    """A class for accessing and manipulating grid-based data from a parent image object.

    This class is designed to facilitate operations on grid structures within a parent image. It provides methods
    for determining grid properties such as the number of rows and columns, retrieving grid-related information,
    and performing visualizations. The use of a parent image ensures that the grid operations are closely tied
    to a specific data context.

    Args:
        parent_image (GridImage): The parent image object which provides data and attributes to the grid accessor.

    Attributes:
        _parent_image (GridImage): Internal reference to the parent image object, used to derive grid data
            and perform grid-related operations.
    """

    def __init__(self, parent_image: GridImage):
        self._parent_image = parent_image

    @property
    def nrows(self) -> int:
        return self._parent_image._grid_setter.nrows

    @nrows.setter
    def nrows(self, nrows: int):
        if nrows < 1: raise ValueError('Number of rows must be greater than 0')
        if type(nrows) != int: raise TypeError('Number of rows must be an integer')

        self._parent_image._grid_setter.nrows = nrows

    @property
    def ncols(self) -> int:
        return self._parent_image._grid_setter.ncols

    @ncols.setter
    def ncols(self, ncols: int):
        if ncols < 1: raise ValueError('Number of columns must be greater than 0')
        if type(ncols) != int: raise TypeError('Number of columns must be an integer')

        self._parent_image._grid_setter.ncols = ncols

    def info(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing basic bounding box measurement data plus any object's grid membership.

        Returns:
            pd.DataFrame: A DataFrame with measurement data derived from the
            parent's image grid settings.
        """
        return self._parent_image._grid_setter.measure(self._parent_image)

    @property
    def _idx_ref_matrix(self):
        """Returns a matrix of grid positions to help with indexing"""
        return np.reshape(np.arange(self.nrows * self.ncols), newshape=(self.nrows, self.ncols))

    def __getitem__(self, idx):
        """
        Returns a crop of the grid section based on its flattened index.

        The grid is ordered from left to right, top to bottom. If no objects
        are present in the parent image, the original image is returned.

        Args:
            idx (int): The flattened index of the grid section to be
                extracted.

        Returns:
            phenotypic.Image: The cropped grid section as defined by the
            given flattened index, or the original parent image if no
            objects are present.
        """
        if self._parent_image.objects.num_objects != 0:
            min_coords, max_coords = self._adv_get_grid_section_slices(idx)
            min_rr, min_cc = min_coords
            max_rr, max_cc = max_coords

            section_image = phenotypic.Image(self._parent_image[int(min_rr):int(max_rr), int(min_cc):int(max_cc)])

            # Remove objects that don't belong in that grid section from the subimage
            objmap = section_image.objmap[:].copy()
            objmap[~np.isin(objmap, self._get_section_labels(idx))] = 0
            section_image.objmap = objmap

            return section_image
        else:
            return phenotypic.Image(self._parent_image)

    # This feels out of place. Maybe move to a measurement module in future updates
    def get_linreg_info(self, axis) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Returns the slope and intercept of a line of best fit across the objects of a certain axis.
        Args:
            axis: (int) 0=row-wise & 1=column-wise
        """
        if self._parent_image.objects.num_objects == 0:
            raise NoObjectsError(self._parent_image.name)
        if axis == 0:
            num_vectors = self.nrows
            x_group = GRID.GRID_ROW_NUM
            x_val = OBJECT_INFO.CENTER_CC
            y_val = OBJECT_INFO.CENTER_RR
        elif axis == 1:
            num_vectors = self.ncols
            x_group = GRID.GRID_COL_NUM
            x_val = OBJECT_INFO.CENTER_RR
            y_val = OBJECT_INFO.CENTER_CC
        else:
            raise ValueError('Axis should be 0 or 1.')

        # create persistent grid_info
        grid_info = self.info()

        # allocate empty vectors to store m & b for all values
        m_slope = np.full(shape=num_vectors, fill_value=np.nan)
        b_intercept = np.full(shape=num_vectors, fill_value=np.nan)

        # Collect slope & intercept for the rows or columns
        # Use 2D covariance/variance method for finding linear regression
        for idx in range(num_vectors):
            x = grid_info.loc[grid_info.loc[:, x_group] == idx, x_val].to_numpy()
            x_mean = np.mean(x)

            y = grid_info.loc[grid_info.loc[:, x_group] == idx, y_val].to_numpy()
            y_mean = np.mean(y)

            covariance = ((x - x_mean) * (y - y_mean)).sum()
            variance = ((x - x_mean) ** 2).sum()
            if variance != 0:
                m_slope[idx] = covariance / variance
                b_intercept[idx] = y_mean - m_slope[idx] * x_mean
            else:
                m_slope[idx] = 0
                b_intercept[idx] = y_mean if axis == 0 else x_mean

        return m_slope, np.round(b_intercept)

    """
    Grid Columns
    """

    def get_col_edges(self) -> np.ndarray:
        """Returns the column edges of the grid"""
        intervals = self.info().loc[:, GRID.GRID_COL_INTERVAL]
        left_edges = intervals.apply(
            lambda x: math.floor(x[0]) if math.floor(x[0]) > 0 else math.ceil(x[0])
        ).to_numpy()
        right_edges = intervals.apply(
            lambda x: math.ceil(x[1]) if math.ceil(x[1]) > 0 else math.floor(x[1])
        ).to_numpy()
        edges = np.unique(np.concatenate([left_edges, right_edges]))
        return edges.astype(int)

    def get_col_map(self) -> np.ndarray:
        """Returns a version of the object map with each object numbered according to their grid column number"""
        grid_info = self.info()
        col_map = self._parent_image.objmap[:]
        for n, col_bidx in enumerate(np.sort(grid_info.loc[:, GRID.GRID_COL_NUM].unique())):
            subtable = grid_info.loc[grid_info.loc[:, GRID.GRID_COL_NUM] == col_bidx, :]

            # Edit the new map's objects to equal the column number
            col_map[np.isin(
                element=self._parent_image.objmap[:],
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
            func_ax.imshow(label2rgb(label=self.get_col_map(), image=self._parent_image.enh_matrix[:]))
        else:
            func_ax.imshow(label2rgb(label=self.get_col_map(), image=self._parent_image.matrix[:]))

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
        intervals = self.info().loc[:, GRID.GRID_ROW_INTERVAL]
        left_edges = intervals.apply(
            lambda x: math.floor(x[0]) if math.floor(x[0]) > 0 else math.ceil(x[0])
        ).to_numpy()
        right_edges = intervals.apply(
            lambda x: math.ceil(x[1]) if math.ceil(x[1]) > 0 else math.floor(x[1])
        ).to_numpy()
        edges = np.unique(np.concatenate([left_edges, right_edges]))
        return edges.astype(int)

    def get_row_map(self) -> np.ndarray:
        """Returns a version of the object map with each object numbered according to their grid row number"""
        grid_info = self.info()
        row_map = self._parent_image.objmap[:]
        for n, col_bidx in enumerate(np.sort(grid_info.loc[:, GRID.GRID_ROW_NUM].unique())):
            subtable = grid_info.loc[grid_info.loc[:, GRID.GRID_ROW_NUM] == col_bidx, :]

            # Edit the new map's objects to equal the column number
            row_map[np.isin(
                element=self._parent_image.objmap[:],
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
            func_ax.imshow(label2rgb(label=self.get_row_map(), image=self._parent_image.enh_matrix[:]))
        else:
            func_ax.imshow(label2rgb(label=self.get_row_map(), image=self._parent_image.matrix[:]))

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

        section_map = self._parent_image.objmap[:]
        for n, bidx in enumerate(np.sort(grid_info.loc[:, GRID.GRID_SECTION_NUM].unique())):
            subtable = grid_info.loc[grid_info.loc[:, GRID.GRID_SECTION_NUM] == bidx, :]
            section_map[np.isin(
                element=self._parent_image.objmap[:],
                test_elements=subtable.index.to_numpy()
            )] = n + 1

        return section_map

    def get_section_counts(self, ascending=False) -> pd.DataFrame:
        """Returns a sorted dataframe with the number of objects within each section"""
        return self.info().loc[:, GRID.GRID_SECTION_NUM].value_counts().sort_values(ascending=ascending)

    def get_info_by_section(self, section_number):
        """ Get the grid info based on the section. Can be accessed by section number or row/column indexes

        Args:
            section_number:

        Returns:

        """
        if isinstance(section_number, int):  # Access by section number
            grid_info = self.info()
            return grid_info.loc[grid_info.loc[:, GRID.GRID_SECTION_NUM] == section_number, :]
        elif isinstance(section_number, tuple) and len(section_number) == 2:  # Access by row and col number
            grid_info = self.info()
            grid_info = grid_info.loc[grid_info.loc[:, GRID.GRID_ROW_NUM] == section_number[0], :]
            return grid_info.loc[grid_info.loc[:, GRID.GRID_ROW_NUM] == section_number[1], :]
        else:
            raise ValueError('Section index should be int or a tuple of indexes')

    def _naive_get_grid_section_slices(self, idx) -> ((int, int), (int, int)):
        """Returns the exact slices of a grid section based on its flattened index

        Note:
            - Can crop objects in the image

        Return:
            (int, int, int, int): ((MinRow, MinCol), (MaxRow, MaxCol)) The slices to extract the grid section from the image.
        """
        row_edges, col_edges = self.get_row_edges(), self.get_col_edges()
        row_pos, col_pos = np.where(self._idx_ref_matrix == idx)
        min_cc = col_edges[col_pos]
        max_cc = col_edges[col_pos + 1]
        min_rr = row_edges[row_pos]
        max_rr = row_edges[row_pos + 1]
        return (min_rr, min_cc), (max_rr, max_cc)

    def _adv_get_grid_section_slices(self, idx) -> ((int, int), (int, int)):
        """Returns the slices of a grid section based on its flattened index, and accounts for objects boundaries.

            Note:
                - Can crop objects in the image

            Return:
                (int, int, int, int): ((MinRow, MinCol), (MaxRow, MaxCol)) The slices to extract the grid section from the image.
        """
        grid_min, grid_max = self._naive_get_grid_section_slices(idx)
        grid_min_rr, grid_min_cc = grid_min
        grid_max_rr, grid_max_cc = grid_max

        grid_info = self.info()
        section_info = grid_info.loc[grid_info.loc[:, GRID.GRID_SECTION_NUM] == idx, :]

        obj_min_cc = section_info.loc[:, OBJECT_INFO.MIN_CC].min()
        min_cc = min(grid_min_cc, obj_min_cc)
        if min_cc < 0: min_cc = 0

        obj_max_cc = section_info.loc[:, OBJECT_INFO.MAX_CC].max()
        max_cc = max(grid_max_cc, obj_max_cc)
        if max_cc > self._parent_image.shape[1] - 1: max_cc = self._parent_image.shape[1] - 1

        obj_min_rr = section_info.loc[:, OBJECT_INFO.MIN_RR].min()
        min_rr = min(grid_min_rr, obj_min_rr)
        if min_rr < 0: min_rr = 0

        obj_max_rr = section_info.loc[:, OBJECT_INFO.MAX_RR].max()
        max_rr = max(grid_max_rr, obj_max_rr)
        if max_rr > self._parent_image.shape[0] - 1: max_rr = self._parent_image.shape[0] - 1

        return (min_rr, min_cc), (max_rr, max_cc)

    def _get_section_labels(self, idx) -> list[int]:
        """Returns a list of labels for a grid section based on its flattened index"""
        grid_info = self.info()
        section_info = grid_info.loc[grid_info.loc[:, GRID.GRID_SECTION_NUM] == idx, :]
        return section_info.index.to_list()
