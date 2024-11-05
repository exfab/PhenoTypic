from __future__ import annotations
from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING: from phenoscope.grid.interface import GridExtractor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from matplotlib.patches import Rectangle
from itertools import cycle
from skimage.color import label2rgb

from phenoscope import Image
from phenoscope.grid.grid_extractor import OptimalCenterGridExtractor
from phenoscope.feature_extraction import BoundaryExtractor


class GriddedImage(Image):
    def __init__(self, image: Union[np.ndarray, Image], n_rows=8, n_cols=12, gridding_method: GridExtractor = None):
        if isinstance(image, np.ndarray):
            super().__init__(image)
        elif isinstance(image, Image):
            super().__init__(image)
            if hasattr(image, '_grid_extractor'):
                self._grid_extractor = image.grid_extractor
        else:
            raise ValueError('Input should be either an image array or a phenoscope.Image object.')

        if gridding_method is None:
            self._grid_extractor = OptimalCenterGridExtractor(n_rows=n_rows, n_cols=n_cols)
        else:
            self._grid_extractor = gridding_method
            self._grid_extractor.n_rows = n_rows
            self._grid_extractor.n_cols = n_cols

        self._bound_extractor = BoundaryExtractor()

    def __getitem__(self, index):
        if len(index) != 2: raise ValueError(
                'Image objects only support 2-dimensional slicing. RGB images will be sliced evenly across each channel.')
        if self.color_array is not None:
            new_img = Image(self.color_array[index])
            new_img.array = self.array[index]
            new_img.enhanced_array = self.enhanced_array[index]
        else:
            new_img = Image(self.array[index])
            new_img.enhanced_array = self.enhanced_array[index]

        if self.object_mask is not None:
            new_img.object_mask = self.object_mask[index]

        if self.object_map is not None:
            new_img.object_map = self.object_map[index]

        return new_img

    @property
    def bound_extractor(self):
        return self._bound_extractor

    @property
    def grid_extractor(self) -> GridExtractor:
        return self._grid_extractor

    @grid_extractor.setter
    def grid_extractor(self, grid_extractor):
        if issubclass(grid_extractor, GridExtractor):
            self._grid_extractor = grid_extractor
        else:
            raise ValueError('grid_extractor should be an subclass of GridExtractor.')

    @property
    def n_rows(self):
        return self._grid_extractor.n_rows

    @n_rows.setter
    def n_rows(self, value):
        self._grid_extractor.n_rows = value

    @property
    def n_cols(self):
        return self._grid_extractor.n_cols

    @n_cols.setter
    def n_cols(self, value):
        self._grid_extractor.n_cols = value

    @property
    def grid_info(self) -> pd.DataFrame:
        if self.object_map is None: raise ValueError('Image object map is empty. Apply a detector first.')
        return self._grid_extractor.extract(self)

    def get_linreg_info(self, axis) -> Tuple[np.ndarray[float], np.ndarray[float]]:
        """
        Returns the slope and intercept of a line of best fit across the objects of a certain axis.
        :param axis: (int) 0=row-wise & 1=column-wise
        :return:
        """
        if axis == 0:
            N = self.n_rows
            x_group = self.grid_extractor.LABEL_GRID_ROW_NUM
            x_val = self._bound_extractor.LABEL_CENTER_CC
            y_val = self._bound_extractor.LABEL_CENTER_RR
        elif axis == 1:
            N = self.n_cols
            x_group = self.grid_extractor.LABEL_GRID_COL_NUM
            x_val = self._bound_extractor.LABEL_CENTER_RR
            y_val = self._bound_extractor.LABEL_CENTER_CC
        else:
            raise ValueError('Axis should be 0 or 1.')

        # Generate & temporarilty cache grid_info to reduce runtime
        grid_info = self.grid_info

        # Create empty vectores to store m & b for all values
        m_slope = np.full(shape=N, fill_value=np.nan)
        b_intercept = np.full(shape=N, fill_value=np.nan)

        # Collect slope & intercept for the rows or columns
        for idx in range(N):
            warnings.simplefilter('ignore', np.RankWarning)  # TODO: When upgrading numpy version this will need to change
            m_slope[idx], b_intercept[idx] = np.polyfit(
                    x=grid_info.loc[grid_info.loc[:, x_group] == idx, x_val],
                    y=grid_info.loc[grid_info.loc[:, x_group] == idx, y_val],
                    deg=1
            )
        return m_slope, np.round(b_intercept)

    # Grid Column Implementation
    @property
    def grid_col_edges(self) -> np.ndarray:
        """
        returns the grid's column edges used to create boundaries
        :return:
        """
        left_edges = self.grid_info.loc[:, self._grid_extractor.LABEL_GRID_COL_INTERVAL].apply(lambda x: x[0]).to_numpy()
        right_edges = self.grid_info.loc[:, self._grid_extractor.LABEL_GRID_COL_INTERVAL].apply(lambda x: x[1]).to_numpy()

        edges = np.unique(np.concatenate([left_edges, right_edges]))
        return edges.astype(int)

    @property
    def grid_col_map(self) -> np.ndarray:
        _tmp_table: pd.DataFrame = self.grid_info
        _new_map: np.ndarray = self.object_map
        for n, col_bindex in enumerate(np.sort(_tmp_table.loc[:, self._grid_extractor.LABEL_GRID_COL_NUM].unique())):
            subtable = _tmp_table.loc[_tmp_table.loc[:, self._grid_extractor.LABEL_GRID_COL_NUM] == col_bindex, :]
            _new_map[np.isin(element=self.object_map, test_elements=subtable.index.to_numpy())] = n + 1
        return _new_map

    def get_col_info(self, idx):
        if idx not in range(self.n_cols):
            raise ValueError(f'Index {idx} is out of range. Index for this image should be between 0 - {self.n_cols - 1}.')

        grid_info = self.grid_info
        return grid_info.loc[grid_info.loc[:, self._grid_extractor.LABEL_GRID_COL_NUM] == idx, :]

    def show_column_overlay(self, use_enhanced=False, show_gridlines=True, ax=None, figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(label2rgb(label=self.grid_col_map, image=self.enhanced_array))
        else:
            func_ax.imshow(label2rgb(label=self.grid_col_map, image=self.array))

        if show_gridlines:
            col_edges = self.grid_col_edges
            row_edges = self.grid_row_edges
            func_ax.vlines(x=col_edges, ymin=row_edges.min(), ymax=row_edges.max(), colors='c', linestyles='--')

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    # Grid Row Implementation
    @property
    def grid_row_edges(self) -> np.ndarray:
        left_edges = self.grid_info.loc[:, self._grid_extractor.LABEL_GRID_ROW_INTERVAL].apply(lambda x: x[0]).to_numpy()
        right_edges = self.grid_info.loc[:, self._grid_extractor.LABEL_GRID_ROW_INTERVAL].apply(lambda x: x[1]).to_numpy()

        edges = np.unique(np.concatenate([left_edges, right_edges]))
        return edges.astype(int)

    @property
    def grid_row_map(self) -> np.ndarray:
        _tmp_table: pd.DataFrame = self.grid_info
        _new_map: np.ndarray = self.object_map
        for n, row_bindex in enumerate(np.sort(_tmp_table.loc[:, self._grid_extractor.LABEL_GRID_ROW_NUM].unique())):
            subtable = _tmp_table.loc[_tmp_table.loc[:, self._grid_extractor.LABEL_GRID_ROW_NUM] == row_bindex, :]
            _new_map[np.isin(element=self.object_map, test_elements=subtable.index.to_numpy())] = n + 1
        return _new_map

    def get_row_info(self, idx: int):
        if idx not in range(self.n_rows):
            raise ValueError(f'Index {idx} is out of range. Index for this image should be between 0 - {self.n_rows - 1}.')

        grid_info = self.grid_info
        return grid_info.loc[grid_info.loc[:, self._grid_extractor.LABEL_GRID_ROW_NUM] == idx, :]

    def show_row_overlay(self, use_enhanced=False, show_gridlines=True, ax=None, figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(label2rgb(label=self.grid_row_map, image=self.enhanced_array))
        else:
            func_ax.imshow(label2rgb(label=self.grid_row_map, image=self.array))

        if show_gridlines:
            col_edges = self.grid_col_edges
            row_edges = self.grid_row_edges
            func_ax.hlines(y=row_edges, xmin=col_edges.min(), xmax=col_edges.max(), color='c', linestyles='--')

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    # Grid Section Implementation
    @property
    def grid_section_map(self) -> np.ndarray:
        _tmp_table: pd.DataFrame = self.grid_info
        _new_map: np.ndarray = self.object_map

        # Get a map with each object label being changed to its bin representation
        for n, section_bindex in enumerate(np.sort(_tmp_table.loc[:, self._grid_extractor.LABEL_GRID_SECTION_IDX].unique())):
            subtable = _tmp_table.loc[_tmp_table.loc[:, self._grid_extractor.LABEL_GRID_SECTION_IDX] == section_bindex, :]
            _new_map[np.isin(element=self.object_map, test_elements=subtable.index.to_numpy())] = n + 1
        return _new_map

    def get_section_info(self, row_idx, col_idx):
        if row_idx not in range(self.n_rows):
            raise ValueError(f'Index {row_idx} is out of range. Index for this image should be between 0 - {self.n_rows - 1}.')
        if col_idx not in range(self.n_cols):
            raise ValueError(f'Index {col_idx} is out of range. Index for this image should be between 0 - {self.n_cols - 1}.')

        grid_info = self.grid_info
        grid_info = grid_info.loc[grid_info.loc[:, self._grid_extractor.LABEL_GRID_ROW_NUM] == row_idx, :]
        return grid_info.loc[grid_info.loc[:, self._grid_extractor.LABEL_GRID_COL_NUM] == col_idx, :]

    def get_section_count(self, ascending=False):
        return self.grid_info.loc[:, self.grid_extractor.LABEL_GRID_SECTION_NUM].value_counts().sort_values(ascending=ascending)

    def show_overlay(self, use_enhanced=False, show_gridlines=True, show_linreg=False, ax=None, figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(label2rgb(label=self.grid_section_map, image=self.enhanced_array))
        else:
            func_ax.imshow(label2rgb(label=self.grid_section_map, image=self.array))

        if show_gridlines:
            col_edges = self.grid_col_edges
            row_edges = self.grid_row_edges
            func_ax.vlines(x=col_edges, ymin=row_edges.min(), ymax=row_edges.max(), colors='c', linestyles='--')
            func_ax.hlines(y=row_edges, xmin=col_edges.min(), xmax=col_edges.max(), color='c', linestyles='--')

        cmap = plt.get_cmap('tab20')
        cmap_cycle = cycle(cmap(i) for i in range(cmap.N))
        img = self.copy()
        img.object_map = self.grid_section_map
        gs_table = self._bound_extractor.extract(img)
        for obj_label in gs_table.index.unique():
            subtable = gs_table.loc[obj_label, :]
            min_rr = subtable.loc[self._bound_extractor.LABEL_MIN_RR]
            max_rr = subtable.loc[self._bound_extractor.LABEL_MAX_RR]
            min_cc = subtable.loc[self._bound_extractor.LABEL_MIN_CC]
            max_cc = subtable.loc[self._bound_extractor.LABEL_MAX_CC]

            width = max_cc - min_cc
            height = max_rr - min_rr

            func_ax.add_patch(Rectangle(
                    (min_cc, min_rr), width=width, height=height,
                    edgecolor=next(cmap_cycle),
                    facecolor='none'
            ))

        if ax is None:
            return fig, func_ax
        else:
            return func_ax
