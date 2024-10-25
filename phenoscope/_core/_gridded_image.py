from ._image import Image

from ..interface import GridFeatureExtractor
from ..feature_extraction import GridSectionExtractor, BoundaryExtractor
from ..util.type_checks import is_binary_mask
from ..util.error_message import INVALID_MASK_SHAPE_MSG, INVALID_MAP_SHAPE_MSG

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import cycle
from skimage.color import label2rgb
from typing import Union
from skimage.measure import label


class GriddedImage(Image):
    def __init__(self, image: Union[np.ndarray, Image], n_rows=8, n_cols=12, gridding_method: GridFeatureExtractor = None):
        if isinstance(image, np.ndarray) or isinstance(image, Image):
            super().__init__(image)
        else:
            raise ValueError('Input should be either an image array or a phenoscope.Image object.')
        if gridding_method is None is None:
            self._gs_extractor = GridSectionExtractor(n_rows=n_rows, n_cols=n_cols)
        else:
            self._gs_extractor = gridding_method

    @property
    def grid_info(self) -> pd.DataFrame:
        if self.object_map is None: raise ValueError('Image object map is empty. Apply a detector first.')
        return self._gs_extractor.extract(self)

    # Grid Column Kit
    @property
    def grid_col_edges(self) -> np.ndarray:
        left_edges = self.grid_info.loc[:, 'grid_col_intervals'].apply(lambda x: x.left).to_numpy()
        right_edges = self.grid_info.loc[:, 'grid_col_intervals'].apply(lambda x: x.right).to_numpy()

        edges = np.unique(np.concatenate([left_edges, right_edges]))
        return edges

    @property
    def grid_col_map(self) -> np.ndarray:
        _tmp_table: pd.DataFrame = self.grid_info
        _new_map: np.ndarray = self.object_map
        for n, col_bindex in enumerate(np.sort(_tmp_table.loc[:, 'grid_col_bin'].unique())):
            subtable = _tmp_table.loc[_tmp_table.loc[:, 'grid_col_bin'] == col_bindex, :]
            _new_map[np.isin(element=self.object_map, test_elements=subtable.index.to_numpy())] = n + 1
        return _new_map

    def show_column_overlay(self, use_enhanced=False, show_edges=False, ax=None, figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(label2rgb(label=self.grid_col_map, image=self.enhanced_array))
        else:
            func_ax.imshow(label2rgb(label=self.grid_col_map, image=self.array))

        if show_edges:
            col_edges = self.grid_col_edges
            row_edges = self.grid_row_edges
            func_ax.vlines(x=col_edges, ymin=row_edges.min(), ymax=row_edges.max(), colors='c', linestyles='--')

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    # Grid Row Kit
    @property
    def grid_row_edges(self) -> np.ndarray:
        left_edges = self.grid_info.loc[:, 'grid_row_intervals'].apply(lambda x: x.left).to_numpy()
        right_edges = self.grid_info.loc[:, 'grid_row_intervals'].apply(lambda x: x.right).to_numpy()

        edges = np.unique(np.concatenate([left_edges, right_edges]))
        return edges

    @property
    def grid_row_map(self) -> np.ndarray:
        _tmp_table: pd.DataFrame = self.grid_info
        _new_map: np.ndarray = self.object_map
        for n, row_bindex in enumerate(np.sort(_tmp_table.loc[:, 'grid_row_bin'].unique())):
            subtable = _tmp_table.loc[_tmp_table.loc[:, 'grid_row_bin'] == row_bindex, :]
            _new_map[np.isin(element=self.object_map, test_elements=subtable.index.to_numpy())] = n + 1
        return _new_map

    def show_row_overlay(self, use_enhanced=False, show_edges=False, ax=None, figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(label2rgb(label=self.grid_row_map, image=self.enhanced_array))
        else:
            func_ax.imshow(label2rgb(label=self.grid_row_map, image=self.array))

        if show_edges:
            col_edges = self.grid_col_edges
            row_edges = self.grid_row_edges
            func_ax.hlines(y=row_edges, xmin=col_edges.min(), xmax=col_edges.max(), color='c', linestyles='--')

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    # Grid Section Kit
    @property
    def grid_section_map(self) -> np.ndarray:
        _tmp_table: pd.DataFrame = self.grid_info
        _new_map: np.ndarray = self.object_map
        for n, section_bindex in enumerate(np.sort(_tmp_table.loc[:, 'grid_section_bin'].unique())):
            subtable = _tmp_table.loc[_tmp_table.loc[:, 'grid_section_bin'] == section_bindex, :]
            _new_map[np.isin(element=self.object_map, test_elements=subtable.index.to_numpy())] = n + 1
        return _new_map

    def show_section_overlay(self, use_enhanced=False, show_edges=False, ax=None, figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(label2rgb(label=self.grid_section_map, image=self.enhanced_array))
        else:
            func_ax.imshow(label2rgb(label=self.grid_section_map, image=self.array))

        if show_edges:
            col_edges = self.grid_col_edges
            row_edges = self.grid_row_edges
            func_ax.vlines(x=col_edges, ymin=row_edges.min(), ymax=row_edges.max(), colors='c', linestyles='--')
            func_ax.hlines(y=row_edges, xmin=col_edges.min(), xmax=col_edges.max(), color='c', linestyles='--')

        cmap = plt.get_cmap('tab20')
        cmap_cycle = cycle(cmap(i) for i in range(cmap.N))
        img = self.copy()
        img.object_map = self.grid_section_map
        gs_table = BoundaryExtractor().extract(img)
        for obj_label in gs_table.index.unique():
            color = next(cmap_cycle)

            subtable = gs_table.loc[obj_label, :]
            min_rr = subtable.loc['min_rr']
            max_rr = subtable.loc['max_rr']
            min_cc = subtable.loc['min_cc']
            max_cc = subtable.loc['max_cc']

            width = max_cc - min_cc
            height = max_rr - min_rr

            func_ax.add_patch(Rectangle(
                    (min_cc, min_rr), width=width, height=height,
                    edgecolor=color,
                    facecolor='none'
            ))

        if ax is None:
            return fig, func_ax
        else:
            return func_ax
