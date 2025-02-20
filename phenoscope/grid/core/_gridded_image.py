from typing import Union, Tuple, TypeVar, Type, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from matplotlib.patches import Rectangle
from itertools import cycle
from skimage.color import label2rgb

from phenoscope import Image
from phenoscope.grid.grid_setter import OptimalCenterGridSetter
from phenoscope.features import BoundaryExtractor
from phenoscope.grid.interface import GridSetter
from phenoscope.grid.core._grid_handler import GridSubhandler
from phenoscope.util.constants import C_ImageHandler, C_ObjectInfo

GT = TypeVar('GT', bound='GriddedImage')


class GriddedImage(Image):
    def __init__(self, image: Optional[Union[np.ndarray, Image]] = None, grid_setter: Optional[GridSetter] = None, nrows=8, ncols=12):
        super().__init__(image)

        if grid_setter is None:
            grid_setter = OptimalCenterGridSetter(nrows=nrows, ncols=ncols)

        self._grid_setter = grid_setter
        self.__grid_subhandler = GridSubhandler(self)

    @property
    def grid(self):
        return self.__grid_subhandler

    @grid.setter
    def grid(self, grid):
        raise C_ImageHandler.IllegalAssignmentError('grid')

    def show_overlay(self, object_label=None,
                     show_gridlines: bool = True,
                     show_linreg: bool = False,
                     ax: plt.Axes = None,
                     figsize: Tuple[int, int] = (9, 10)
                     ) -> (plt.Figure, plt.Axes):
        fig, ax = super().show_overlay(object_label=object_label, ax=ax, figsize=figsize)

        if show_gridlines:
            col_edges = self.grid.get_col_edges()
            row_edges = self.grid.get_row_edges()
            ax.vlines(x=col_edges, ymin=row_edges.min(), ymax=row_edges.max(), colors='c', linestyles='--')
            ax.hlines(y=row_edges, xmin=col_edges.min(), xmax=col_edges.max(), color='c', linestyles='--')

            cmap = plt.get_cmap('tab20')
            cmap_cycle = cycle(cmap(i) for i in range(cmap.N))
            img = self.copy()
            img.obj_map = self.grid.get_section_map()
            gs_table = BoundaryExtractor().extract(img)

            # Add squares that denote object grid belonging. Useful for cases where objects are larger than grid sections
            for obj_label in gs_table.index.unique():
                subtable = gs_table.loc[obj_label, :]
                min_rr = subtable.loc[C_ObjectInfo.MIN_RR]
                max_rr = subtable.loc[C_ObjectInfo.MAX_RR]
                min_cc = subtable.loc[C_ObjectInfo.MIN_CC]
                max_cc = subtable.loc[C_ObjectInfo.MAX_CC]

                width = max_cc - min_cc
                height = max_rr - min_rr

                ax.add_patch(Rectangle(
                    (min_cc, min_rr), width=width, height=height,
                    edgecolor=next(cmap_cycle),
                    facecolor='none'
                )
                )

        return fig, ax