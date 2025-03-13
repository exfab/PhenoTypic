from typing import Union, Tuple, TypeVar, Type, Optional

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from itertools import cycle

from phenoscope import Image
from phenoscope.measure import BoundaryExtractor
from phenoscope.grid.abstract import GridSetter
from phenoscope.util.constants import C_ImageHandler, C_ObjectInfo, C_ImageFormats
from phenoscope.grid import OptimalCenterGridSetter

from .accessors import GridAccessor

GT = TypeVar('GT', bound='GridImage')


class GridImage(Image):
    """
    Represents an image that supports grid-based processing and overlay visualization.

    This class extends the base `Image` class functionality to include grid handling,
    grid-based slicing, and advanced visualization capabilities such as displaying overlay information
    with gridlines and annotations. It interacts with the provided grid handling utilities
    to determine grid structure and assign/overlay it effectively on the image.

    Attributes:
        _grid_setter (Optional[GridSetter]): An object responsible for defining and optimizing the grid
            layout over the image, defaulting to an `OptimalCenterGridSetter` instance if none is provided.
        __grid_subhandler (GridAccessor): An internal utility for managing grid-based operations such as
            accessing row and column edges and generating section maps for the image's grid system.
    """

    def __init__(self, input_image: Optional[Union[np.ndarray, Type[Image]]] = None, input_schema: str = None,
                 grid_setter: Optional[GridSetter] = None,
                 nrows: int = 8, ncols: int = 12):
        super().__init__(input_image=input_image, input_schema=input_schema)

        if hasattr(input_image, '_grid_setter'):
            grid_setter = input_image._grid_setter
        elif grid_setter is None:
            grid_setter = OptimalCenterGridSetter(nrows=nrows, ncols=ncols)

        self._grid_setter: Optional[GridSetter] = grid_setter
        self.__grid_subhandler = GridAccessor(self)

    @property
    def grid(self) -> GridAccessor:
        """Returns the GridAccessor object for managing grid-related operations.

        Returns:
            GridAccessor: Provides access to Grid-related operations.

        See Also :class:`GridAccessor`
        """
        return self.__grid_subhandler

    @grid.setter
    def grid(self, grid):
        raise C_ImageHandler.IllegalAssignmentError('grid')

    def __getitem__(self, key) -> Image:
        """Returns a copy of the image at the slices specified as a regular Image object.

        Returns:
            Image: A copy of the image at the slices indicated
        """
        if self.schema not in C_ImageFormats.MATRIX_FORMATS:
            subimage = Image(input_image=self.array[key], input_schema=self.schema)
        else:
            subimage = Image(input_image=self.matrix[key], input_schema=self.schema)

        subimage.enh_matrix[:] = self.enh_matrix[key]
        subimage.omap[:] = self.omap[key]
        return subimage

    def show_overlay(self, object_label: Optional[int] = None,
                     show_gridlines: bool = True,
                     show_linreg: bool = False,
                     figsize: Tuple[int, int] = (9, 10),
                     annotate: bool = False,
                     annotation_size: int = 12,
                     annotation_color: str = 'white',
                     annotation_facecolor: str = 'red',
                     ax: plt.Axes = None,
                     ) -> (plt.Figure, plt.Axes):
        """
        Displays an overlay of data with optional annotations, linear regression lines, and gridlines on a
        grid-based figure. The figure can be customized with various parameters to suit visualization needs.

        Args:
            object_label (Optional): Specific label of the object to highlight or focus on in the overlay.
            show_gridlines (bool): Whether or not to include gridlines on the overlay. Defaults to True.
            show_linreg (bool): Indicate whether to display linear regression lines on the overlay. Defaults to False.
            figsize (Tuple[int, int]): Size of the figure, specified as a tuple of width and height values (in inches).
                Defaults to (9, 10).
            annotate (bool): Determines whether points or objects should be annotated. Defaults to False.
            annotation_size (int): Font size for the annotations. Defaults to 12.
            annotation_color (str): Color for annotation text. Defaults to 'white'.
            annotation_facecolor (str): Background color for annotation highlights. Defaults to 'red'.
            ax (plt.Axes, optional): Axis on which to draw the overlay; can be provided externally. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Modified figure and axis containing the rendered overlay.
        """
        fig, ax = super().show_overlay(object_label=object_label, ax=ax, figsize=figsize,
                                       annotate=annotate, annotation_size=annotation_size,
                                       annotation_color=annotation_color, annotation_facecolor=annotation_facecolor
                                       )

        if show_gridlines:
            col_edges = self.grid.get_col_edges()
            row_edges = self.grid.get_row_edges()
            ax.vlines(x=col_edges, ymin=row_edges.min(), ymax=row_edges.max(), colors='c', linestyles='--')
            ax.hlines(y=row_edges, xmin=col_edges.min(), xmax=col_edges.max(), color='c', linestyles='--')

            cmap = plt.get_cmap('tab20')
            cmap_cycle = cycle(cmap(i) for i in range(cmap.N))
            img = self.copy()
            img.omap = self.grid.get_section_map()
            gs_table = BoundaryExtractor().measure(img)

            # Add squares that denote object grid belonging. Useful for cases where objects are larger than grid sections
            for obj_label in gs_table.index.unique():
                subtable = gs_table.loc[obj_label, :]
                min_rr = subtable.loc[C_ObjectInfo.MIN_RR]
                max_rr = subtable.loc[C_ObjectInfo.MAX_RR]
                min_cc = subtable.loc[C_ObjectInfo.MIN_CC]
                max_cc = subtable.loc[C_ObjectInfo.MAX_CC]

                width = max_cc - min_cc
                height = max_rr - min_rr

                ax.add_patch(
                    Rectangle(
                        (min_cc, min_rr), width=width, height=height,
                        edgecolor=next(cmap_cycle),
                        facecolor='none'
                    )
                )

        return fig, ax
