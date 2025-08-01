from typing import Union, Tuple, Type, Optional, Literal

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Rectangle
from itertools import cycle

from .._image import Image
from phenotypic.measure import MeasureBounds
from phenotypic.abstract import GridFinder
from phenotypic.util.constants_ import IMAGE_TYPES, BBOX, METADATA_LABELS
from phenotypic.util.exceptions_ import IllegalAssignmentError
from phenotypic.grid import OptimalCenterGridFinder

from phenotypic.core._image_parts.accessors import GridAccessor


class ImageGridHandler(Image):
    """
    A specialized Image object that supports grid-based processing and overlay visualization.

    This class extends the base `Image` class functionality to include grid handling,
    grid-based slicing, and advanced visualization capabilities such as displaying overlay information
    with gridlines and annotations. It interacts with the provided grid handling utilities
    to determine grid structure and assign/overlay it effectively on the image.

    Args:
            input_image (Optional[Union[np.ndarray, Type[Image]]]): The input_image
                image, which can be a NumPy array or an image-like object. If
                this parameter is not provided, it defaults to None.
            imformat (str): A string representing the schema of the input_image
                image. It defaults to None if not provided.
            grid_finder (Optional[GridFinder]): An optional GridFinder instance
                for defining grids on the image. If not provided, it defaults to
                a center grid setter.
            nrows (int): An integer passed to the grid setter to specify the number of rows in the grid
                (Defaults to 8).
            ncols (int): An integer passed to the grid setter to specify the number of columns in the grid
                (Defaults to 12).

    Attributes:
        _grid_setter (Optional[GridFinder]): An object responsible for defining and optimizing the grid
            layout over the image, defaulting to an `OptimalCenterGridSetter` instance if none is provided.
        _accessors.grid (GridAccessor): An internal utility for managing grid-based operations such as
            accessing row and column edges and generating section maps for the image's grid system.
    """

    def __init__(self, input_image: Optional[Union[np.ndarray, Image]] = None, imformat: str = None,
                 name: str = None, grid_finder: Optional[GridFinder] = None,
                 nrows: int = 8, ncols: int = 12):
        """
        Initializes the instance with the given image, format, grid finding
        mechanism, and dimensions of the grid.

        Args:
            input_image (Optional[Union[np.ndarray, Image]]): The input image provided
                as a NumPy array or an image object. Can be None if the image is
                optional for initialization.
            imformat (str): The string representing the image format.
            name (str): The name identifier for the image.
            grid_finder (Optional[GridFinder]): Mechanism responsible for finding a grid
                within the image. If None, an optimal center grid finder is instantiated.
            nrows (int): Number of rows in the grid. Defaults to 8.
            ncols (int): Number of columns in the grid. Defaults to 12.

        Attributes:
            _grid_setter (Optional[GridFinder]): Private attribute storing the grid finding
                mechanism, which is either passed as input or is generated internally.
            _accessors.grid (GridAccessor): The grid accessor object for managing and
                accessing grid-related functionalities.
        """
        super().__init__(input_image=input_image, imformat=imformat, name=name)

        if hasattr(input_image, '_grid_setter'):
            grid_finder = input_image._grid_setter
        elif grid_finder is None:
            grid_finder = OptimalCenterGridFinder(nrows=nrows, ncols=ncols)

        self._grid_setter: Optional[GridFinder] = grid_finder
        self._accessors.grid = GridAccessor(self)
        self.metadata[METADATA_LABELS.IMAGE_TYPE] = IMAGE_TYPES.GRID.value

    @property
    def grid(self) -> GridAccessor:
        """Returns the GridAccessor object for grid-related operations.

        Returns:
            GridAccessor: Provides access to Grid-related operations.

        See Also :class:`GridAccessor`
        """
        return self._accessors.grid

    @grid.setter
    def grid(self, grid):
        raise IllegalAssignmentError('grid')

    def info(self, include_metadata:bool=True)->pd.DataFrame:
        return self.grid.info(include_metadata=include_metadata)

    @property
    def nrows(self) -> int:
        """
        Retrieves the number of rows in the grid.

        This property is used to access the number of rows present in the grid
        object. It encapsulates the `nrows` attribute of the `grid` and returns
        it as an integer.

        Returns:
            int: The number of rows in the grid.
        """
        return self._grid_setter.nrows

    @nrows.setter
    def nrows(self, nrows):
        """
        Sets the number of rows in the grid. Ensures that the provided value is of the correct type.

        Args:
            nrows (int): The number of rows to set. Must be an integer.

        Raises:
            TypeError: If the provided value for nrows is not of type int.
        """
        if not isinstance(nrows, int):
            raise TypeError(f'Expected int, got {type(nrows)}')
        self._grid_setter.nrows = nrows

    @property
    def ncols(self) -> int:
        """
        Gets the number of columns in the grid.

        This property retrieves the total number of columns in the grid
        by accessing the corresponding attribute of the underlying grid
        instance. It provides a read-only interface to the `ncols` value.

        Returns:
            int: The number of columns in the grid.
        """
        return self._grid_setter.ncols

    @ncols.setter
    def ncols(self, ncols):
        """
        Setter for the 'ncols' attribute ensuring it is assigned a valid value. The 'ncols'
        attribute defines the number of columns in the grid structure. The method validates
        the data type of the input and guarantees that only integers are accepted. If an invalid
        type is provided, a TypeError exception is raised.

        Args:
            ncols: The number of columns to set for the grid structure.

        Raises:
            TypeError: If the provided value for 'ncols' is not of type int.
        """
        if not isinstance(ncols, int):
            raise TypeError(f'Expected int, got {type(ncols)}')
        self._grid_setter.ncols = ncols

    def __getitem__(self, key) -> Image:
        """Returns a copy of the image at the slices specified as a regular Image object.

        Returns:
            Image: A copy of the image at the slices indicated
        """
        if self._image_format.is_array():
            subimage = Image(input_image=self.array[key], imformat=self.imformat)
        else:
            subimage = Image(input_image=self.matrix[key], imformat=self.imformat)

        subimage.enh_matrix[:] = self.enh_matrix[key]
        subimage.objmap[:] = self.objmap[key]
        return subimage

    def show_overlay(self, object_label: Optional[int] = None,
                     show_gridlines: bool = True,
                     show_linreg: bool = False,
                     figsize: Tuple[int, int] = (9, 10),
                     show_labels: bool = False,
                     annotation_kwargs: None | dict = None,
                     ax: plt.Axes = None,
                     ) -> (plt.Figure, plt.Axes):
        """
        Displays an overlay of data with optional annotations, linear regression lines, and gridlines on a
        grid-based figure. The figure can be customized with various parameters to suit visualization needs.

        Args:
            object_label (Optional): Specific label of the object to highlight or focus on in the overlay.
            show_gridlines (bool): Whether to include gridlines on the overlay. Defaults to True.
            show_linreg (bool): Indicate whether to display linear regression lines on the overlay. Defaults to False.
            figsize (Tuple[int, int]): Size of the figure, specified as a tuple of width and height values (in inches).
                Defaults to (9, 10).
            show_labels (bool): Determines whether points or objects should be annotated. Defaults to False.
            annotation_kwargs (None | dict): Additional parameters for customization of the
                object annotations. Defaults: size=12, color='white', facecolor='red'. Other kwargs
                are passed to the matplotlib.axes.text () method.
            ax (plt.Axes, optional): Axis on which to draw the overlay; can be provided externally. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]: Modified figure and axis containing the rendered overlay.
        """
        fig, ax = super().show_overlay(
            object_label=object_label, ax=ax, figsize=figsize,
            show_labels=show_labels, annotation_kwargs=annotation_kwargs,
        )

        if show_gridlines and self.num_objects > 0:
            col_edges = self.grid.get_col_edges()
            upper_col_edges = col_edges[1:]
            lower_col_edges = col_edges[:-1]

            # Set x-axes labels to grid column numbers
            col_centers = ((upper_col_edges - lower_col_edges) // 2) + lower_col_edges
            ax.set_xticks(col_centers)
            ax.set_xticklabels(np.arange(self.ncols))

            row_edges = self.grid.get_row_edges()
            upper_row_edges = row_edges[1:]
            lower_row_edges = row_edges[:-1]

            # Set y-axis labels to grid row numbers
            row_centers = ((upper_row_edges - lower_row_edges) // 2) + lower_row_edges
            ax.set_yticks(row_centers)
            ax.set_yticklabels(np.arange(self.nrows))

            # Draw grid lines
            ax.vlines(x=col_edges, ymin=row_edges.min(), ymax=row_edges.max(), colors='c', linestyles='--')
            ax.hlines(y=row_edges, xmin=col_edges.min(), xmax=col_edges.max(), color='c', linestyles='--')

            cmap = plt.get_cmap('tab20')
            cmap_cycle = cycle(cmap(i) for i in range(cmap.N))
            img = self.copy()
            img.objmap = self.grid.get_section_map()
            gs_table = MeasureBounds().measure(img)

            # Add squares that denote object grid belonging. Useful for cases where objects are larger than grid sections
            for obj_label in gs_table.index.unique():
                subtable = gs_table.loc[obj_label, :]
                min_rr = subtable.loc[str(BBOX.MIN_RR)]
                max_rr = subtable.loc[str(BBOX.MAX_RR)]
                min_cc = subtable.loc[str(BBOX.MIN_CC)]
                max_cc = subtable.loc[str(BBOX.MAX_CC)]

                width = max_cc - min_cc
                height = max_rr - min_rr

                ax.add_patch(
                    Rectangle(
                        (min_cc, min_rr), width=width, height=height,
                        edgecolor=next(cmap_cycle),
                        facecolor='none',
                    ),
                )

        return fig, ax
