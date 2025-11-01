from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import pandas as pd
import numpy as np

from phenotypic.ABC_ import GridFinder


class ManualGridFinder(GridFinder):
    """
    A GridFinder implementation where users directly specify grid row and column coordinates.

    This class allows for complete manual control over grid placement by accepting
    explicit row and column edge coordinates. No optimization or automatic calculation
    is performed - the grid is defined exactly as specified by the user.

    Attributes:
        nrows (int): Number of rows in the grid (derived from row_edges).
        ncols (int): Number of columns in the grid (derived from col_edges).
        row_edges (np.ndarray): Array of row edge coordinates defining grid rows.
        col_edges (np.ndarray): Array of column edge coordinates defining grid columns.

    Example:
        >>> # Create a 3x4 grid with specific coordinates
        >>> row_edges = np.array([0, 100, 200, 300])  # 3 rows
        >>> col_edges = np.array([0, 80, 160, 240, 320])  # 4 columns
        >>> finder = ManualGridFinder(row_edges=row_edges, col_edges=col_edges)
        >>> grid_info = finder.measure(image)
    """

    def __init__(self, row_edges: np.ndarray, col_edges: np.ndarray):
        """
        Initialize a ManualGridFinder with explicit row and column edge coordinates.

        Args:
            row_edges (np.ndarray): Array of row edge coordinates. Length should be nrows + 1.
                Example: [0, 100, 200, 300] defines 3 rows.
            col_edges (np.ndarray): Array of column edge coordinates. Length should be ncols + 1.
                Example: [0, 80, 160, 240, 320] defines 4 columns.

        Raises:
            ValueError: If row_edges or col_edges have fewer than 2 elements.
        """
        if len(row_edges) < 2:
            raise ValueError("row_edges must have at least 2 elements to define at least 1 row")
        if len(col_edges) < 2:
            raise ValueError("col_edges must have at least 2 elements to define at least 1 column")

        self._row_edges = np.asarray(row_edges, dtype=int)
        self._col_edges = np.asarray(col_edges, dtype=int)

        # Ensure edges are sorted
        self._row_edges.sort()
        self._col_edges.sort()

        # Set nrows and ncols based on edge arrays
        self.nrows: int = len(self._row_edges) - 1
        self.ncols: int = len(self._col_edges) - 1

    def _operate(self, image: Image) -> pd.DataFrame:
        """
        Processes an image to assign objects to grid cells based on manually specified edges.

        Args:
            image (Image): The image containing objects to be gridded.

        Returns:
            pd.DataFrame: A DataFrame containing the grid results including boundary intervals,
                grid indices, and section numbers corresponding to the manually defined grid.
        """
        # Use base class method to assemble grid info with our predefined edges
        return self._get_grid_info(image=image, row_edges=self._row_edges, col_edges=self._col_edges)

    def get_row_edges(self, image: Image) -> np.ndarray:
        """
        Returns the manually specified row edges.

        Args:
            image (Image): The image (not used, but required by interface).

        Returns:
            np.ndarray: Array of row edge coordinates.
        """
        return self._row_edges.copy()

    def get_col_edges(self, image: Image) -> np.ndarray:
        """
        Returns the manually specified column edges.

        Args:
            image (Image): The image (not used, but required by interface).

        Returns:
            np.ndarray: Array of column edge coordinates.
        """
        return self._col_edges.copy()
