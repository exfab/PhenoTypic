from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import pandas as pd
import numpy as np

from phenotypic.abc_ import GridMeasureFeatures
from phenotypic.tools.constants_ import BBOX, GRID
from abc import ABC


class GridFinder(GridMeasureFeatures, ABC):
    """Abstract base class for detecting grid structure and assigning objects to wells.

    GridFinder is the foundation for grid detection algorithms in arrayed plate imaging.
    It detects the row and column spacing of colonies on agar plates and assigns each
    detected object to its corresponding grid cell (well). This is essential for
    high-throughput phenotyping experiments where samples are arranged in regular grids
    (e.g., 96-well, 384-well formats).

    **What it does:**

    GridFinder implementations analyze the spatial distribution of detected objects in
    an image and determine the underlying grid structure. They compute pixel coordinates
    where grid rows and columns are located (row_edges and col_edges), then use these
    edges to assign each object to a row number, column number, and section number
    (unique well identifier).

    **Why it's important for colony phenotyping:**

    In arrayed plate experiments, colonies are grown at fixed positions corresponding to
    wells in a microplate. By mapping detected colonies to grid positions, downstream
    analysis can:

    - Correlate colony measurements with sample metadata (what was inoculated in each well)
    - Track growth across replicate wells
    - Identify spatial patterns or contamination
    - Export results organized by well coordinates for database import

    Without grid assignment, measurements are just unorganized lists of objects with no
    link to experimental design.

    **Grid concepts:**

    - **Row edges:** Array of pixel row coordinates where rows begin/end. For an 8-row
      grid, this is an array of 9 values: [0, y1, y2, ..., y8, image_height].
    - **Column edges:** Array of pixel column coordinates where columns begin/end. For a
      12-column grid, this is an array of 13 values: [0, x1, x2, ..., x12, image_width].
    - **Grid cell assignment:** Each object's center is tested against row/column edges
      using pd.cut(), assigning it to row i (0 to nrows-1) and column j (0 to ncols-1).
    - **Section number:** A unique well ID computed as row*ncols + col, ordered from
      top-left (0) to bottom-right (nrows*ncols - 1).

    **Typical plate formats:**

    - 96-well plate: 8 rows × 12 columns (A1-H12)
    - 384-well plate: 16 rows × 24 columns (A1-P24)

    **Attributes:**
        nrows (int): Number of rows in the grid. For 96-well plates, this is 8.
        ncols (int): Number of columns in the grid. For 96-well plates, this is 12.

    **Abstract Methods:**

        You must implement these two methods in subclasses:

        - _operate(image: Image) -> pd.DataFrame: Main entry point. Should compute
          row and column edges, then call _get_grid_info() to assemble and return
          the complete grid DataFrame.
        - get_row_edges(image: Image) -> np.ndarray: Return array of row edge pixel
          coordinates. Length must be nrows + 1.
        - get_col_edges(image: Image) -> np.ndarray: Return array of column edge pixel
          coordinates. Length must be ncols + 1.

    **Helper Methods for Implementation:**

        These protected methods reduce code duplication when implementing _operate():

        - _get_grid_info(image, row_edges, col_edges) -> pd.DataFrame: Assembles
          complete grid information from pre-computed edge coordinates. This method
          automatically calls _add_row_number_info(), _add_col_number_info(), and
          _add_section_number_info() to populate all required columns. Use this in
          your _operate() implementation after computing edges.

    **Output Format:**

        The _operate() method returns a pandas DataFrame with detected objects and
        their grid assignments:

        - ROW_NUM: Grid row index (0 to nrows-1)
        - COL_NUM: Grid column index (0 to ncols-1)
        - SECTION_NUM: Well identifier (0 to nrows*ncols-1), ordered left-to-right,
          top-to-bottom
        - Additional columns: Object metadata (centroid, bounding box, etc.) from
          image.objects.info()

        Objects that fall outside all grid cells (due to edge clipping or misalignment)
        will have NaN values in grid columns.

    **Concrete Implementations:**

        PhenoTypic provides two built-in implementations:

        - **AutoGridFinder:** Automatically optimizes row and column edge positions
          using scipy.optimize.minimize_scalar to minimize the MSE between object
          centroids and grid bin midpoints. Useful when grid position is unknown.
        - **ManualGridFinder:** User specifies exact row and column edge coordinates
          (e.g., from manual measurement or calibration). Use when you know the exact
          grid position.

    **Notes:**

        - GridFinder subclasses can work with regular Image objects, not just GridImage.
        - Edge coordinates should always be sorted in ascending order (handled by
          _clip_row_edges and _clip_col_edges).
        - Ensure row_edges and col_edges are clipped to image bounds to prevent indexing
          errors.
        - Grid assignment uses pandas.cut() with include_lowest=True and right=True,
          meaning objects are assigned based on which interval they fall into.

    **Examples:**

        .. dropdown:: Create a ManualGridFinder for a 96-well plate with known geometry

            For example, if a microscope image of a 96-well plate is 2048×3072 pixels
            and wells are evenly spaced, you might manually define:

            .. code-block:: python

                import numpy as np
                from phenotypic import Image
                from phenotypic.grid import ManualGridFinder
                from phenotypic.detect import OtsuDetector

                # Load image of 96-well plate
                image = Image.from_image_path("plate_scan.jpg")

                # Detect colonies
                detector = OtsuDetector()
                image_with_objects = detector.operate(image)

                # Define grid for 8 rows × 12 columns
                # Rows: 8 wells vertically, spaced from pixel 100 to 2000
                row_edges = np.array([100, 350, 600, 850, 1100, 1350, 1600, 1850, 2100])
                # Columns: 12 wells horizontally, spaced from pixel 50 to 3050
                col_edges = np.linspace(50, 3050, 13, dtype=int)

                # Create grid finder and assign colonies to wells
                grid_finder = ManualGridFinder(row_edges=row_edges, col_edges=col_edges)
                grid_df = grid_finder.measure(image_with_objects)

                # Result has columns: ROW_NUM, COL_NUM, SECTION_NUM, plus object info
                print(grid_df[['ROW_NUM', 'COL_NUM', 'SECTION_NUM']])

        .. dropdown:: Use AutoGridFinder when grid position is unknown

            When the image is rotated, shifted, or otherwise misaligned, let
            AutoGridFinder automatically compute optimal edge positions:

            .. code-block:: python

                from phenotypic.grid import AutoGridFinder
                from phenotypic import Image
                from phenotypic.detect import OtsuDetector

                # Load and detect colonies
                image = Image.from_image_path("rotated_plate.jpg")
                detector = OtsuDetector()
                image_with_objects = detector.operate(image)

                # AutoGridFinder optimizes edge positions to align with detected colonies
                grid_finder = AutoGridFinder(nrows=8, ncols=12, tol=0.01)
                grid_df = grid_finder.measure(image_with_objects)

                # Grid assignment is robust to rotation and minor misalignment
                print(f"Found {len(grid_df)} colonies assigned to grid")

        .. dropdown:: Understanding SECTION_NUM for well mapping

            SECTION_NUM provides a single integer ID for each well, useful for
            organizing results or looking up sample metadata:

            .. code-block:: python

                # Example: 8×12 grid (96-well plate)
                # SECTION_NUM runs 0-95, numbered left-to-right, top-to-bottom
                # Section 0 = Row 0, Col 0 (top-left, A1)
                # Section 11 = Row 0, Col 11 (top-right, A12)
                # Section 12 = Row 1, Col 0 (second row left, B1)
                # Section 95 = Row 7, Col 11 (bottom-right, H12)

                grid_df = grid_finder.measure(image_with_objects)

                # Filter colonies in a specific well
                section_5_objects = grid_df[grid_df['SectionNum'] == 5]

                # Map section numbers back to well coordinates
                well_row = section_num // 12
                well_col = section_num % 12
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
        This method is to returns the row edges of the grid as numpy rgb.
        Args:
            image (Image): Image object.
        Returns:
            np.ndarray: Row-edges of the grid.
        """
        pass

    @abc.abstractmethod
    def get_col_edges(self, image: Image) -> np.ndarray:
        """
        This method is to returns the column edges of the grid as numpy rgb.
        Args:
            image:

        Returns:
            np.ndarray: Column-edges of the grid.

        """
        pass

    @staticmethod
    def _clip_row_edges(row_edges, imshape: (int, int, ...)) -> np.ndarray:
        return np.clip(a=row_edges, a_min=0, a_max=imshape[0])

    def _add_row_number_info(
        self, table: pd.DataFrame, row_edges: np.array, imshape: (int, int)
    ) -> pd.DataFrame:
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

    def _add_col_number_info(
        self, table: pd.DataFrame, col_edges: np.array, imshape: (int, int)
    ) -> pd.DataFrame:
        col_edges = self._clip_col_edges(col_edges=col_edges, imshape=imshape)
        table.loc[:, str(GRID.COL_NUM)] = pd.cut(
            table.loc[:, str(BBOX.CENTER_CC)],
            bins=col_edges,
            labels=range(self.ncols),
            include_lowest=True,
            right=True,
        )
        return table

    def _add_section_number_info(
        self,
        table: pd.DataFrame,
        row_edges: np.array,
        col_edges: np.array,
        imshape: (int, int),
    ) -> pd.DataFrame:
        # Ensure ROW_NUM and COL_NUM exist
        if str(GRID.ROW_NUM) not in table.columns:
            self._add_row_number_info(table=table, row_edges=row_edges, imshape=imshape)
        if str(GRID.COL_NUM) not in table.columns:
            self._add_col_number_info(table=table, col_edges=col_edges, imshape=imshape)

        # Create section number directly from row and column indices
        idx_map = np.reshape(
            np.arange(self.nrows * self.ncols), (self.nrows, self.ncols)
        )

        # Compute section number for each row using vectorized operations
        row_nums = table.loc[:, str(GRID.ROW_NUM)].values
        col_nums = table.loc[:, str(GRID.COL_NUM)].values

        # Handle NaN values by masking
        valid_mask = pd.notna(row_nums) & pd.notna(col_nums)
        section_nums = np.full(len(table), np.nan)

        if valid_mask.any():
            section_nums[valid_mask] = idx_map[
                row_nums[valid_mask].astype(int), col_nums[valid_mask].astype(int)
            ]

        # Create a new column with proper dtype handling
        section_series = pd.Series(section_nums, index=table.index)
        # Convert to nullable integer type first to handle NaN, then to categorical
        table[str(GRID.SECTION_NUM)] = (
            section_series.astype("Int64").astype(np.uint16).astype("category")
        )
        return table

    def _get_grid_info(
        self, image: Image, row_edges: np.ndarray, col_edges: np.ndarray
    ) -> pd.DataFrame:
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
        info_table = self._add_row_number_info(
            table=info_table, row_edges=row_edges, imshape=image.shape
        )

        # Add column information
        info_table = self._add_col_number_info(
            table=info_table, col_edges=col_edges, imshape=image.shape
        )

        # Add section information
        info_table = self._add_section_number_info(
            table=info_table,
            row_edges=row_edges,
            col_edges=col_edges,
            imshape=image.shape,
        )

        return info_table
