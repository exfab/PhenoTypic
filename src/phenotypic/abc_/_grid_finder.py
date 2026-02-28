from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import pandas as pd
import numpy as np

from phenotypic.abc_ import GridMeasureFeatures
from phenotypic.tools_.measurement_info_ import BBOX, GRID
from abc import ABC


class GridFinder(GridMeasureFeatures, ABC):
    """Abstract base class for detecting grid structure and assigning objects to wells.

    GridFinder is the foundation for grid detection algorithms in arrayed plate imaging.
    It detects the row and column spacing of colonies on agar plates and assigns each
    detected object to its corresponding grid cell (well). This is essential for
    high-throughput phenotyping experiments where samples are arranged in regular grids
    (e.g., 96-well, 384-well formats).

    **Quick Decision Guide**

    Use [AutoGridFinder](src/phenotypic/grid/_auto_grid_finder.py) when:
    - Grid position is unknown or image is rotated/shifted
    - Colonies are detected but well boundaries are unclear
    - You want automatic optimization of row/column edge positions
    - Tolerance parameter allows tuning optimization precision

    Use [ManualGridFinder](src/phenotypic/grid/_manual_grid_finder.py) when:
    - You know exact grid geometry from microscope calibration
    - Grid position is fixed and repeatable across images
    - You have pre-measured row and column edge coordinates
    - You want deterministic, non-optimized grid assignment

    Combining with detection pipelines:
    - Use GridFinder after ObjectDetector to map colonies to wells
    - AutoGridFinder works with any detection result
    - ManualGridFinder requires pre-computed edge coordinates
    - Grid assignment is independent of detection algorithm

    **What it does**

    GridFinder implementations analyze the spatial distribution of detected objects in
    an image and determine the underlying grid structure. They compute pixel coordinates
    where grid rows and columns are located (row_edges and col_edges), then use these
    edges to assign each object to a row number, column number, and section number
    (unique well identifier).

    **Why it's important for colony phenotyping**

    In arrayed plate experiments, colonies are grown at fixed positions corresponding to
    wells in a microplate. By mapping detected colonies to grid positions, downstream
    analysis can:

    - **Sample tracking:** Correlate colony measurements with sample metadata inoculated
      in each well
    - **Replicate analysis:** Track growth across identical replicate wells
    - **Spatial detection:** Identify contamination patterns or edge effects
    - **Data export:** Organize results by well coordinates for database import and
      statistical analysis

    Without grid assignment, measurements are just unorganized lists of objects with no
    link to experimental design.

    **Grid concepts**

    - **Row edges:** Array of pixel row coordinates marking row boundaries. For 8 rows,
      array has 9 values: [0, y1, y2, ..., y8, image_height]. Objects between row_edges[i]
      and row_edges[i+1] belong to row i.
    - **Column edges:** Array of pixel column coordinates marking column boundaries. For
      12 columns, array has 13 values: [0, x1, x2, ..., x12, image_width]. Objects between
      col_edges[j] and col_edges[j+1] belong to column j.
    - **Grid cell assignment:** Each object's centroid (center_rr, center_cc) is tested
      against row/column edges using pd.cut(), assigning it to row i (0 to nrows-1) and
      column j (0 to ncols-1).
    - **Section number:** A unique well ID computed as row*ncols + col, ordered left-to-right,
      top-to-bottom (top-left well = 0, bottom-right well = nrows*ncols - 1).

    **Typical plate formats**

    - **96-well plate:** 8 rows × 12 columns (A1-H12)
    - **384-well plate:** 16 rows × 24 columns (A1-P24)
    - **1536-well plate:** 32 rows × 48 columns (very high-throughput)

    **Attributes:**
        nrows (int): Number of rows in the grid. For 96-well plates, typically 8.
        ncols (int): Number of columns in the grid. For 96-well plates, typically 12.

    **Abstract Methods**

    Subclasses must implement these methods:

    - **_operate(image: Image) -> pd.DataFrame:** Main entry point. Compute row and
      column edges, then call _get_grid_info() to assemble the complete grid DataFrame.
      Return the DataFrame with all grid assignments.
    - **get_row_edges(image: Image) -> np.ndarray:** Return array of row edge pixel
      coordinates. Length must be exactly nrows + 1 (e.g., 9 values for 8 rows).
    - **get_col_edges(image: Image) -> np.ndarray:** Return array of column edge pixel
      coordinates. Length must be exactly ncols + 1 (e.g., 13 values for 12 columns).

    **Helper Methods for Implementation**

    These protected methods reduce code duplication and handle grid assignment:

    - **_get_grid_info(image, row_edges, col_edges) -> pd.DataFrame:** Assembles
      complete grid information from pre-computed edge coordinates. Calls internal
      methods to populate ROW_NUM, COL_NUM, and ROW_MAJOR_IDX columns. Use this after
      computing edges in your _operate() implementation.
    - **_add_row_number_info():** Assigns row indices using pd.cut() with object
      centroids and row edges.
    - **_add_col_number_info():** Assigns column indices using pd.cut() with object
      centroids and column edges.
    - **_add_section_number_info():** Computes section numbers from row and column
      indices using vectorized operations.
    - **_clip_row_edges() / _clip_col_edges():** Ensures edge coordinates are clipped
      to image bounds (prevents indexing errors).

    **Output Format**

    The _operate() method returns a pandas DataFrame with detected objects and their
    grid assignments:

    - **ROW_NUM:** Grid row index (0 to nrows-1), representing vertical well position
    - **COL_NUM:** Grid column index (0 to ncols-1), representing horizontal well position
    - **ROW_MAJOR_IDX:** Well identifier (0 to nrows*ncols-1), ordered left-to-right,
      top-to-bottom for convenient database mapping
    - **Additional columns:** Object metadata (centroid, bounding box, morphology) from
      image.objects.info()

    Objects that fall outside all grid cells (due to edge clipping or misalignment)
    have NaN values in ROW_NUM, COL_NUM, and ROW_MAJOR_IDX columns.

    **Concrete Implementations**

    PhenoTypic provides two built-in GridFinder implementations:

    - [AutoGridFinder](src/phenotypic/grid/_auto_grid_finder.py): Automatically optimizes
      row and column edge positions using scipy.optimize.minimize_scalar. Minimizes MSE
      between object centroids and grid bin midpoints. Use when grid position is unknown.
    - [ManualGridFinder](src/phenotypic/grid/_manual_grid_finder.py): User specifies exact
      row and column edge coordinates from calibration or measurement. Use when grid geometry
      is known and fixed.

    **Optimization Strategy (for AutoGridFinder)**

    AutoGridFinder uses an iterative optimization approach:

    - **Objective function:** Minimize mean squared error (MSE) between object centroids
      and their nearest grid bin midpoints
    - **Optimization method:** scipy.optimize.minimize_scalar with bounded search
    - **Starting point:** Heuristic estimates from object distribution (dividing image
      into nrows/ncols regions)
    - **Convergence:** Stops when relative tolerance is met or max iterations reached
    - **Result:** Optimal row_edges and col_edges that best align detected colonies to
      grid structure

    Example optimization pattern:

        >>> from scipy.optimize import minimize_scalar
        >>> # Pseudo-code for optimization loop
        >>> # def objective(edges): return mse(colony_positions, grid_bins(edges))
        >>> # result = minimize_scalar(objective, bounds=(min_edge, max_edge), method='bounded')
        >>> # optimal_edges = result.x

    **Notes**

    - GridFinder subclasses work with regular Image objects (not just GridImage)
    - Edge coordinates must be sorted in ascending order (handled by _clip_row_edges
      and _clip_col_edges)
    - Ensure row_edges and col_edges are clipped to image bounds to prevent indexing
      errors
    - Grid assignment uses pandas.cut() with include_lowest=True and right=True, meaning
      objects are assigned based on which interval they fall into
    - NaN values in grid columns indicate objects outside all grid cells

    **Examples**

        Use AutoGridFinder when grid position is unknown:

        When the image is rotated, shifted, or geometry is unclear, let AutoGridFinder
        automatically compute optimal edge positions by optimizing alignment:

        >>> from phenotypic.grid import AutoGridFinder
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> # Load and detect colonies on plate
        >>> image = load_synth_yeast_plate()
        >>> detector = OtsuDetector()
        >>> image_with_objects = detector.apply(image)
        >>> # AutoGridFinder optimizes edge positions to align with colonies
        >>> grid_finder = AutoGridFinder(nrows=8, ncols=12, tol=0.01)
        >>> grid_df = grid_finder.measure(image_with_objects)
        >>> # Access well assignments
        >>> print(f"Found {len(grid_df)} colonies assigned to grid")
        >>> print(grid_df[['ROW_NUM', 'COL_NUM', 'ROW_MAJOR_IDX']].head())

        Create a ManualGridFinder for a 96-well plate with known geometry:

        When grid geometry is known from microscope calibration, manually specify
        row and column edges for reproducible grid assignment:

        >>> import numpy as np
        >>> from phenotypic.grid import ManualGridFinder
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> # Load and detect colonies
        >>> image = load_synth_yeast_plate()
        >>> detector = OtsuDetector()
        >>> image_with_objects = detector.apply(image)
        >>> # Define grid for 8 rows x 12 columns (96-well)
        >>> # Rows: 8 wells vertically, evenly spaced from pixel 100 to 2000
        >>> row_edges = np.linspace(100, 2000, 9, dtype=int)
        >>> # Columns: 12 wells horizontally, evenly spaced from pixel 50 to 3050
        >>> col_edges = np.linspace(50, 3050, 13, dtype=int)
        >>> # Create grid finder with known edge coordinates
        >>> grid_finder = ManualGridFinder(row_edges=row_edges, col_edges=col_edges)
        >>> grid_df = grid_finder.measure(image_with_objects)
        >>> # Result includes grid assignments plus object metadata
        >>> print(grid_df[['ROW_NUM', 'COL_NUM', 'ROW_MAJOR_IDX']].head())

        Understanding ROW_MAJOR_IDX for well mapping:

        ROW_MAJOR_IDX provides a single integer ID for each well, useful for organizing
        results and correlating with sample metadata:

        >>> from phenotypic.grid import AutoGridFinder
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> # Detect and assign colonies to grid
        >>> image = load_synth_yeast_plate()
        >>> detector = OtsuDetector()
        >>> image_with_objects = detector.apply(image)
        >>> grid_finder = AutoGridFinder(nrows=8, ncols=12)
        >>> grid_df = grid_finder.measure(image_with_objects)
        >>> # Example: 8x12 grid (96-well plate)
        >>> # ROW_MAJOR_IDX runs 0-95, numbered left-to-right, top-to-bottom
        >>> # Section 0 = Row 0, Col 0 (top-left, A1)
        >>> # Section 11 = Row 0, Col 11 (top-right, A12)
        >>> # Section 12 = Row 1, Col 0 (second row left, B1)
        >>> # Section 95 = Row 7, Col 11 (bottom-right, H12)
        >>> # Filter colonies in a specific well
        >>> section_5_objects = grid_df[grid_df['ROW_MAJOR_IDX'] == 5]
        >>> # Map section numbers back to well coordinates
        >>> well_row = 5 // 12  # Row index
        >>> well_col = 5 % 12   # Column index
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
        table[str(GRID.ROW_MAJOR_IDX)] = (
            section_series.astype("Int64").astype(np.uint16).astype("category")
        )

        # Column-major index (col * nrows + row)
        col_major_map = np.reshape(
            np.arange(self.nrows * self.ncols),
            (self.nrows, self.ncols),
            order="F",
        )
        col_major_nums = np.full(len(table), np.nan)
        if valid_mask.any():
            col_major_nums[valid_mask] = col_major_map[
                row_nums[valid_mask].astype(int),
                col_nums[valid_mask].astype(int),
            ]
        col_major_series = pd.Series(col_major_nums, index=table.index)
        table[str(GRID.COL_MAJOR_IDX)] = (
            col_major_series.astype("Int64").astype(np.uint16).astype("category")
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
            pd.DataFrame: Complete grid information table with ROW_NUM, COL_NUM, ROW_MAJOR_IDX, and COL_MAJOR_IDX columns.
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
