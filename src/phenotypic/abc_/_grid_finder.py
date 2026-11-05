from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import pandas as pd
import numpy as np

from phenotypic.abc_ import GridMeasureFeatures
from phenotypic.schema import OBJECT
from phenotypic.schema import BBOX, GRID


class GridFinder(GridMeasureFeatures, ABC):
    """Abstract base class for detecting grid structure and assigning objects to wells.

    GridFinder is the foundation for grid detection algorithms in arrayed plate imaging.
    It detects the row and column spacing of colonies on agar plates and assigns each
    detected object to its corresponding grid cell (well). This is essential for
    high-throughput phenotyping experiments where samples are arranged in regular grids
    (e.g., 96-well, 384-well formats).

    **Quick Decision Guide**

    Use [CenteredAutoGridFinder](src/phenotypic/grid/_centered_auto_grid_finder.py)
    (the default, which a ``GridImage`` builds when given no finder) when:
    - Grid position is unknown and the plate is roughly centered in the frame
    - Some rows or columns are empty, at the edges or inside the plate: it fits
      the grid by its periodicity rather than by the span of the colonies
    - Colonies are detected but well boundaries are unclear

    Use [ManualGridFinder](src/phenotypic/grid/_manual_grid_finder.py) when:
    - You know exact grid geometry from microscope calibration
    - Grid position is fixed and repeatable across images
    - You have pre-measured row and column edge coordinates
    - You want deterministic, non-optimized grid assignment

    Combining with detection pipelines:
    - Use GridFinder after ObjectDetector to map colonies to wells
    - CenteredAutoGridFinder works with any detection result
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

    - **ROW_NUM** (``Grid_RowNum``): Grid row index (0 to nrows-1), representing
      vertical well position
    - **COL_NUM** (``Grid_ColNum``): Grid column index (0 to ncols-1), representing
      horizontal well position
    - **ROW_MAJOR_IDX** (``Grid_RowMajorIdx``): Well identifier (0 to nrows*ncols-1),
      ordered left-to-right, top-to-bottom for convenient database mapping
    - **Additional columns:** Object metadata (centroid, bounding box, morphology) from
      image.objects.info()

    Objects whose centers fall outside the grid edges are clamped to the nearest
    edge cell (row 0 or nrows-1, col 0 or ncols-1).

    **Concrete Implementations**

    PhenoTypic provides three built-in GridFinder implementations:

    - [CenteredAutoGridFinder](src/phenotypic/grid/_centered_auto_grid_finder.py): the
      default. Fits a regular axis-aligned grid to the colony centers by their
      periodicity, so empty edge or interior rows do not shift it. Assumes the plate
      is roughly centered in the frame.
    - [AutoGridFinder](src/phenotypic/grid/_auto_grid_finder.py): Deterministic robust fit
      using weighted object centers. Estimates pitch from center range, fits grid indices
      via least-squares, rejects outliers, and refits. Robust to protruding colonies
      (e.g., filamentous fungi). Its pitch comes from the span of the centers, so it
      suits plates whose outer rows and columns are occupied.
    - [ManualGridFinder](src/phenotypic/grid/_manual_grid_finder.py): User specifies exact
      row and column edge coordinates from calibration or measurement. Use when grid geometry
      is known and fixed.

    **Fitting Strategy (for AutoGridFinder)**

    AutoGridFinder uses a deterministic center-based robust fit:

    - **Centers:** Extract weighted centroids from detected objects, sorted along each axis
    - **Pitch estimate:** ``(max_center - min_center) / (n_expected - 1)`` (robust to multiple objects per cell)
    - **Grid indices:** ``round((center - min_center) / pitch)``
    - **Least-squares fit:** ``center = pitch * idx + offset`` via closed-form normal equations
    - **Outlier rejection:** Remove centers where ``|residual| > pitch * residual_fraction``
    - **Refit:** Refined pitch and offset from inliers only
    - **Symmetry anchoring:** When detected span < expected, center the grid in the image
    - **Edges:** ``offset + pitch * i - pitch/2`` for ``i = 0..n``, clipped to image bounds

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

        Assign colonies to wells with the default grid finder:

        When the grid position is unknown, ``CenteredAutoGridFinder`` fits the
        grid to the detected colonies. Select the grid columns by their schema
        names:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.grid import CenteredAutoGridFinder
        >>> from phenotypic.schema import GRID
        >>> # Load and detect colonies on plate
        >>> image = OtsuDetector().apply(load_synth_yeast_plate())
        >>> grid_finder = CenteredAutoGridFinder(nrows=8, ncols=12)
        >>> grid_df = grid_finder.measure(image)
        >>> # One row per detected object, with its well assignment
        >>> print(f"Assigned {len(grid_df)} objects to the grid")
        Assigned 552 objects to the grid
        >>> well_columns = [str(GRID.ROW_NUM), str(GRID.COL_NUM), str(GRID.ROW_MAJOR_IDX)]
        >>> print(grid_df[well_columns].head())
          Grid_RowNum Grid_ColNum Grid_RowMajorIdx
        0           0           1                1
        1           0          10               10
        2           0           1                1
        3           0           1                1
        4           0          10               10
        >>> # 8 rows x 12 columns need 9 row edges and 13 column edges
        >>> len(grid_finder.get_row_edges(image)), len(grid_finder.get_col_edges(image))
        (9, 13)

        Create a ManualGridFinder for a 96-well plate with known geometry:

        When grid geometry is known from calibration, specify the row and column
        edges for reproducible grid assignment. Here the edges divide the
        600 x 800 pixel synthetic plate evenly:

        >>> import numpy as np
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.grid import ManualGridFinder
        >>> from phenotypic.schema import GRID
        >>> image = OtsuDetector().apply(load_synth_yeast_plate())
        >>> height, width = image.shape[:2]
        >>> # 9 row edges for 8 rows, 13 column edges for 12 columns
        >>> row_edges = np.linspace(0, height, 9, dtype=int)
        >>> col_edges = np.linspace(0, width, 13, dtype=int)
        >>> print(row_edges)
        [  0  75 150 225 300 375 450 525 600]
        >>> grid_finder = ManualGridFinder(row_edges=row_edges, col_edges=col_edges)
        >>> grid_df = grid_finder.measure(image)
        >>> well_columns = [str(GRID.ROW_NUM), str(GRID.COL_NUM), str(GRID.ROW_MAJOR_IDX)]
        >>> print(grid_df[well_columns].head())
          Grid_RowNum Grid_ColNum Grid_RowMajorIdx
        0           0           0                0
        1           0          10               10
        2           0           0                0
        3           0           0                0
        4           0          10               10

        Understanding ROW_MAJOR_IDX for well mapping:

        ROW_MAJOR_IDX provides a single integer ID for each well, useful for
        organizing results and correlating with sample metadata. On an 8 x 12
        grid it runs 0-95, left-to-right, top-to-bottom: section 0 is A1 (row 0,
        column 0), 11 is A12, 12 is B1, and 95 is H12.

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.grid import CenteredAutoGridFinder
        >>> from phenotypic.schema import GRID
        >>> image = OtsuDetector().apply(load_synth_yeast_plate())
        >>> grid_df = CenteredAutoGridFinder(nrows=8, ncols=12).measure(image)
        >>> # Filter the objects in one well
        >>> section_5 = grid_df[grid_df[str(GRID.ROW_MAJOR_IDX)] == 5]
        >>> len(section_5)
        5
        >>> # Map the section number back to its well coordinates
        >>> well_row, well_col = divmod(5, 12)
        >>> (well_row, well_col)
        (0, 5)
        >>> section_5[[str(GRID.ROW_NUM), str(GRID.COL_NUM)]].drop_duplicates().values.tolist()
        [[0, 5]]
    """

    nrows: int
    ncols: int

    @abstractmethod
    def _operate(self, image: Image) -> pd.DataFrame:
        return pd.DataFrame()

    @abstractmethod
    def get_row_edges(self, image: Image) -> np.ndarray:
        """
        This method is to returns the row edges of the grid as a numpy array.
        Args:
            image (Image): Image object.
        Returns:
            np.ndarray: Row-edges of the grid.
        """
        pass

    @abstractmethod
    def get_col_edges(self, image: Image) -> np.ndarray:
        """
        This method is to returns the column edges of the grid as a numpy array.
        Args:
            image:

        Returns:
            np.ndarray: Column-edges of the grid.

        """
        pass

    @staticmethod
    def _clip_row_edges(row_edges, imshape: tuple[int, ...]) -> np.ndarray:
        return np.clip(a=row_edges, a_min=0, a_max=imshape[0])

    def _add_row_number_info(
            self, table: pd.DataFrame, row_edges: np.ndarray, imshape: tuple[int, ...]
    ) -> pd.DataFrame:
        row_edges = self._clip_row_edges(row_edges=row_edges, imshape=imshape)
        col = str(BBOX.CENTER_RR)
        table.loc[:, str(GRID.ROW_NUM)] = pd.cut(
                table.loc[:, col],
                bins=row_edges,
                labels=range(self.nrows),
                include_lowest=True,
                right=True,
        )
        # Clamp out-of-bounds objects to nearest edge cell
        nan_mask = table[str(GRID.ROW_NUM)].isna()
        if nan_mask.any():
            centers = table.loc[nan_mask, col].values.astype(float)
            nearest = np.clip(
                np.searchsorted(row_edges, centers, side="right") - 1,
                0, self.nrows - 1,
            )
            table.loc[nan_mask, str(GRID.ROW_NUM)] = nearest
        return table

    @staticmethod
    def _clip_col_edges(col_edges, imshape: tuple[int, ...]) -> np.ndarray:
        return np.clip(a=col_edges, a_min=0, a_max=imshape[1] - 1)

    def _add_col_number_info(
            self, table: pd.DataFrame, col_edges: np.ndarray, imshape: tuple[int, ...]
    ) -> pd.DataFrame:
        col_edges = self._clip_col_edges(col_edges=col_edges, imshape=imshape)
        col = str(BBOX.CENTER_CC)
        table.loc[:, str(GRID.COL_NUM)] = pd.cut(
                table.loc[:, col],
                bins=col_edges,
                labels=range(self.ncols),
                include_lowest=True,
                right=True,
        )
        # Clamp out-of-bounds objects to nearest edge cell
        nan_mask = table[str(GRID.COL_NUM)].isna()
        if nan_mask.any():
            centers = table.loc[nan_mask, col].values.astype(float)
            nearest = np.clip(
                np.searchsorted(col_edges, centers, side="right") - 1,
                0, self.ncols - 1,
            )
            table.loc[nan_mask, str(GRID.COL_NUM)] = nearest
        return table

    def _add_section_number_info(
            self,
            table: pd.DataFrame,
            row_edges: np.ndarray,
            col_edges: np.ndarray,
            imshape: tuple[int, ...],
    ) -> pd.DataFrame:
        # Ensure ROW_NUM and COL_NUM exist
        if str(GRID.ROW_NUM) not in table.columns:
            self._add_row_number_info(table=table, row_edges=row_edges, imshape=imshape)
        if str(GRID.COL_NUM) not in table.columns:
            self._add_col_number_info(table=table, col_edges=col_edges, imshape=imshape)

        row_nums = table.loc[:, str(GRID.ROW_NUM)].values
        col_nums = table.loc[:, str(GRID.COL_NUM)].values

        valid_mask = pd.notna(row_nums) & pd.notna(col_nums)
        n = len(table)

        if valid_mask.any():
            valid_rows = row_nums[valid_mask].astype(int)
            valid_cols = col_nums[valid_mask].astype(int)
        else:
            valid_rows = valid_cols = np.array([], dtype=int)

        # Row-major index (row * ncols + col)
        idx_map = np.reshape(
            np.arange(self.nrows * self.ncols), (self.nrows, self.ncols)
        )
        row_major_arr = pd.arrays.IntegerArray(
            np.zeros(n, dtype=np.uint16),
            mask=np.ones(n, dtype=bool),
        )
        if valid_mask.any():
            row_major_arr[valid_mask] = idx_map[valid_rows, valid_cols]
        table[str(GRID.ROW_MAJOR_IDX)] = pd.Categorical(row_major_arr)

        # Column-major index (col * nrows + row)
        col_major_map = np.reshape(
            np.arange(self.nrows * self.ncols),
            (self.nrows, self.ncols),
            order="F",
        )
        col_major_arr = pd.arrays.IntegerArray(
            np.zeros(n, dtype=np.uint16),
            mask=np.ones(n, dtype=bool),
        )
        if valid_mask.any():
            col_major_arr[valid_mask] = col_major_map[valid_rows, valid_cols]
        table[str(GRID.COL_MAJOR_IDX)] = pd.Categorical(col_major_arr)

        return table

    def _get_grid_info(
            self,
            image: Image,
            row_edges: np.ndarray,
            col_edges: np.ndarray,
            info_table: pd.DataFrame | None = None,
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
        if image.num_objects == 0:
            columns = [
                OBJECT.LABEL,
                str(BBOX.CENTER_RR), str(BBOX.CENTER_CC),
                str(BBOX.MIN_RR), str(BBOX.MIN_CC),
                str(BBOX.MAX_RR), str(BBOX.MAX_CC),
                str(BBOX.INTENSITY_WEIGHTED_CENTER_RR), str(BBOX.INTENSITY_WEIGHTED_CENTER_CC),
                str(BBOX.DIST_WEIGHTED_CENTER_RR), str(BBOX.DIST_WEIGHTED_CENTER_CC),
                str(GRID.ROW_NUM), str(GRID.COL_NUM),
                str(GRID.ROW_MAJOR_IDX), str(GRID.COL_MAJOR_IDX),
            ]
            return pd.DataFrame(columns=columns)

        if info_table is None:
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
