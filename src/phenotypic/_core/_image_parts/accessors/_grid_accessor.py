from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari
    from phenotypic import GridImage

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.color import label2rgb

import phenotypic
from phenotypic.tools_.constants_ import METADATA, IMAGE_TYPES, OBJECT
from phenotypic.tools_.measurement_info_ import BBOX, GRID
from phenotypic.tools_.exceptions_ import NoObjectsError


class GridAccessor:
    """Provides grid-based access and analysis for microbial colony arrays on agar plates.

    This class facilitates operations on grid structures within a GridImage, enabling analysis
    of robotically-pinned microbial colonies arranged in a regular rectangular array pattern.
    It provides methods for determining grid properties, retrieving colony information by grid
    location, and visualizing grid overlays with row and column assignments.

    The grid divides an agar plate image into a regular matrix of sections, each potentially
    containing one or more detected colonies. The grid is ordered left-to-right, top-to-bottom
    when using flattened indexing.

    Attributes:
        nrows (int): Number of rows in the grid (read/write property). Corresponds to the
            number of row pins in a colony pinning robot. Must be >= 1.
        ncols (int): Number of columns in the grid (read/write property). Corresponds to the
            number of column pins in a colony pinning robot. Must be >= 1.

    Examples:
        Access grid information for a 96-well colony plate:

        >>> from phenotypic import GridImage
        >>> # Load image and create grid accessor
        >>> grid_image = GridImage('agar_plate.png', nrows=8, ncols=12)
        >>> # Get grid information as a DataFrame
        >>> grid_info = grid_image.grid.info()
        >>> print(f"Found {len(grid_info)} colonies across {grid_image.grid.nrows} rows "
        ...       f"and {grid_image.grid.ncols} columns")
        >>> # Extract a single grid section (colony at row 2, column 3)
        >>> section_idx = 2 * grid_image.grid.ncols + 3  # Flattened index
        >>> colony_image = grid_image.grid[section_idx]
        >>> # Visualize grid columns with color-coded labels
        >>> fig, ax = grid_image.grid.show_column_overlay(show_gridlines=True)

        Get colony counts by grid section:

        >>> # Count colonies in each grid section
        >>> section_counts = grid_image.grid.get_section_counts(ascending=False)
        >>> print("Colonies per section (sorted):")
        >>> print(section_counts)
        >>> # Get all colony information for row 0
        >>> row_info = grid_image.grid.get_info_by_section((0, slice(None)))
    """

    def __init__(self, root_image: GridImage):
        self._root_image: GridImage = root_image

    @property
    def _accessor_property_name(self) -> str:
        return "grid"

    @property
    def nrows(self) -> int:
        """Get the number of rows in the grid.

        Returns:
            int: Number of rows in the grid array. Must be >= 1.
        """
        return self._root_image.grid_finder.nrows

    @nrows.setter
    def nrows(self, nrows: int):
        """Set the number of rows in the grid.

        Args:
            nrows (int): Number of rows in the grid. Must be a positive integer
                (>= 1). Typically corresponds to the number of row pins in a
                colony pinning robot.

        Raises:
            ValueError: If nrows is less than 1.
            TypeError: If nrows is not an integer type.
        """
        if nrows < 1:
            raise ValueError("Number of nrows must be greater than 0")
        if type(nrows) != int:
            raise TypeError("Number of nrows must be an integer")

        self._root_image.grid_finder.nrows = nrows

    @property
    def ncols(self) -> int:
        """Get the number of columns in the grid.

        Returns:
            int: Number of columns in the grid array. Must be >= 1.
        """
        return self._root_image.grid_finder.ncols

    @ncols.setter
    def ncols(self, ncols: int):
        """Set the number of columns in the grid.

        Args:
            ncols (int): Number of columns in the grid. Must be a positive
                integer (>= 1). Typically corresponds to the number of column
                pins in a colony pinning robot.

        Raises:
            ValueError: If ncols is less than 1.
            TypeError: If ncols is not an integer type.
        """
        if ncols < 1:
            raise ValueError("Number of columns must be greater than 0")
        if type(ncols) != int:
            raise TypeError("Number of columns must be an integer")

        self._root_image.grid_finder.ncols = ncols

    def info(self, include_metadata=True) -> pd.DataFrame:
        """Get grid information for all detected colonies.

        Returns a DataFrame with bounding box measurements and grid location (row, column,
        section) assignments for each detected object (colony). This is the primary method
        for accessing detailed colony positioning and measurement data.

        Args:
            include_metadata (bool, optional): Whether to include image metadata columns
                in the output DataFrame. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with one row per detected colony. Columns include:
                - ObjectLabel: Unique identifier for the colony
                - CenterRR, CenterCC: Row and column coordinates of colony center
                - MinRR, MaxRR, MinCC, MaxCC: Bounding box coordinates
                - RowNum: Grid row index (0-indexed)
                - ColNum: Grid column index (0-indexed)
                - SectionNum: Flattened grid section index (0 to nrows*ncols-1)
                - Additional columns if include_metadata=True

        Examples:
            Retrieve and analyze grid information:

            >>> # Get full grid information
            >>> grid_info = grid_image.grid.info()
            >>> # Count colonies by row
            >>> colonies_per_row = grid_info.groupby('RowNum').size()
            >>> # Find largest colony in grid section 10
            >>> section_10 = grid_info[grid_info['SectionNum'] == 10]
            >>> largest = section_10.loc[section_10['Area'].idxmax()]
            >>> # Get colonies without metadata
            >>> grid_info_minimal = grid_image.grid.info(include_metadata=False)
        """
        info = self._root_image.grid_finder.measure(self._root_image)
        if include_metadata:
            return self._root_image.metadata.insert_metadata(info)
        else:
            return info

    @property
    def _idx_ref_matrix(self) -> np.ndarray:
        """Internal property: matrix mapping grid positions to flattened indices.

        Creates a reference matrix that converts 2D grid coordinates (row, col)
        to flattened section indices. Grid sections are ordered left-to-right,
        top-to-bottom (row-major order).

        Returns:
            np.ndarray: 2D integer array of shape (nrows, ncols) where element
                [i, j] contains the flattened section index corresponding to grid
                position (i, j). For an 8x12 grid: [0, 0] = 0 (top-left),
                [0, 11] = 11 (top-right), [7, 0] = 84 (bottom-left),
                [7, 11] = 95 (bottom-right).
        """
        return np.reshape(
                np.arange(self.nrows * self.ncols, dtype=np.uint16),
                shape=(self.nrows, self.ncols)
        )

    def _parse_slice_to_section_indices(
            self,
            idx: int | tuple | slice,
    ) -> list[int]:
        """Convert various index formats to list of flattened section indices.

        Supports single indices (int), slices on flattened indices, and tuple-based
        indexing for (row, column) patterns with optional slicing on each dimension.

        Args:
            idx: Index in various forms:
                - int: Single flattened section index
                - slice: Range of flattened section indices
                - tuple[int, int]: Single section as (row, col)
                - tuple[int, slice]: Specific row, range of columns
                - tuple[slice, int]: Range of rows, specific column
                - tuple[slice, slice]: Range of rows and columns (only if one is full)

        Returns:
            list[int]: List of flattened section indices.

        Raises:
            ValueError: If slicing in both row AND column dimensions simultaneously
                (except when one dimension is a full slice like ':').
            IndexError: If index is out of bounds or tuple has wrong length.
            TypeError: If index type is not supported.
        """
        # Case 1: int -> single section
        if isinstance(idx, int):
            return [idx]

        # Case 2: slice -> range of flattened sections
        if isinstance(idx, slice):
            total_sections = self.nrows * self.ncols
            return list(range(total_sections)[idx])

        # Case 3: tuple (row_part, col_part)
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise IndexError(
                        "Grid section index tuple must have length 2: (row, col)."
                )

            row_part, col_part = idx

            # Check if both parts are slices
            if isinstance(row_part, slice) and isinstance(col_part, slice):
                # Only allow if at least one is a full slice (:)
                row_is_full = row_part == slice(None)
                col_is_full = col_part == slice(None)

                if not (row_is_full or col_is_full):
                    raise ValueError(
                            "Cannot slice in both dimensions simultaneously. "
                            "Use grid[rows, col] or grid[row, cols], not grid[rows, cols]."
                    )

            # Convert to list of row indices
            if isinstance(row_part, slice):
                rows = list(range(self.nrows)[row_part])
            else:
                rows = [row_part]

            # Convert to list of column indices
            if isinstance(col_part, slice):
                cols = list(range(self.ncols)[col_part])
            else:
                cols = [col_part]

            # Generate section indices using _idx_ref_matrix (row-major order)
            sections = []
            for r in rows:
                for c in cols:
                    sections.append(int(self._idx_ref_matrix[r, c]))

            return sections

        raise TypeError(f"Invalid index type: {type(idx)}")

    def _get_multi_section_bounds(
            self,
            section_indices: list[int],
            grid_info: pd.DataFrame | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Get union of bounding boxes for multiple grid sections.

        For each section index, retrieves the object-aware bounding box using
        `_adv_get_grid_section_slices`, then computes the union (min of mins,
        max of maxs) across all sections.

        Args:
            section_indices: List of flattened section indices.
            grid_info: Optional precomputed grid info table to avoid recomputation.

        Returns:
            tuple[tuple[float, float], tuple[float, float]]: A tuple containing:
                - (min_rr, min_cc): Minimum pixel coordinates (top-left)
                - (max_rr, max_cc): Maximum pixel coordinates (bottom-right)
                These bounds encompass all objects in all specified sections.
        """
        # Handle empty selection
        if len(section_indices) == 0:
            return (0, 0), (self._root_image.shape[0], self._root_image.shape[1])

        if grid_info is None:
            grid_info = self.info()

        min_rr_all, max_rr_all, min_cc_all, max_cc_all = (
            self._get_section_bounds_arrays(grid_info=grid_info)
        )
        idx_array = np.asarray(section_indices, dtype=int)

        # Compute union: min of mins, max of maxs
        global_min_rr = float(np.min(min_rr_all[idx_array]))
        global_min_cc = float(np.min(min_cc_all[idx_array]))
        global_max_rr = float(np.max(max_rr_all[idx_array]))
        global_max_cc = float(np.max(max_cc_all[idx_array]))

        return (global_min_rr, global_min_cc), (global_max_rr, global_max_cc)

    def _get_section_bounds_arrays(
            self,
            grid_info: pd.DataFrame | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-section bounds using grid edges and object extents."""
        if grid_info is None:
            grid_info = self.info()

        nrows, ncols = self.nrows, self.ncols
        section_count = nrows * ncols
        row_edges = self.get_row_edges()
        col_edges = self.get_col_edges()

        grid_min_rr = np.repeat(row_edges[:-1], ncols)
        grid_max_rr = np.repeat(row_edges[1:], ncols)
        grid_min_cc = np.tile(col_edges[:-1], nrows)
        grid_max_cc = np.tile(col_edges[1:], nrows)

        if grid_info.empty:
            obj_min_rr = np.full(section_count, np.nan)
            obj_max_rr = np.full(section_count, np.nan)
            obj_min_cc = np.full(section_count, np.nan)
            obj_max_cc = np.full(section_count, np.nan)
        else:
            bounds = (
                grid_info.groupby(str(GRID.ROW_MAJOR_IDX), observed=False)
                .agg(
                        min_rr=(str(BBOX.MIN_RR), "min"),
                        max_rr=(str(BBOX.MAX_RR), "max"),
                        min_cc=(str(BBOX.MIN_CC), "min"),
                        max_cc=(str(BBOX.MAX_CC), "max"),
                )
            )
            bounds = bounds.reset_index()
            bounds[str(GRID.ROW_MAJOR_IDX)] = bounds[str(GRID.ROW_MAJOR_IDX)].astype(
                int)
            bounds = bounds.set_index(str(GRID.ROW_MAJOR_IDX)).reindex(
                    range(section_count)
            )
            obj_min_rr = bounds["min_rr"].to_numpy()
            obj_max_rr = bounds["max_rr"].to_numpy()
            obj_min_cc = bounds["min_cc"].to_numpy()
            obj_max_cc = bounds["max_cc"].to_numpy()

        min_rr = np.where(
                np.isnan(obj_min_rr), grid_min_rr, np.minimum(grid_min_rr, obj_min_rr)
        )
        max_rr = np.where(
                np.isnan(obj_max_rr), grid_max_rr, np.maximum(grid_max_rr, obj_max_rr)
        )
        min_cc = np.where(
                np.isnan(obj_min_cc), grid_min_cc, np.minimum(grid_min_cc, obj_min_cc)
        )
        max_cc = np.where(
                np.isnan(obj_max_cc), grid_max_cc, np.maximum(grid_max_cc, obj_max_cc)
        )

        min_rr = np.clip(min_rr, 0, self._root_image.shape[0] - 1)
        max_rr = np.clip(max_rr, 0, self._root_image.shape[0] - 1)
        min_cc = np.clip(min_cc, 0, self._root_image.shape[1] - 1)
        max_cc = np.clip(max_cc, 0, self._root_image.shape[1] - 1)

        return min_rr, max_rr, min_cc, max_cc

    def _get_multi_section_labels(
            self,
            section_indices: list[int],
            grid_info: pd.DataFrame | None = None,
    ) -> list[int]:
        """Get all object labels from multiple grid sections.

        Retrieves the labels (object identifiers) for all objects belonging to
        the specified sections. The returned list contains each label exactly
        once, even if an object appears in multiple sections.

        Args:
            section_indices: List of flattened section indices.
            grid_info: Optional precomputed grid info table to avoid recomputation.

        Returns:
            list[int]: List of unique object labels across all specified sections.
        """
        if len(section_indices) == 0:
            return []

        if grid_info is None:
            grid_info = self.info()

        labels = grid_info.loc[
            grid_info.loc[:, str(GRID.ROW_MAJOR_IDX)].isin(section_indices),
            OBJECT.LABEL,
        ]
        if labels.empty:
            return []

        return labels.unique().tolist()

    def __getitem__(
            self,
            idx: int | tuple[int, int] | slice | tuple[slice | int, slice | int],
    ) -> phenotypic.Image:
        """Extract grid section(s) as a subimage.

        Returns a cropped image corresponding to one or more grid sections. Supports
        flexible indexing including single sections, row/column slices, and flattened
        index ranges. The grid is indexed left-to-right, top-to-bottom (row-major order).
        Only objects belonging to the specified grid sections are included. For multiple
        sections, the subimage is cropped to the union of all section bounding boxes.

        Args:
            idx: Grid section identifier(s). Supported formats:
                - int: Single flattened section index (0 to nrows*ncols-1).
                  Example: grid[54] for center section in 8x12 grid.
                - tuple[int, int]: Single section as (row_index, col_index).
                  Example: grid[4, 6] for row 4, column 6.
                - slice: Range of flattened section indices.
                  Example: grid[0:12] for first row in 8x12 grid.
                - tuple[int, slice]: Specific row, range of columns.
                  Example: grid[2, 0:6] for row 2, columns 0-5.
                - tuple[slice, int]: Range of rows, specific column.
                  Example: grid[0:4, 3] for rows 0-3, column 3.
                - tuple[slice, slice]: Slices in both dimensions (only if at least
                  one is a full slice ':'). Examples: grid[2, :], grid[:, 5],
                  grid[:, :], grid[:].

                Note: Slicing in both dimensions is not allowed unless one is a
                full slice. For example, grid[0:5, 2:7] raises ValueError.
                Use grid[rows, col] OR grid[row, cols], but not grid[rows, cols].

        Returns:
            phenotypic.Image: A subimage containing the selected grid section(s).
                For a single section, pixels and objects are relative to that section's
                top-left corner. For multiple sections, the image is cropped to the
                union of all selected section bounding boxes, preserving complete
                objects even if they extend beyond ideal grid boundaries. Object labels
                are preserved for objects in the selected sections; objects from other
                sections have their labels removed (set to 0). The subimage is marked
                with IMAGE_TYPE=GRID_SECTION metadata. If no objects are present in
                the parent image, returns a copy of the entire parent image.

        Raises:
            IndexError: If idx is out of bounds for the grid dimensions, or if idx
                is a tuple with length != 2.
            ValueError: If attempting to slice in both row AND column dimensions
                simultaneously with non-full slices (e.g., grid[0:5, 2:7]).
            TypeError: If idx is not a supported type.

        Examples:

            .. code-block:: python

                # Single section by flattened index
                top_left = grid_image.grid[0]
                print(f"Section size: {top_left.shape}")

                # Single section by (row, col) indexing
                center = grid_image.grid[4, 6]

                # Flattened index slice - first row (sections 0-11 for 8x12 grid)
                first_row = grid_image.grid[0:12]

                # All columns in row 2
                row_2 = grid_image.grid[2, :]

                # All rows in column 5
                col_5 = grid_image.grid[:, 5]

                # Rows 0-3 in column 5
                subset = grid_image.grid[0:4, 5]

                # Row 2, columns 0-5
                subset = grid_image.grid[2, 0:6]

                # Every other section (step slicing)
                checkerboard = grid_image.grid[::2]

                # Full grid extraction
                all_sections = grid_image.grid[:]

                # This raises ValueError: cannot slice both dimensions
                # grid_image.grid[0:5, 2:7]
        """
        # Parse input to list of section indices
        section_indices = self._parse_slice_to_section_indices(idx)

        # Handle empty image (no objects detected)
        if self._root_image.objects.num_objects == 0:
            return phenotypic.Image(self._root_image)

        # Handle empty selection
        if len(section_indices) == 0:
            # Return empty image with parent's shape
            return phenotypic.Image(self._root_image)

        grid_info = self.info()

        # Get union bounding box for all selected sections
        min_coords, max_coords = self._get_multi_section_bounds(
                section_indices, grid_info=grid_info
        )
        min_rr, min_cc = min_coords
        max_rr, max_cc = max_coords

        # Crop parent image to union bounds

        section_image = phenotypic.Image(
                self._root_image[
                    self.__to_int(min_rr):self.__to_int(max_rr),
                    self.__to_int(min_cc):self.__to_int(max_cc)]
        )

        # Filter object map to only include selected sections
        objmap = section_image.objmap[:]
        valid_labels = self._get_multi_section_labels(
                section_indices, grid_info=grid_info
        )
        objmap[~np.isin(objmap, valid_labels)] = 0
        section_image.objmap = objmap

        # Mark as grid section
        section_image.metadata[METADATA.IMAGE_TYPE] = IMAGE_TYPES.GRID_SECTION.value

        return section_image

    @staticmethod
    def __to_int(val):
        """Safely convert value to int, handling numpy scalars."""
        if hasattr(val, 'item'):
            return int(val.item())
        return int(val)

    def get_centroid_alignment_info(self, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Calculate linear regression fit for colony centroids along a grid axis.

        Computes the slope and intercept of a best-fit line through the centroids of
        colonies arranged along a specified axis (rows or columns). This quantifies
        alignment quality and any systematic drift in the pinned colony array. Uses
        standard least-squares linear regression to fit the line: y = m*x + b.

        For row-wise analysis (axis=0), the function groups colonies by their row
        index and fits a line to the relationship between column position and column
        coordinate. For column-wise analysis (axis=1), it groups by column index and
        fits a line to the relationship between row position and row coordinate.

        Args:
            axis (int): Axis along which to compute alignment:
                - 0: Row-wise alignment. For each row, measures how colony centers
                  vary along the column (CC) axis as a function of their grid column
                  position. Slope indicates pixels of drift per grid column.
                - 1: Column-wise alignment. For each column, measures how colony
                  centers vary along the row (RR) axis as a function of their grid
                  row position. Slope indicates pixels of drift per grid row.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - m_slope (np.ndarray[float]): Slopes for each row or column. Length
                  is nrows if axis=0, ncols if axis=1. Values represent pixels of
                  drift per grid position unit. NaN indicates no colonies in that
                  row/column, 0 indicates single colony with no drift measurable.
                - b_intercept (np.ndarray): Y-intercepts for each row/column, rounded
                  to nearest integer. NaN indicates no colonies in that row/column.

        Raises:
            NoObjectsError: If the parent image contains no detected objects (colonies).
            ValueError: If axis is neither 0 nor 1.

        Examples:
            Analyze colony alignment across grid axes:

            >>> # Check row alignment (horizontal drift of colonies across each row)
            >>> row_slopes, row_intercepts = grid_image.grid.get_centroid_alignment_info(axis=0)
            >>> print(f"Row alignment slopes (pixels/column): {row_slopes}")
            >>> # Check column alignment (vertical drift of colonies across each column)
            >>> col_slopes, col_intercepts = grid_image.grid.get_centroid_alignment_info(axis=1)
            >>> print(f"Column alignment slopes (pixels/row): {col_slopes}")
            >>> # Identify rows with significant drift indicating pinning issues
            >>> drift_threshold = 0.05  # pixels per grid position
            >>> problematic_rows = np.where(np.abs(row_slopes) > drift_threshold)[0]
            >>> print(f"Rows with significant drift: {problematic_rows}")
        """
        if self._root_image.objects.num_objects == 0:
            raise NoObjectsError(self._root_image.name)
        if axis == 0:
            num_vectors = self.nrows
            x_group = str(GRID.ROW_NUM)
            x_val = str(BBOX.CENTER_CC)
            y_val = str(BBOX.CENTER_RR)
        elif axis == 1:
            num_vectors = self.ncols
            x_group = str(GRID.COL_NUM)
            x_val = str(BBOX.CENTER_RR)
            y_val = str(BBOX.CENTER_CC)
        else:
            raise ValueError("Axis should be 0 or 1.")

        # create persistent grid_info
        grid_info = self.info()

        # allocate empty vectors to store m & b for all values
        m_slope = np.full(shape=num_vectors, fill_value=np.nan)
        b_intercept = np.full(shape=num_vectors, fill_value=np.nan)

        # Collect slope & intercept for the nrows or columns
        # Use 2D covariance/variance method for finding linear regression
        for idx in range(num_vectors):
            x = grid_info.loc[grid_info.loc[:, x_group] == idx, x_val].to_numpy()
            x_mean = np.mean(x) if x.size > 0 else np.nan

            y = grid_info.loc[grid_info.loc[:, x_group] == idx, y_val].to_numpy()
            y_mean = np.mean(y) if y.size > 0 else np.nan

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
        """Get the column boundary positions in pixel coordinates.

        Returns the x-coordinates (column indices) that define the vertical boundaries
        of each grid column in the image. For an ncols-column grid, returns ncols+1
        boundary values: the left edge of column 0, internal boundaries between
        adjacent columns, and the right edge of column ncols-1.

        Returns:
            np.ndarray: 1D array of strictly increasing column edge positions (pixel
                column indices). Length is ncols+1. First value is 0 or the left edge
                of the first column, last value is the image width or right boundary.

        Examples:
            Retrieve and use column edge positions:

            >>> col_edges = grid_image.grid.get_col_edges()
            >>> print(f"Column edges: {col_edges}")
            >>> # Output: [0.0, 106.5, 213.0, 319.5, ...]  for a 12-column grid
            >>> # Calculate column width
            >>> col_width = col_edges[1] - col_edges[0]
            >>> print(f"Column width: {col_width} pixels")
            >>> # Extract pixels for column 3
            >>> col_3_min, col_3_max = int(col_edges[3]), int(col_edges[4])
            >>> column_3_data = grid_image.gray[:, col_3_min:col_3_max]
            >>> # Visualize grid column positions
            >>> fig, ax = plt.subplots()
            >>> ax.imshow(grid_image.gray)
            >>> ax.vlines(x=col_edges, ymin=0, ymax=grid_image.shape[0], colors='cyan')
        """
        return self._root_image.grid_finder.get_col_edges(self._root_image)

    def get_col_map(self) -> np.ndarray:
        """Get an object map with objects labeled by their grid column number.

        Creates a copy of the object map where each detected colony is relabeled
        according to its grid column assignment. All pixels belonging to colonies
        in the same grid column receive the same label. This is useful for
        visualizing or analyzing all colonies in a particular column together.

        Returns:
            np.ndarray: 2D integer array with same shape as the parent image. Each
                pixel belonging to a colony is set to that colony's grid column number
                (1-indexed, ranging from 1 to ncols). Pixels not belonging to any
                colony are 0. Can be passed directly to label2rgb for visualization.

        Examples:
            Get and visualize column-labeled colony map:

            >>> col_map = grid_image.grid.get_col_map()
            >>> # All colonies in column 0 have value 1, column 1 have value 2, etc.
            >>> print(f"Unique values in col_map: {np.unique(col_map)}")
            >>> # Output: [0, 1, 2, 3, ..., 12]  for a 12-column grid
            >>> # Count total pixels belonging to each column
            >>> for col_num in range(1, grid_image.grid.ncols + 1):
            ...     col_pixels = np.sum(col_map == col_num)
            ...     print(f"Column {col_num}: {col_pixels} pixels")
            >>> # Visualize columns with distinct colors
            >>> from skimage.color import label2rgb
            >>> colored_columns = label2rgb(label=col_map, image=grid_image.gray[:])
            >>> plt.imshow(colored_columns)
        """
        grid_info = self.info()
        col_map = self._root_image.objmap[:].copy()
        for n, col_bidx in enumerate(
                np.sort(grid_info.loc[:, str(GRID.COL_NUM)].unique())
        ):
            subtable = grid_info.loc[grid_info.loc[:, str(GRID.COL_NUM)] == col_bidx, :]

            # Edit the new map's objects to equal the column number
            col_map[
                np.isin(
                        element=self._root_image.objmap[:],
                        test_elements=subtable[OBJECT.LABEL].to_numpy(),
                )
            ] = n + 1
        return col_map

    def show_column_overlay(
            self,
            use_enhanced: bool = False,
            show_gridlines: bool = True,
            ax: plt.Axes | None = None,
            figsize: tuple[int, int] = (9, 10),
    ) -> tuple[plt.Figure, plt.Axes]:
        """Visualize colonies with column-based color coding and optional grid overlay.

        Displays the image with an overlay where each colony is colored according to
        its grid column assignment. This helps visualize the column structure of the
        pinned array and identify any column-wise positioning issues or misalignment.

        Args:
            use_enhanced (bool, optional): If True, use the detection matrix version
                of the parent image (detect_mat) for better contrast and visibility.
                If False, use the standard grayscale image (gray). Defaults to False.
            show_gridlines (bool, optional): If True, overlay cyan dashed vertical lines
                marking the column boundaries and horizontal lines for row boundaries.
                Defaults to True.
            ax (plt.Axes | None, optional): Existing Matplotlib Axes object to plot into.
                If None, a new figure and axes are created with the specified figsize.
                Defaults to None.
            figsize (tuple[int, int], optional): Figure size as (width, height) in inches,
                only used when ax is None. Defaults to (9, 10).

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib Figure and
                Axes objects. If ax was provided as input, the function returns the
                created figure and the input ax object (not func_ax). If ax is None,
                returns the newly created figure and axes.

        Examples:
            Display column overlay visualization with options:

            >>> # Display column overlay with gridlines
            >>> fig, ax = grid_image.grid.show_column_overlay(show_gridlines=True)
            >>> plt.title("Colony Array - Column Overlay")
            >>> plt.show()
            >>> # Use enhanced image for better contrast
            >>> fig, ax = grid_image.grid.show_column_overlay(
            ...     use_enhanced=True,
            ...     show_gridlines=True,
            ...     figsize=(12, 14)
            ... )
            >>> # Plot on existing axes
            >>> fig, axes = plt.subplots(1, 2, figsize=(16, 10))
            >>> grid_image.grid.show_column_overlay(ax=axes[0])
            >>> grid_image.grid.show_row_overlay(ax=axes[1])
        """
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(
                    label2rgb(label=self.get_col_map(),
                              image=self._root_image.detect_mat[:])
            )
        else:
            func_ax.imshow(
                    label2rgb(label=self.get_col_map(), image=self._root_image.gray[:])
            )

        if show_gridlines:
            col_edges = self.get_col_edges()
            row_edges = self.get_row_edges()
            func_ax.vlines(
                    x=col_edges,
                    ymin=row_edges.min(),
                    ymax=row_edges.max(),
                    colors="c",
                    linestyles="--",
            )

        return fig, ax

    """
    Grid Rows
    """

    def get_row_edges(self) -> np.ndarray:
        """Get the row boundary positions in pixel coordinates.

        Returns the y-coordinates (row indices) that define the horizontal boundaries
        of each grid row in the image. For an nrows-row grid, returns nrows+1
        boundary values: the top edge of row 0, internal boundaries between
        adjacent rows, and the bottom edge of row nrows-1.

        Returns:
            np.ndarray: 1D array of strictly increasing row edge positions (pixel
                row indices). Length is nrows+1. First value is 0 or the top edge
                of the first row, last value is the image height or bottom boundary.

        Examples:
            Retrieve and use row edge positions:

            >>> row_edges = grid_image.grid.get_row_edges()
            >>> print(f"Row edges: {row_edges}")
            >>> # Output: [0.0, 95.2, 190.4, 285.6, ...]  for an 8-row grid
            >>> # Calculate row height
            >>> row_height = row_edges[1] - row_edges[0]
            >>> print(f"Row height: {row_height} pixels")
            >>> # Extract pixels for row 4
            >>> row_4_min, row_4_max = int(row_edges[4]), int(row_edges[5])
            >>> row_4_data = grid_image.gray[row_4_min:row_4_max, :]
            >>> # Visualize grid row positions
            >>> fig, ax = plt.subplots()
            >>> ax.imshow(grid_image.gray)
            >>> ax.hlines(y=row_edges, xmin=0, xmax=grid_image.shape[1], colors='cyan')
        """
        return self._root_image.grid_finder.get_row_edges(self._root_image)

    def get_row_map(self) -> np.ndarray:
        """Get an object map with objects labeled by their grid row number.

        Creates a copy of the object map where each detected colony is relabeled
        according to its grid row assignment. All pixels belonging to colonies
        in the same grid row receive the same label. This is useful for
        visualizing or analyzing all colonies in a particular row together.

        Returns:
            np.ndarray: 2D integer array with same shape as the parent image. Each
                pixel belonging to a colony is set to that colony's grid row number
                (1-indexed, ranging from 1 to nrows). Pixels not belonging to any
                colony are 0. Can be passed directly to label2rgb for visualization.

        Examples:
            Get and visualize row-labeled colony map:

            >>> row_map = grid_image.grid.get_row_map()
            >>> # All colonies in row 0 have value 1, row 1 have value 2, etc.
            >>> print(f"Unique values in row_map: {np.unique(row_map)}")
            >>> # Output: [0, 1, 2, 3, ..., 8]  for an 8-row grid
            >>> # Count total pixels belonging to each row
            >>> for row_num in range(1, grid_image.grid.nrows + 1):
            ...     row_pixels = np.sum(row_map == row_num)
            ...     print(f"Row {row_num}: {row_pixels} pixels")
            >>> # Visualize rows with distinct colors
            >>> from skimage.color import label2rgb
            >>> colored_rows = label2rgb(label=row_map, image=grid_image.gray[:])
            >>> plt.imshow(colored_rows)
        """
        grid_info = self.info()
        row_map = self._root_image.objmap[:].copy()
        for n, col_bidx in enumerate(
                np.sort(grid_info.loc[:, str(GRID.ROW_NUM)].unique())
        ):
            subtable = grid_info.loc[grid_info.loc[:, str(GRID.ROW_NUM)] == col_bidx, :]

            # Edit the new map's objects to equal the column number
            row_map[
                np.isin(
                        element=self._root_image.objmap[:],
                        test_elements=subtable[OBJECT.LABEL].to_numpy(),
                )
            ] = n + 1
        return row_map

    def show_row_overlay(
            self,
            use_enhanced: bool = False,
            show_gridlines: bool = True,
            ax: plt.Axes | None = None,
            figsize: tuple[int, int] = (9, 10),
    ) -> tuple[plt.Figure, plt.Axes]:
        """Visualize colonies with row-based color coding and optional grid overlay.

        Displays the image with an overlay where each colony is colored according to
        its grid row assignment. This helps visualize the row structure of the pinned
        array and identify any row-wise positioning issues or misalignment.

        Args:
            use_enhanced (bool, optional): If True, use the detection matrix version
                of the parent image (detect_mat) for better contrast and visibility.
                If False, use the standard grayscale image (gray). Defaults to False.
            show_gridlines (bool, optional): If True, overlay cyan dashed horizontal
                lines marking the row boundaries and vertical lines for column boundaries.
                Defaults to True.
            ax (plt.Axes | None, optional): Existing Matplotlib Axes object to plot into.
                If None, a new figure and axes are created with the specified figsize.
                Defaults to None.
            figsize (tuple[int, int], optional): Figure size as (width, height) in inches,
                only used when ax is None. Defaults to (9, 10).

        Returns:
            tuple[plt.Figure, plt.Axes]: A tuple containing the Matplotlib Figure and
                Axes objects. If ax is None, returns the created figure and axes.
                If ax is provided, returns the created figure and the input ax object.

        Examples:
            Display row overlay visualization with options:

            >>> # Display row overlay with gridlines
            >>> fig, ax = grid_image.grid.show_row_overlay(show_gridlines=True)
            >>> plt.title("Colony Array - Row Overlay")
            >>> plt.show()
            >>> # Use enhanced image for better contrast
            >>> fig, ax = grid_image.grid.show_row_overlay(
            ...     use_enhanced=True,
            ...     show_gridlines=True,
            ...     figsize=(12, 14)
            ... )
            >>> # Create side-by-side comparison
            >>> fig, axes = plt.subplots(1, 2, figsize=(16, 10))
            >>> grid_image.grid.show_column_overlay(ax=axes[0])
            >>> grid_image.grid.show_row_overlay(ax=axes[1])
            >>> plt.suptitle("Column vs Row Grid Visualization")
            >>> plt.show()
        """
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(
                    label2rgb(label=self.get_row_map(),
                              image=self._root_image.detect_mat[:])
            )
        else:
            func_ax.imshow(
                    label2rgb(label=self.get_row_map(),
                              image=self._root_image.gray[:])
            )

        if show_gridlines:
            col_edges = self.get_col_edges()
            row_edges = self.get_row_edges()
            func_ax.hlines(
                    y=row_edges,
                    xmin=col_edges.min(),
                    xmax=col_edges.max(),
                    colors="c",
                    linestyles="--",
            )

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    """
    Grid Sections
    """

    def get_section_map(self) -> np.ndarray:
        """Get an object map with objects labeled by their grid section number.

        Creates a copy of the object map where each detected colony is relabeled
        according to its grid section assignment (flattened grid index). Section
        numbering is 0-indexed, ordered left-to-right, top-to-bottom (row-major).

        Returns:
            np.ndarray: 2D integer array with same shape as the parent image. Each
                pixel belonging to a colony is set to that colony's grid section
                number (0-indexed, ranging from 0 to nrows*ncols-1). Pixels not
                belonging to any colony are 0. Can be passed directly to label2rgb
                for visualization.

        Examples:
            Get and visualize section-labeled colony map:

            >>> section_map = grid_image.grid.get_section_map()
            >>> # For an 8x12 grid:
            >>> # Section 0: top-left (row 0, col 0)
            >>> # Section 11: top-right (row 0, col 11)
            >>> # Section 84: bottom-left (row 7, col 0)
            >>> # Section 95: bottom-right (row 7, col 11)
            >>> # Identify empty sections
            >>> empty_sections = []
            >>> for section_num in range(grid_image.grid.nrows * grid_image.grid.ncols):
            ...     if np.sum(section_map == section_num) == 0:
            ...         empty_sections.append(section_num)
            >>> print(f"Empty sections: {empty_sections}")
            >>> # Visualize section distribution
            >>> from skimage.color import label2rgb
            >>> colored_sections = label2rgb(label=section_map, image=grid_image.gray[:])
            >>> plt.imshow(colored_sections)
        """
        grid_info = self.info()

        section_map = self._root_image.objmap[:]
        for n, bidx in enumerate(
                np.sort(grid_info.loc[:, GRID.ROW_MAJOR_IDX].unique())):
            subtable = grid_info.loc[grid_info.loc[:, GRID.ROW_MAJOR_IDX] == bidx, :]
            section_map[
                np.isin(
                        element=self._root_image.objmap[:],
                        test_elements=subtable.loc[:, OBJECT.LABEL].to_numpy(),
                )
            ] = n + 1

        return section_map

    def get_section_counts(self, ascending: bool = False) -> pd.Series:
        """Count the number of objects (colonies) in each grid section.

        Returns a Series showing how many colonies were detected in each grid section,
        sorted by count. Useful for quality control to identify problematic sections
        with unexpected colony counts (e.g., empty sections, multiple colonies in
        single pinned location, indicating pinning errors or detection artifacts).

        Args:
            ascending (bool, optional): If False (default), sort counts in descending
                order (sections with most colonies first). If True, sort ascending
                (fewest colonies first, useful for identifying empty sections).
                Defaults to False.

        Returns:
            pd.Series: A pandas Series where:
                - Index: Grid section number (0 to nrows*ncols-1), unsorted sections
                  (those with no colonies) are not included
                - Values: Count of colonies in that section
                - Index name: GRID.ROW_MAJOR_IDX constant

        Examples:
            Count and analyze colonies per grid section:

            >>> section_counts = grid_image.grid.get_section_counts()
            >>> # Find sections with multiple colonies (potential pinning errors)
            >>> problem_sections = section_counts[section_counts > 1]
            >>> print(f"Sections with multiple colonies: {problem_sections}")
            >>> # Output:
            >>> # SectionNum
            >>> # 5      2
            >>> # 12     3
            >>> # dtype: int64
            >>> # Find empty sections (no colony detected)
            >>> expected_sections = set(range(grid_image.grid.nrows * grid_image.grid.ncols))
            >>> detected_sections = set(section_counts.index)
            >>> empty_sections = expected_sections - detected_sections
            >>> print(f"Empty sections: {empty_sections}")
            >>> # Statistics on detection completeness
            >>> num_expected = grid_image.grid.nrows * grid_image.grid.ncols
            >>> num_detected = len(section_counts)
            >>> completeness = 100 * num_detected / num_expected
            >>> print(f"Array completeness: {completeness:.1f}%")
        """
        return (
            self.info()
            .loc[:, GRID.ROW_MAJOR_IDX]
            .value_counts()
            .sort_values(ascending=ascending)
        )

    def get_info_by_section(
            self, section_number: int | tuple[int, int]
    ) -> pd.DataFrame:
        """Get grid information for colonies in a specific grid section.

        Retrieves detailed colony information (bounding box coordinates, centroid,
        area, etc.) for all objects within a given grid section. The section can be
        specified either by flattened index or by (row, column) tuple. Returns an
        empty DataFrame if no colonies are present in the requested section.

        Args:
            section_number (int | tuple[int, int]): Grid section identifier:
                - If int: flattened section index (0 to nrows*ncols-1)
                - If tuple[int, int]: (row_index, col_index) pair specifying grid
                  position, with both indices 0-based

        Returns:
            pd.DataFrame: DataFrame with one row per colony in the specified section.
                Contains the same columns as the info() method, including ObjectLabel,
                CenterRR, CenterCC, bounding box coordinates, grid position columns
                (RowNum, ColNum, SectionNum), and optionally metadata columns.
                Returns empty DataFrame if section contains no colonies.

        Raises:
            ValueError: If section_number is neither an int nor a 2-tuple.

        Examples:
            Retrieve colony information for specific grid sections:

            >>> # Get colonies using flattened index (section 25)
            >>> section_info = grid_image.grid.get_info_by_section(25)
            >>> print(f"Colonies in section 25: {len(section_info)}")
            >>> # Get colonies using (row, column) notation
            >>> # Get colonies in grid position (row=2, col=5)
            >>> section_info = grid_image.grid.get_info_by_section((2, 5))
            >>> if len(section_info) > 0:
            ...     # Analyze properties of colonies in this section
            ...     colony = section_info.iloc[0]
            ...     print(f"Colony area: {colony['Area']} pixels")
            ...     print(f"Colony center: ({colony['CenterRR']}, {colony['CenterCC']})")
            ... else:
            ...     print("No colony detected in this section")
            >>> # Find largest colony in section 10
            >>> section_10 = grid_image.grid.get_info_by_section(10)
            >>> if len(section_10) > 0:
            ...     largest = section_10.loc[section_10['Area'].idxmax()]
            ...     print(f"Largest colony: label={largest.name}, area={largest['Area']}")
        """
        if isinstance(section_number, int):  # Access by section number
            grid_info = self.info()
            return grid_info.loc[
                grid_info.loc[:, str(GRID.ROW_MAJOR_IDX)] == section_number, :
            ]
        elif (
                isinstance(section_number, tuple) and len(section_number) == 2
        ):  # Access by row and col number
            grid_info = self.info()
            grid_info = grid_info.loc[
                grid_info.loc[:, str(GRID.ROW_NUM)] == section_number[0], :
            ]
            return grid_info.loc[
                grid_info.loc[:, str(GRID.ROW_NUM)] == section_number[1], :
            ]
        else:
            raise ValueError("Section index should be int or a tuple of label_subset")

    def _naive_get_grid_section_slices(
            self, idx: int
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Internal method: get pixel slices for a grid section based on grid edges.

        Returns the exact pixel boundaries of a grid section without considering
        the actual objects within it. Uses grid edge positions to determine section
        bounds. This may result in cropping objects that extend beyond the grid
        section boundaries.

        Args:
            idx (int): Flattened grid section index (0 to nrows*ncols-1).

        Returns:
            tuple[tuple[float, float], tuple[float, float]]: A tuple containing:
                - (min_row, min_col): Minimum pixel coordinates (top-left corner)
                - (max_row, max_col): Maximum pixel coordinates (bottom-right corner)
                These values can be used for slicing the parent image.
        """
        row_edges, col_edges = self.get_row_edges(), self.get_col_edges()
        row_pos, col_pos = np.where(self._idx_ref_matrix == idx)
        min_cc = col_edges[col_pos]
        max_cc = col_edges[col_pos + 1]
        min_rr = row_edges[row_pos]
        max_rr = row_edges[row_pos + 1]
        return (min_rr, min_cc), (max_rr, max_cc)

    def _adv_get_grid_section_slices(
            self,
            idx: int,
            grid_info: pd.DataFrame | None = None,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Internal method: get pixel slices for a grid section accounting for object boundaries.

        Returns pixel boundaries for a grid section, expanded if necessary to fully
        include all objects that belong to this section. This preserves complete
        objects that might extend slightly beyond the ideal grid boundaries, and
        clips to image boundaries.

        Args:
            idx (int): Flattened grid section index (0 to nrows*ncols-1).
            grid_info (pd.DataFrame | None): Optional precomputed grid info table.

        Returns:
            tuple[tuple[float, float], tuple[float, float]]: A tuple containing:
                - (min_row, min_col): Minimum pixel coordinates (top-left corner)
                - (max_row, max_col): Maximum pixel coordinates (bottom-right corner)
                Coordinates are clipped to valid image boundaries [0, image_width/height].
                These values can be used for slicing the parent image.
        """
        grid_min, grid_max = self._naive_get_grid_section_slices(idx)
        grid_min_rr, grid_min_cc = grid_min
        grid_max_rr, grid_max_cc = grid_max

        if grid_info is None:
            grid_info = self.info()
        section_info = grid_info.loc[
            grid_info.loc[:, str(GRID.ROW_MAJOR_IDX)] == idx, :]

        obj_min_cc = section_info.loc[:, str(BBOX.MIN_CC)].min()
        min_cc = min(grid_min_cc, obj_min_cc)
        if min_cc < 0:
            min_cc = 0

        obj_max_cc = section_info.loc[:, str(BBOX.MAX_CC)].max()
        max_cc = max(grid_max_cc, obj_max_cc)
        if max_cc > self._root_image.shape[1] - 1:
            max_cc = self._root_image.shape[1] - 1

        obj_min_rr = section_info.loc[:, str(BBOX.MIN_RR)].min()
        min_rr = min(grid_min_rr, obj_min_rr)
        if min_rr < 0:
            min_rr = 0

        obj_max_rr = section_info.loc[:, str(BBOX.MAX_RR)].max()
        max_rr = max(grid_max_rr, obj_max_rr)
        if max_rr > self._root_image.shape[0] - 1:
            max_rr = self._root_image.shape[0] - 1

        return (min_rr, min_cc), (max_rr, max_cc)

    def _get_section_labels(
            self,
            idx: int,
            grid_info: pd.DataFrame | None = None,
    ) -> list[int]:
        """Internal method: get object labels belonging to a grid section.

        Retrieves all object labels (colony identifiers) that are assigned to
        the specified grid section based on centroid-based grid assignment.

        Args:
            idx (int): Flattened grid section index (0 to nrows*ncols-1).
            grid_info (pd.DataFrame | None): Optional precomputed grid info table.

        Returns:
            list[int]: List of object labels assigned to this grid section.
                Returns empty list if no colonies are in the section.
        """
        if grid_info is None:
            grid_info = self.info()
        section_info = grid_info.loc[
            grid_info.loc[:, str(GRID.ROW_MAJOR_IDX)] == idx, :]
        return section_info[OBJECT.LABEL].to_list()

    def _build_gridline_shapes(self) -> list[np.ndarray]:
        """Build line shape arrays for all grid boundaries.

        Returns:
            list[np.ndarray]: List of 2x2 arrays ``[[row_start, col_start],
                [row_end, col_end]]``, one per grid edge. Produces
                ``(ncols + 1) + (nrows + 1)`` lines total.
        """
        row_edges = self.get_row_edges()
        col_edges = self.get_col_edges()
        lines: list[np.ndarray] = []

        # Vertical lines span from first row edge to last row edge
        for c in col_edges:
            lines.append(np.array([[row_edges[0], c], [row_edges[-1], c]]))

        # Horizontal lines span from first col edge to last col edge
        for r in row_edges:
            lines.append(np.array([[r, col_edges[0]], [r, col_edges[-1]]]))

        return lines

    def _build_section_box_shapes(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Build rectangle shapes and per-section colors for all grid sections.

        Uses grid edge positions so every section gets a box regardless of
        whether colonies are detected. Colors cycle through the ``tab20``
        colormap.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple of
                ``(rectangles, colors)`` where each rectangle is a 4x2 array
                of corner vertices and each color is an RGBA array.
        """
        row_edges = self.get_row_edges()
        col_edges = self.get_col_edges()
        cmap = plt.get_cmap("tab20")

        rectangles: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        idx = 0
        for ri in range(len(row_edges) - 1):
            r0 = row_edges[ri]
            r1 = row_edges[ri + 1]
            for ci in range(len(col_edges) - 1):
                c0 = col_edges[ci]
                c1 = col_edges[ci + 1]
                rectangles.append(
                    np.array([[r0, c0], [r0, c1], [r1, c1], [r1, c0]])
                )
                colors.append(np.asarray(cmap(idx % 20)))
                idx += 1

        return rectangles, colors

    def napari(
        self,
        name: str | None = None,
        reset: bool = False,
        *,
        show_gridlines: bool = True,
        show_section_boxes: bool = True,
        gridline_color: str = "cyan",
        gridline_edge_width: float = 2.0,
        section_box_edge_width: float = 2.0,
        opacity: float = 1.0,
    ) -> napari.Viewer:
        """Add grid overlay layers to a persistent global napari viewer.

        Creates or reuses a single napari viewer instance and adds Shapes
        layers for gridlines and/or section boxes. Each layer can be toggled
        independently inside the napari GUI.

        Args:
            name: Optional custom name used in layer naming. If None, uses the
                image's ``name`` attribute. Defaults to None.
            reset: If True, closes the current napari viewer and creates a
                fresh one. Defaults to False.
            show_gridlines: If True, add a Shapes layer with lines at every
                grid boundary. Defaults to True.
            show_section_boxes: If True, add a Shapes layer with colored
                rectangles for each grid section. Defaults to True.
            gridline_color: Edge color for gridlines (any napari color spec).
                Defaults to ``"cyan"``.
            gridline_edge_width: Stroke width in pixels for gridlines.
                Defaults to 2.0.
            section_box_edge_width: Stroke width in pixels for section box
                edges. Defaults to 2.0.
            opacity: Layer opacity from 0.0 (transparent) to 1.0 (opaque).
                Defaults to 1.0.

        Returns:
            napari.Viewer: The global napari viewer instance with grid layers.

        Raises:
            ImportError: If napari is not installed.
            ValueError: If opacity is not in [0.0, 1.0].

        Examples:
            Overlay grid on a grayscale image:

            >>> from phenotypic.data import load_synth_yeast_plate
            >>> gi = load_synth_yeast_plate()
            >>> viewer = gi.gray.napari()
            >>> viewer = gi.grid.napari()

            Show only gridlines without section boxes:

            >>> viewer = gi.grid.napari(show_section_boxes=False)
        """
        if not 0.0 <= opacity <= 1.0:
            raise ValueError(f"opacity must be in range [0.0, 1.0], got {opacity}")

        from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import (
            _HAS_NAPARI,
            _viewer_is_alive,
        )
        from phenotypic._core._image_parts.accessor_abstracts import (
            _image_accessor_base,
        )

        if not _HAS_NAPARI:
            raise ImportError(
                "napari is required for interactive visualization. "
                "Install with: pip install phenotypic[gui]"
            )
        import napari as _napari

        viewer = _image_accessor_base._global_napari_viewer

        # Reset viewer if requested
        if reset and _viewer_is_alive(viewer):
            viewer.close()
            _image_accessor_base._global_napari_viewer = None
            viewer = None

        # Create new viewer if needed
        if not _viewer_is_alive(viewer):
            viewer = _napari.Viewer()
            _image_accessor_base._global_napari_viewer = viewer

        # Resolve layer name prefix
        if name is not None:
            image_name = name
        else:
            image_name = getattr(self._root_image, "name", "image")

        # --- Gridlines layer ---
        if show_gridlines:
            gridline_layer_name = f"grid_{image_name}_gridlines"
            gridline_shapes = self._build_gridline_shapes()
            try:
                layer = viewer.layers[gridline_layer_name]
                layer.data = gridline_shapes
                layer.edge_color = gridline_color
                layer.edge_width = gridline_edge_width
                layer.opacity = opacity
            except KeyError:
                viewer.add_shapes(
                    gridline_shapes,
                    shape_type="line",
                    edge_color=gridline_color,
                    edge_width=gridline_edge_width,
                    name=gridline_layer_name,
                    opacity=opacity,
                )

        # --- Section boxes layer ---
        if show_section_boxes:
            section_layer_name = f"grid_{image_name}_sections"
            section_rectangles, section_colors = self._build_section_box_shapes()
            try:
                layer = viewer.layers[section_layer_name]
                layer.data = section_rectangles
                layer.edge_color = section_colors
                layer.edge_width = section_box_edge_width
                layer.opacity = opacity
            except KeyError:
                viewer.add_shapes(
                    section_rectangles,
                    shape_type="rectangle",
                    edge_color=section_colors,
                    face_color="transparent",
                    edge_width=section_box_edge_width,
                    name=section_layer_name,
                    opacity=opacity,
                )

        return viewer
