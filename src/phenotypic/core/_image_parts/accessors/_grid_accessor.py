from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import GridImage

import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import pandas as pd
from skimage.color import label2rgb

import phenotypic
from phenotypic.core._image_parts.accessor_abstracts import ImageAccessorBase
from phenotypic.tools.constants_ import METADATA, IMAGE_TYPES, BBOX, GRID, OBJECT
from phenotypic.tools.exceptions_ import NoObjectsError


class GridAccessor(ImageAccessorBase):
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
        Access grid information for a 96-well colony plate (8 rows x 12 columns):

        ```python
        from phenotypic import GridImage

        # Load image and create grid accessor
        grid_image = GridImage('agar_plate.png', nrows=8, ncols=12)

        # Get grid information as a DataFrame
        grid_info = grid_image.grid.info()
        print(f"Found {len(grid_info)} colonies across {grid_image.grid.nrows} rows "
              f"and {grid_image.grid.ncols} columns")

        # Extract a single grid section (colony at row 2, column 3)
        section_idx = 2 * grid_image.grid.ncols + 3  # Flattened index
        colony_image = grid_image.grid[section_idx]

        # Visualize grid columns with color-coded labels
        fig, ax = grid_image.grid.show_column_overlay(show_gridlines=True)
        ```

        Get colony counts by grid section:

        ```python
        # Count colonies in each grid section
        section_counts = grid_image.grid.get_section_counts(ascending=False)
        print("Colonies per section (sorted):")
        print(section_counts)

        # Get all colony information for row 0
        row_info = grid_image.grid.get_info_by_section((0, slice(None)))
        ```
    """

    def __init__(self, root_image: GridImage):
        super().__init__(root_image)
        self._root_image: GridImage = root_image

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
            nrows (int): Number of rows. Must be a positive integer.

        Raises:
            ValueError: If nrows < 1.
            TypeError: If nrows is not an integer.
        """
        if nrows < 1: raise ValueError('Number of nrows must be greater than 0')
        if type(nrows) != int: raise TypeError('Number of nrows must be an integer')

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
            ncols (int): Number of columns. Must be a positive integer.

        Raises:
            ValueError: If ncols < 1.
            TypeError: If ncols is not an integer.
        """
        if ncols < 1: raise ValueError('Number of columns must be greater than 0')
        if type(ncols) != int: raise TypeError('Number of columns must be an integer')

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
            ```python
            # Get full grid information
            grid_info = grid_image.grid.info()

            # Count colonies by row
            colonies_per_row = grid_info.groupby('RowNum').size()

            # Find largest colony in grid section 10
            section_10 = grid_info[grid_info['SectionNum'] == 10]
            largest = section_10.loc[section_10['Area'].idxmax()]

            # Get colonies without metadata
            grid_info_minimal = grid_image.grid.info(include_metadata=False)
            ```
        """
        info = self._root_image.grid_finder.measure(self._root_image)
        if include_metadata:
            return self._root_image.metadata.insert_metadata(info)
        else:
            return info

    @property
    def _idx_ref_matrix(self):
        """Internal property: matrix mapping grid positions to flattened indices.

        Returns:
            np.ndarray: 2D array of shape (nrows, ncols) where element [i, j] contains
                the flattened index for that grid position.
        """
        return np.reshape(np.arange(self.nrows*self.ncols), newshape=(self.nrows, self.ncols))

    def __getitem__(self, idx):
        """Extract a grid section as a subimage.

        Returns a cropped image corresponding to a specific grid section based on either
        its flattened index or a (row, column) grid coordinate. The grid is indexed
        left-to-right, top-to-bottom (row-major order). Only objects belonging to the
        specified grid section are included in the subimage.

        Args:
            idx (int | tuple[int, int]): Grid section identifier.
                - If int: flattened grid section index, ranging from 0 to
                  ``nrows * ncols - 1``. For an 8x12 grid: section 0 is top-left,
                  section 11 is top-right, section 84 is bottom-left,
                  section 95 is bottom-right.
                - If tuple: ``(row_index, col_index)`` pair specifying the grid
                  location, with both indices 0-based.

        Returns:
            phenotypic.Image: A subimage containing the grid section. Pixel coordinates
                are adjusted to the section origin. Object labels are preserved for objects
                belonging to this section only; other objects are removed (label set to 0).
                If no objects are present in the parent image, returns a copy of the entire
                parent image.

        Raises:
            IndexError: If idx is out of bounds (for either flattened or
                ``(row, col)`` indexing), or if a tuple index does not have
                length 2.

        Examples:

            ```python
            # Extract top-left grid section (row 0, col 0)
            top_left = grid_image.grid[0]
            print(f"Section size: {top_left.shape}")

            # Extract center section for an 8x12 grid (row 4, col 6)
            # Using flattened index
            center_idx = 4 * 12 + 6  # = 54
            center_section = grid_image.grid[center_idx]

            # The same section accessed using (row, col) indexing
            center_section_2 = grid_image.grid[4, 6]

            # Process colonies in top row (row 0, columns 0-11)
            for col in range(grid_image.grid.ncols):
                # Access by (row, col) index
                section = grid_image.grid[0, col]
                analyze_colony(section)
            ```
        """
        # Allow access either by flattened index or by (row, col) tuple
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise IndexError('Grid section index tuple must have length 2: (row, col).')
            row_idx, col_idx = idx
            # This will naturally raise IndexError for out-of-range indices
            idx = int(self._idx_ref_matrix[row_idx, col_idx])

        if self._root_image.objects.num_objects != 0:
            min_coords, max_coords = self._adv_get_grid_section_slices(idx)
            min_rr, min_cc = min_coords
            max_rr, max_cc = max_coords

            section_image = phenotypic.Image(self._root_image[int(min_rr):int(max_rr), int(min_cc):int(max_cc)])

            # Remove objects that don't belong in that grid section from the subimage
            objmap = section_image.objmap[:]
            objmap[~np.isin(objmap, self._get_section_labels(idx))] = 0
            section_image.objmap = objmap
            section_image.metadata[METADATA.IMAGE_TYPE] = IMAGE_TYPES.GRID_SECTION.value

            return section_image
        else:
            return phenotypic.Image(self._root_image)

    def get_centroid_alignment_info(self, axis) -> Tuple[np.ndarray[float], np.ndarray[int]]:
        """Calculate linear regression fit for colony centroids along a grid axis.

        Computes the slope and intercept of a best-fit line through the centroids of
        colonies arranged along a specified axis (rows or columns). This quantifies
        alignment quality and any systematic drift in the pinned colony array.

        The linear regression uses the standard least-squares method (2D covariance/variance)
        to find the best-fit line: y = m*x + b.

        Args:
            axis (int): Axis along which to compute alignment:
                - 0: Row-wise alignment. For each row, finds how colony centers vary
                  along the column (CC) axis as a function of their column index.
                - 1: Column-wise alignment. For each column, finds how colony centers
                  vary along the row (RR) axis as a function of their row index.

        Returns:
            Tuple[np.ndarray[float], np.ndarray[int]]: A tuple containing:
                - m_slope (np.ndarray): Slopes for each row/column (length nrows or ncols).
                  NaN indicates no colonies in that row/column.
                - b_intercept (np.ndarray): Y-intercepts, rounded to nearest integer.
                  NaN indicates no colonies in that row/column.

        Raises:
            NoObjectsError: If the parent image contains no detected objects (colonies).
            ValueError: If axis is not 0 or 1.

        Examples:
            ```python
            # Check row alignment (how much do colonies shift left-right across their row?)
            row_slopes, row_intercepts = grid_image.grid.get_centroid_alignment_info(axis=0)
            print(f"Row alignment slopes (pixels/column): {row_slopes}")

            # Check column alignment (how much do colonies shift top-bottom across their column?)
            col_slopes, col_intercepts = grid_image.grid.get_centroid_alignment_info(axis=1)

            # Identify problematic rows with high drift
            drift_threshold = 0.05  # pixels per grid position
            problematic_rows = np.where(np.abs(row_slopes) > drift_threshold)[0]
            print(f"Rows with significant drift: {problematic_rows}")
            ```
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
            raise ValueError('Axis should be 0 or 1.')

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

            covariance = ((x - x_mean)*(y - y_mean)).sum()
            variance = ((x - x_mean) ** 2).sum()
            if variance != 0:
                m_slope[idx] = covariance/variance
                b_intercept[idx] = y_mean - m_slope[idx]*x_mean
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
        of each grid column. For an ncols-column grid, returns ncols+1 values representing
        the left edge of column 0, boundaries between adjacent columns, and the right edge
        of column ncols-1.

        Returns:
            np.ndarray: 1D array of column edge positions (pixel column indices).
                Length is ncols+1. Values are in ascending order.

        Examples:
            ```python
            col_edges = grid_image.grid.get_col_edges()
            print(f"Column edges: {col_edges}")
            # Output: [0.0, 106.5, 213.0, 319.5, ...]  for a 12-column grid

            # Calculate column width
            col_width = col_edges[1] - col_edges[0]
            print(f"Column width: {col_width} pixels")

            # Extract pixels for column 3
            col_3_min, col_3_max = int(col_edges[3]), int(col_edges[4])
            column_3_data = grid_image.gray[:, col_3_min:col_3_max]
            ```
        """
        return self._root_image.grid_finder.get_col_edges(self._root_image)

    def get_col_map(self) -> np.ndarray:
        """Get an object map with objects labeled by their grid column number.

        Creates a copy of the object map where each detected colony is relabeled according
        to its grid column assignment. This is useful for visualizing or analyzing all
        colonies in a particular column together.

        Returns:
            np.ndarray: 2D integer array with same shape as the parent image. Each pixel
                belonging to a colony is set to that colony's grid column number (1-indexed,
                ranging from 1 to ncols). Pixels not belonging to any colony are 0.

        Examples:
            ```python
            col_map = grid_image.grid.get_col_map()

            # All colonies in column 0 have value 1, column 1 have value 2, etc.
            print(f"Unique values in col_map: {np.unique(col_map)}")
            # Output: [0, 1, 2, 3, ..., 12]  for a 12-column grid

            # Count total pixels belonging to each column
            for col_num in range(1, grid_image.grid.ncols + 1):
                col_pixels = np.sum(col_map == col_num)
                print(f"Column {col_num}: {col_pixels} pixels")
            ```
        """
        grid_info = self.info()
        col_map = self._root_image.objmap[:].copy()
        for n, col_bidx in enumerate(np.sort(grid_info.loc[:, str(GRID.COL_NUM)].unique())):
            subtable = grid_info.loc[grid_info.loc[:, str(GRID.COL_NUM)] == col_bidx, :]

            # Edit the new map's objects to equal the column number
            col_map[np.isin(
                    element=self._root_image.objmap[:],
                    test_elements=subtable[OBJECT.LABEL].to_numpy(),
            )] = n + 1
        return col_map

    def show_column_overlay(self, use_enhanced=False, show_gridlines=True, ax=None,
                            figsize=(9, 10)) -> Tuple[plt.Figure, plt.Axes]:
        """Visualize colonies with column-based color coding and optional grid overlay.

        Displays the image with an overlay showing each colony colored by its grid column.
        This helps visualize the column structure of the pinned array and identify any
        column-wise positioning issues.

        Args:
            use_enhanced (bool, optional): If True, use the enhanced grayscale version
                of the image for better visibility. Defaults to False.
            show_gridlines (bool, optional): If True, overlay cyan dashed lines marking
                the grid column and row boundaries. Defaults to True.
            ax (plt.Axes, optional): Existing Matplotlib Axes to plot into. If None,
                a new figure and axes are created. Defaults to None.
            figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
                Only used if ax is None. Defaults to (9, 10).

        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib Figure and Axes objects. If ax was
                provided as input, ax is returned instead of the original input value.

        Examples:
            ```python
            # Display column overlay with gridlines
            fig, ax = grid_image.grid.show_column_overlay(show_gridlines=True)
            plt.title("Colony Array - Column Overlay")
            plt.show()

            # Use enhanced image for better contrast
            fig, ax = grid_image.grid.show_column_overlay(
                use_enhanced=True,
                show_gridlines=True,
                figsize=(12, 14)
            )

            # Plot on existing axes
            fig, axes = plt.subplots(1, 2, figsize=(16, 10))
            grid_image.grid.show_column_overlay(ax=axes[0])
            grid_image.grid.show_row_overlay(ax=axes[1])
            ```
        """
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(label2rgb(label=self.get_col_map(), image=self._root_image.enh_gray[:]))
        else:
            func_ax.imshow(label2rgb(label=self.get_col_map(), image=self._root_image.gray[:]))

        if show_gridlines:
            col_edges = self.get_col_edges()
            row_edges = self.get_row_edges()
            func_ax.vlines(x=col_edges, ymin=row_edges.min(), ymax=row_edges.max(), colors='c', linestyles='--')

        return fig, ax

    """
    Grid Rows
    """

    def get_row_edges(self) -> np.ndarray:
        """Get the row boundary positions in pixel coordinates.

        Returns the y-coordinates (row indices) that define the horizontal boundaries
        of each grid row. For an nrows-row grid, returns nrows+1 values representing
        the top edge of row 0, boundaries between adjacent rows, and the bottom edge
        of row nrows-1.

        Returns:
            np.ndarray: 1D array of row edge positions (pixel row indices).
                Length is nrows+1. Values are in ascending order.

        Examples:
            ```python
            row_edges = grid_image.grid.get_row_edges()
            print(f"Row edges: {row_edges}")
            # Output: [0.0, 95.2, 190.4, 285.6, ...]  for an 8-row grid

            # Calculate row height
            row_height = row_edges[1] - row_edges[0]
            print(f"Row height: {row_height} pixels")

            # Extract pixels for row 4
            row_4_min, row_4_max = int(row_edges[4]), int(row_edges[5])
            row_4_data = grid_image.gray[row_4_min:row_4_max, :]
            ```
        """
        return self._root_image.grid_finder.get_row_edges(self._root_image)

    def get_row_map(self) -> np.ndarray:
        """Get an object map with objects labeled by their grid row number.

        Creates a copy of the object map where each detected colony is relabeled according
        to its grid row assignment. This is useful for visualizing or analyzing all
        colonies in a particular row together.

        Returns:
            np.ndarray: 2D integer array with same shape as the parent image. Each pixel
                belonging to a colony is set to that colony's grid row number (1-indexed,
                ranging from 1 to nrows). Pixels not belonging to any colony are 0.

        Examples:
            ```python
            row_map = grid_image.grid.get_row_map()

            # All colonies in row 0 have value 1, row 1 have value 2, etc.
            print(f"Unique values in row_map: {np.unique(row_map)}")
            # Output: [0, 1, 2, 3, ..., 8]  for an 8-row grid

            # Count total pixels belonging to each row
            for row_num in range(1, grid_image.grid.nrows + 1):
                row_pixels = np.sum(row_map == row_num)
                print(f"Row {row_num}: {row_pixels} pixels")
            ```
        """
        grid_info = self.info()
        row_map = self._root_image.objmap[:].copy()
        for n, col_bidx in enumerate(np.sort(grid_info.loc[:, str(GRID.ROW_NUM)].unique())):
            subtable = grid_info.loc[grid_info.loc[:, str(GRID.ROW_NUM)] == col_bidx, :]

            # Edit the new map's objects to equal the column number
            row_map[
                np.isin(
                        element=self._root_image.objmap[:],
                        test_elements=subtable[OBJECT.LABEL].to_numpy(),
                )
            ] = n + 1
        return row_map

    def show_row_overlay(self, use_enhanced=False, show_gridlines=True, ax=None,
                         figsize=(9, 10)) -> (plt.Figure, plt.Axes):
        """Visualize colonies with row-based color coding and optional grid overlay.

        Displays the image with an overlay showing each colony colored by its grid row.
        This helps visualize the row structure of the pinned array and identify any
        row-wise positioning issues.

        Args:
            use_enhanced (bool, optional): If True, use the enhanced grayscale version
                of the image for better visibility. Defaults to False.
            show_gridlines (bool, optional): If True, overlay cyan dashed lines marking
                the grid row and column boundaries. Defaults to True.
            ax (plt.Axes, optional): Existing Matplotlib Axes to plot into. If None,
                a new figure and axes are created. Defaults to None.
            figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
                Only used if ax is None. Defaults to (9, 10).

        Returns:
            Tuple[plt.Figure, plt.Axes]: Matplotlib Figure and Axes objects. If ax was
                provided as input, only the axes are returned.

        Examples:
            ```python
            # Display row overlay with gridlines
            fig, ax = grid_image.grid.show_row_overlay(show_gridlines=True)
            plt.title("Colony Array - Row Overlay")
            plt.show()

            # Use enhanced image for better contrast
            fig, ax = grid_image.grid.show_row_overlay(
                use_enhanced=True,
                show_gridlines=True,
                figsize=(12, 14)
            )

            # Create side-by-side comparison
            fig, axes = plt.subplots(1, 2, figsize=(16, 10))
            grid_image.grid.show_column_overlay(ax=axes[0])
            grid_image.grid.show_row_overlay(ax=axes[1])
            plt.suptitle("Column vs Row Grid Visualization")
            plt.show()
            ```
        """
        if ax is None:
            fig, func_ax = plt.subplots(tight_layout=True, figsize=figsize)
        else:
            func_ax = ax

        func_ax.grid(False)

        if use_enhanced:
            func_ax.imshow(label2rgb(label=self.get_row_map(), image=self._root_image.enh_gray[:]))
        else:
            func_ax.imshow(label2rgb(label=self.get_row_map(), image=self._root_image.gray[:]))

        if show_gridlines:
            col_edges = self.get_col_edges()
            row_edges = self.get_row_edges()
            func_ax.hlines(y=row_edges, xmin=col_edges.min(), xmax=col_edges.max(), colors='c', linestyles='--')

        if ax is None:
            return fig, func_ax
        else:
            return func_ax

    """
    Grid Sections
    """

    def get_section_map(self) -> np.ndarray:
        """Get an object map with objects labeled by their grid section number.

        Creates a copy of the object map where each detected colony is relabeled according
        to its grid section assignment (flattened grid index). Section numbering is 0-indexed,
        ordered left-to-right, top-to-bottom.

        Returns:
            np.ndarray: 2D integer array with same shape as the parent image. Each pixel
                belonging to a colony is set to that colony's grid section number (0-indexed,
                ranging from 0 to nrows*ncols-1). Pixels not belonging to any colony are 0.

        Examples:
            ```python
            section_map = grid_image.grid.get_section_map()

            # For an 8x12 grid:
            # Section 0: top-left (row 0, col 0)
            # Section 11: top-right (row 0, col 11)
            # Section 84: bottom-left (row 7, col 0)
            # Section 95: bottom-right (row 7, col 11)

            # Identify empty sections
            empty_sections = []
            for section_num in range(grid_image.grid.nrows * grid_image.grid.ncols):
                if np.sum(section_map == section_num) == 0:
                    empty_sections.append(section_num)
            print(f"Empty sections: {empty_sections}")

            # Visualize section distribution
            from skimage.color import label2rgb
            colored_sections = label2rgb(label=section_map, image=grid_image.gray[:])
            plt.imshow(colored_sections)
            ```
        """
        grid_info = self.info()

        section_map = self._root_image.objmap[:]
        for n, bidx in enumerate(np.sort(grid_info.loc[:, GRID.SECTION_NUM].unique())):
            subtable = grid_info.loc[grid_info.loc[:, GRID.SECTION_NUM] == bidx, :]
            section_map[np.isin(
                    element=self._root_image.objmap[:],
                    test_elements=subtable.loc[:, OBJECT.LABEL].to_numpy(),
            )] = n + 1

        return section_map

    def get_section_counts(self, ascending=False) -> pd.DataFrame:
        """Count the number of objects (colonies) in each grid section.

        Returns a Series showing how many colonies were detected in each grid section.
        Useful for quality control to identify problematic sections with unexpected
        colony counts (e.g., empty sections, multiple colonies in single pinned location).

        Args:
            ascending (bool, optional): If False (default), sort counts in descending order
                (sections with most colonies first). If True, sort ascending (fewest first).
                Defaults to False.

        Returns:
            pd.Series: A pandas Series where:
                - Index: Grid section number (0 to nrows*ncols-1)
                - Values: Count of colonies in that section

        Examples:
            ```python
            section_counts = grid_image.grid.get_section_counts()

            # Find sections with multiple colonies (potential pinning errors)
            problem_sections = section_counts[section_counts > 1]
            print(f"Sections with multiple colonies: {problem_sections}")
            # Output:
            # SectionNum
            # 5      2
            # 12     3
            # dtype: int64

            # Find empty sections (no colony detected)
            expected_sections = set(range(grid_image.grid.nrows * grid_image.grid.ncols))
            detected_sections = set(section_counts.index)
            empty_sections = expected_sections - detected_sections
            print(f"Empty sections: {empty_sections}")

            # Statistics on detection completeness
            num_expected = grid_image.grid.nrows * grid_image.grid.ncols
            num_detected = len(section_counts)
            completeness = 100 * num_detected / num_expected
            print(f"Array completeness: {completeness:.1f}%")
            ```
        """
        return self.info().loc[:, GRID.SECTION_NUM].value_counts().sort_values(ascending=ascending)

    def get_info_by_section(self, section_number):
        """Get grid information for colonies in a specific grid section.

        Retrieves detailed colony information (bounding box, centroid, etc.) for all
        objects within a given grid section. The section can be specified either by
        flattened index or by (row, column) tuple.

        Args:
            section_number (int or tuple[int, int]): Grid section identifier:
                - If int: flattened section index (0 to nrows*ncols-1)
                - If tuple: (row_index, col_index) pair specifying the grid location

        Returns:
            pd.DataFrame: DataFrame with one row per colony in the specified section.
                Contains same columns as info() method. Empty DataFrame if section
                contains no colonies.

        Raises:
            ValueError: If section_number is neither an int nor a 2-tuple.

        Examples:
            ```python
            # Get colonies using flattened index (section 25)
            section_info = grid_image.grid.get_info_by_section(25)
            print(f"Colonies in section 25: {len(section_info)}")

            # Get colonies using (row, column) notation
            # Get colonies in grid position (row=2, col=5)
            section_info = grid_image.grid.get_info_by_section((2, 5))

            if len(section_info) > 0:
                # Analyze properties of colonies in this section
                colony = section_info.iloc[0]
                print(f"Colony area: {colony['Area']} pixels")
                print(f"Colony center: ({colony['CenterRR']}, {colony['CenterCC']})")
            else:
                print("No colony detected in this section")

            # Find largest colony in section 10
            section_10 = grid_image.grid.get_info_by_section(10)
            if len(section_10) > 0:
                largest = section_10.loc[section_10['Area'].idxmax()]
                print(f"Largest colony: label={largest.name}, area={largest['Area']}")
            ```
        """
        if isinstance(section_number, int):  # Access by section number
            grid_info = self.info()
            return grid_info.loc[grid_info.loc[:, str(GRID.SECTION_NUM)] == section_number, :]
        elif isinstance(section_number, tuple) and len(section_number) == 2:  # Access by row and col number
            grid_info = self.info()
            grid_info = grid_info.loc[grid_info.loc[:, str(GRID.ROW_NUM)] == section_number[0], :]
            return grid_info.loc[grid_info.loc[:, str(GRID.ROW_NUM)] == section_number[1], :]
        else:
            raise ValueError('Section index should be int or a tuple of label_subset')

    def _naive_get_grid_section_slices(self, idx) -> ((int, int), (int, int)):
        """Internal method: get pixel slices for a grid section based on grid edges.

        Returns the exact pixel boundaries of a grid section without considering
        the actual objects within it. This may result in cropping objects that
        extend beyond the grid section boundaries.

        Args:
            idx (int): Flattened grid section index.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: ((min_row, min_col), (max_row, max_col))
                giving pixel coordinates for slicing the parent image.
        """
        row_edges, col_edges = self.get_row_edges(), self.get_col_edges()
        row_pos, col_pos = np.where(self._idx_ref_matrix == idx)
        min_cc = col_edges[col_pos]
        max_cc = col_edges[col_pos + 1]
        min_rr = row_edges[row_pos]
        max_rr = row_edges[row_pos + 1]
        return (min_rr, min_cc), (max_rr, max_cc)

    def _adv_get_grid_section_slices(self, idx) -> ((int, int), (int, int)):
        """Internal method: get pixel slices for a grid section accounting for object boundaries.

        Returns pixel boundaries for a grid section, expanded if necessary to fully
        include all objects that belong to this section. This preserves complete
        objects that might extend slightly beyond the ideal grid boundaries.

        Args:
            idx (int): Flattened grid section index.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: ((min_row, min_col), (max_row, max_col))
                giving pixel coordinates for slicing the parent image.
        """
        grid_min, grid_max = self._naive_get_grid_section_slices(idx)
        grid_min_rr, grid_min_cc = grid_min
        grid_max_rr, grid_max_cc = grid_max

        grid_info = self.info()
        section_info = grid_info.loc[grid_info.loc[:, str(GRID.SECTION_NUM)] == idx, :]

        obj_min_cc = section_info.loc[:, str(BBOX.MIN_CC)].min()
        min_cc = min(grid_min_cc, obj_min_cc)
        if min_cc < 0: min_cc = 0

        obj_max_cc = section_info.loc[:, str(BBOX.MAX_CC)].max()
        max_cc = max(grid_max_cc, obj_max_cc)
        if max_cc > self._root_image.shape[1] - 1: max_cc = self._root_image.shape[1] - 1

        obj_min_rr = section_info.loc[:, str(BBOX.MIN_RR)].min()
        min_rr = min(grid_min_rr, obj_min_rr)
        if min_rr < 0: min_rr = 0

        obj_max_rr = section_info.loc[:, str(BBOX.MAX_RR)].max()
        max_rr = max(grid_max_rr, obj_max_rr)
        if max_rr > self._root_image.shape[0] - 1: max_rr = self._root_image.shape[0] - 1

        return (min_rr, min_cc), (max_rr, max_cc)

    def _get_section_labels(self, idx) -> list[int]:
        """Internal method: get object labels belonging to a grid section.

        Args:
            idx (int): Flattened grid section index.

        Returns:
            list[int]: List of object labels assigned to this grid section.
        """
        grid_info = self.info()
        section_info = grid_info.loc[grid_info.loc[:, str(GRID.SECTION_NUM)] == idx, :]
        return section_info.index.to_list()
