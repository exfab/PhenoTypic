from __future__ import annotations

import numpy as np
import pandas as pd

from .abc_ import SetAnalyzer


class EdgeCorrector(SetAnalyzer):
    """Analyzer for detecting and correcting edge effects in colony detection.
    
    This class identifies colonies at grid edges (missing orthogonal neighbors) and
    caps their measurement values to prevent edge effects in growth assays. Edge
    colonies often show artificially inflated measurements due to lack of competition
    for resources.
    
    Args:
        on: Column name for grouping/aggregation operations.
        groupby: List of column names to group by (e.g., ['ImageName', 'Metadata_Plate']).
        measurement_col: Name of measurement column to correct (e.g., 'Area', 'MeanRadius').
        nrows: Number of rows in the grid layout. Default is 8.
        ncols: Number of columns in the grid layout. Default is 12.
        top_n: Number of top values to average for correction threshold. Default is 3.
        connectivity: Neighbor pattern - 4 for orthogonal (N/S/E/W) or 8 for all adjacent. Default is 4.
        agg_func: Aggregation function for parent class. Default is 'mean'.
        num_workers: Number of parallel workers. Default is 1.
    
    Attributes:
        nrows: Grid row count.
        ncols: Grid column count.
        top_n: Number of top values for threshold calculation.
        connectivity: Neighbor connectivity pattern (4 or 8).
        measurement_col: Column to apply edge correction to.
    
    Examples:
        >>> import pandas as pd
        >>> from phenotypic.analysis import EdgeCorrector
        >>> from phenotypic.tools.constants_ import GRID
        >>> 
        >>> # Create sample data with grid info and measurements
        >>> data = pd.DataFrame({
        ...     'ImageName': ['img1'] * 96,
        ...     str(GRID.SECTION_NUM): range(96),
        ...     'Area': np.random.uniform(100, 500, 96)
        ... })
        >>> 
        >>> # Initialize corrector
        >>> corrector = EdgeCorrector(
        ...     on='Area',
        ...     groupby=['ImageName'],
        ...     measurement_col='Area',
        ...     nrows=8,
        ...     ncols=12,
        ...     top_n=10,
        ...     connectivity=4
        ... )
        >>> 
        >>> # Apply edge correction
        >>> corrected_data = corrector.analyze(data)
    """

    def __init__(
            self,
            on: str,
            groupby: list[str],
            measurement_col: str,
            nrows: int = 8,
            ncols: int = 12,
            top_n: int = 3,
            connectivity: int = 4,
            agg_func: str = 'mean',
            num_workers: int = 1
    ):
        """Initialize EdgeCorrector with grid and correction parameters.
        
        Args:
            on: Column name for grouping/aggregation operations.
            groupby: List of column names to group by.
            measurement_col: Name of measurement column to correct.
            nrows: Number of rows in grid. Default is 8.
            ncols: Number of columns in grid. Default is 12.
            top_n: Number of top values for averaging threshold. Default is 3.
            connectivity: Neighbor pattern (4 or 8). Default is 4.
            agg_func: Aggregation function. Default is 'mean'.
            num_workers: Number of workers. Default is 1.
            
        Raises:
            ValueError: If connectivity is not 4 or 8.
            ValueError: If nrows, ncols, or top_n are not positive.
        """
        super().__init__(on=on, groupby=groupby, agg_func=agg_func, num_workers=num_workers)

        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
        if nrows <= 0 or ncols <= 0:
            raise ValueError(f"nrows and ncols must be positive, got nrows={nrows}, ncols={ncols}")
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")

        self.nrows = nrows
        self.ncols = ncols
        self.top_n = top_n
        self.connectivity = connectivity
        self.on = measurement_col
        self._original_data: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def _surrounded_positions(
            active_idx: np.ndarray | list[int],
            shape: tuple[int, int],
            connectivity: int = 4,
            min_neighbors: int | None = None,
            return_counts: bool = False,
            dtype: np.dtype = np.int64,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Find grid cells that are surrounded by active neighbors.
        
        This function identifies cells in a 2D grid that have a sufficient number
        of active neighbors based on the specified connectivity pattern. Input uses
        flattened indices in C-order (row-major).
        
        Args:
            active_idx: Flattened indices of active cells. Will be deduplicated.
            shape: Grid dimensions as (rows, cols).
            connectivity: Neighbor pattern. Must be 4 (N,S,E,W) or 8 (adds diagonals).
            min_neighbors: Minimum number of active neighbors required. If None,
                requires all neighbors in the connectivity pattern to be active
                (fully surrounded). Border cells cannot qualify when None.
            return_counts: If True, also return the neighbor counts for selected indices.
            dtype: Data type for output arrays.
        
        Returns:
            If return_counts is False:
                Sorted array of flattened indices meeting the neighbor criterion.
            If return_counts is True:
                Tuple of (indices, counts) where counts[i] is the number of active
                neighbors for indices[i].
        
        Raises:
            ValueError: If connectivity is not 4 or 8, if any active_idx is out of
                bounds, if min_neighbors is invalid, or if shape is invalid.
        
        Notes:
            - Flattening uses C-order: idx = row * cols + col
            - When min_neighbors=None, border cells are geometrically excluded since
              they cannot have all neighbors active
            - Results are always sorted for deterministic output
        
        Examples:
            >>> import numpy as np
            >>> # 8×12 plate; 3×3 active block centered at (4,6)
            >>> rows, cols = 8, 12
            >>> block_rc = [(r, c) for r in range(3, 6) for c in range(5, 8)]
            >>> active = np.array([r*cols + c for r, c in block_rc], dtype=np.int64)
            >>> 
            >>> # Fully surrounded (default, since min_neighbors=None → all)
            >>> res_all = EdgeCorrector._surrounded_positions(active, (rows, cols), connectivity=4)
            >>> assert np.array_equal(res_all, np.array([4*cols + 6], dtype=np.int64))
            >>> 
            >>> # Threshold: at least 3 of 4 neighbors
            >>> idxs, counts = EdgeCorrector._surrounded_positions(
            ...     active, (rows, cols), connectivity=4, min_neighbors=3, return_counts=True
            ... )
            >>> assert (counts >= 3).all()
            >>> assert (4*cols + 6) in idxs  # center has 4
        """
        # Validate connectivity
        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

        # Validate shape
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"shape must be two positive integers, got {shape}")

        rows, cols = shape
        total_cells = rows*cols

        # Coerce active_idx to 1D unique array
        active_idx = np.asarray(active_idx, dtype=dtype).ravel()
        active_idx = np.unique(active_idx)

        # Validate bounds
        if len(active_idx) > 0:
            if active_idx.min() < 0 or active_idx.max() >= total_cells:
                raise ValueError(
                        f"All active_idx must be in [0, {total_cells}), "
                        f"got range [{active_idx.min()}, {active_idx.max()}]"
                )

        # Determine max_neighbors and validate min_neighbors
        max_neighbors = connectivity
        if min_neighbors is None:
            min_neighbors = max_neighbors
        else:
            if not (1 <= min_neighbors <= max_neighbors):
                raise ValueError(
                        f"min_neighbors must be in [1, {max_neighbors}], got {min_neighbors}"
                )

        # Handle empty input
        if len(active_idx) == 0:
            if return_counts:
                return np.array([], dtype=dtype), np.array([], dtype=dtype)
            return np.array([], dtype=dtype)

        # Build active mask
        active_mask = np.zeros((rows, cols), dtype=bool)
        rows_idx = active_idx//cols
        cols_idx = active_idx%cols
        active_mask[rows_idx, cols_idx] = True

        # Define neighbor offsets based on connectivity
        if connectivity == 4:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:  # connectivity == 8
            offsets = [
                (-1, 0), (1, 0), (0, -1), (0, 1),  # cardinal
                (-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonal
            ]

        # Accumulate neighbor counts using aligned slicing
        neighbor_count = np.zeros((rows, cols), dtype=np.int32)

        for dr, dc in offsets:
            # Calculate slice bounds for source (active_mask)
            src_r_start = max(0, -dr)
            src_r_end = rows - max(0, dr)
            src_c_start = max(0, -dc)
            src_c_end = cols - max(0, dc)

            # Calculate slice bounds for destination (neighbor_count)
            dst_r_start = max(0, dr)
            dst_r_end = rows - max(0, -dr)
            dst_c_start = max(0, dc)
            dst_c_end = cols - max(0, -dc)

            # Extract views
            src_view = active_mask[src_r_start:src_r_end, src_c_start:src_c_end]
            dst_view = neighbor_count[dst_r_start:dst_r_end, dst_c_start:dst_c_end]

            # Accumulate
            dst_view += src_view.astype(np.int32)

        # Select cells that are active AND have sufficient neighbors
        sufficient_neighbors = neighbor_count >= min_neighbors
        selected_mask = active_mask & sufficient_neighbors

        # Convert back to flattened indices
        selected_rows, selected_cols = np.where(selected_mask)
        result_idx = (selected_rows*cols + selected_cols).astype(dtype)
        result_idx = np.sort(result_idx)

        if return_counts:
            # Get counts for selected indices
            counts = neighbor_count[selected_rows, selected_cols].astype(dtype)
            # Sort counts to match sorted indices
            sort_order = np.argsort(selected_rows*cols + selected_cols)
            counts = counts[sort_order]
            return result_idx, counts

        return result_idx

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze and apply edge correction to grid-based colony measurements.
        
        This method processes the input DataFrame by grouping according to specified
        columns and applying edge correction to each group independently. Edge colonies
        (those missing orthogonal neighbors) have their measurements capped to prevent
        artificially inflated values.
        
        Args:
            data: DataFrame containing grid section numbers (GRID.SECTION_NUM) and
                measurement data. Must include all columns specified in self.groupby
                and self.on.
        
        Returns:
            DataFrame with corrected measurement values. Original structure is preserved
            with only the measurement column modified for edge-affected rows.
        
        Raises:
            KeyError: If required columns are missing from input DataFrame.
            ValueError: If data is empty or malformed.
        
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from phenotypic.analysis import EdgeCorrector
            >>> from phenotypic.tools.constants_ import GRID
            >>> 
            >>> # Create sample grid data with measurements
            >>> np.random.seed(42)
            >>> data = pd.DataFrame({
            ...     'ImageName': ['img1'] * 96,
            ...     GRID.SECTION_NUM: range(96),
            ...     'Area': np.random.uniform(100, 500, 96)
            ... })
            >>> 
            >>> # Apply edge correction
            >>> corrector = EdgeCorrector(
            ...     on='Area',
            ...     groupby=['ImageName'],
            ...     measurement_col='Area',
            ...     nrows=8,
            ...     ncols=12,
            ...     top_n=10
            ... )
            >>> corrected = corrector.analyze(data)
            >>> 
            >>> # Check results
            >>> results = corrector.results()
        
        Notes:
            - Stores original data in self._original_data for comparison
            - Stores corrected data in self._latest_measurements for retrieval
            - Groups are processed independently with their own thresholds
        """
        from phenotypic.tools.constants_ import GRID

        # Validate input
        if data is None or len(data) == 0:
            raise ValueError("Input data cannot be empty")

        # Store original data for comparison
        self._original_data = data

        # Check required columns
        section_col = str(GRID.SECTION_NUM)
        required_cols = set(self.groupby + [section_col, self.on])
        missing_cols = required_cols - set(data.columns)

        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        # Prepare configuration for _apply2group_func
        config = {
            'nrows'          : self.nrows,
            'ncols'          : self.ncols,
            'top_n'          : self.top_n,
            'connectivity'   : self.connectivity,
            'measurement_col': self.on,
            'section_col'    : section_col
        }

        # Apply correction to each group
        if len(self.groupby) > 0:
            # Suppress FutureWarning by explicitly including groups
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning,
                                        message='.*grouping columns.*')
                corrected_data = data.groupby(
                        self.groupby,
                        group_keys=False,
                        observed=True
                ).apply(
                        lambda group: self._apply2group_func(group, config)
                )
        else:
            # No grouping - apply to entire dataset
            corrected_data = self._apply2group_func(data, config)

        # Store results
        self._latest_measurements = corrected_data.reset_index(drop=True)

        return self._latest_measurements

    def show(self, figsize: tuple[int, int] = (14, 5)):
        """Display visualization of edge correction results.
        
        Creates a multi-panel figure showing:
        1. Histogram comparing original vs corrected value distributions
        2. Scatter plot highlighting which colonies were corrected
        3. Summary statistics
        
        Args:
            figsize: Figure size as (width, height) in inches. Default is (14, 5).
        
        Raises:
            RuntimeError: If analyze() has not been called yet.
        
        Examples:
            >>> corrector = EdgeCorrector(
            ...     on='Size_Area',
            ...     groupby=['ImageName'],
            ...     measurement_col='Size_Area'
            ... )
            >>> corrector.analyze(data)
            >>> corrector.show()  # Display comparison plots
        
        Notes:
            - Requires analyze() to be called first to populate data
            - Shows original vs corrected distributions
            - Highlights corrected data points in red
            - Displays summary statistics in the plot
        """
        import matplotlib.pyplot as plt
        from phenotypic.tools.constants_ import GRID

        # Check if analyze has been called
        if self._latest_measurements.empty or self._original_data.empty:
            raise RuntimeError("No data to display. Call analyze() first.")

        # Get measurement column
        meas_col = self.on

        # Identify which rows were corrected
        original_values = self._original_data[meas_col]
        corrected_values = self._latest_measurements[meas_col]
        was_corrected = ~np.isclose(original_values.values, corrected_values.values)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: Histogram comparison
        ax1 = axes[0]
        ax1.hist(original_values, bins=30, alpha=0.6, label='Original', color='blue', edgecolor='black')
        ax1.hist(corrected_values, bins=30, alpha=0.6, label='Corrected', color='green', edgecolor='black')
        ax1.set_xlabel(meas_col)
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution: {meas_col}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Scatter plot with corrections highlighted
        ax2 = axes[1]

        # Plot all points
        ax2.scatter(
                self._original_data.index[~was_corrected],
                original_values[~was_corrected],
                alpha=0.5,
                s=30,
                color='blue',
                label='Unchanged'
        )

        # Highlight corrected points
        if was_corrected.any():
            ax2.scatter(
                    self._original_data.index[was_corrected],
                    original_values[was_corrected],
                    alpha=0.7,
                    s=50,
                    color='red',
                    marker='x',
                    label='Original (corrected)'
            )
            ax2.scatter(
                    self._original_data.index[was_corrected],
                    corrected_values[was_corrected],
                    alpha=0.7,
                    s=30,
                    color='green',
                    marker='o',
                    label='Corrected value'
            )

        ax2.set_xlabel('Row Index')
        ax2.set_ylabel(meas_col)
        ax2.set_title('Edge Correction Applied')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add summary statistics as text
        n_corrected = was_corrected.sum()
        n_total = len(original_values)
        pct_corrected = 100*n_corrected/n_total if n_total > 0 else 0

        summary_text = (
            f"Corrected: {n_corrected}/{n_total} ({pct_corrected:.1f}%)\n"
            f"Original mean: {original_values.mean():.2f}\n"
            f"Corrected mean: {corrected_values.mean():.2f}"
        )

        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show()

    def results(self) -> pd.DataFrame:
        """Return the corrected measurement DataFrame.
        
        Returns the DataFrame with edge-corrected measurements from the most recent
        call to analyze(). This allows retrieval of results after processing.
        
        Returns:
            DataFrame with corrected measurements. If analyze() has not been called,
            returns an empty DataFrame.
        
        Examples:
            >>> corrector = EdgeCorrector(
            ...     on='Area',
            ...     groupby=['ImageName'],
            ...     measurement_col='Area'
            ... )
            >>> corrected = corrector.analyze(data)
            >>> results = corrector.results()  # Same as corrected
            >>> assert results.equals(corrected)
        
        Notes:
            - Returns the DataFrame stored in self._latest_measurements
            - Contains the same structure as input but with corrected values
            - Use this method to retrieve results after calling analyze()
        """
        return self._latest_measurements

    @staticmethod
    def _apply2group_func(group: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply edge correction to a single group of data.
        
        This method identifies grid sections with missing orthogonal neighbors (edge sections),
        calculates a correction threshold from the top N values, and caps measurements that
        exceed this threshold.
        
        Args:
            group: DataFrame containing grid and measurement data for a single group.
            config: Configuration dictionary containing:
                - 'nrows': Number of grid rows
                - 'ncols': Number of grid columns  
                - 'top_n': Number of top values for threshold calculation
                - 'connectivity': Neighbor pattern (4 or 8)
                - 'measurement_col': Column name to correct
                - 'section_col': Column name containing section numbers
        
        Returns:
            DataFrame with corrected measurement values for edge sections.
        
        Notes:
            - Only sections at edges (missing neighbors) are corrected
            - Only values exceeding the threshold are capped
            - Values below threshold and interior sections remain unchanged
            - Handles cases with fewer than top_n values gracefully
        
        Examples:
            >>> config = {
            ...     'nrows': 8, 'ncols': 12, 'top_n': 10,
            ...     'connectivity': 4, 'measurement_col': 'Area',
            ...     'section_col': 'SectionNum'
            ... }
            >>> corrected_group = EdgeCorrector._apply2group_func(group, config)
        """
        from phenotypic.tools.constants_ import GRID

        # Extract configuration
        nrows = config['nrows']
        ncols = config['ncols']
        top_n = config['top_n']
        connectivity = config['connectivity']
        measurement_col = config['measurement_col']
        section_col = config.get('section_col', str(GRID.SECTION_NUM))

        # Make a copy to avoid modifying the original
        group = group.copy()

        # Handle empty groups
        if len(group) == 0:
            return group

        # Get unique section numbers present in the data
        present_sections = group[section_col].dropna().unique()

        # Handle case where no sections are present
        if len(present_sections) == 0:
            return group

        # Convert section numbers to 0-indexed flattened indices
        # Assuming section numbers are 0-indexed already (row * ncols + col)
        active_indices = present_sections.astype(int)

        # Find fully-surrounded (interior) sections
        try:
            surrounded_indices = EdgeCorrector._surrounded_positions(
                    active_idx=active_indices,
                    shape=(nrows, ncols),
                    connectivity=connectivity,
                    min_neighbors=None,  # Require all neighbors (fully surrounded)
                    return_counts=False
            )
        except ValueError:
            # If validation fails, return group unchanged
            return group

        # Identify edge sections (all sections - surrounded sections)
        surrounded_set = set(surrounded_indices)
        edge_sections = [sec for sec in present_sections if sec not in surrounded_set]

        # If no edge sections, return unchanged
        if len(edge_sections) == 0:
            return group

        # Calculate threshold from top N values across entire group
        if measurement_col not in group.columns:
            return group

        # Use actual number of values if fewer than top_n available
        actual_top_n = min(top_n, len(group))

        if actual_top_n == 0:
            return group

        # Get top N values and calculate threshold
        top_values = group.nlargest(actual_top_n, measurement_col)[measurement_col]
        threshold = top_values.mean()

        # Apply correction: cap edge values that exceed threshold
        edge_mask = group[section_col].isin(edge_sections)
        exceeds_threshold = group[measurement_col] > threshold
        correction_mask = edge_mask & exceeds_threshold

        # Apply the correction
        group.loc[correction_mask, measurement_col] = threshold

        return group
