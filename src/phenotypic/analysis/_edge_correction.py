from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .abc_ import SetAnalyzer
from phenotypic.tools.constants_ import MeasurementInfo


class EDGE_CORRECTION(MeasurementInfo):
    @classmethod
    def category(cls) -> str:
        return "EdgeCorrection"

    CORRECTED_CAP = "CorrectedCap", "The carrying capacity for the target measurement"


class EdgeCorrector(SetAnalyzer):
    """Analyzer for detecting and correcting edge effects in colony detection.
    
    This class identifies colonies at grid edges (missing orthogonal neighbors) and
    caps their measurement values to prevent edge effects in growth assays. Edge
    colonies often show artificially inflated measurements due to lack of competition
    for resources.

    """

    def __init__(
            self,
            on: str,
            groupby: list[str],
            time_label: str = "Metadata_Time",
            nrows: int = 8,
            ncols: int = 12,
            top_n: int = 3,
            pvalue: float = 0.05,
            connectivity: int = 4,
            agg_func: str = 'mean',
            num_workers: int = 1
    ):
        """
        Initializes the class with specified parameters to configure the state of the object.
        The class is aimed at processing and analyzing connectivity data with multiple grouping
        and aggregation options, while ensuring input validation.

        Args:
            on (str): The dataset column to analyze or process.
            groupby (list[str]): List of column names for grouping the data.
            time_label (str): Specific time reference column, defaulting to "Metadata_Time".
            nrows (int): Number of rows in the dataset, must be positive.
            ncols (int): Number of columns in the dataset, must be positive.
            top_n (int): Number of top results to analyze. Must be a positive integer.
            pvalue (float): Statistical threshold for significance testing between the surrounded and edge colonies.
                defaults to 0.05. Set to 0.0 to apply to all plates.
            connectivity (int): The connectivity mode to use. Must be either 4 or 8.
            agg_func (str): Aggregation function to apply, defaulting to 'mean'.
            num_workers (int): Number of workers for parallel processing.

        Raises:
            ValueError: If `connectivity` is not 4 or 8.
            ValueError: If `nrows` or `ncols` are not positive integers.
            ValueError: If `top_n` is not a positive integer.
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
        self.time_label = time_label
        self.pvalue = pvalue

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
            'nrows'       : self.nrows,
            'ncols'       : self.ncols,
            'top_n'       : self.top_n,
            'connectivity': self.connectivity,
            'on'          : self.on,
            'pvalue'      : self.pvalue,
        }

        agg_data = data.groupby(by=self.grouby + [self.time_label], as_index=False).agg(
                {self.on: self.agg_func}
        )

        grouped = agg_data.groupby(by=self.groupby, as_index=False)
        corrected_data = Parallel(n_jobs=self.n_jobs)(
                delayed(self.__class__._apply2group_func)(
                        group, **config
                ) for _, group in grouped
        )

        # Store results
        self._latest_measurements = corrected_data.reset_index(drop=True)

        return self._latest_measurements

    def show(self):
        raise NotImplementedError()

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
    def _apply2group_func(group: pd.DataFrame,
                          on: str,
                          nrows: int,
                          ncols: int,
                          top_n: int,
                          time_label: str,
                          connectivity: int,
                          pvalue: float
                          ) -> pd.DataFrame:
        """
        Note:
            - assumes "Grid_SectionNum" from a `GridFinder` is in the dataframe groups
            = applies permutation test on the last time-point to see if theres a statistically significant difference
            - caps clips all the data to the last time point
        """
        from phenotypic.tools.constants_ import GRID

        section_col = GRID.SECTION_NUM

        # Handle empty groups
        if len(group) == 0:
            return group

        # Make a copy to avoid modifying the original
        group: pd.DataFrame = group.copy()
        tmax = group.loc[:, time_label].max()

        last_time_group = group.loc[group.loc[:, time_label] == tmax, :]

        # Get unique section numbers present in the data
        present_sections = last_time_group.loc[:, section_col].dropna().unique()

        # Handle case where no sections are present
        if len(present_sections) == 0:
            return group

        # Convert section numbers to 0-indexed flattened indices
        # Assuming section numbers are 0-indexed already (row * ncols + col)
        active_indices = present_sections.astype(int)

        # Find fully-surrounded (interior) sections
        try:
            surrounded_idx = EdgeCorrector._surrounded_positions(
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
        surrounded_idx = set(surrounded_idx)
        edge_idx = [sec for sec in present_sections if sec not in surrounded_idx]

        # If no inner sections, return unchanged
        if len(surrounded_idx) == 0:
            return group

        # TODO: Add permutation test for statistical significance before correction.

        # Calculate threshold from top N inner values
        # ===========================================
        if on not in group.columns:
            return group

        last_inner_values: pd.Series = \
            last_time_group.loc[last_time_group.loc[:, GRID.SECTION_NUM].isin(surrounded_idx), on]

        # Use actual number of values if fewer than top_n available
        actual_top_n = min(top_n, len(last_inner_values))

        if actual_top_n == 0:  # If no inner colonies
            return group

        # Get top N values and calculate threshold
        top_values = last_inner_values.nlargest(actual_top_n)
        threshold = top_values.mean()

        # Apply correction: cap edge values that exceed threshold
        edge_mask = group.loc[:, GRID.SECTION_NUM].isin(edge_idx)
        group.loc[edge_mask, on] = np.clip(group.loc[edge_mask, on], a_min=0, a_max=threshold)
        return group

    @staticmethod
    def _permutation_test(
            group1: np.ndarray,
            group2: np.ndarray,
            n_permutations: int = 10000,
            statistic: Literal["mean", "median"] = "mean",
            alternative: Literal["two-sided", "less", "greater"] = "two-sided",
            random_seed: int = None
    ) -> Tuple[float, float, np.ndarray]:
        """Perform a permutation test to compare two independent groups.

        This test randomly shuffles group labels to generate a null distribution
        of the test statistic, then compares the observed statistic to this
        distribution to calculate a p-value.

        Args:
            group1: Array of measurements from group 1 (e.g., edge colonies).
            group2: Array of measurements from group 2 (e.g., inner colonies).
            n_permutations: Number of random permutations to perform.
            statistic: Test statistic to use. Options are:
                - "mean": Difference in means (group1 - group2)
                - "median": Difference in medians (group1 - group2)
            alternative: Type of alternative hypothesis:
                - "two-sided": Test if groups differ in either direction
                - "less": Test if group1 < group2
                - "greater": Test if group1 > group2
            random_seed: Seed for random number generator for reproducibility.

        Returns:
            p_value: The permutation test p-value.
            observed_stat: The observed test statistic.
            null_distribution: Array of test statistics from permutations.

        Raises:
            ValueError: If groups are empty or statistic/alternative are invalid.

        Examples:
            >>> edge_sizes = np.array([2.3, 2.5, 2.1, 2.8])
            >>> inner_sizes = np.array([3.1, 3.3, 3.0, 3.2, 2.9])
            >>> p_val, obs_stat, null_dist = permutation_test(
            ...     edge_sizes, inner_sizes, n_permutations=10000
            ... )
            >>> print(f"p-value: {p_val:.4f}, observed difference: {obs_stat:.3f}")
        """
        if len(group1) == 0 or len(group2) == 0:
            raise ValueError("Both groups must contain at least one observation")

        if statistic not in ["mean", "median"]:
            raise ValueError("statistic must be 'mean' or 'median'")

        if alternative not in ["two-sided", "less", "greater"]:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)

        # Convert to numpy arrays
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        # Choose test statistic function
        if statistic == "mean":
            stat_func = np.mean
        else:  # median
            stat_func = np.median

        # Calculate observed test statistic
        observed_stat = stat_func(group1) - stat_func(group2)

        # Pool all observations
        pooled = np.concatenate([group1, group2])
        n1 = len(group1)
        n_total = len(pooled)

        # Generate null distribution through permutation
        null_distribution = np.zeros(n_permutations)

        for i in range(n_permutations):
            # Randomly shuffle the pooled data
            np.random.shuffle(pooled)

            # Split into two groups with original sizes
            perm_group1 = pooled[:n1]
            perm_group2 = pooled[n1:]

            # Calculate test statistic for this permutation
            null_distribution[i] = stat_func(perm_group1) - stat_func(perm_group2)

        # Calculate p-value based on alternative hypothesis
        if alternative == "two-sided":
            # Count permutations as or more extreme in either direction
            p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_stat))
        elif alternative == "less":
            # Count permutations as small or smaller
            p_value = np.mean(null_distribution <= observed_stat)
        else:  # greater
            # Count permutations as large or larger
            p_value = np.mean(null_distribution >= observed_stat)

        return p_value, observed_stat, null_distribution


EdgeCorrector.__doc__ = EDGE_CORRECTION.append_rst_to_doc(EdgeCorrector.__doc__)
