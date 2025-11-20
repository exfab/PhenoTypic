from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from scipy.stats import permutation_test
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from phenotypic.tools.constants_ import MeasurementInfo
from .abc_ import SetAnalyzer


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
            'time_label'  : self.time_label,
        }

        # Build aggregation dictionary to preserve all columns
        groupby_cols = self.groupby + [section_col]
        if self.time_label in data:
            groupby_cols = groupby_cols + [self.time_label]
        
        # Determine which columns to aggregate
        agg_dict = {}
        for col in data.columns:
            if col not in groupby_cols:
                # Use specified agg_func for measurement column, 'first' for others
                if col == self.on:
                    agg_dict[col] = self.agg_func
                else:
                    agg_dict[col] = 'first'
        
        agg_data = data.groupby(by=groupby_cols, as_index=False).agg(agg_dict)

        # Handle empty groupby case
        if len(self.groupby) == 0:
            # Process entire dataset as single group
            corrected_data = [self.__class__._apply2group_func(agg_data, **config)]
        else:
            grouped = agg_data.groupby(by=self.groupby, as_index=False)
            corrected_data = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.__class__._apply2group_func)(
                            group, **config
                    ) for _, group in grouped
            )

        # Store results
        if corrected_data:
            self._latest_measurements = pd.concat(corrected_data, ignore_index=True)
        else:
            self._latest_measurements = pd.DataFrame()

        return self._latest_measurements

    def show(self,
             figsize: tuple[int, int] | None = None,
             max_groups: int = 20,
             collapsed: bool = True,
             criteria: dict[str, any] | None = None
             ) -> tuple[Figure, plt.Axes]:
        """Visualize edge correction results.

        Displays the distribution of measurements for the last time point, highlighting
        surrounded vs. edge colonies and the calculated correction threshold.

        Args:
            figsize: Figure size (width, height).
            max_groups: Maximum number of groups to display.
            collapsed: If True, show groups stacked vertically.
            criteria: Filtering criteria.

        Returns:
            Tuple of (Figure, Axes).
        """
        if self._original_data.empty:
            raise RuntimeError("No results to display. Call analyze() first.")

        data = self._original_data.copy()

        if criteria is not None:
            data = self._filter_by(df=data, criteria=criteria, copy=False)
            if data.empty:
                raise ValueError("No data matches the specified criteria")

        # Determine groups
        if len(self.groupby) == 1:
            groups = data[self.groupby[0]].unique()
            group_col = self.groupby[0]
        else:
            data['_group_key'] = data[self.groupby].astype(str).agg(' | '.join, axis=1)
            groups = data['_group_key'].unique()
            group_col = '_group_key'

        if len(groups) > max_groups:
            print(f"Warning: Displaying first {max_groups} groups out of {len(groups)}")
            groups = groups[:max_groups]

        if collapsed:
            return self._show_collapsed(data, groups, group_col, figsize)
        else:
            return self._show_individual(data, groups, group_col, figsize)

    def _show_collapsed(self, data: pd.DataFrame, groups, group_col: str,
                        figsize: tuple[int, int] | None) -> tuple[Figure, plt.Axes]:
        n_groups = len(groups)
        if figsize is None:
            figsize = (10, max(6, 0.5*n_groups + 2))

        fig, ax = plt.subplots(figsize=figsize)

        added_labels = set()

        for idx, group_name in enumerate(groups):
            y_pos = n_groups - idx
            group_data = data[data[group_col] == group_name]

            stats = self._calculate_group_stats(group_data)
            if stats is None:
                continue

            lt_df = stats['last_time_df']
            threshold = stats['threshold']
            surrounded_mask = stats['surrounded_mask']
            edge_mask = stats['edge_mask']

            # Range line
            vals = lt_df[self.on].values
            if len(vals) > 0:
                ax.hlines(y_pos, vals.min(), vals.max(), colors='lightgray', lw=1.5, zorder=1)

            # Threshold
            if not np.isinf(threshold):
                lbl = 'Threshold'
                if lbl not in added_labels:
                    added_labels.add(lbl)
                else:
                    lbl = None
                ax.plot([threshold, threshold], [y_pos - 0.2, y_pos + 0.2],
                        color='#F4A261', lw=2.5, label=lbl, zorder=2)

            # Jitter
            y_jitter = np.random.normal(y_pos, 0.05, len(lt_df))

            is_clipped = lt_df[self.on] > threshold

            # Helper for scatter plots
            def add_scatter(mask, color, marker, label_key):
                if mask.any():
                    lbl = label_key
                    if lbl not in added_labels:
                        added_labels.add(lbl)
                    else:
                        lbl = None
                    ax.scatter(lt_df.loc[mask, self.on], y_jitter[mask],
                               c=color, marker=marker, s=30 if marker == 'o' else 40,
                               alpha=0.6 if marker == 'o' else 0.8,
                               label=lbl, zorder=3)

            # Inner Pass
            add_scatter(surrounded_mask & (~is_clipped), '#2E86AB', 'o', 'Inner (Pass)')
            # Inner Clipped
            add_scatter(surrounded_mask & is_clipped, '#2E86AB', 'x', 'Inner (Clipped)')
            # Edge Pass
            add_scatter(edge_mask & (~is_clipped), '#E63946', 'o', 'Edge (Pass)')
            # Edge Clipped
            add_scatter(edge_mask & is_clipped, '#E63946', 'x', 'Edge (Clipped)')

            # Means
            inner_vals = lt_df.loc[surrounded_mask, self.on]
            edge_vals = lt_df.loc[edge_mask, self.on]

            if len(inner_vals) > 0:
                lbl = 'Inner Mean'
                if lbl not in added_labels:
                    added_labels.add(lbl)
                else:
                    lbl = None
                mean_val = inner_vals.mean()
                ax.plot([mean_val, mean_val], [y_pos - 0.25, y_pos + 0.25],
                        color='#2E86AB', linewidth=2.5, label=lbl, zorder=4,
                        linestyle="--")

            if len(edge_vals) > 0:
                lbl = 'Edge Mean'
                if lbl not in added_labels:
                    added_labels.add(lbl)
                else:
                    lbl = None
                mean_val = edge_vals.mean()
                ax.plot([mean_val, mean_val], [y_pos - 0.25, y_pos + 0.25],
                        color='#E63946', linewidth=2.5, label=lbl, zorder=4,
                        linestyle="--")

            # P-value
            if self.pvalue != 0 and len(inner_vals) > 0 and len(edge_vals) > 0:
                pval = self._perm_test(inner_vals, edge_vals)
                mean_inner = inner_vals.mean()
                mean_edge = edge_vals.mean()

                # Bracket parameters
                bracket_y = y_pos + 0.3
                bracket_h = 0.05

                # Draw bracket
                ax.plot([mean_inner, mean_inner, mean_edge, mean_edge],
                        [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                        color='black', linewidth=1, zorder=5)

                # Add p-value text
                mid_x = (mean_inner + mean_edge)/2
                ax.text(mid_x, bracket_y + bracket_h + 0.05, f"p={pval:.3f}",
                        ha='center', va='bottom', fontsize=8)

        ax.set_yticks(range(1, n_groups + 1))
        ax.set_yticklabels(groups[::-1])
        ax.set_xlabel(self.on)
        ax.set_title(f"Edge Correction (Top N={self.top_n}, p={self.pvalue})")
        ax.legend(loc='best')
        plt.tight_layout()
        return fig, ax

    def _show_individual(self, data: pd.DataFrame, groups, group_col: str,
                         figsize: tuple[int, int] | None) -> tuple[Figure, plt.Axes]:
        n_groups = len(groups)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1)//n_cols

        if figsize is None:
            figsize = (5*n_cols, 4*n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, group_name in enumerate(groups):
            ax = axes[idx]
            group_data = data[data[group_col] == group_name]

            stats = self._calculate_group_stats(group_data)
            if stats is None:
                ax.text(0.5, 0.5, "Insufficient Data", ha='center')
                continue

            lt_df = stats['last_time_df']
            threshold = stats['threshold']
            surrounded_mask = stats['surrounded_mask']
            edge_mask = stats['edge_mask']

            vals = lt_df[self.on].values
            is_clipped = lt_df[self.on] > threshold

            ax.boxplot([vals], positions=[1], widths=0.3,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor='lightgray', alpha=0.3))

            x_jitter = np.random.normal(1, 0.04, len(lt_df))

            # Inner Pass
            mask_ip = surrounded_mask & (~is_clipped)
            if mask_ip.any():
                ax.scatter(x_jitter[mask_ip], lt_df.loc[mask_ip, self.on],
                           c='#2E86AB', marker='o', s=30, alpha=0.6, label='Inner (Pass)')
            # Inner Clipped
            mask_ic = surrounded_mask & is_clipped
            if mask_ic.any():
                ax.scatter(x_jitter[mask_ic], lt_df.loc[mask_ic, self.on],
                           c='#2E86AB', marker='x', s=40, alpha=0.8, label='Inner (Clipped)')
            # Edge Pass
            mask_ep = edge_mask & (~is_clipped)
            if mask_ep.any():
                ax.scatter(x_jitter[mask_ep], lt_df.loc[mask_ep, self.on],
                           c='#E63946', marker='o', s=30, alpha=0.6, label='Edge (Pass)')
            # Edge Clipped
            mask_ec = edge_mask & is_clipped
            if mask_ec.any():
                ax.scatter(x_jitter[mask_ec], lt_df.loc[mask_ec, self.on],
                           c='#E63946', marker='x', s=40, alpha=0.8, label='Edge (Clipped)')

            if not np.isinf(threshold):
                ax.axhline(y=threshold, color='#F4A261', linestyle='--', label='Threshold')

            ax.set_title(group_name)
            ax.set_ylabel(self.on)
            ax.set_xticks([])

            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=8)

        for idx in range(n_groups, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def _calculate_group_stats(self, group: pd.DataFrame):
        from phenotypic.tools.constants_ import GRID

        if len(group) == 0:
            return None

        tmax = group[self.time_label].max()
        last_time_group = group[group[self.time_label] == tmax].copy()

        present_sections = last_time_group[GRID.SECTION_NUM].dropna().unique()
        if len(present_sections) == 0:
            return None

        active_indices = present_sections.astype(int)

        try:
            surrounded_idx = self._surrounded_positions(
                    active_idx=active_indices,
                    shape=(self.nrows, self.ncols),
                    connectivity=self.connectivity,
                    min_neighbors=None,
                    return_counts=False
            )
        except ValueError:
            return None

        surrounded_idx_set = set(surrounded_idx)

        if len(surrounded_idx_set) == 0:
            return {
                'last_time_df'   : last_time_group,
                'threshold'      : np.inf,
                'surrounded_mask': pd.Series(False, index=last_time_group.index),
                'edge_mask'      : pd.Series(True, index=last_time_group.index)
            }

        surrounded_mask = last_time_group[GRID.SECTION_NUM].isin(surrounded_idx_set)
        edge_mask = ~surrounded_mask & last_time_group[GRID.SECTION_NUM].isin(present_sections)

        if self.on not in group.columns:
            return None

        last_inner_values = last_time_group.loc[surrounded_mask, self.on]
        threshold = np.inf
        should_correct = True

        if self.pvalue != 0:
            last_edge_values = last_time_group.loc[edge_mask, self.on]
            if len(last_edge_values) > 0 and len(last_inner_values) > 0:
                perm_results = permutation_test(
                        data=(last_inner_values, last_edge_values),
                        statistic=lambda x, y: np.mean(x) - np.mean(y),
                        permutation_type="independent",
                        n_resamples=1000,
                        alternative="two-sided",
                )
                if perm_results.pvalue > self.pvalue:
                    should_correct = False

        if should_correct:
            actual_top_n = min(self.top_n, len(last_inner_values))
            if actual_top_n > 0:
                top_values = last_inner_values.nlargest(actual_top_n)
                threshold = top_values.mean()

        return {
            'last_time_df'   : last_time_group,
            'threshold'      : threshold,
            'surrounded_mask': surrounded_mask,
            'edge_mask'      : edge_mask
        }

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
            ...     groupby=['ImageName']
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
        if time_label in group.columns:
            tmax = group.loc[:, time_label].max()

            last_time_group = group.loc[group.loc[:, time_label] == tmax, :]
        else:
            last_time_group = group

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

        # Calculate threshold from top N inner values
        # ===========================================
        if on not in group.columns:
            return group

        last_inner_values: pd.Series = \
            last_time_group.loc[last_time_group.loc[:, GRID.SECTION_NUM].isin(surrounded_idx), on]

        # TODO: Add permutation test for statistical significance before correction.
        if pvalue != 0:
            last_edge_values: pd.Series = \
                last_time_group.loc[last_time_group.loc[:, GRID.SECTION_NUM].isin(edge_idx), on]

            # If difference is not statistically significant, don't apply correction
            if EdgeCorrector._perm_test(last_inner_values, last_edge_values) > pvalue:
                return group

        # Use actual number of values if fewer than top_n available
        actual_top_n = min(top_n, len(last_inner_values))

        if actual_top_n == 0:  # If no inner colonies
            return group

        # Get top N values and calculate threshold
        top_values = last_inner_values.nlargest(actual_top_n)
        threshold = top_values.mean()

        # Apply correction: cap ALL values that exceed for fairness
        group.loc[:, on] = np.clip(group.loc[:, on], a_min=0, a_max=threshold)
        return group

    @staticmethod
    def _perm_test(surrounded, edge):
        return permutation_test(
                data=(surrounded, edge),
                statistic=lambda x, y: np.mean(x) - np.mean(y),
                permutation_type="independent",
                n_resamples=1000,
                alternative="two-sided",
        ).pvalue


EdgeCorrector.__doc__ = EDGE_CORRECTION.append_rst_to_doc(EdgeCorrector.__doc__)
