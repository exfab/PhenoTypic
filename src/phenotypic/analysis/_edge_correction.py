from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
from scipy.stats import permutation_test
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from phenotypic.tools.constants_ import MeasurementInfo
from .abc_ import SetAnalyzer


class EDGE_CORRECTION(MeasurementInfo):
    """Measurement info container for edge correction analysis results.

    Provides metadata for measurement values produced by the EdgeCorrector,
    organizing corrected colony measurements under the "EdgeCorrection" category.
    """

    @classmethod
    def category(cls) -> str:
        return "EdgeCorrection"

    CORRECTED_CAP = "Cap", "The carrying capacity for the target measurement"
    NEW_VAL = "NewVal", "The new value of the target measurement"


class EdgeCorrector(SetAnalyzer):
    """Analyzer for detecting and correcting edge effects in arrayed colony growth.

    This class identifies colonies at grid edges (missing orthogonal neighbors) and
    caps their measurement values to prevent edge effects in high-throughput phenotyping
    assays. Edge colonies often show artificially inflated measurements (larger areas,
    higher color intensity) due to lack of competition for resources from missing
    neighbors. The corrector uses permutation testing to determine if edge and interior
    colonies are statistically different before applying correction.

    **Intuition:** In plate-based assays (96-well, 384-well), colonies at grid edges
    experience fundamentally different growth conditions: they lack orthogonal neighbors
    that would otherwise compete for nutrients and space. This causes edge colonies to
    appear larger/brighter than interior colonies under identical conditions, biasing
    downstream analyses. EdgeCorrector detects this asymmetry and caps measurements to
    a threshold derived from top interior colonies, preventing this systematic bias.

    **Use cases:**
        - High-throughput phenotyping on standard plate layouts (8x12, 16x24, etc.)
        - Growth assays where colony size/intensity is a fitness proxy
        - Comparing genotypes across plates with multiple replicates per condition
        - Any analysis where spatial position should not correlate with phenotype

    **Caveats:**
        - Requires multiple interior colonies to establish a reliable threshold
        - Edge correction assumes interior and edge colonies *should* have similar
          distributions; this may not hold in some experimental designs
        - If too many wells are empty or dead, surrounded position detection may fail
        - Permutation testing requires adequate sample sizes for statistical power
        - All measurements (not just edge colonies) are capped when correction is applied

    Attributes:
        nrows (int): Number of rows in the grid layout.
        ncols (int): Number of columns in the grid layout.
        top_n (int): Number of top-valued interior colonies to use for threshold calculation.
        connectivity (int): Neighbor pattern: 4 (orthogonal) or 8 (with diagonals).
        time_label (str): Column name containing time point information.
        pvalue (float): P-value threshold for permutation test (0.0 disables test).
        on (str): Name of measurement column to analyze and correct.
        groupby (list[str]): Column names for grouping data by experiment/plate/condition.
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
            agg_func: str = "mean",
            num_workers: int = 1,
    ):
        """Initialize EdgeCorrector with grid layout and correction parameters.

        Args:
            on (str): Name of the measurement column to analyze (e.g., 'Area', 'Intensity').
                Corrected values will be placed in a new column EDGE_CORRECTION.NEW_VAL-{on}.
                Original column is preserved unchanged.
            groupby (list[str]): Column names for grouping data independently (e.g.,
                ['ImageName', 'Condition']). Each group gets its own threshold calculation.
            time_label (str, optional): Column name containing time point information.
                Defaults to "Metadata_Time". The maximum time point per group is used
                to identify interior vs. edge colonies.
            nrows (int, optional): Number of rows in the grid layout. Defaults to 8
                (standard 96-well plate). Must be positive. Affects edge detection logic.
            ncols (int, optional): Number of columns in the grid layout. Defaults to 12
                (standard 96-well plate). Must be positive. Affects edge detection logic.
            top_n (int, optional): Number of top-valued interior colonies to use for
                threshold calculation. Defaults to 3. The threshold is the mean of the
                top_n interior colonies; larger values give more stable thresholds but
                may miss subtle edge effects.
            pvalue (float, optional): P-value threshold for permutation test comparing
                interior vs. edge distributions. Defaults to 0.05. Set to 0.0 to disable
                statistical testing and apply correction to all groups. Values are passed
                to scipy.stats.permutation_test with 1000 resamples.
            connectivity (int, optional): Neighbor pattern for interior cell detection.
                Defaults to 4 (orthogonal: North, South, East, West). Set to 8 to include
                diagonal neighbors. Affects how strictly "surrounded" is defined.
            agg_func (str, optional): Aggregation function for multiple measurements per
                section (well). Defaults to "mean". See pandas.DataFrame.agg for options.
            num_workers (int, optional): Number of parallel workers for group processing.
                Defaults to 1 (serial). Use -1 for all CPU cores via joblib.Parallel.

        Raises:
            ValueError: If connectivity is not 4 or 8.
            ValueError: If nrows or ncols are not positive integers.
            ValueError: If top_n is not a positive integer.

        Examples:
            .. dropdown:: Basic initialization with 96-well plate defaults

                >>> from phenotypic.analysis import EdgeCorrector
                >>> corrector = EdgeCorrector(
                ...     on='Area',
                ...     groupby=['ImageName'],
                ...     top_n=3,
                ...     pvalue=0.05
                ... )
                >>> # nrows=8, ncols=12 are defaults for 96-well format

            .. dropdown:: Custom grid layout (384-well format, 16x24)

                >>> corrector = EdgeCorrector(
                ...     on='ColonyIntensity',
                ...     groupby=['Plate', 'Condition'],
                ...     nrows=16,
                ...     ncols=24,
                ...     top_n=3,
                ...     connectivity=8,  # Include diagonal neighbors
                ...     num_workers=4
                ... )

            .. dropdown:: Aggressive correction (no statistical test)

                >>> corrector = EdgeCorrector(
                ...     on='Area',
                ...     groupby=['ImageName'],
                ...     pvalue=0.0,  # Apply to all groups regardless of stats
                ...     top_n=1  # Use single top value as threshold
                ... )
        """
        super().__init__(
                on=on, groupby=groupby, agg_func=agg_func, num_workers=num_workers
        )

        if connectivity not in (4, 8):
            raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
        if nrows <= 0 or ncols <= 0:
            raise ValueError(
                    f"nrows and ncols must be positive, got nrows={nrows}, ncols={ncols}"
            )
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
            .. dropdown:: Finding fully surrounded and partially surrounded cells on an 8×12 grid

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
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),  # cardinal
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),  # diagonal
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
        columns and applying edge correction to each group independently. For each group,
        it identifies edge colonies (those missing orthogonal neighbors at the final time
        point), compares their distributions to interior colonies via permutation test,
        and caps all measurements to a threshold derived from top interior colonies.

        Edge correction assumes that interior and edge colonies under identical conditions
        should have similar phenotypic distributions. When they differ significantly
        (p < pvalue threshold), measurements are capped to prevent edge-driven bias in
        downstream analyses.

        Args:
            data (pd.DataFrame): Input DataFrame containing grid measurements. Must include:
                - GRID.SECTION_NUM (str): Column with well/section indices (0-indexed
                  flattened position: row * ncols + col)
                - self.on (str): Measurement column to analyze and correct
                - All columns in self.groupby: For independent group processing
                - self.time_label (str, optional): Time point column if not all observations
                  are at the same time

        Returns:
            pd.DataFrame: Measurements with two new correction columns added:
                - EdgeCorrection_Size-{on}: Capped measurement values (clipped
                  to threshold where edge effect detected)
                - EdgeCorrection_-{self.on}: Threshold value used for correction
                Original measurement column (self.on) remains unchanged. All other columns
                preserved from input. One row per well per group.

        Raises:
            KeyError: If required columns (GRID.SECTION_NUM, self.on, or any in
                self.groupby) are missing.
            ValueError: If data is empty or has zero rows.

        Notes:
            - Stores original data in self._original_data for later visualization
            - Stores corrected data in self._latest_measurements for retrieval via results()
            - Groups are processed independently via joblib.Parallel if num_workers > 1
            - Aggregation (default: mean) is applied to multiple measurements per well
            - Edge correction is only applied if permutation test p-value < self.pvalue
            - If pvalue=0.0, correction is applied to all groups regardless of statistics

        Examples:
            .. dropdown:: Basic edge correction on 96-well data

                >>> import pandas as pd
                >>> import numpy as np
                >>> from phenotypic.analysis import EdgeCorrector
                >>> from phenotypic.tools.constants_ import GRID
                >>>
                >>> # Create sample 96-well data (8 rows x 12 cols)
                >>> np.random.seed(42)
                >>> data = pd.DataFrame({
                ...     'ImageName': ['img1'] * 96,
                ...     GRID.SECTION_NUM: range(96),
                ...     'Metadata_Time': [1] * 96,
                ...     'Shape_Area': np.random.uniform(100, 500, 96)
                ... })
                >>>
                >>> # Edge colonies (row/col 0 or 7/11) have larger areas
                >>> edge_idx = [i for i in range(96) if i//12 in (0,7) or i%12 in (0,11)]
                >>> data.loc[edge_idx, 'Shape_Area'] *= 1.5
                >>>
                >>> # Apply correction
                >>> corrector = EdgeCorrector(
                ...     on='Shape_Area',
                ...     groupby=['ImageName'],
                ...     top_n=5,
                ...     pvalue=0.05
                ... )
                >>> corrected = corrector.analyze(data)
                >>> # New columns created:
                >>> # - 'EdgeCorrection_NewVal-Area': Capped area values at threshold
                >>> # - 'EdgeCorrection_Cap-Area': Threshold value used
                >>> # Original 'Area' column unchanged

            .. dropdown:: Multi-group edge correction (multiple plates and conditions)

                >>> # Data from multiple plates and conditions
                >>> data = pd.DataFrame({
                ...     'Plate': ['P1']*96 + ['P2']*96,
                ...     'Condition': ['WT']*48 + ['KO']*48 + ['WT']*48 + ['KO']*48,
                ...     GRID.SECTION_NUM: list(range(96))*2,
                ...     'Metadata_Time': [1]*192,
                ...     'Area': np.random.uniform(100, 500, 192)
                ... })
                >>>
                >>> corrector = EdgeCorrector(
                ...     on='Area',
                ...     groupby=['Plate', 'Condition'],  # 4 independent corrections
                ...     nrows=8, ncols=12,
                ...     num_workers=4
                ... )
                >>> corrected = corrector.analyze(data)
                >>> # Each plate-condition combo gets its own threshold
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
            "nrows"       : self.nrows,
            "ncols"       : self.ncols,
            "top_n"       : self.top_n,
            "connectivity": self.connectivity,
            "on"          : self.on,
            "pvalue"      : self.pvalue,
            "time_label"  : self.time_label,
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
                    agg_dict[col] = "first"

        agg_data = data.groupby(by=groupby_cols, as_index=False).agg(agg_dict)

        # Handle empty groupby case
        if len(self.groupby) == 0:
            # Process entire dataset as single group
            corrected_data = [self.__class__._apply2group_func(agg_data, **config)]
        else:
            grouped = agg_data.groupby(by=self.groupby, as_index=False)
            corrected_data = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.__class__._apply2group_func)(group, **config)
                    for _, group in grouped
            )

        # Store results
        if corrected_data:
            self._latest_measurements = pd.concat(corrected_data, ignore_index=True)
        else:
            self._latest_measurements = pd.DataFrame()

        return self._latest_measurements

    def show(
            self,
            figsize: tuple[int, int] | None = None,
            max_groups: int = 20,
            collapsed: bool = True,
            criteria: dict[str, any] | None = None,
            **kwargs,
    ) -> tuple[Figure, plt.Axes]:
        """Visualize edge correction results with interior/edge colony comparisons.

        Displays the distribution of measurements for the last time point per group,
        highlighting interior (surrounded) vs. edge colonies. Shows the calculated
        correction threshold and permutation test p-values. Interior colonies are shown
        in blue, edge colonies in red. Circles indicate measurements passing the threshold,
        X's indicate capped measurements.

        Args:
            figsize (tuple[int, int], optional): Figure size as (width, height) in inches.
                If None, auto-sized based on number of groups (single-group: 10x6,
                many groups: 10x max(6, 0.5*ngroups+2)).
            max_groups (int, optional): Maximum number of groups to display. Defaults to 20.
                If data has more groups, a warning is printed and only the first 20 are shown.
            collapsed (bool, optional): If True (default), show all groups stacked vertically
                on a single axis with y-offsets. If False, create a grid of subplots with
                one group per subplot.
            criteria (dict[str, any], optional): Filter groups before visualization using
                column-value criteria (e.g., {'Plate': 'P1', 'Condition': ['WT', 'KO']}).
                Filtering uses SetAnalyzer._filter_by with AND logic across criteria.
            **kwargs: Additional matplotlib parameters:
                - dpi (int): Figure resolution, passed to plt.subplots()
                - facecolor (str): Figure background color
                - edgecolor (str): Figure edge color
                - legend_fontsize (int): Font size for legend (default 9 for collapsed,
                  8 for individual)

        Returns:
            tuple[Figure, plt.Axes]: Tuple of (matplotlib Figure, Axes object(s)):
                - If collapsed=True: (Figure, single Axes)
                - If collapsed=False: (Figure, array of Axes)

        Raises:
            RuntimeError: If analyze() has not been called (no results to display).
            ValueError: If criteria filter leaves no matching data.

        Notes:
            - Interior colonies are those with all orthogonal neighbors present (4-connectivity)
            - Edge colonies are detected but lack all orthogonal neighbors
            - Threshold line (orange) is derived from top interior colonies
            - P-values displayed between interior and edge means (if pvalue != 0)
            - Permutation test uses 1000 resamples with two-sided alternative
            - Call analyze() before show()

        Examples:
            .. dropdown:: Basic visualization of edge correction results

                >>> corrector = EdgeCorrector(on='Area', groupby=['ImageName'])
                >>> corrected = corrector.analyze(data)
                >>> fig, ax = corrector.show()
                >>> # Single collapsed plot with all groups stacked vertically

            .. dropdown:: Individual subplots per group

                >>> fig, axes = corrector.show(
                ...     collapsed=False,
                ...     figsize=(15, 10)
                ... )
                >>> # Grid of subplots, max 3 columns

            .. dropdown:: Filtered visualization for specific plate

                >>> fig, ax = corrector.show(
                ...     criteria={'Plate': 'P1'},
                ...     max_groups=10,
                ...     figsize=(12, 8)
                ... )
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
            data["_group_key"] = data[self.groupby].astype(str).agg(" | ".join, axis=1)
            groups = data["_group_key"].unique()
            group_col = "_group_key"

        if len(groups) > max_groups:
            print(f"Warning: Displaying first {max_groups} groups out of {len(groups)}")
            groups = groups[:max_groups]

        if collapsed:
            return self._show_collapsed(data, groups, group_col, figsize, **kwargs)
        else:
            return self._show_individual(data, groups, group_col, figsize, **kwargs)

    def _show_collapsed(
            self,
            data: pd.DataFrame,
            groups,
            group_col: str,
            figsize: tuple[int, int] | None,
            **kwargs,
    ) -> tuple[Figure, plt.Axes]:
        """Helper method to create collapsed visualization (all groups on one axis).

        Internal method used by show() to display all groups stacked vertically on
        a single matplotlib axis. Each group is offset vertically, with interior
        colonies in blue circles, edge colonies in red circles, and capped values
        marked with X's.

        Args:
            data (pd.DataFrame): Original (uncorrected) measurement data.
            groups (array-like): Unique group identifiers to display.
            group_col (str): Column name containing group labels.
            figsize (tuple[int, int] | None): Figure size as (width, height).
            **kwargs: Matplotlib figure parameters (dpi, facecolor, edgecolor,
                legend_fontsize).

        Returns:
            tuple[Figure, plt.Axes]: (Figure object, Axes object)

        Notes:
            - Groups are displayed from top to bottom (groups[::-1] order)
            - Y-axis has positions 1 to n_groups; y_pos = n_groups - idx
            - Threshold line is orange; interior means are blue dashed; edge means are red dashed
        """
        # Extract figure-level kwargs
        fig_kwargs = {
            k: v for k, v in kwargs.items() if k in ("dpi", "facecolor", "edgecolor")
        }
        legend_fontsize = kwargs.get("legend_fontsize", 9)

        n_groups = len(groups)
        if figsize is None:
            figsize = (10, max(6, 0.5*n_groups + 2))

        fig, ax = plt.subplots(figsize=figsize, **fig_kwargs)

        added_labels = set()

        for idx, group_name in enumerate(groups):
            y_pos = n_groups - idx
            group_data = data[data[group_col] == group_name]

            stats = self._calculate_group_stats(group_data)
            if stats is None:
                continue

            lt_df = stats["last_time_df"]
            threshold = stats["threshold"]
            surrounded_mask = stats["surrounded_mask"]
            edge_mask = stats["edge_mask"]

            # Range line
            vals = lt_df[self.on].values
            if len(vals) > 0:
                ax.hlines(
                        y_pos, vals.min(), vals.max(), colors="lightgray", lw=1.5,
                        zorder=1
                )

            # Threshold
            if not np.isinf(threshold):
                lbl = "Threshold"
                if lbl not in added_labels:
                    added_labels.add(lbl)
                else:
                    lbl = None
                ax.plot(
                        [threshold, threshold],
                        [y_pos - 0.2, y_pos + 0.2],
                        color="#F4A261",
                        lw=2.5,
                        label=lbl,
                        zorder=2,
                )

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
                    ax.scatter(
                            lt_df.loc[mask, self.on],
                            y_jitter[mask],
                            c=color,
                            marker=marker,
                            s=30 if marker == "o" else 40,
                            alpha=0.6 if marker == "o" else 0.8,
                            label=lbl,
                            zorder=3,
                    )

            # Inner Pass
            add_scatter(surrounded_mask & (~is_clipped), "#2E86AB", "o", "Inner (Pass)")
            # Inner Clipped
            add_scatter(surrounded_mask & is_clipped, "#2E86AB", "x", "Inner (Clipped)")
            # Edge Pass
            add_scatter(edge_mask & (~is_clipped), "#E63946", "o", "Edge (Pass)")
            # Edge Clipped
            add_scatter(edge_mask & is_clipped, "#E63946", "x", "Edge (Clipped)")

            # Means
            inner_vals = lt_df.loc[surrounded_mask, self.on]
            edge_vals = lt_df.loc[edge_mask, self.on]

            if len(inner_vals) > 0:
                lbl = "Inner Mean"
                if lbl not in added_labels:
                    added_labels.add(lbl)
                else:
                    lbl = None
                mean_val = inner_vals.mean()
                ax.plot(
                        [mean_val, mean_val],
                        [y_pos - 0.25, y_pos + 0.25],
                        color="#2E86AB",
                        linewidth=2.5,
                        label=lbl,
                        zorder=4,
                        linestyle="--",
                )

            if len(edge_vals) > 0:
                lbl = "Edge Mean"
                if lbl not in added_labels:
                    added_labels.add(lbl)
                else:
                    lbl = None
                mean_val = edge_vals.mean()
                ax.plot(
                        [mean_val, mean_val],
                        [y_pos - 0.25, y_pos + 0.25],
                        color="#E63946",
                        linewidth=2.5,
                        label=lbl,
                        zorder=4,
                        linestyle="--",
                )

            # P-value
            # permutation test requires at least 2 observations per side
            if self.pvalue != 0 and len(inner_vals) >= 2 and len(edge_vals) >= 2:
                pval = self._perm_test(inner_vals, edge_vals)
                mean_inner = inner_vals.mean()
                mean_edge = edge_vals.mean()

                # Bracket parameters
                bracket_y = y_pos + 0.3
                bracket_h = 0.05

                # Draw bracket
                ax.plot(
                        [mean_inner, mean_inner, mean_edge, mean_edge],
                        [
                            bracket_y,
                            bracket_y + bracket_h,
                            bracket_y + bracket_h,
                            bracket_y,
                        ],
                        color="black",
                        linewidth=1,
                        zorder=5,
                )

                # Add p-value text
                mid_x = (mean_inner + mean_edge)/2
                ax.text(
                        mid_x,
                        bracket_y + bracket_h + 0.05,
                        f"p={pval:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                )

        ax.set_yticks(range(1, n_groups + 1))
        ax.set_yticklabels(groups[::-1])
        ax.set_xlabel(self.on)
        ax.set_title(f"Edge Correction (Top N={self.top_n}, p={self.pvalue})")
        ax.legend(loc="best", fontsize=legend_fontsize)
        plt.tight_layout()
        return fig, ax

    def _show_individual(
            self,
            data: pd.DataFrame,
            groups,
            group_col: str,
            figsize: tuple[int, int] | None,
            **kwargs,
    ) -> tuple[Figure, plt.Axes]:
        """Helper method to create individual subplots for each group.

        Internal method used by show() to display each group in a separate subplot.
        Creates a grid of subplots (max 3 columns) with box plots and scatter overlays.
        Interior colonies shown in blue, edge colonies in red. Capped measurements
        marked with X's.

        Args:
            data (pd.DataFrame): Original (uncorrected) measurement data.
            groups (array-like): Unique group identifiers to display.
            group_col (str): Column name containing group labels.
            figsize (tuple[int, int] | None): Figure size as (width, height). If None,
                auto-calculated as (5*ncols, 4*nrows) where ncols=min(3, len(groups)).
            **kwargs: Matplotlib figure parameters (dpi, facecolor, edgecolor,
                legend_fontsize).

        Returns:
            tuple[Figure, np.ndarray[plt.Axes]]: (Figure object, array of Axes)
                Array shape is (nrows, ncols) with empty subplots hidden.

        Notes:
            - One group per subplot with title showing group name
            - Box plots show overall data distribution (light gray)
            - X-axis not used (x-jitter adds visual separation)
            - Legend shown only on first subplot to avoid clutter
            - Unused subplot axes are hidden (set_visible(False))
        """
        # Extract figure-level kwargs
        fig_kwargs = {
            k: v for k, v in kwargs.items() if k in ("dpi", "facecolor", "edgecolor")
        }
        legend_fontsize = kwargs.get("legend_fontsize", 8)

        n_groups = len(groups)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1)//n_cols

        if figsize is None:
            figsize = (5*n_cols, 4*n_rows)

        fig, axes = plt.subplots(
                n_rows, n_cols, figsize=figsize, squeeze=False, **fig_kwargs
        )
        axes = axes.flatten()

        for idx, group_name in enumerate(groups):
            ax = axes[idx]
            group_data = data[data[group_col] == group_name]

            stats = self._calculate_group_stats(group_data)
            if stats is None:
                ax.text(0.5, 0.5, "Insufficient Data", ha="center")
                continue

            lt_df = stats["last_time_df"]
            threshold = stats["threshold"]
            surrounded_mask = stats["surrounded_mask"]
            edge_mask = stats["edge_mask"]

            vals = lt_df[self.on].values
            is_clipped = lt_df[self.on] > threshold

            ax.boxplot(
                    [vals],
                    positions=[1],
                    widths=0.3,
                    patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor="lightgray", alpha=0.3),
            )

            x_jitter = np.random.normal(1, 0.04, len(lt_df))

            # Inner Pass
            mask_ip = surrounded_mask & (~is_clipped)
            if mask_ip.any():
                ax.scatter(
                        x_jitter[mask_ip],
                        lt_df.loc[mask_ip, self.on],
                        c="#2E86AB",
                        marker="o",
                        s=30,
                        alpha=0.6,
                        label="Inner (Pass)",
                )
            # Inner Clipped
            mask_ic = surrounded_mask & is_clipped
            if mask_ic.any():
                ax.scatter(
                        x_jitter[mask_ic],
                        lt_df.loc[mask_ic, self.on],
                        c="#2E86AB",
                        marker="x",
                        s=40,
                        alpha=0.8,
                        label="Inner (Clipped)",
                )
            # Edge Pass
            mask_ep = edge_mask & (~is_clipped)
            if mask_ep.any():
                ax.scatter(
                        x_jitter[mask_ep],
                        lt_df.loc[mask_ep, self.on],
                        c="#E63946",
                        marker="o",
                        s=30,
                        alpha=0.6,
                        label="Edge (Pass)",
                )
            # Edge Clipped
            mask_ec = edge_mask & is_clipped
            if mask_ec.any():
                ax.scatter(
                        x_jitter[mask_ec],
                        lt_df.loc[mask_ec, self.on],
                        c="#E63946",
                        marker="x",
                        s=40,
                        alpha=0.8,
                        label="Edge (Clipped)",
                )

            if not np.isinf(threshold):
                ax.axhline(
                        y=threshold, color="#F4A261", linestyle="--", label="Threshold"
                )

            ax.set_title(group_name)
            ax.set_ylabel(self.on)
            ax.set_xticks([])

            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(
                        by_label.values(),
                        by_label.keys(),
                        loc="best",
                        fontsize=legend_fontsize,
                )

        for idx in range(n_groups, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def _calculate_group_stats(self, group: pd.DataFrame):
        """Calculate statistics for edge correction visualization of a single group.

        Helper method for show() that computes interior/edge classifications and
        correction threshold for a single group. Identifies fully surrounded colonies
        at the maximum time point, performs permutation testing if enabled, and
        calculates threshold from top interior values.

        Args:
            group (pd.DataFrame): Subset of data for a single group (e.g., single plate).
                Must contain GRID.SECTION_NUM, self.on, and self.time_label columns.

        Returns:
            dict | None: Dictionary with keys:
                - 'last_time_df' (pd.DataFrame): Rows at maximum time point
                - 'threshold' (float): Correction threshold (np.inf if no correction)
                - 'surrounded_mask' (pd.Series[bool]): Interior colony indicators
                - 'edge_mask' (pd.Series[bool]): Edge colony indicators
            Returns None if group is empty, lacks colonies, or lacks measurement column.

        Notes:
            - Uses maximum time point per group for interior/edge identification
            - Interior = all orthogonal neighbors present (4-connectivity)
            - Edge = present but not fully surrounded
            - Permutation test compares inner vs edge distributions (scipy.stats.permutation_test)
            - If p-value > self.pvalue, threshold = np.inf (no correction applied)
            - Threshold = mean of top_n interior values if correction applies
        """
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
                    return_counts=False,
            )
        except ValueError:
            return None

        surrounded_idx_set = set(surrounded_idx)

        if len(surrounded_idx_set) == 0:
            return {
                "last_time_df"   : last_time_group,
                "threshold"      : np.inf,
                "surrounded_mask": pd.Series(False, index=last_time_group.index),
                "edge_mask"      : pd.Series(True, index=last_time_group.index),
            }

        surrounded_mask = last_time_group[GRID.SECTION_NUM].isin(surrounded_idx_set)
        edge_mask = ~surrounded_mask & last_time_group[GRID.SECTION_NUM].isin(
                present_sections
        )

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
            "last_time_df"   : last_time_group,
            "threshold"      : threshold,
            "surrounded_mask": surrounded_mask,
            "edge_mask"      : edge_mask,
        }

    def results(self) -> pd.DataFrame:
        """Return the corrected measurement DataFrame from the last analyze() call.

        Retrieves the DataFrame with edge-corrected measurements produced by the most
        recent call to analyze(). Provides convenient access to results without retaining
        a local reference.

        Returns:
            pd.DataFrame: Edge-corrected measurements with original data plus two new
                correction columns:
                - EDGE_CORRECTION.NEW_VAL-{self.on}: Capped measurement values
                - EDGE_CORRECTION.CORRECTED_CAP-{self.on}: Threshold value used
                Original measurement column (self.on) is preserved unchanged. If analyze()
                has not been called, returns an empty DataFrame.

        Examples:
            .. dropdown:: Retrieving corrected measurements after analysis

                >>> corrector = EdgeCorrector(
                ...     on='Area',
                ...     groupby=['ImageName']
                ... )
                >>> corrected = corrector.analyze(data)
                >>> results = corrector.results()
                >>> assert results.equals(corrected)
                >>> # Access corrected values
                >>> corrected_areas = results['Size-Area']
                >>> thresholds = results['Cap-Area']
                >>> # Original 'Area' column also available for comparison
                >>> original_areas = results['Area']

        Notes:
            - Returns the DataFrame stored in self._latest_measurements
            - Same as the return value of analyze()
            - Always use this method rather than direct attribute access
        """
        return self._latest_measurements

    @staticmethod
    def _apply2group_func(
            group: pd.DataFrame,
            on: str,
            nrows: int,
            ncols: int,
            top_n: int,
            time_label: str,
            connectivity: int,
            pvalue: float,
    ) -> pd.DataFrame:
        """Apply edge correction logic to a single group of measurements.

        Static method called by analyze() via joblib.Parallel to process each group
        independently. Identifies interior colonies, performs permutation testing, and
        creates new corrected columns. Original measurement column remains unchanged.
        Called once per group.

        Args:
            group (pd.DataFrame): Measurement data for a single group. Must contain:
                - GRID.SECTION_NUM: Flattened well indices (row*ncols + col)
                - on: Measurement column to correct
                - time_label: Time point column (optional)
            on (str): Name of measurement column to analyze. Used as basis for new
                corrected columns: EDGE_CORRECTION.NEW_VAL-{on} and
                EDGE_CORRECTION.CORRECTED_CAP-{on}.
            nrows (int): Grid rows (e.g., 8 for 96-well).
            ncols (int): Grid columns (e.g., 12 for 96-well).
            top_n (int): Number of top interior values for threshold.
            time_label (str): Time point column name.
            connectivity (int): 4 or 8 neighbor connectivity.
            pvalue (float): P-value threshold for permutation test. If interior and
                edge distributions differ significantly (p < pvalue), apply correction.
                Set to 0.0 to apply to all groups.

        Returns:
            pd.DataFrame: Input group with two new correction columns added:
                - EDGE_CORRECTION.NEW_VAL-{on}: Capped measurement values at threshold
                  (clipped if correction applied, original otherwise)
                - EDGE_CORRECTION.CORRECTED_CAP-{on}: Threshold value computed
                Original measurement column (on) is preserved unchanged. All rows get
                corrected values (not just edge wells) for consistency and reproducibility.

        Notes:
            - Interior = all orthogonal neighbors present (4-connectivity determined by grid)
            - Edge = detected but missing >= 1 orthogonal neighbor
            - Uses last time point per group to identify interior/edge classification
            - Permutation test: scipy.stats.permutation_test with 1000 resamples
            - Threshold = mean of top_n interior values at last time point
            - If no interior colonies exist, returns group with new columns set to original values
            - If interior/edge difference not significant (p > pvalue), returns with new columns
              set to original values (threshold = inf)
            - New columns always created for consistency, even if no correction applied

        Examples:
            .. dropdown:: Direct use in batch processing

                >>> from phenotypic.analysis import EdgeCorrector
                >>> group_data = data[data['Plate'] == 'P1'] #doctest: +SKIP
                >>> corrected = EdgeCorrector._apply2group_func(
                ...     group_data,
                ...     on='Area',
                ...     nrows=8, ncols=12,
                ...     top_n=5,
                ...     time_label='Time',
                ...     connectivity=4,
                ...     pvalue=0.05
                ... )
        """
        from phenotypic.tools.constants_ import GRID

        section_col = GRID.SECTION_NUM

        # Set base case
        group.loc[:, f"{EDGE_CORRECTION.NEW_VAL}-{on}"] = group.loc[:, on]

        # TODO: Should this be the max or np.inf
        group.loc[:, f"{EDGE_CORRECTION.CORRECTED_CAP}-{on}"] = group.loc[:, on].max()

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
                    return_counts=False,
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

        last_inner_values: pd.Series = last_time_group.loc[
            last_time_group.loc[:, GRID.SECTION_NUM].isin(surrounded_idx), on
        ]

        if pvalue != 0:
            last_edge_values: pd.Series = last_time_group.loc[
                last_time_group.loc[:, GRID.SECTION_NUM].isin(edge_idx), on
            ]

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
        group.loc[:, f"{EDGE_CORRECTION.NEW_VAL}-{on}"] = np.clip(group.loc[:, on],
                                                                  a_min=0,
                                                                  a_max=threshold)
        group.loc[:, f"{EDGE_CORRECTION.CORRECTED_CAP}-{on}"] = threshold
        return group

    @staticmethod
    def _perm_test(surrounded, edge):
        """Perform permutation test comparing interior vs. edge colony distributions.

        Determines if interior (surrounded) and edge colonies have statistically
        significantly different measurements using permutation testing. Tests the
        null hypothesis that distributions are identical.

        Args:
            surrounded (array-like): Measurements for interior colonies.
            edge (array-like): Measurements for edge colonies.

        Returns:
            float: Two-sided p-value from permutation test (range [0, 1]).
                Small p-values indicate significant difference between distributions.

        Notes:
            - Uses scipy.stats.permutation_test with 1000 resamples
            - Test statistic: mean(surrounded) - mean(edge)
            - Two-sided alternative: tests if means differ in either direction
            - Returns p-value >= 0.05 suggests no significant edge effect
            - Returns p-value < 0.05 suggests colonies ARE significantly different
        """
        return permutation_test(
                data=(surrounded, edge),
                statistic=lambda x, y: np.mean(x) - np.mean(y),
                permutation_type="independent",
                n_resamples=1000,
                alternative="two-sided",
        ).pvalue


EdgeCorrector.__doc__ = EDGE_CORRECTION.append_rst_to_doc(EdgeCorrector.__doc__)
