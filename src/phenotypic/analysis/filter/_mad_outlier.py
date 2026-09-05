from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
# matplotlib is imported inside the methods that draw. A module-scope import
# put pyplot (541 ms, the single largest remaining cost) on the
# `import phenotypic` path for every run, most of which draw nothing.
from pydantic import field_validator, PrivateAttr

from .._helper import _qc_math
from ..abc_ import SetAnalyzer
from ..abc_._set_analyzer import normalize_measurement_metadata_columns

if TYPE_CHECKING:  # pragma: no cover - annotations only
    import matplotlib.pyplot as plt

# Iglewicz & Hoaglin (1993) consistency constant: for a normal distribution
# sigma ~= 1.4826 * MAD, and 0.6745 ~= 1 / 1.4826. Multiplying the absolute
# deviation by 0.6745 / MAD therefore estimates the deviation in units of
# sigma, putting the modified Z-score on the same scale as a standard Z-score.
# The canonical definition now lives in ``_qc_math`` so QC checks reuse it.
_MAD_CONSISTENCY = _qc_math.MAD_CONSISTENCY


class MADOutlierRemover(SetAnalyzer):
    """Analyzer for removing outliers using the modified Z-score (MAD) method.

    This class removes outliers from measurement data by applying the
    Iglewicz-Hoaglin modified Z-score test within groups. For each group it
    computes the median and the median absolute deviation (MAD) of the
    measurement column, scores every row as
    ``0.6745 * |value - median| / MAD``, and removes rows whose score exceeds
    ``threshold``.

    Unlike Tukey's fence (see :class:`TukeyOutlierRemover`), the MAD method
    estimates spread from the median absolute deviation, which has a 50%
    breakdown point: the test stays accurate even when up to half the rows in
    a group are contaminated. This makes it a robust default for small or
    skewed colony-measurement groups.

    If MAD is zero for a group (all values identical, or a > 50% tie), the
    test falls back to the raw absolute deviation from the median scaled by
    the *mean* absolute deviation, preserving the breakdown point while
    avoiding division by zero. If every value is identical, no rows are
    removed.

    Args:
        on: Name of measurement column to test for outliers (e.g., 'Shape_Area',
            'Intensity_IntegratedIntensity').
        groupby: List of column names to group by (e.g., ['StrainID', 'Time']).
        threshold: Modified Z-score cutoff. Iglewicz & Hoaglin (1993) recommend
            3.5 for general use. Lower values (e.g., 2.5) are more aggressive;
            higher values (e.g., 5.0) more conservative. Default is 3.5.
        n_jobs: Number of parallel workers. Default is 1.

    Attributes:
        on: Column to test for outliers.
        groupby: List of column names to group by.
        threshold: Modified Z-score cutoff used for outlier identification.
        n_jobs: Number of parallel workers. Default is 1.

    Examples:
        Remove outliers and visualize results:

        >>> import pandas as pd
        >>> import numpy as np
        >>> from phenotypic.analysis import MADOutlierRemover
        >>> # Create sample data with some outliers
        >>> np.random.seed(42)
        >>> data = pd.DataFrame({
        ...     'ImageName': ['img1'] * 50 + ['img2'] * 50,
        ...     'Area': np.concatenate([
        ...         np.random.normal(200, 30, 48),
        ...         [500, 550],  # outliers in img1
        ...         np.random.normal(180, 25, 48),
        ...         [50, 600]  # outliers in img2
        ...     ])
        ... })
        >>> # Initialize detector
        >>> detector = MADOutlierRemover(
        ...     on='Area',
        ...     groupby=['ImageName'],
        ...     threshold=3.5
        ... )
        >>> # Remove outliers
        >>> filtered_data = detector.analyze(data)
        >>> # Check how many were removed
        >>> print(f"Original: {len(data)}, Filtered: {len(filtered_data)}")  # doctest: +SKIP
        >>> # Visualize removed outliers
        >>> fig = detector.show()  # doctest: +SKIP

    References:
        Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle
        Outliers*. ASQC Quality Press.
    """

    agg_func: Callable | str | list | dict | None = None
    threshold: float = 3.5

    _original_data: pd.DataFrame = PrivateAttr(default_factory=pd.DataFrame)

    @field_validator("threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        """Reject non-positive modified Z-score thresholds.

        Args:
            value: The candidate ``threshold`` cutoff.

        Returns:
            The validated ``threshold`` value.

        Raises:
            ValueError: If ``threshold`` is not positive.
        """
        if value <= 0:
            raise ValueError(f"threshold must be positive, got {value}")
        return value

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from data using the modified Z-score (MAD) method.

        This method processes the input DataFrame by grouping according to
        specified columns and removing outliers within each group
        independently. Outliers are identified using the Iglewicz-Hoaglin
        modified Z-score and filtered out. The original data is stored
        internally for visualization purposes.

        Args:
            data: DataFrame containing measurement data. Must include all columns
                specified in self.groupby and self.on.

        Returns:
            DataFrame with outliers removed. Contains only the original columns
            (no additional outlier flag columns).

        Raises:
            KeyError: If required columns are missing from input DataFrame.
            ValueError: If data is empty or malformed.

        Examples:
            Analyze and filter outliers from measurement data:

            >>> import pandas as pd
            >>> import numpy as np
            >>> from phenotypic.analysis import MADOutlierRemover
            >>> # Create sample data
            >>> np.random.seed(42)
            >>> data = pd.DataFrame({
            ...     'ImageName': ['img1'] * 100,
            ...     'Area': np.concatenate([
            ...         np.random.normal(200, 30, 98),
            ...         [500, 50]  # outliers
            ...     ])
            ... })
            >>> # Remove outliers
            >>> detector = MADOutlierRemover(
            ...     on='Area',
            ...     groupby=['ImageName'],
            ...     threshold=3.5
            ... )
            >>> filtered_data = detector.analyze(data)
            >>> # Check results
            >>> print(f"Original: {len(data)} rows, Filtered: {len(filtered_data)} rows")  # doctest: +SKIP
            >>> print(f"Removed {len(data) - len(filtered_data)} outliers")  # doctest: +SKIP

        Notes:
            - Stores original data in self._original_data for visualization
            - Stores filtered results in self._latest_measurements for retrieval
            - Groups are processed independently with their own median and MAD
            - NaN values in measurement column are preserved in output
        """
        data = normalize_measurement_metadata_columns(data)
        # Validate input
        if data is None or len(data) == 0:
            raise ValueError("Input data cannot be empty")

        # Store original data for visualization
        self._original_data = data.copy()

        # Check required columns
        required_cols = set(self.groupby + [self.on])
        missing_cols = required_cols - set(data.columns)

        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        # Prepare configuration for _apply2group_func
        config = {"threshold": self.threshold, "on": self.on}

        # Apply outlier removal to each group
        grouped = data.groupby(by=self.groupby, as_index=True)
        if self.n_jobs == 1:
            results = []
            for key, group in grouped:
                results.append(self.__class__._apply2group_func(key, group, **config))
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self.__class__._apply2group_func)(key, group, **config)
                    for key, group in grouped
            )

        # Concatenate all group results
        self._latest_measurements = pd.concat(results, ignore_index=True)

        return self._latest_measurements

    def show(
            self,
            figsize: tuple[int, int] | None = None,
            max_groups: int = 20,
            collapsed: bool = True,
            criteria: dict[str, Any] | None = None,
            **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Visualize outlier detection results.

        Creates a visualization showing the distribution of values with outliers
        highlighted and modified Z-score bounds displayed. Can display as
        individual subplots or as a collapsed stacked view with all groups in a
        single plot. Outlier flags are computed dynamically for visualization
        only.

        Args:
            figsize: Figure size as (width, height). If None, automatically determined
                based on number of groups and mode.
            max_groups: Maximum number of groups to display. If there are more groups,
                only the first max_groups will be shown. Default is 20.
            collapsed: If True, show all groups stacked vertically in a single plot.
                If False, show each group in its own subplot. Default is True.
            criteria: Optional dictionary specifying filtering criteria for data selection.
                When provided, only groups matching the criteria will be displayed.
                Format: {'column_name': value} or {'column_name': [value1, value2]}.
                Default is None (show all groups).
            **kwargs: Additional matplotlib parameters to customize the plot. Common options include:
                - dpi: Figure resolution (default 100)
                - facecolor: Figure background color
                - edgecolor: Figure edge color
                - grid_alpha: Alpha value for grid lines (default 0.3)
                - grid_axis: Which axis to apply grid to ('both', 'x', 'y')
                - legend_loc: Legend location (default 'best')
                - legend_fontsize: Font size for legend (default 8)

        Returns:
            Tuple of (Figure, Axes) containing the visualization.

        Raises:
            ValueError: If analyze() has not been called yet (no results to display).
            KeyError: If criteria references columns not present in the data.

        Examples:
            Visualize outlier detection with multiple grouping options:

            >>> import pandas as pd
            >>> import numpy as np
            >>> from phenotypic.analysis import MADOutlierRemover
            >>> # Create sample data with multiple grouping columns
            >>> np.random.seed(42)
            >>> data = pd.DataFrame({
            ...     'ImageName': ['img1', 'img2'] * 50,
            ...     'Plate': ['P1'] * 50 + ['P2'] * 50,
            ...     'Area': np.concatenate([
            ...         np.random.normal(200, 30, 48), [500, 550],
            ...         np.random.normal(180, 25, 48), [50, 600]
            ...     ])
            ... })
            >>> # Remove outliers and visualize all groups
            >>> detector = MADOutlierRemover(
            ...     on='Area',
            ...     groupby=['Plate', 'ImageName'],
            ...     threshold=3.5
            ... )
            >>> results = detector.analyze(data)  # doctest: +SKIP
            >>> fig, axes = detector.show(figsize=(12, 5))  # doctest: +SKIP
            >>> # Visualize only specific plate
            >>> fig, axes = detector.show(criteria={'Plate': 'P1'})  # doctest: +SKIP

        Notes:
            Individual mode (collapsed=False):
            - Each group gets its own subplot with box plot
            - Outliers shown in red, normal values in blue
            - Horizontal lines show the modified Z-score bounds

            Collapsed mode (collapsed=True):
            - All groups stacked vertically in single plot
            - Each group shown as horizontal line with median marker
            - Vertical bars show the modified Z-score bounds
            - Normal points as circles, outliers as diamonds

            Filtering with criteria:
            - Only groups matching all criteria are displayed
            - Useful for focusing on specific plates, conditions, or subsets
        """
        if self._original_data.empty:
            raise ValueError("No results to display. Call analyze() first.")

        # Use original data for visualization and dynamically compute outlier flags
        data = self._original_data.copy()

        # Apply filtering if criteria provided
        if criteria is not None:
            data = self._filter_by(df=data, criteria=criteria, copy=False)

            if data.empty:
                raise ValueError("No data matches the specified criteria")

        # Get unique groups
        if len(self.groupby) == 1:
            groups = data[self.groupby[0]].unique()
            group_col = self.groupby[0]
        else:
            # Create a combined group identifier for multiple groupby columns
            data["_group_key"] = data[self.groupby].astype(str).agg(" | ".join, axis=1)
            groups = data["_group_key"].unique()
            group_col = "_group_key"

        # Limit number of groups if needed
        if len(groups) > max_groups:
            groups = groups[:max_groups]
            print(
                    f"Warning: Displaying only first {max_groups} of {len(data[group_col].unique())} groups"
            )

        # Branch based on visualization mode
        if collapsed:
            return self._show_collapsed(data, groups, group_col, figsize, **kwargs)
        else:
            return self._show_individual(data, groups, group_col, figsize, **kwargs)

    def _show_individual(
            self,
            data: pd.DataFrame,
            groups,
            group_col: str,
            figsize: tuple[int, int] | None,
            **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create individual subplots for each group."""
        import matplotlib.pyplot as plt

        # Extract figure-level kwargs
        fig_kwargs = {
            k: v for k, v in kwargs.items() if k in ("dpi", "facecolor", "edgecolor")
        }
        grid_alpha = kwargs.get("grid_alpha", 0.3)
        grid_axis = kwargs.get("grid_axis", "y")
        legend_fontsize = kwargs.get("legend_fontsize", 8)

        # Calculate layout
        n_groups = len(groups)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols

        # Set figure size
        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(
                n_rows, n_cols, figsize=figsize, squeeze=False, **fig_kwargs
        )
        axes = axes.flatten()

        total_outliers = 0
        total_count = 0

        # Plot each group
        for idx, group_name in enumerate(groups):
            ax = axes[idx]
            group_data = data[data[group_col] == group_name].copy()

            # Dynamically compute outlier flags for this group
            values = group_data[self.on].to_numpy(dtype=float, na_value=np.nan)
            median, lower_bound, upper_bound, is_outlier = self._mad_bounds(
                    values, self.threshold
            )
            group_data["_is_outlier"] = is_outlier

            # Separate inliers and outliers
            inliers = group_data[~group_data["_is_outlier"]]
            outliers = group_data[group_data["_is_outlier"]]

            total_outliers += len(outliers)
            total_count += len(group_data)

            # Create x-coordinates for scatter plot
            x_inliers = np.random.normal(1, 0.04, len(inliers))
            x_outliers = np.random.normal(1, 0.04, len(outliers))

            # Plot inliers
            if len(inliers) > 0:
                ax.scatter(
                        x_inliers,
                        inliers[self.on].values,
                        alpha=0.6,
                        s=40,
                        c="#2E86AB",
                        label="Normal",
                        zorder=3,
                )

            # Plot outliers
            if len(outliers) > 0:
                ax.scatter(
                        x_outliers,
                        outliers[self.on].values,
                        alpha=0.8,
                        s=50,
                        c="#E63946",
                        marker="D",
                        label="Outlier",
                        zorder=4,
                )

            # Create box plot (drop NaN, which boxplot cannot handle)
            finite_values = values[~np.isnan(values)]
            if len(finite_values) > 0:
                ax.boxplot(
                        [finite_values],
                        positions=[1],
                        widths=0.3,
                        patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor="lightgray", alpha=0.3),
                        medianprops=dict(color="black", linewidth=2),
                )

            # Add bound lines (only when a finite cutoff exists)
            if np.isfinite(lower_bound) and np.isfinite(upper_bound):
                ax.axhline(
                        y=lower_bound,
                        color="#F4A261",
                        linestyle="--",
                        linewidth=1.5,
                        label="Lower Bound",
                        zorder=2,
                )
                ax.axhline(
                        y=upper_bound,
                        color="#F4A261",
                        linestyle="--",
                        linewidth=1.5,
                        label="Upper Bound",
                        zorder=2,
                )

            # Formatting
            ax.set_title(
                    f"{group_name}\n({len(outliers)} outliers / {len(group_data)} total)",
                    fontsize=10,
                    fontweight="bold",
            )
            ax.set_ylabel(self.on, fontsize=9)
            ax.set_xticks([])
            ax.grid(True, alpha=grid_alpha, axis=grid_axis)

            # Add legend only to first subplot
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                # Remove duplicate labels
                by_label = dict(zip(labels, handles))
                ax.legend(
                        by_label.values(),
                        by_label.keys(),
                        loc="best",
                        fontsize=legend_fontsize,
                        framealpha=0.9,
                )

        # Hide unused subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].set_visible(False)

        # Overall title
        outlier_pct = 100 * total_outliers / total_count if total_count > 0 else 0

        fig.suptitle(
                f"MAD Outlier Detection (threshold={self.threshold})\n"
                f"{total_outliers} outliers detected ({outlier_pct:.1f}% of {total_count} measurements)",
                fontsize=14,
                fontweight="bold",
                y=0.995,
        )

        plt.tight_layout()

        return fig, axes

    def _show_collapsed(
            self,
            data: pd.DataFrame,
            groups,
            group_col: str,
            figsize: tuple[int, int] | None,
            **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create collapsed stacked view with all groups in single plot."""
        import matplotlib.pyplot as plt

        # Extract figure-level kwargs
        fig_kwargs = {
            k: v for k, v in kwargs.items() if k in ("dpi", "facecolor", "edgecolor")
        }
        grid_alpha = kwargs.get("grid_alpha", 0.2)
        legend_fontsize = kwargs.get("legend_fontsize", 9)

        n_groups = len(groups)

        # Set figure size
        if figsize is None:
            figsize = (10, max(6, 0.5 * n_groups + 2))

        fig, ax = plt.subplots(1, 1, figsize=figsize, **fig_kwargs)

        total_outliers = 0
        total_count = 0

        added_labels = set()

        # Process each group and plot stacked vertically
        for idx, group_name in enumerate(groups):
            y_pos = n_groups - idx  # Stack from top to bottom
            group_data = data[data[group_col] == group_name].copy()

            # Dynamically compute outlier flags for this group
            values = group_data[self.on].to_numpy(dtype=float, na_value=np.nan)
            median, lower_bound, upper_bound, is_outlier = self._mad_bounds(
                    values, self.threshold
            )
            group_data["_is_outlier"] = is_outlier

            # Separate inliers and outliers
            inliers = group_data[~group_data["_is_outlier"]]
            outliers = group_data[group_data["_is_outlier"]]

            total_outliers += len(outliers)
            total_count += len(group_data)

            # Draw horizontal line for full data range
            finite_values = values[~np.isnan(values)]
            if len(finite_values) > 0:
                ax.hlines(
                        y_pos,
                        finite_values.min(),
                        finite_values.max(),
                        colors="lightgray",
                        linewidth=1.5,
                        alpha=0.6,
                        zorder=1,
                )

            # Add vertical tick marks for bounds and median
            tick_height = 0.15  # Height of tick marks

            if np.isfinite(lower_bound) and np.isfinite(upper_bound):
                # Lower bound tick
                lbl = "Bounds"
                if lbl not in added_labels:
                    added_labels.add(lbl)
                else:
                    lbl = None

                ax.plot(
                        [lower_bound, lower_bound],
                        [y_pos - tick_height, y_pos + tick_height],
                        color="#F4A261",
                        linewidth=2.5,
                        linestyle="-",
                        label=lbl,
                        zorder=3,
                )

                # Upper bound tick
                ax.plot(
                        [upper_bound, upper_bound],
                        [y_pos - tick_height, y_pos + tick_height],
                        color="#F4A261",
                        linewidth=2.5,
                        linestyle="-",
                        zorder=3,
                )

            # Median marker
            lbl = "Median"
            if lbl not in added_labels:
                added_labels.add(lbl)
            else:
                lbl = None

            ax.plot(
                    [median, median],
                    [y_pos - tick_height, y_pos + tick_height],
                    color="black",
                    linewidth=2.5,
                    linestyle="-",
                    label=lbl,
                    zorder=3,
            )

            # Create y-coordinates with jitter for scatter plot
            y_inliers = np.random.normal(y_pos, 0.06, len(inliers))
            y_outliers = np.random.normal(y_pos, 0.06, len(outliers))

            # Plot inliers
            if len(inliers) > 0:
                lbl = "Normal"
                if lbl not in added_labels:
                    added_labels.add(lbl)
                else:
                    lbl = None
                ax.scatter(
                        inliers[self.on].values,
                        y_inliers,
                        alpha=0.6,
                        s=30,
                        c="#2E86AB",
                        label=lbl,
                        zorder=4,
                )

            # Plot outliers
            if len(outliers) > 0:
                lbl = "Outlier"
                if lbl not in added_labels:
                    added_labels.add(lbl)
                else:
                    lbl = None
                ax.scatter(
                        outliers[self.on].values,
                        y_outliers,
                        alpha=0.8,
                        s=35,
                        c="#E63946",
                        marker="D",
                        label=lbl,
                        zorder=5,
                )

        # Formatting
        ax.set_yticks(range(1, n_groups + 1))
        ax.set_yticklabels(groups[::-1])  # Reverse to match top-to-bottom order
        ax.set_xlabel(self.on, fontsize=11, fontweight="bold")
        ax.set_ylabel("Group", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=grid_alpha, axis="x")
        ax.set_ylim(0.5, n_groups + 0.5)

        # Add legend
        ax.legend(loc="best", fontsize=legend_fontsize, framealpha=0.9)

        # Overall title
        outlier_pct = 100 * total_outliers / total_count if total_count > 0 else 0
        fig.suptitle(
                f"MAD Outlier Detection (threshold={self.threshold})\n"
                f"{total_outliers} outliers detected ({outlier_pct:.1f}% of {total_count} measurements)",
                fontsize=13,
                fontweight="bold",
        )

        plt.tight_layout()

        return fig, ax

    def results(self) -> pd.DataFrame:
        """Return the filtered results (outliers removed).

        Returns the DataFrame with outliers removed from the most recent call
        to analyze().

        Returns:
            DataFrame with outliers filtered out. Contains only the original columns
            without additional outlier flag columns. If analyze() has not been called,
            returns an empty DataFrame.

        Examples:
            Retrieve filtered results after analysis:

            >>> detector = MADOutlierRemover(
            ...     on='Area',
            ...     groupby=['ImageName']
            ... )
            >>> filtered_data = detector.analyze(data)  # doctest: +SKIP
            >>> results_copy = detector.results()  # Same as filtered_data  # doctest: +SKIP
            >>> assert results_copy.equals(filtered_data)  # doctest: +SKIP

        Notes:
            - Returns the DataFrame stored in self._latest_measurements
            - Contains only inliers (outliers have been removed)
            - Use this method to retrieve results after calling analyze()
        """
        return self._latest_measurements

    @staticmethod
    def _modified_z_scores(values: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Compute the modified Z-scores for an array of values.

        Implements the Iglewicz-Hoaglin modified Z-score
        ``0.6745 * |value - median| / MAD``. When MAD is zero (all values
        identical, or a > 50% tie) the test falls back to the raw absolute
        deviation scaled by the *mean* absolute deviation.

        Args:
            values: Array of numeric values; NaN entries are ignored when
                computing the median and deviations.

        Returns:
            A tuple ``(scores, median, half_width)`` where ``scores`` is the
            per-value modified Z-score array (NaN where the input was NaN),
            ``median`` is the group median, and ``half_width`` is the absolute
            deviation that corresponds to a score of exactly 1.0 -- i.e. a row
            is an outlier when ``|value - median| > threshold * half_width``.
            ``half_width`` is ``0.0`` only when every value is identical.
        """
        # Compute the median / absolute deviations / MAD once and derive both
        # the scores and the half-width from them, mirroring
        # ``_qc_math.modified_z_scores`` without re-reducing the array.
        values = np.asarray(values, dtype=float)
        median = float(np.nanmedian(values))
        abs_dev = np.abs(values - median)
        mad = float(np.nanmedian(abs_dev))

        if mad == 0.0:
            mean_ad = float(np.nanmean(abs_dev))
            if mean_ad == 0.0:
                # All values identical -- no outliers possible.
                return np.zeros_like(values, dtype=float), median, 0.0
            return abs_dev / mean_ad, median, mean_ad

        return _MAD_CONSISTENCY * abs_dev / mad, median, mad / _MAD_CONSISTENCY

    @staticmethod
    def _mad_bounds(
            values: np.ndarray, threshold: float
    ) -> tuple[float, float, float, np.ndarray]:
        """Compute the median, outlier bounds, and outlier mask for a group.

        Args:
            values: Array of numeric values for one group.
            threshold: Modified Z-score cutoff.

        Returns:
            A tuple ``(median, lower_bound, upper_bound, is_outlier)``. The
            bounds are the values where the modified Z-score equals
            ``threshold``; both are ``inf``/``-inf`` when no finite cutoff
            exists (all values identical). ``is_outlier`` is a boolean array
            aligned with ``values`` (always ``False`` for NaN entries).
        """
        scores, median, half_width = MADOutlierRemover._modified_z_scores(values)
        is_outlier = scores > threshold
        if half_width == 0.0:
            return median, -np.inf, np.inf, is_outlier
        bound = threshold * half_width
        return median, median - bound, median + bound, is_outlier

    @staticmethod
    def _apply2group_func(
            key, group: pd.DataFrame, on: str, threshold: float
    ) -> pd.DataFrame:
        """Apply modified Z-score (MAD) outlier removal on a DataFrame group.

        This static method filters out rows in the DataFrame group where the
        values of the ``on`` column have a modified Z-score exceeding
        ``threshold``. Rows whose ``on`` value is NaN are preserved (their
        score is NaN, which never exceeds the threshold).

        Args:
            key: The group key (not used but required for joblib).
            group: A group of DataFrame rows to which the MAD filtering is applied.
            on: The column in the DataFrame on which the modified Z-score is computed.
            threshold: Modified Z-score cutoff for identifying outlier rows.

        Returns:
            Filtered DataFrame containing rows whose modified Z-score does not
            exceed ``threshold``.
        """
        values = group[on].to_numpy(dtype=float, na_value=np.nan)
        scores, _, _ = MADOutlierRemover._modified_z_scores(values)
        return group[~(scores > threshold)]
