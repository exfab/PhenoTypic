from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import delayed, Parallel
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .abc_ import SetAnalyzer


class TukeyOutlierRemover(SetAnalyzer):
    """Analyzer for removing outliers using Tukey's fence method.
    
    This class removes outliers from measurement data by applying Tukey's fence test
    within groups. The method calculates the interquartile range (IQR) and removes values
    that fall outside Q1 - k*IQR or Q3 + k*IQR, where k is a tunable multiplier
    (typically 1.5 for outliers or 3.0 for extreme outliers).
    
    Args:
        on: Name of measurement column to test for outliers (e.g., 'Shape_Area', 'Intensity_IntegratedIntensity').
        groupby: List of column names to group by (e.g., ['ImageName', 'Metadata_Plate']).
        k: IQR multiplier for fence calculation. Default is 1.5 (standard outliers).
            Use 3.0 for extreme outliers only.
        num_workers: Number of parallel workers. Default is 1.
    
    Attributes:
        groupby: List of column names to group by.
        on: Column to test for outliers.
        k: IQR multiplier used for fence calculation.
        num_workers: Number of parallel workers. Default is 1.
        
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from phenotypic.analysis import TukeyOutlierRemover
        >>> 
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
        >>> 
        >>> # Initialize detector
        >>> detector = TukeyOutlierRemover(
        ...     on='Area',
        ...     groupby=['ImageName'],
        ...     k=1.5
        ... )
        >>> 
        >>> # Remove outliers
        >>> filtered_data = detector.analyze(data)
        >>> 
        >>> # Check how many were removed
        >>> print(f"Original: {len(data)}, Filtered: {len(filtered_data)}")
        >>> 
        >>> # Visualize removed outliers
        >>> fig = detector.show()
    """

    def __init__(
            self,
            on: str,
            groupby: list[str],
            k: float = 1.5,
            num_workers: int = 1
    ):
        """Initialize TukeyOutlierRemover with test parameters.
        
        Args:
            on: Column name for grouping/aggregation operations.
            groupby: List of column names to group by.
            measurement_col: Name of measurement column to test for outliers.
            k: IQR multiplier for fence calculation. Default is 1.5.
            agg_func: Aggregation function. Default is 'mean'.
            num_workers: Number of workers. Default is 1.
            
        Raises:
            ValueError: If k is not positive.
        """
        super().__init__(on=on, groupby=groupby, agg_func=None, num_workers=num_workers)

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        self.k = k
        self._original_data: pd.DataFrame = pd.DataFrame()

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from data using Tukey's fence method.
        
        This method processes the input DataFrame by grouping according to specified
        columns and removing outliers within each group independently. Outliers are
        identified using the IQR method and filtered out. The original data is stored
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
            >>> import pandas as pd
            >>> import numpy as np
            >>> from phenotypic.analysis import TukeyOutlierRemover
            >>> 
            >>> # Create sample data
            >>> np.random.seed(42)
            >>> data = pd.DataFrame({
            ...     'ImageName': ['img1'] * 100,
            ...     'Area': np.concatenate([
            ...         np.random.normal(200, 30, 98),
            ...         [500, 50]  # outliers
            ...     ])
            ... })
            >>> 
            >>> # Remove outliers
            >>> detector = TukeyOutlierRemover(
            ...     on='Area',
            ...     groupby=['ImageName'],
            ...     k=1.5
            ... )
            >>> filtered_data = detector.analyze(data)
            >>> 
            >>> # Check results
            >>> print(f"Original: {len(data)} rows, Filtered: {len(filtered_data)} rows")
            >>> print(f"Removed {len(data) - len(filtered_data)} outliers")
        
        Notes:
            - Stores original data in self._original_data for visualization
            - Stores filtered results in self._latest_measurements for retrieval
            - Groups are processed independently with their own fences
            - NaN values in measurement column are preserved in output
        """
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
        config = {
            'k' : self.k,
            'on': self.on
        }

        # Apply outlier removal to each group
        # Create groups
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

    def show(self,
             figsize: tuple[int, int] | None = None,
             max_groups: int = 20,
             collapsed: bool = True,
             criteria: dict[str, any] | None = None
             ) -> (plt.Figure, plt.Axes):
        """Visualize outlier detection results.
        
        Creates a visualization showing the distribution of values with outliers highlighted
        and fence boundaries displayed. Can display as individual subplots or as a collapsed
        stacked view with all groups in a single plot. Outlier flags are computed dynamically
        for visualization only.
        
        Args:
            figsize: Figure size as (width, height). If None, automatically determined
                based on number of groups and mode.
            max_groups: Maximum number of groups to display. If there are more groups,
                only the first max_groups will be shown. Default is 20.
            collapsed: If True, show all groups stacked vertically in a single plot.
                If False, show each group in its own subplot. Default is False.
            criteria: Optional dictionary specifying filtering criteria for data selection.
                When provided, only groups matching the criteria will be displayed.
                Format: {'column_name': value} or {'column_name': [value1, value2]}.
                Default is None (show all groups).
        
        Returns:
            Tuple of (Figure, Axes) containing the visualization.
        
        Raises:
            ValueError: If analyze() has not been called yet (no results to display).
            KeyError: If criteria references columns not present in the data.
        
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from phenotypic.analysis import TukeyOutlierRemover
            >>> 
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
            >>> 
            >>> # Remove outliers and visualize all groups
            >>> detector = TukeyOutlierRemover(
            ...     on='Area',
            ...     groupby=['Plate', 'ImageName'],
            ...     k=1.5
            ... )
            >>> results = detector.analyze(data)
            >>> fig, axes = detector.show(figsize=(12, 5))
            >>> 
            >>> # Visualize only specific plate
            >>> fig, axes = detector.show(criteria={'Plate': 'P1'})
            >>> 
            >>> # Visualize specific images across plates using collapsed view
            >>> fig, ax = detector.show(criteria={'ImageName': 'img1'}, collapsed=True)
        
        Notes:
            Individual mode (collapsed=False):
            - Each group gets its own subplot with box plot
            - Outliers shown in red, normal values in blue
            - Horizontal lines show fence boundaries
            
            Collapsed mode (collapsed=True):
            - All groups stacked vertically in single plot
            - Each group shown as horizontal line with median marker
            - Vertical bars show fence boundaries
            - Normal points as circles, outliers as diamonds
            - More compact for comparing many groups
            
            Filtering with criteria:
            - Only groups matching all criteria are displayed
            - Useful for focusing on specific plates, conditions, or subsets
            - Can be combined with both individual and collapsed modes
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
            data['_group_key'] = data[self.groupby].astype(str).agg(' | '.join, axis=1)
            groups = data['_group_key'].unique()
            group_col = '_group_key'

        # Limit number of groups if needed
        if len(groups) > max_groups:
            groups = groups[:max_groups]
            print(f"Warning: Displaying only first {max_groups} of {len(data[group_col].unique())} groups")

        # Branch based on visualization mode
        if collapsed:
            return self._show_collapsed(data, groups, group_col, figsize)
        else:
            return self._show_individual(data, groups, group_col, figsize)

    def _show_individual(self, data: pd.DataFrame, groups, group_col: str,
                         figsize: tuple[int, int] | None) -> (plt.Figure, plt.Axes):
        """Create individual subplots for each group."""
        # Calculate layout
        n_groups = len(groups)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1)//n_cols

        # Set figure size
        if figsize is None:
            figsize = (5*n_cols, 4*n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        total_outliers = 0
        total_count = 0

        # Plot each group
        for idx, group_name in enumerate(groups):
            ax = axes[idx]
            group_data = data[data[group_col] == group_name].copy()

            # Dynamically compute outlier flags for this group
            values = group_data[self.on].values
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_fence = q1 - (iqr*self.k)
            upper_fence = q3 + (iqr*self.k)

            is_outlier = (values < lower_fence) | (values > upper_fence)
            group_data['_is_outlier'] = is_outlier

            # Separate inliers and outliers
            inliers = group_data[~group_data['_is_outlier']]
            outliers = group_data[group_data['_is_outlier']]

            total_outliers += len(outliers)
            total_count += len(group_data)

            # Create x-coordinates for scatter plot
            x_inliers = np.random.normal(1, 0.04, len(inliers))
            x_outliers = np.random.normal(1, 0.04, len(outliers))

            # Plot inliers
            if len(inliers) > 0:
                ax.scatter(x_inliers, inliers[self.on].values,
                           alpha=0.6, s=40, c='#2E86AB', label='Normal', zorder=3)

            # Plot outliers
            if len(outliers) > 0:
                ax.scatter(x_outliers, outliers[self.on].values,
                           alpha=0.8, s=50, c='#E63946', marker='D',
                           label='Outlier', zorder=4)

            # Create box plot
            bp = ax.boxplot([values], positions=[1], widths=0.3,
                            patch_artist=True, showfliers=False,
                            boxprops=dict(facecolor='lightgray', alpha=0.3),
                            medianprops=dict(color='black', linewidth=2))

            # Add fence lines
            ax.axhline(y=lower_fence, color='#F4A261', linestyle='--',
                       linewidth=1.5, label='Lower Fence', zorder=2)
            ax.axhline(y=upper_fence, color='#F4A261', linestyle='--',
                       linewidth=1.5, label='Upper Fence', zorder=2)

            # Formatting
            ax.set_title(f'{group_name}\n({len(outliers)} outliers / {len(group_data)} total)',
                         fontsize=10, fontweight='bold')
            ax.set_ylabel(self.on, fontsize=9)
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')

            # Add legend only to first subplot
            if idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                # Remove duplicate labels
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(),
                          loc='best', fontsize=8, framealpha=0.9)

        # Hide unused subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].set_visible(False)

        # Overall title
        outlier_pct = 100*total_outliers/total_count if total_count > 0 else 0

        fig.suptitle(
                f'Tukey Outlier Detection (k={self.k})\n'
                f'{total_outliers} outliers detected ({outlier_pct:.1f}% of {total_count} measurements)',
                fontsize=14, fontweight='bold', y=0.995
        )

        plt.tight_layout()

        return fig, axes

    def _show_collapsed(self, data: pd.DataFrame, groups, group_col: str,
                        figsize: tuple[int, int] | None) -> (plt.Figure, plt.Axes):
        """Create collapsed stacked view with all groups in single plot."""
        n_groups = len(groups)

        # Set figure size
        if figsize is None:
            figsize = (10, max(6, 0.5*n_groups + 2))

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        total_outliers = 0
        total_count = 0
        
        added_labels = set()

        # Process each group and plot stacked vertically
        for idx, group_name in enumerate(groups):
            y_pos = n_groups - idx  # Stack from top to bottom
            group_data = data[data[group_col] == group_name].copy()

            # Dynamically compute outlier flags for this group
            values = group_data[self.on].values
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            median = np.percentile(values, 50)
            lower_fence = q1 - (iqr*self.k)
            upper_fence = q3 + (iqr*self.k)

            is_outlier = (values < lower_fence) | (values > upper_fence)
            group_data['_is_outlier'] = is_outlier

            # Separate inliers and outliers
            inliers = group_data[~group_data['_is_outlier']]
            outliers = group_data[group_data['_is_outlier']]

            total_outliers += len(outliers)
            total_count += len(group_data)

            # Draw horizontal line for full data range
            data_min = values.min()
            data_max = values.max()
            ax.hlines(y_pos, data_min, data_max, colors='lightgray',
                      linewidth=1.5, alpha=0.6, zorder=1)

            # Add vertical tick marks for fences and mean
            tick_height = 0.15  # Height of tick marks

            # Lower fence tick
            lbl = 'Fences'
            if lbl not in added_labels:
                added_labels.add(lbl)
            else:
                lbl = None
                
            ax.plot([lower_fence, lower_fence], [y_pos - tick_height, y_pos + tick_height],
                    color='#F4A261', linewidth=2.5, linestyle='-',
                    label=lbl, zorder=3)

            # Upper fence tick
            ax.plot([upper_fence, upper_fence], [y_pos - tick_height, y_pos + tick_height],
                    color='#F4A261', linewidth=2.5, linestyle='-', zorder=3)

            # Median marker
            lbl = 'Median'
            if lbl not in added_labels:
                added_labels.add(lbl)
            else:
                lbl = None
            
            ax.plot([median, median], [y_pos - tick_height, y_pos + tick_height],
                    color='black', linewidth=2.5, linestyle='-',
                    label=lbl, zorder=3)

            # Create y-coordinates with jitter for scatter plot
            y_inliers = np.random.normal(y_pos, 0.06, len(inliers))
            y_outliers = np.random.normal(y_pos, 0.06, len(outliers))

            # Plot inliers
            if len(inliers) > 0:
                lbl = 'Normal'
                if lbl not in added_labels:
                    added_labels.add(lbl)
                else:
                    lbl = None
                ax.scatter(inliers[self.on].values, y_inliers,
                           alpha=0.6, s=30, c='#2E86AB', label=lbl, zorder=4)

            # Plot outliers
            if len(outliers) > 0:
                lbl = 'Outlier'
                if lbl not in added_labels:
                    added_labels.add(lbl)
                else:
                    lbl = None
                ax.scatter(outliers[self.on].values, y_outliers,
                           alpha=0.8, s=35, c='#E63946', marker='D',
                           label=lbl, zorder=5)

        # Formatting
        ax.set_yticks(range(1, n_groups + 1))
        ax.set_yticklabels(groups[::-1])  # Reverse to match top-to-bottom order
        ax.set_xlabel(self.on, fontsize=11, fontweight='bold')
        ax.set_ylabel('Group', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        ax.set_ylim(0.5, n_groups + 0.5)

        # Add legend
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

        # Overall title
        outlier_pct = 100*total_outliers/total_count if total_count > 0 else 0
        fig.suptitle(
                f'Tukey Outlier Detection (k={self.k})\n'
                f'{total_outliers} outliers detected ({outlier_pct:.1f}% of {total_count} measurements)',
                fontsize=13, fontweight='bold'
        )

        plt.tight_layout()

        return fig, ax

    def results(self) -> pd.DataFrame:
        """Return the filtered results (outliers removed).
        
        Returns the DataFrame with outliers removed from the most recent call to analyze().
        
        Returns:
            DataFrame with outliers filtered out. Contains only the original columns
            without additional outlier flag columns. If analyze() has not been called,
            returns an empty DataFrame.
        
        Examples:
            >>> detector = TukeyOutlierRemover(
            ...     on='Area',
            ...     groupby=['ImageName']
            ... )
            >>> filtered_data = detector.analyze(data)
            >>> results_copy = detector.results()  # Same as filtered_data
            >>> assert results_copy.equals(filtered_data)
            >>> 
            >>> # Check how many rows were removed
            >>> num_removed = len(data) - len(filtered_data)
            >>> print(f"Removed {num_removed} outliers")
        
        Notes:
            - Returns the DataFrame stored in self._latest_measurements
            - Contains only inliers (outliers have been removed)
            - Use this method to retrieve results after calling analyze()
        """
        return self._latest_measurements

    @staticmethod
    def _apply2group_func(key, group: pd.DataFrame, on: str, k: float) -> pd.DataFrame:
        """
        Applies Tukey's outlier removal on a DataFrame group.

        This static method filters out rows in the DataFrame group where the values of a specific
        column (`on`) are considered outliers. Outliers are determined by the provided multiplier
        for the IQR. Rows with values outside the range defined by the lower and upper thresholds
        (calculated using the IQR method) are excluded.

        Args:
            key: The group key (not used but required for joblib).
            group: A group of DataFrame rows to which the IQR-based filtering is applied.
            on: The column in the DataFrame on which the IQR thresholding is computed.
            k: The factor by which the IQR is multiplied to determine the
                threshold for identifying outlier rows in the DataFrame group.

        Returns:
            Filtered DataFrame containing rows that fall within the calculated IQR thresholds.
        """
        values = group[on]
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_fence = q1 - (iqr*k)
        upper_fence = q3 + (iqr*k)

        return group[(values >= lower_fence) & (values <= upper_fence)]
