from __future__ import annotations

import numpy as np
import pandas as pd

from .abc_ import SetAnalyzer


class TukeyOutlierDetector(SetAnalyzer):
    """Analyzer for detecting outliers using Tukey's fence method.
    
    This class identifies outliers in measurement data by applying Tukey's fence test
    within groups. The method calculates the interquartile range (IQR) and flags values
    that fall outside Q1 - k*IQR or Q3 + k*IQR, where k is a tunable multiplier
    (typically 1.5 for outliers or 3.0 for extreme outliers).
    
    Args:
        on: Column name for grouping/aggregation operations.
        groupby: List of column names to group by (e.g., ['ImageName', 'Metadata_Plate']).
        measurement_col: Name of measurement column to test for outliers (e.g., 'Area', 'MeanRadius').
        k: IQR multiplier for fence calculation. Default is 1.5 (standard outliers).
            Use 3.0 for extreme outliers only.
        agg_func: Aggregation function for parent class. Default is 'mean'.
        num_workers: Number of parallel workers. Default is 1.
    
    Attributes:
        measurement_col: Column to test for outliers.
        k: IQR multiplier used for fence calculation.
        
    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from phenotypic.analysis import TukeyOutlierDetector
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
        >>> detector = TukeyOutlierDetector(
        ...     on='Area',
        ...     groupby=['ImageName'],
        ...     measurement_col='Area',
        ...     k=1.5
        ... )
        >>> 
        >>> # Detect outliers
        >>> results = detector.analyze(data)
        >>> 
        >>> # View results with outlier flags
        >>> outliers = results[results['is_outlier']]
    """

    def __init__(
            self,
            on: str,
            groupby: list[str],
            k: float = 1.5,
            agg_func: str = 'mean',
            num_workers: int = 1
    ):
        """Initialize TukeyOutlierDetector with test parameters.
        
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
        super().__init__(on=on, groupby=groupby, agg_func=agg_func, num_workers=num_workers)

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        self.k = k
        self._original_data: pd.DataFrame = pd.DataFrame()

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze data and flag outliers using Tukey's fence method.
        
        This method processes the input DataFrame by grouping according to specified
        columns and identifying outliers within each group independently. Outliers are
        flagged but not removed, allowing downstream analysis to decide how to handle them.
        
        Args:
            data: DataFrame containing measurement data. Must include all columns
                specified in self.groupby and self.on.
        
        Returns:
            DataFrame with added columns:
                - 'is_outlier': Boolean flag indicating outlier status
                - 'lower_fence': Lower bound for this group
                - 'upper_fence': Upper bound for this group
                - 'outlier_type': 'low', 'high', or None indicating outlier direction
        
        Raises:
            KeyError: If required columns are missing from input DataFrame.
            ValueError: If data is empty or malformed.
        
        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from phenotypic.analysis import TukeyOutlierDetector
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
            >>> # Detect outliers
            >>> detector = TukeyOutlierDetector(
            ...     on='Area',
            ...     groupby=['ImageName'],
            ...     measurement_col='Area',
            ...     k=1.5
            ... )
            >>> results = detector.analyze(data)
            >>> 
            >>> # Check outliers
            >>> print(f"Found {results['is_outlier'].sum()} outliers")
            >>> outliers = results[results['is_outlier']]
        
        Notes:
            - Stores original data in self._original_data for comparison
            - Stores results in self._latest_measurements for retrieval
            - Groups are processed independently with their own fences
            - NaN values in measurement column are not flagged as outliers
        """
        # Validate input
        if data is None or len(data) == 0:
            raise ValueError("Input data cannot be empty")

        # Store original data for comparison
        self._original_data = data.copy()

        # Check required columns
        required_cols = set(self.groupby + [self.on])
        missing_cols = required_cols - set(data.columns)

        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")

        # Prepare configuration for _apply2group_func
        config = {
            'k'              : self.k,
            'measurement_col': self.on
        }

        # Apply outlier detection to each group
        if len(self.groupby) > 0:
            # Suppress FutureWarning by explicitly including groups
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning,
                                        message='.*grouping columns.*')
                results_data = data.groupby(
                        self.groupby,
                        group_keys=False,
                        observed=True
                ).apply(
                        lambda group: self._apply2group_func(group, config)
                )
        else:
            # No grouping - apply to entire dataset
            results_data = self._apply2group_func(data, config)

        # Store results
        self._latest_measurements = results_data.reset_index(drop=True)

        return self._latest_measurements

    def show(self, figsize: tuple[int, int] = (14, 5)):
        """Display visualization of outlier detection results.
        
        Creates a multi-panel figure showing:
        1. Box plot with outliers highlighted
        2. Scatter plot of values with outliers marked
        3. Summary statistics
        
        Args:
            figsize: Figure size as (width, height) in inches. Default is (14, 5).
        
        Raises:
            RuntimeError: If analyze() has not been called yet.
        
        Examples:
            >>> detector = TukeyOutlierDetector(
            ...     on='Area',
            ...     groupby=['ImageName'],
            ...     measurement_col='Area'
            ... )
            >>> detector.analyze(data)
            >>> detector.show()  # Display outlier plots
        
        Notes:
            - Requires analyze() to be called first to populate data
            - Shows distribution with outliers highlighted in red
            - Displays fence boundaries and summary statistics
            - For multiple groups, shows aggregate view across all groups
        """
        import matplotlib.pyplot as plt

        # Check if analyze has been called
        if self._latest_measurements.empty or self._original_data.empty:
            raise RuntimeError("No data to display. Call analyze() first.")

        # Get measurement column
        meas_col = self.on

        # Get outlier information
        is_outlier = self._latest_measurements['is_outlier']
        values = self._latest_measurements[meas_col]

        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: Box plot with outliers
        ax1 = axes[0]

        # Separate inliers and outliers for plotting
        inliers = values[~is_outlier]
        outliers = values[is_outlier]

        # Create box plot for inliers
        bp = ax1.boxplot([inliers.dropna()], positions=[1], widths=0.6,
                         patch_artist=True, showfliers=False)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_edgecolor('black')

        # Overlay outliers as red points
        if len(outliers) > 0:
            ax1.scatter(
                    np.ones(len(outliers)),
                    outliers,
                    color='red',
                    s=50,
                    alpha=0.7,
                    marker='x',
                    label='Outliers',
                    zorder=3
            )

        ax1.set_ylabel(meas_col)
        ax1.set_xticks([1])
        ax1.set_xticklabels(['All Groups'])
        ax1.set_title(f'Distribution: {meas_col}')
        if len(outliers) > 0:
            ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Panel 2: Scatter plot with outliers highlighted
        ax2 = axes[1]

        # Plot all points
        ax2.scatter(
                self._latest_measurements.index[~is_outlier],
                values[~is_outlier],
                alpha=0.5,
                s=30,
                color='blue',
                label='Inliers'
        )

        # Highlight outliers
        if is_outlier.any():
            # Color by outlier type
            high_outliers = is_outlier & (self._latest_measurements['outlier_type'] == 'high')
            low_outliers = is_outlier & (self._latest_measurements['outlier_type'] == 'low')

            if high_outliers.any():
                ax2.scatter(
                        self._latest_measurements.index[high_outliers],
                        values[high_outliers],
                        alpha=0.7,
                        s=80,
                        color='red',
                        marker='^',
                        label='High outliers'
                )

            if low_outliers.any():
                ax2.scatter(
                        self._latest_measurements.index[low_outliers],
                        values[low_outliers],
                        alpha=0.7,
                        s=80,
                        color='orange',
                        marker='v',
                        label='Low outliers'
                )

        ax2.set_xlabel('Row Index')
        ax2.set_ylabel(meas_col)
        ax2.set_title('Outlier Detection Results')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add summary statistics as text
        n_outliers = is_outlier.sum()
        n_total = len(values)
        pct_outliers = 100*n_outliers/n_total if n_total > 0 else 0

        high_count = (self._latest_measurements['outlier_type'] == 'high').sum()
        low_count = (self._latest_measurements['outlier_type'] == 'low').sum()

        summary_text = (
            f"Outliers: {n_outliers}/{n_total} ({pct_outliers:.1f}%) | "
            f"High: {high_count}, Low: {low_count}\n"
            f"k = {self.k} | "
            f"Median: {values.median():.2f} | "
            f"Mean: {values.mean():.2f}"
        )

        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout(rect=[0, 0.08, 1, 1])
        plt.show()

    def results(self) -> pd.DataFrame:
        """Return the DataFrame with outlier detection results.
        
        Returns the DataFrame with outlier flags and fence information from the most
        recent call to analyze().
        
        Returns:
            DataFrame with original data plus outlier detection columns:
                - 'is_outlier': Boolean flag
                - 'lower_fence': Lower bound
                - 'upper_fence': Upper bound
                - 'outlier_type': Direction of outlier ('low', 'high', or None)
            If analyze() has not been called, returns an empty DataFrame.
        
        Examples:
            >>> detector = TukeyOutlierDetector(
            ...     on='Area',
            ...     groupby=['ImageName'],
            ...     measurement_col='Area'
            ... )
            >>> results = detector.analyze(data)
            >>> results_copy = detector.results()  # Same as results
            >>> assert results_copy.equals(results)
            >>> 
            >>> # Filter to outliers only
            >>> outliers_only = results[results['is_outlier']]
        
        Notes:
            - Returns the DataFrame stored in self._latest_measurements
            - Contains original columns plus outlier detection information
            - Use this method to retrieve results after calling analyze()
        """
        return self._latest_measurements

    @staticmethod
    def _apply2group_func(group: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply Tukey's outlier test to a single group of data.
        
        This method calculates quartiles and IQR for the group, determines fence
        boundaries, and flags values falling outside these boundaries as outliers.
        
        Args:
            group: DataFrame containing measurement data for a single group.
            config: Configuration dictionary containing:
                - 'k': IQR multiplier for fence calculation
                - 'measurement_col': Column name to test
        
        Returns:
            DataFrame with added columns for outlier detection:
                - 'is_outlier': Boolean flag
                - 'lower_fence': Lower boundary
                - 'upper_fence': Upper boundary
                - 'outlier_type': 'low', 'high', or None
        
        Notes:
            - NaN values in measurement column are flagged as non-outliers
            - If IQR is 0 (all values identical), no values are flagged as outliers
            - Fences are constant within a group but may vary between groups
        
        Examples:
            >>> config = {'k': 1.5, 'measurement_col': 'Area'}
            >>> group_results = TukeyOutlierDetector._apply2group_func(group, config)
        """
        # Extract configuration
        k = config['k']
        measurement_col = config['measurement_col']

        # Make a copy to avoid modifying the original
        group = group.copy()

        # Handle empty groups
        if len(group) == 0:
            group['is_outlier'] = pd.Series(dtype=bool)
            group['lower_fence'] = pd.Series(dtype=float)
            group['upper_fence'] = pd.Series(dtype=float)
            group['outlier_type'] = pd.Series(dtype=object)
            return group

        # Check if measurement column exists
        if measurement_col not in group.columns:
            group['is_outlier'] = False
            group['lower_fence'] = np.nan
            group['upper_fence'] = np.nan
            group['outlier_type'] = None
            return group

        # Get values, excluding NaN
        values = group[measurement_col]
        valid_mask = values.notna()

        # Initialize output columns
        group['is_outlier'] = False
        group['lower_fence'] = np.nan
        group['upper_fence'] = np.nan
        group['outlier_type'] = None

        # Need at least some valid values to compute quartiles
        if valid_mask.sum() < 2:
            return group

        # Calculate quartiles and IQR
        Q1 = values[valid_mask].quantile(0.25)
        Q3 = values[valid_mask].quantile(0.75)
        IQR = Q3 - Q1

        # Calculate fences
        lower_fence = Q1 - k*IQR
        upper_fence = Q3 + k*IQR

        # Store fence values for all rows in this group
        group['lower_fence'] = lower_fence
        group['upper_fence'] = upper_fence

        # Flag outliers (only for valid values)
        is_low_outlier = (values < lower_fence) & valid_mask
        is_high_outlier = (values > upper_fence) & valid_mask

        group.loc[is_low_outlier, 'is_outlier'] = True
        group.loc[is_high_outlier, 'is_outlier'] = True

        group.loc[is_low_outlier, 'outlier_type'] = 'low'
        group.loc[is_high_outlier, 'outlier_type'] = 'high'

        return group
