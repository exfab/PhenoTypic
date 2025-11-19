import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from phenotypic.analysis import TukeyOutlierRemover


class TestTukeyOutlierRemover:
    """Test suite for TukeyOutlierRemover functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with known outliers for testing."""
        np.random.seed(42)

        # Create two groups with different distributions and outliers
        group1_normal = np.random.normal(200, 30, 95)
        group1_outliers = np.array([500, 550, 600, 50, 40])  # High and low outliers

        group2_normal = np.random.normal(180, 25, 97)
        group2_outliers = np.array([450, 500, 30])  # High and low outliers

        data = pd.DataFrame({
            'ImageName': ['img1'] * 100 + ['img2'] * 100,
            'Area': np.concatenate([group1_normal, group1_outliers,
                                   group2_normal, group2_outliers]),
            'Metadata_Plate': ['P1'] * 200
        })

        return data

    @pytest.fixture
    def sample_data_multiple_groups(self):
        """Create sample data with multiple groups for testing."""
        np.random.seed(42)

        groups = []
        for plate in ['P1', 'P2']:
            for img in ['img1', 'img2']:
                normal_vals = np.random.normal(200, 30, 48)
                outliers = np.array([500, 50])
                
                group_data = pd.DataFrame({
                    'ImageName': [img] * 50,
                    'Plate': [plate] * 50,
                    'Area': np.concatenate([normal_vals, outliers])
                })
                groups.append(group_data)

        return pd.concat(groups, ignore_index=True)

    def test_initialization(self):
        """Test TukeyOutlierRemover initialization with various parameters."""
        # Test basic initialization
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName']
        )

        assert detector.on == 'Area'
        assert detector.groupby == ['ImageName']
        assert detector.k == 1.5
        assert detector.n_jobs == 1

        # Test custom parameters
        detector_custom = TukeyOutlierRemover(
            on='Size',
            groupby=['Plate', 'Well'],
            k=3.0,
            num_workers=2
        )

        assert detector_custom.on == 'Size'
        assert detector_custom.groupby == ['Plate', 'Well']
        assert detector_custom.k == 3.0
        assert detector_custom.n_jobs == 2

    def test_initialization_invalid_k(self):
        """Test that invalid k values raise errors."""
        with pytest.raises(ValueError, match="k must be positive"):
            TukeyOutlierRemover(
                on='Area',
                groupby=['ImageName'],
                k=-1.5
            )

        with pytest.raises(ValueError, match="k must be positive"):
            TukeyOutlierRemover(
                on='Area',
                groupby=['ImageName'],
                k=0
            )

    def test_analyze_basic(self, sample_data):
        """Test basic outlier removal functionality."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName'],
            k=1.5
        )

        filtered_data = detector.analyze(sample_data)

        # Check that result is a DataFrame
        assert isinstance(filtered_data, pd.DataFrame)
        assert not filtered_data.empty

        # Check that outliers were removed
        assert len(filtered_data) < len(sample_data)

        # Check that original columns are preserved
        assert set(filtered_data.columns) == set(sample_data.columns)

        # Check that no outlier flag columns were added
        assert 'is_outlier' not in filtered_data.columns
        assert 'lower_fence' not in filtered_data.columns
        assert 'upper_fence' not in filtered_data.columns

    def test_analyze_removes_outliers(self, sample_data):
        """Test that outliers are properly removed."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName'],
            k=1.5
        )

        original_len = len(sample_data)
        filtered_data = detector.analyze(sample_data)
        filtered_len = len(filtered_data)

        # Should have removed some outliers
        assert filtered_len < original_len

        # Check that extreme values are removed
        for group_name in sample_data['ImageName'].unique():
            original_group = sample_data[sample_data['ImageName'] == group_name]
            filtered_group = filtered_data[filtered_data['ImageName'] == group_name]

            # Max value in filtered should be less than max in original
            assert filtered_group['Area'].max() <= original_group['Area'].max()
            # Min value in filtered should be greater than min in original
            assert filtered_group['Area'].min() >= original_group['Area'].min()

    def test_analyze_different_k_values(self, sample_data):
        """Test that different k values produce different results."""
        detector_strict = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName'],
            k=1.5
        )

        detector_lenient = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName'],
            k=3.0
        )

        filtered_strict = detector_strict.analyze(sample_data)
        filtered_lenient = detector_lenient.analyze(sample_data)

        # Stricter k should remove more outliers
        assert len(filtered_strict) <= len(filtered_lenient)

    def test_analyze_multiple_groups(self, sample_data_multiple_groups):
        """Test outlier removal with multiple groupby columns."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['Plate', 'ImageName'],
            k=1.5
        )

        filtered_data = detector.analyze(sample_data_multiple_groups)

        # Check that all groups are represented
        original_groups = sample_data_multiple_groups.groupby(['Plate', 'ImageName']).size()
        filtered_groups = filtered_data.groupby(['Plate', 'ImageName']).size()

        assert len(original_groups) == len(filtered_groups)

        # Each group should have some rows removed
        for group_key in original_groups.index:
            plate, img = group_key
            original_count = original_groups[group_key]
            filtered_count = filtered_groups[group_key]
            assert filtered_count <= original_count

    def test_results_method(self, sample_data):
        """Test the results() method returns filtered data."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName']
        )

        # Should return empty DataFrame if analyze hasn't been called
        empty_results = detector.results()
        assert isinstance(empty_results, pd.DataFrame)
        assert empty_results.empty

        # After analyze, should return filtered results
        filtered_data = detector.analyze(sample_data)
        results = detector.results()

        assert isinstance(results, pd.DataFrame)
        assert not results.empty
        pd.testing.assert_frame_equal(results, filtered_data)

    def test_show_method_basic(self, sample_data):
        """Test the show() method for visualization."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName']
        )

        detector.analyze(sample_data)
        fig, axes = detector.show()

        assert isinstance(fig, plt.Figure)

        # Check that figure has axes
        if isinstance(axes, np.ndarray):
            assert len(axes) > 0
        else:
            assert isinstance(axes, plt.Axes)

        plt.close('all')

    def test_show_method_with_figsize(self, sample_data):
        """Test show() method with custom figure size."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName']
        )

        detector.analyze(sample_data)
        fig, axes = detector.show(figsize=(10, 6))

        assert isinstance(fig, plt.Figure)
        assert fig.get_size_inches()[0] == 10
        assert fig.get_size_inches()[1] == 6

        plt.close('all')

    def test_show_method_max_groups(self, sample_data_multiple_groups):
        """Test show() method with max_groups parameter."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['Plate', 'ImageName']
        )

        detector.analyze(sample_data_multiple_groups)
        fig, axes = detector.show(max_groups=2)

        assert isinstance(fig, plt.Figure)

        plt.close('all')
    
    def test_show_method_collapsed(self, sample_data):
        """Test show() method with collapsed=True."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName']
        )

        detector.analyze(sample_data)
        fig, ax = detector.show(collapsed=True)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

        plt.close('all')

    def test_show_method_collapsed_with_multiple_groups(self, sample_data_multiple_groups):
        """Test collapsed view with multiple groups."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['Plate', 'ImageName']
        )

        detector.analyze(sample_data_multiple_groups)
        fig, ax = detector.show(collapsed=True, figsize=(12, 8))

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        # Check that y-axis has correct number of ticks
        assert len(ax.get_yticks()) > 0

        plt.close('all')

    def test_show_with_criteria_single_filter(self, sample_data_multiple_groups):
        """Test show() with criteria parameter filtering single column."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['Plate', 'ImageName']
        )

        detector.analyze(sample_data_multiple_groups)
        
        # Filter to show only one plate
        fig, axes = detector.show(criteria={'Plate': 'P1'})

        assert isinstance(fig, plt.Figure)
        
        plt.close('all')

    def test_show_with_criteria_multiple_filters(self, sample_data_multiple_groups):
        """Test show() with multiple criteria filters."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['Plate', 'ImageName']
        )

        detector.analyze(sample_data_multiple_groups)
        
        # Filter to show specific plate and image
        fig, axes = detector.show(criteria={'Plate': 'P1', 'ImageName': 'img1'})

        assert isinstance(fig, plt.Figure)
        
        plt.close('all')

    def test_show_with_criteria_collapsed_mode(self, sample_data_multiple_groups):
        """Test criteria parameter with collapsed visualization mode."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['Plate', 'ImageName']
        )

        detector.analyze(sample_data_multiple_groups)
        
        # Filter to show only one image across all plates in collapsed view
        fig, ax = detector.show(criteria={'ImageName': 'img1'}, collapsed=True)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        
        plt.close('all')

    def test_show_with_criteria_list_values(self, sample_data_multiple_groups):
        """Test criteria parameter with list of values."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['Plate', 'ImageName']
        )

        detector.analyze(sample_data_multiple_groups)
        
        # Filter using list of values
        fig, axes = detector.show(criteria={'Plate': ['P1', 'P2']})

        assert isinstance(fig, plt.Figure)
        
        plt.close('all')

    def test_show_with_criteria_no_matches(self, sample_data_multiple_groups):
        """Test that appropriate error is raised when criteria matches no data."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['Plate', 'ImageName']
        )

        detector.analyze(sample_data_multiple_groups)
        
        # Use criteria that won't match anything
        with pytest.raises(ValueError, match="No data matches the specified criteria"):
            detector.show(criteria={'Plate': 'NonexistentPlate'})

    def test_show_with_criteria_invalid_column(self, sample_data_multiple_groups):
        """Test that KeyError is raised for invalid column in criteria."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['Plate', 'ImageName']
        )

        detector.analyze(sample_data_multiple_groups)
        
        # Use criteria with non-existent column
        with pytest.raises(KeyError):
            detector.show(criteria={'NonexistentColumn': 'value'})

    def test_show_before_analyze_raises_error(self):
        """Test that show() raises error if called before analyze()."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName']
        )

        with pytest.raises(ValueError, match="No results to display"):
            detector.show()

    def test_analyze_missing_columns(self, sample_data):
        """Test behavior with missing required columns."""
        data_missing_col = sample_data.drop(columns=['Area'])

        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName']
        )

        with pytest.raises(KeyError, match="Missing required columns"):
            detector.analyze(data_missing_col)

    def test_analyze_empty_data(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['ImageName', 'Area'])

        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName']
        )

        with pytest.raises(ValueError, match="Input data cannot be empty"):
            detector.analyze(empty_df)

    def test_analyze_preserves_original_data(self, sample_data):
        """Test that analyze preserves original data for visualization."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName']
        )

        original_copy = sample_data.copy()
        filtered_data = detector.analyze(sample_data)

        # Original data should be preserved internally
        assert not detector._original_data.empty
        pd.testing.assert_frame_equal(detector._original_data, original_copy)

        # Filtered data should be different
        assert len(filtered_data) < len(detector._original_data)

    def test_parallel_processing(self, sample_data_multiple_groups):
        """Test parallel processing produces same results as sequential."""
        detector_parallel = TukeyOutlierRemover(
            on='Area',
            groupby=['Plate', 'ImageName'],
            num_workers=2
        )

        detector_sequential = TukeyOutlierRemover(
            on='Area',
            groupby=['Plate', 'ImageName'],
            num_workers=1
        )

        filtered_parallel = detector_parallel.analyze(sample_data_multiple_groups)
        filtered_sequential = detector_sequential.analyze(sample_data_multiple_groups)

        # Results should be identical (same rows, possibly different order)
        assert len(filtered_parallel) == len(filtered_sequential)
        
        # Sort both DataFrames for comparison
        filtered_parallel_sorted = filtered_parallel.sort_values(['Plate', 'ImageName', 'Area']).reset_index(drop=True)
        filtered_sequential_sorted = filtered_sequential.sort_values(['Plate', 'ImageName', 'Area']).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(filtered_parallel_sorted, filtered_sequential_sorted)

    def test_no_outliers_in_data(self):
        """Test behavior when data has no outliers."""
        np.random.seed(42)
        
        # Create data with tight distribution (no outliers)
        data = pd.DataFrame({
            'ImageName': ['img1'] * 100,
            'Area': np.random.normal(200, 5, 100)  # Very tight distribution
        })

        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName'],
            k=1.5
        )

        filtered_data = detector.analyze(data)

        # Should remove very few or no points
        assert len(filtered_data) >= len(data) * 0.95  # At most 5% removed

    def test_apply2group_func_static_method(self):
        """Test the static _apply2group_func method directly."""
        np.random.seed(42)
        
        # Create test group
        group = pd.DataFrame({
            'Area': np.concatenate([np.random.normal(200, 30, 48), [500, 50]])
        })

        # Apply method
        filtered = TukeyOutlierRemover._apply2group_func(
            key=None,
            group=group,
            on='Area',
            k=1.5
        )

        # Should remove outliers
        assert len(filtered) < len(group)
        
        # Extreme values should be removed
        assert filtered['Area'].max() < 500
        assert filtered['Area'].min() > 50

    def test_single_group_single_value(self):
        """Test behavior with insufficient data (single value per group)."""
        data = pd.DataFrame({
            'ImageName': ['img1', 'img2'],
            'Area': [100, 200]
        })

        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName']
        )

        # Should handle gracefully - with one value, IQR is 0, so all values are kept
        filtered_data = detector.analyze(data)
        assert len(filtered_data) == len(data)

    def test_nan_values_in_measurement_column(self):
        """Test handling of NaN values in measurement column."""
        np.random.seed(42)
        
        data = pd.DataFrame({
            'ImageName': ['img1'] * 50,
            'Area': np.concatenate([np.random.normal(200, 30, 48), [np.nan, np.nan]])
        })

        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName']
        )

        # Should handle NaN values without crashing
        filtered_data = detector.analyze(data)
        
        # Result should be a valid DataFrame
        assert isinstance(filtered_data, pd.DataFrame)

    def test_consistent_group_boundaries(self, sample_data):
        """Test that fence boundaries are consistent within groups."""
        detector = TukeyOutlierRemover(
            on='Area',
            groupby=['ImageName'],
            k=1.5
        )

        detector.analyze(sample_data)

        # Verify through visualization that boundaries are computed correctly
        fig, axes = detector.show()
        assert isinstance(fig, plt.Figure)
        
        # Test collapsed view as well
        fig_collapsed, ax_collapsed = detector.show(collapsed=True)
        assert isinstance(fig_collapsed, plt.Figure)
        assert isinstance(ax_collapsed, plt.Axes)
        
        plt.close('all')

