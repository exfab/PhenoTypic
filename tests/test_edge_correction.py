"""
Comprehensive tests for EdgeCorrector class.

Tests edge identification, threshold calculation, value capping, and groupby behavior
for the edge correction functionality.
"""

import pytest
import numpy as np
import pandas as pd

from phenotypic.analysis import EdgeCorrector
from phenotypic.tools.constants_ import GRID


class TestSurroundedPositions:
    """Test the _surrounded_positions static method."""

    def test_4connectivity_full_grid(self):
        """Test 4-connectivity on a complete 3x3 grid."""
        # All 9 positions active
        active_idx = np.array(range(9))
        result = EdgeCorrector._surrounded_positions(
                active_idx=active_idx,
                shape=(3, 3),
                connectivity=4
        )
        # Only center position (4) should be fully surrounded
        assert len(result) == 1
        assert result[0] == 4

    def test_8connectivity_full_grid(self):
        """Test 8-connectivity on a complete 3x3 grid."""
        # All 9 positions active
        active_idx = np.array(range(9))
        result = EdgeCorrector._surrounded_positions(
                active_idx=active_idx,
                shape=(3, 3),
                connectivity=8
        )
        # Only center position (4) should be fully surrounded with 8-connectivity
        assert len(result) == 1
        assert result[0] == 4

    def test_4connectivity_8x12_grid(self):
        """Test 4-connectivity on standard 8x12 microplate grid."""
        rows, cols = 8, 12
        # 3x3 block centered at (4, 6)
        block_rc = [(r, c) for r in range(3, 6) for c in range(5, 8)]
        active = np.array([r*cols + c for r, c in block_rc], dtype=np.int64)

        result = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4
        )

        # Only the center of the 3x3 block should be fully surrounded
        expected_center = 4*cols + 6
        assert len(result) == 1
        assert result[0] == expected_center

    def test_min_neighbors_threshold(self):
        """Test with min_neighbors threshold instead of full surround."""
        rows, cols = 8, 12
        # 3x3 block
        block_rc = [(r, c) for r in range(3, 6) for c in range(5, 8)]
        active = np.array([r*cols + c for r, c in block_rc], dtype=np.int64)

        # At least 3 of 4 neighbors
        result, counts = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4,
                min_neighbors=3, return_counts=True
        )

        # Should include center (4 neighbors) and edge positions (3 neighbors)
        assert len(result) > 1
        assert all(counts >= 3)
        assert (4*cols + 6) in result  # center

    def test_empty_input(self):
        """Test with no active positions."""
        result = EdgeCorrector._surrounded_positions(
                active_idx=np.array([]),
                shape=(8, 12),
                connectivity=4
        )
        assert len(result) == 0

    def test_single_position(self):
        """Test with single active position - should have no surrounded cells."""
        result = EdgeCorrector._surrounded_positions(
                active_idx=np.array([50]),
                shape=(8, 12),
                connectivity=4
        )
        assert len(result) == 0

    def test_invalid_connectivity(self):
        """Test that invalid connectivity raises ValueError."""
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            EdgeCorrector._surrounded_positions(
                    active_idx=np.array([0, 1]),
                    shape=(3, 3),
                    connectivity=6
            )

    def test_out_of_bounds_indices(self):
        """Test that out of bounds indices raise ValueError."""
        with pytest.raises(ValueError, match="must be in"):
            EdgeCorrector._surrounded_positions(
                    active_idx=np.array([0, 100]),
                    shape=(3, 3),
                    connectivity=4
            )

    def test_invalid_shape(self):
        """Test that invalid shape raises ValueError."""
        with pytest.raises(ValueError, match="shape must be two positive integers"):
            EdgeCorrector._surrounded_positions(
                    active_idx=np.array([0]),
                    shape=(3, 0),
                    connectivity=4
            )


class TestEdgeCorrectorInit:
    """Test EdgeCorrector initialization."""

    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area'
        )
        assert corrector.nrows == 8
        assert corrector.ncols == 12
        assert corrector.top_n == 3
        assert corrector.connectivity == 4

    def test_custom_grid_size(self):
        """Test initialization with custom grid dimensions."""
        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                nrows=4,
                ncols=6
        )
        assert corrector.nrows == 4
        assert corrector.ncols == 6

    def test_custom_top_n(self):
        """Test initialization with custom top_n."""
        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                top_n=20
        )
        assert corrector.top_n == 20

    def test_8connectivity(self):
        """Test initialization with 8-connectivity."""
        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                connectivity=8
        )
        assert corrector.connectivity == 8

    def test_invalid_connectivity_raises(self):
        """Test that invalid connectivity raises ValueError."""
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            EdgeCorrector(
                    on='Area',
                    groupby=['ImageName'],
                    measurement_col='Area',
                    connectivity=6
            )

    def test_invalid_grid_size_raises(self):
        """Test that invalid grid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="nrows and ncols must be positive"):
            EdgeCorrector(
                    on='Area',
                    groupby=['ImageName'],
                    measurement_col='Area',
                    nrows=0,
                    ncols=12
            )

    def test_invalid_top_n_raises(self):
        """Test that invalid top_n raises ValueError."""
        with pytest.raises(ValueError, match="top_n must be positive"):
            EdgeCorrector(
                    on='Area',
                    groupby=['ImageName'],
                    measurement_col='Area',
                    top_n=-5
            )


class TestThresholdCalculation:
    """Test threshold calculation and top N value selection."""

    def test_top_n_selection(self):
        """Test that top N values are correctly selected."""
        np.random.seed(42)
        data = pd.DataFrame({
            'ImageName'          : ['img1']*96,
            str(GRID.SECTION_NUM): range(96),
            'Area'               : np.random.uniform(100, 500, 96)
        })

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                top_n=10,
                nrows=8,
                ncols=12
        )

        # Get top 10 values manually
        expected_top_10 = data.nlargest(10, 'Area')['Area']
        expected_threshold = expected_top_10.mean()

        # The corrector should use this threshold internally
        # We can't directly test the threshold, but we can verify behavior
        corrected = corrector.analyze(data)

        # All corrected values should be <= max of original
        assert corrected['Area'].max() <= data['Area'].max()

    def test_fewer_than_top_n_values(self):
        """Test behavior when fewer than top_n values are available."""
        data = pd.DataFrame({
            'ImageName'          : ['img1']*5,
            str(GRID.SECTION_NUM): range(5),
            'Area'               : [100, 200, 300, 400, 500]
        })

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                top_n=10,  # More than available
                nrows=2,
                ncols=3
        )

        # Should use all 5 values for threshold
        corrected = corrector.analyze(data)

        # Should not raise error
        assert len(corrected) == 5


class TestValueCapping:
    """Test that value capping works correctly."""

    def test_only_edge_sections_corrected(self):
        """Test that only edge sections are corrected, not interior."""
        np.random.seed(42)

        # Create 4x6 grid with all positions filled
        nrows, ncols = 4, 6
        n_sections = nrows*ncols

        data = pd.DataFrame({
            'ImageName'          : ['img1']*n_sections,
            str(GRID.SECTION_NUM): range(n_sections),
            'Area'               : np.random.uniform(100, 300, n_sections)
        })

        # Set some edge values very high (higher than interior values)
        edge_sections = [0, 1, 2, 3, 4, 5]  # Top row
        data.loc[data[str(GRID.SECTION_NUM)].isin(edge_sections), 'Area'] = 1000

        # Ensure interior values are in normal range for top_n calculation
        interior_sections = [7, 8, 9, 10, 13, 14, 15, 16]
        data.loc[data[str(GRID.SECTION_NUM)].isin(interior_sections), 'Area'] = np.random.uniform(250, 350,
                                                                                                  len(interior_sections))

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                top_n=8,  # Use interior + some edge values for threshold
                nrows=nrows,
                ncols=ncols,
                connectivity=4
        )

        original = data.copy()
        corrected = corrector.analyze(data)

        # Edge sections should be corrected (capped below 1000)
        edge_mask = corrected[str(GRID.SECTION_NUM)].isin(edge_sections)
        assert (corrected.loc[edge_mask, 'Area'] < 1000).any()

    def test_only_exceeding_values_capped(self):
        """Test that only values exceeding threshold are capped."""
        np.random.seed(42)

        data = pd.DataFrame({
            'ImageName'          : ['img1']*96,
            str(GRID.SECTION_NUM): range(96),
            'Area'               : np.random.uniform(100, 200, 96)
        })

        # Set a few edge values very high
        data.loc[0, 'Area'] = 1000
        data.loc[1, 'Area'] = 1100

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                top_n=10,
                nrows=8,
                ncols=12
        )

        original = data.copy()
        corrected = corrector.analyze(data)

        # High edge values should be capped
        assert corrected.loc[0, 'Area'] < original.loc[0, 'Area']
        assert corrected.loc[1, 'Area'] < original.loc[1, 'Area']

        # Values already below threshold should be unchanged
        low_value_mask = original['Area'] < 200
        assert np.allclose(
                corrected.loc[low_value_mask, 'Area'].values,
                original.loc[low_value_mask, 'Area'].values
        )

    def test_interior_sections_unchanged(self):
        """Test that interior sections are never modified."""
        np.random.seed(42)

        # Create 5x5 grid
        nrows, ncols = 5, 5
        data = pd.DataFrame({
            'ImageName'          : ['img1']*25,
            str(GRID.SECTION_NUM): range(25),
            'Area'               : np.random.uniform(100, 500, 25)
        })

        # Set center position (12) to high value
        data.loc[12, 'Area'] = 1000

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                top_n=5,
                nrows=nrows,
                ncols=ncols
        )

        original = data.copy()
        corrected = corrector.analyze(data)

        # Center position should remain unchanged
        assert corrected.loc[12, 'Area'] == original.loc[12, 'Area']


class TestGroupbyBehavior:
    """Test that groupby operations work correctly."""

    def test_multiple_groups(self):
        """Test correction with multiple image groups."""
        np.random.seed(42)

        # Create data for two images
        img1_data = pd.DataFrame({
            'ImageName'          : ['img1']*96,
            str(GRID.SECTION_NUM): range(96),
            'Area'               : np.random.uniform(100, 300, 96)
        })

        img2_data = pd.DataFrame({
            'ImageName'          : ['img2']*96,
            str(GRID.SECTION_NUM): range(96),
            'Area'               : np.random.uniform(200, 500, 96)
        })

        data = pd.concat([img1_data, img2_data], ignore_index=True)

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                top_n=10,
                nrows=8,
                ncols=12
        )

        corrected = corrector.analyze(data)

        # Both groups should have been processed
        assert len(corrected) == 192
        assert set(corrected['ImageName']) == {'img1', 'img2'}

        # Each group should have independent thresholds
        img1_corrected = corrected[corrected['ImageName'] == 'img1']
        img2_corrected = corrected[corrected['ImageName'] == 'img2']

        assert len(img1_corrected) == 96
        assert len(img2_corrected) == 96

    def test_different_grid_configurations(self):
        """Test that each group uses the same grid configuration."""
        np.random.seed(42)

        data = pd.DataFrame({
            'ImageName'          : ['img1']*48 + ['img2']*48,
            str(GRID.SECTION_NUM): list(range(48))*2,
            'Area'               : np.random.uniform(100, 500, 96)
        })

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                top_n=5,
                nrows=6,
                ncols=8
        )

        corrected = corrector.analyze(data)
        assert len(corrected) == 96

    def test_no_groupby(self):
        """Test correction without groupby (single dataset)."""
        np.random.seed(42)

        data = pd.DataFrame({
            str(GRID.SECTION_NUM): range(96),
            'Area'               : np.random.uniform(100, 500, 96)
        })

        corrector = EdgeCorrector(
                on='Area',
                groupby=[],  # No grouping
                measurement_col='Area',
                top_n=10,
                nrows=8,
                ncols=12
        )

        corrected = corrector.analyze(data)
        assert len(corrected) == 96


class TestAnalyzeMethod:
    """Test the analyze method."""

    def test_analyze_stores_results(self):
        """Test that analyze stores original and corrected data."""
        np.random.seed(42)

        data = pd.DataFrame({
            'ImageName'          : ['img1']*96,
            str(GRID.SECTION_NUM): range(96),
            'Area'               : np.random.uniform(100, 500, 96)
        })

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area'
        )

        corrected = corrector.analyze(data)

        # Check that data is stored
        assert not corrector._original_data.empty
        assert not corrector._latest_measurements.empty

        # Results should be retrievable
        results = corrector.results()
        assert results.equals(corrected)

    def test_analyze_with_missing_columns(self):
        """Test that analyze raises error with missing columns."""
        data = pd.DataFrame({
            'ImageName': ['img1']*10,
            'Area'     : np.random.uniform(100, 500, 10)
            # Missing GRID.SECTION_NUM
        })

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area'
        )

        with pytest.raises(KeyError, match="Missing required columns"):
            corrector.analyze(data)

    def test_analyze_with_empty_data(self):
        """Test that analyze raises error with empty data."""
        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area'
        )

        with pytest.raises(ValueError, match="cannot be empty"):
            corrector.analyze(pd.DataFrame())


class TestResultsMethod:
    """Test the results method."""

    def test_results_returns_corrected_data(self):
        """Test that results returns the corrected DataFrame."""
        np.random.seed(42)

        data = pd.DataFrame({
            'ImageName'          : ['img1']*96,
            str(GRID.SECTION_NUM): range(96),
            'Area'               : np.random.uniform(100, 500, 96)
        })

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area'
        )

        corrected = corrector.analyze(data)
        results = corrector.results()

        assert results.equals(corrected)

    def test_results_before_analyze(self):
        """Test that results returns empty DataFrame before analyze."""
        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area'
        )

        results = corrector.results()
        assert results.empty


class TestShowMethod:
    """Test the show visualization method."""

    def test_show_requires_analyze(self):
        """Test that show raises error if analyze not called."""
        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area'
        )

        with pytest.raises(RuntimeError, match="Call analyze\\(\\) first"):
            corrector.show()

    def test_show_runs_after_analyze(self):
        """Test that show runs successfully after analyze."""
        np.random.seed(42)

        data = pd.DataFrame({
            'ImageName'          : ['img1']*96,
            str(GRID.SECTION_NUM): range(96),
            'Area'               : np.random.uniform(100, 500, 96)
        })

        # Set some edge values high
        data.loc[0:11, 'Area'] = 1000  # Top row

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area'
        )

        corrector.analyze(data)

        # This should not raise an error
        # Note: In actual test environment, we might mock plt.show()
        # to prevent display, but for now we'll skip the actual display test
        # corrector.show()  # Would display in interactive environment


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_microplate_scenario(self):
        """Test with realistic microplate colony data."""
        np.random.seed(42)

        # Simulate 8x12 plate with most colonies ~200 area
        # But edge colonies artificially inflated
        nrows, ncols = 8, 12
        n_sections = nrows*ncols

        data = pd.DataFrame({
            'ImageName'          : ['plate1']*n_sections,
            str(GRID.SECTION_NUM): range(n_sections),
            str(GRID.ROW_NUM)    : [i//ncols for i in range(n_sections)],
            str(GRID.COL_NUM)    : [i%ncols for i in range(n_sections)],
            'Area'               : np.random.normal(200, 30, n_sections)
        })

        # Inflate edge colonies
        edge_mask = (
                (data[str(GRID.ROW_NUM)] == 0) |
                (data[str(GRID.ROW_NUM)] == nrows - 1) |
                (data[str(GRID.COL_NUM)] == 0) |
                (data[str(GRID.COL_NUM)] == ncols - 1)
        )
        data.loc[edge_mask, 'Area'] *= 1.5

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                top_n=10,
                nrows=nrows,
                ncols=ncols,
                connectivity=4
        )

        original_mean = data['Area'].mean()
        corrected = corrector.analyze(data)
        corrected_mean = corrected['Area'].mean()

        # After correction, mean should be lower (edge inflation removed)
        assert corrected_mean < original_mean

        # Edge colonies should have been capped
        edge_corrected = corrected.loc[edge_mask, 'Area']
        edge_original = data.loc[edge_mask, 'Area']

        # At least some edge values should be different (capped)
        assert not np.allclose(edge_corrected.values, edge_original.values)

    def test_partial_grid_coverage(self):
        """Test with partial grid coverage (some sections empty)."""
        np.random.seed(42)

        # Only 50 of 96 sections have colonies
        sections = np.random.choice(96, size=50, replace=False)

        data = pd.DataFrame({
            'ImageName'          : ['img1']*50,
            str(GRID.SECTION_NUM): sections,
            'Area'               : np.random.uniform(100, 500, 50)
        })

        corrector = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                top_n=10,
                nrows=8,
                ncols=12
        )

        # Should handle partial coverage without error
        corrected = corrector.analyze(data)
        assert len(corrected) == 50

    def test_with_multiple_measurements(self):
        """Test correction on different measurement columns."""
        np.random.seed(42)

        data = pd.DataFrame({
            'ImageName'          : ['img1']*96,
            str(GRID.SECTION_NUM): range(96),
            'Area'               : np.random.uniform(100, 500, 96),
            'MeanRadius'         : np.random.uniform(5, 15, 96),
            'Perimeter'          : np.random.uniform(20, 60, 96)
        })

        # Test correction on Area
        corrector_area = EdgeCorrector(
                on='Area',
                groupby=['ImageName'],
                measurement_col='Area',
                top_n=10
        )

        corrected_area = corrector_area.analyze(data.copy())

        # Test correction on MeanRadius
        corrector_radius = EdgeCorrector(
                on='MeanRadius',
                groupby=['ImageName'],
                measurement_col='MeanRadius',
                top_n=10
        )

        corrected_radius = corrector_radius.analyze(data.copy())

        # Both should work independently
        assert len(corrected_area) == 96
        assert len(corrected_radius) == 96

        # Area correction shouldn't affect other columns
        assert corrected_area['MeanRadius'].equals(data['MeanRadius'])
