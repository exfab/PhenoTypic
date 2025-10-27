"""Tests for EdgeCorrector.surrounded_positions function."""

from __future__ import annotations

import numpy as np
import pytest

from phenotypic.analysis._edge_correction import EdgeCorrector


class TestSurroundedPositionsValidation:
    """Test input validation for surrounded_positions."""

    def test_bad_connectivity_raises(self):
        """Test that invalid connectivity values raise ValueError."""
        active = np.array([5], dtype=np.int64)
        shape = (3, 3)

        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            EdgeCorrector._surrounded_positions(active, shape, connectivity=6)

        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            EdgeCorrector._surrounded_positions(active, shape, connectivity=3)

    def test_out_of_bounds_index_raises(self):
        """Test that out-of-bounds indices raise ValueError."""
        shape = (3, 3)
        total = 9

        # Negative index
        with pytest.raises(ValueError, match=f"All active_idx must be in \\[0, {total}\\)"):
            EdgeCorrector._surrounded_positions([-1], shape)

        # Index too large
        with pytest.raises(ValueError, match=f"All active_idx must be in \\[0, {total}\\)"):
            EdgeCorrector._surrounded_positions([9], shape)

        # Mixed valid and invalid
        with pytest.raises(ValueError, match=f"All active_idx must be in \\[0, {total}\\)"):
            EdgeCorrector._surrounded_positions([0, 4, 10], shape)

    def test_min_neighbors_zero_raises(self):
        """Test that min_neighbors=0 raises ValueError."""
        active = np.array([4], dtype=np.int64)
        shape = (3, 3)

        with pytest.raises(ValueError, match="min_neighbors must be in \\[1, 4\\]"):
            EdgeCorrector._surrounded_positions(active, shape, connectivity=4, min_neighbors=0)

    def test_min_neighbors_exceeds_connectivity_raises(self):
        """Test that min_neighbors > connectivity raises ValueError."""
        active = np.array([4], dtype=np.int64)
        shape = (3, 3)

        with pytest.raises(ValueError, match="min_neighbors must be in \\[1, 4\\]"):
            EdgeCorrector._surrounded_positions(active, shape, connectivity=4, min_neighbors=5)

        with pytest.raises(ValueError, match="min_neighbors must be in \\[1, 8\\]"):
            EdgeCorrector._surrounded_positions(active, shape, connectivity=8, min_neighbors=9)

    def test_invalid_shape_raises(self):
        """Test that invalid shapes raise ValueError."""
        active = np.array([0], dtype=np.int64)

        with pytest.raises(ValueError, match="shape must be two positive integers"):
            EdgeCorrector._surrounded_positions(active, (0, 3))

        with pytest.raises(ValueError, match="shape must be two positive integers"):
            EdgeCorrector._surrounded_positions(active, (3, 0))

        with pytest.raises(ValueError, match="shape must be two positive integers"):
            EdgeCorrector._surrounded_positions(active, (-1, 3))


class TestSurroundedPositionsGeometry:
    """Test geometric properties of surrounded_positions."""

    def test_border_never_qualify_connectivity4_none(self):
        """Test that border cells never qualify when min_neighbors=None with connectivity=4."""
        rows, cols = 5, 5
        # All cells in the grid
        all_active = np.arange(rows*cols, dtype=np.int64)

        result = EdgeCorrector._surrounded_positions(
                all_active, (rows, cols), connectivity=4, min_neighbors=None
        )

        # Only interior cells (not on border) should qualify
        # Border cells: row 0, row 4, col 0, col 4
        for idx in result:
            r, c = idx//cols, idx%cols
            assert 0 < r < rows - 1, f"Row {r} is on border"
            assert 0 < c < cols - 1, f"Col {c} is on border"

    def test_border_never_qualify_connectivity8_none(self):
        """Test that border cells never qualify when min_neighbors=None with connectivity=8."""
        rows, cols = 5, 5
        all_active = np.arange(rows*cols, dtype=np.int64)

        result = EdgeCorrector._surrounded_positions(
                all_active, (rows, cols), connectivity=8, min_neighbors=None
        )

        # Only interior cells should qualify
        for idx in result:
            r, c = idx//cols, idx%cols
            assert 0 < r < rows - 1, f"Row {r} is on border"
            assert 0 < c < cols - 1, f"Col {c} is on border"

    def test_corner_has_two_neighbors_connectivity4(self):
        """Test that corner cells have exactly 2 neighbors with connectivity=4."""
        rows, cols = 3, 3
        all_active = np.arange(rows*cols, dtype=np.int64)

        # Top-left corner (0, 0) -> idx 0
        # Should have neighbors at (0, 1) and (1, 0)
        result, counts = EdgeCorrector._surrounded_positions(
                all_active, (rows, cols), connectivity=4, min_neighbors=2, return_counts=True
        )

        # Find corner in results
        corner_idx = 0
        if corner_idx in result:
            pos = np.where(result == corner_idx)[0][0]
            assert counts[pos] == 2, f"Corner should have 2 neighbors, got {counts[pos]}"


class TestSurroundedPositionsDegenerate:
    """Test degenerate cases for surrounded_positions."""

    def test_empty_active_idx_returns_empty(self):
        """Test that empty active_idx returns empty result."""
        result = EdgeCorrector._surrounded_positions([], (5, 5), connectivity=4)
        assert len(result) == 0
        assert result.dtype == np.int64

    def test_empty_active_idx_with_counts_returns_empty(self):
        """Test that empty active_idx with return_counts=True returns empty arrays."""
        idxs, counts = EdgeCorrector._surrounded_positions(
                [], (5, 5), connectivity=4, return_counts=True
        )
        assert len(idxs) == 0
        assert len(counts) == 0
        assert idxs.dtype == np.int64
        assert counts.dtype == np.int64

    def test_single_active_cell_returns_empty_when_none(self):
        """Test that single active cell returns empty when min_neighbors=None."""
        # Single cell cannot have any neighbors
        result = EdgeCorrector._surrounded_positions([4], (3, 3), connectivity=4)
        assert len(result) == 0

    def test_single_active_cell_returns_empty_when_min1(self):
        """Test that single active cell returns empty even with min_neighbors=1."""
        # Single cell has no active neighbors
        result = EdgeCorrector._surrounded_positions(
                [4], (3, 3), connectivity=4, min_neighbors=1
        )
        assert len(result) == 0


class TestSurroundedPositionsCorrectness:
    """Test correctness of surrounded_positions with specific patterns."""

    def test_3x3_block_connectivity4(self):
        """Test 3×3 active block with connectivity=4."""
        rows, cols = 8, 12
        # 3×3 block centered at (4, 6): rows 3-5, cols 5-7
        block_rc = [(r, c) for r in range(3, 6) for c in range(5, 8)]
        active = np.array([r*cols + c for r, c in block_rc], dtype=np.int64)

        # With min_neighbors=None (all 4 required), only center qualifies
        result = EdgeCorrector._surrounded_positions(active, (rows, cols), connectivity=4)
        expected = np.array([4*cols + 6], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_3x3_block_connectivity4_min3(self):
        """Test 3×3 active block with connectivity=4 and min_neighbors=3."""
        rows, cols = 8, 12
        block_rc = [(r, c) for r in range(3, 6) for c in range(5, 8)]
        active = np.array([r*cols + c for r, c in block_rc], dtype=np.int64)

        idxs, counts = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=3, return_counts=True
        )

        # All counts should be >= 3
        assert (counts >= 3).all()

        # Center should be in results with count=4
        center_idx = 4*cols + 6
        assert center_idx in idxs
        center_pos = np.where(idxs == center_idx)[0][0]
        assert counts[center_pos] == 4

        # Edge midpoints should have 3 neighbors
        # e.g., (3, 6), (5, 6), (4, 5), (4, 7)
        edge_midpoints = [
            3*cols + 6,  # top
            5*cols + 6,  # bottom
            4*cols + 5,  # left
            4*cols + 7,  # right
        ]
        for emp in edge_midpoints:
            if emp in idxs:
                pos = np.where(idxs == emp)[0][0]
                assert counts[pos] == 3

    def test_3x3_block_connectivity8(self):
        """Test 3×3 active block with connectivity=8."""
        rows, cols = 8, 12
        block_rc = [(r, c) for r in range(3, 6) for c in range(5, 8)]
        active = np.array([r*cols + c for r, c in block_rc], dtype=np.int64)

        # With connectivity=8 and min_neighbors=None (all 8 required), only center qualifies
        result = EdgeCorrector._surrounded_positions(active, (rows, cols), connectivity=8)
        expected = np.array([4*cols + 6], dtype=np.int64)
        assert np.array_equal(result, expected)

    def test_3x3_block_connectivity8_min5(self):
        """Test 3×3 active block with connectivity=8 and min_neighbors=5."""
        rows, cols = 8, 12
        block_rc = [(r, c) for r in range(3, 6) for c in range(5, 8)]
        active = np.array([r*cols + c for r, c in block_rc], dtype=np.int64)

        idxs, counts = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=8, min_neighbors=5, return_counts=True
        )

        # All counts should be >= 5
        assert (counts >= 5).all()

        # Center should have 8 neighbors
        center_idx = 4*cols + 6
        assert center_idx in idxs
        center_pos = np.where(idxs == center_idx)[0][0]
        assert counts[center_pos] == 8

        # Edge midpoints should have 5 neighbors
        edge_midpoints = [
            3*cols + 6,  # top
            5*cols + 6,  # bottom
            4*cols + 5,  # left
            4*cols + 7,  # right
        ]
        for emp in edge_midpoints:
            assert emp in idxs
            pos = np.where(idxs == emp)[0][0]
            assert counts[pos] == 5

    def test_subset_property_connectivity4(self):
        """Test that results with min_neighbors=k are subset of results with min_neighbors=k-1."""
        rows, cols = 10, 10
        # Random pattern
        np.random.seed(42)
        active = np.random.choice(rows*cols, size=50, replace=False)

        # Get results for different thresholds
        res4 = set(EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=4
        ))
        res3 = set(EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=3
        ))
        res2 = set(EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=2
        ))
        res1 = set(EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=1
        ))

        # Subset property
        assert res4.issubset(res3)
        assert res3.issubset(res2)
        assert res2.issubset(res1)

    def test_subset_property_connectivity8(self):
        """Test subset property for connectivity=8."""
        rows, cols = 10, 10
        np.random.seed(42)
        active = np.random.choice(rows*cols, size=50, replace=False)

        res8 = set(EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=8, min_neighbors=8
        ))
        res7 = set(EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=8, min_neighbors=7
        ))
        res6 = set(EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=8, min_neighbors=6
        ))

        assert res8.issubset(res7)
        assert res7.issubset(res6)


class TestSurroundedPositionsMisc:
    """Miscellaneous tests for surrounded_positions."""

    def test_deduplication(self):
        """Test that duplicate indices are deduplicated."""
        rows, cols = 5, 5
        # Create duplicates
        active = [12, 12, 13, 13, 13, 17, 17, 18]

        # Should work without error and dedupe
        result = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=1
        )

        # Result should be sorted and unique
        assert len(result) == len(np.unique(result))
        assert np.array_equal(result, np.sort(result))

    def test_list_input(self):
        """Test that list input works correctly."""
        rows, cols = 5, 5
        active_list = [6, 7, 8, 11, 12, 13, 16, 17, 18]

        result = EdgeCorrector._surrounded_positions(
                active_list, (rows, cols), connectivity=4
        )

        # Should return center of 3×3 block
        assert 12 in result

    def test_dtype_parameter(self):
        """Test that dtype parameter is respected."""
        rows, cols = 5, 5
        active = [12]

        result32 = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=1, dtype=np.int32
        )
        assert result32.dtype == np.int32

        result64 = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=1, dtype=np.int64
        )
        assert result64.dtype == np.int64

    def test_return_counts_sorted_correctly(self):
        """Test that counts are sorted to match sorted indices."""
        rows, cols = 5, 5
        # 3×3 block
        active = [6, 7, 8, 11, 12, 13, 16, 17, 18]

        idxs, counts = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=2, return_counts=True
        )

        # Indices should be sorted
        assert np.array_equal(idxs, np.sort(idxs))

        # Counts should correspond to sorted indices
        # Verify by checking a known case
        if 12 in idxs:  # center
            pos = np.where(idxs == 12)[0][0]
            assert counts[pos] == 4

    def test_large_grid(self):
        """Test with a larger grid to ensure performance is reasonable."""
        rows, cols = 100, 100
        # Create a 10×10 block in the middle
        block_rc = [(r, c) for r in range(45, 55) for c in range(45, 55)]
        active = np.array([r*cols + c for r, c in block_rc], dtype=np.int64)

        result = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4
        )

        # Should have 8×8 = 64 fully surrounded cells
        assert len(result) == 64

        # All should be in interior of block
        for idx in result:
            r, c = idx//cols, idx%cols
            assert 46 <= r <= 53
            assert 46 <= c <= 53

    def test_horizontal_line(self):
        """Test with a horizontal line of active cells."""
        rows, cols = 5, 10
        # Row 2, all columns
        active = [2*cols + c for c in range(cols)]

        # With connectivity=4, no cell can be fully surrounded (needs N and S neighbors)
        result = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4
        )
        assert len(result) == 0

        # With min_neighbors=2, all interior cells qualify (have E and W)
        result2 = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=2
        )
        # Should have cols-2 cells (excluding first and last)
        assert len(result2) == cols - 2

    def test_vertical_line(self):
        """Test with a vertical line of active cells."""
        rows, cols = 10, 5
        # Column 2, all rows
        active = [r*cols + 2 for r in range(rows)]

        # With connectivity=4, no cell can be fully surrounded (needs E and W neighbors)
        result = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4
        )
        assert len(result) == 0

        # With min_neighbors=2, all interior cells qualify (have N and S)
        result2 = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=2
        )
        # Should have rows-2 cells (excluding first and last)
        assert len(result2) == rows - 2

    def test_checkerboard_pattern(self):
        """Test with a checkerboard pattern."""
        rows, cols = 6, 6
        # Checkerboard: (r+c) % 2 == 0
        active = [r*cols + c for r in range(rows) for c in range(cols) if (r + c)%2 == 0]

        # With connectivity=4, no cell has any active neighbors (all neighbors are opposite color)
        result = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=4, min_neighbors=1
        )
        assert len(result) == 0

        # With connectivity=8, each cell has 4 diagonal neighbors
        result8 = EdgeCorrector._surrounded_positions(
                active, (rows, cols), connectivity=8, min_neighbors=4
        )
        # Interior checkerboard cells should qualify
        assert len(result8) > 0
