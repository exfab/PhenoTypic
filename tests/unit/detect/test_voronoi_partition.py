import numpy as np
import pytest
from phenotypic.tools_.branch_pathfinding import (
    connectivity_correct_labels,
    euclidean_voronoi_assign,
)


class TestEuclideanVoronoiAssign:
    def test_two_seeds_partition(self):
        """Two seeds split a rectangular mask at the midpoint."""
        mask = np.ones((10, 20), dtype=bool)
        markers = np.zeros((10, 20), dtype=np.int32)
        markers[5, 3] = 1
        markers[5, 17] = 2

        result = euclidean_voronoi_assign(markers, mask)

        assert result[5, 0] == 1
        assert result[5, 19] == 2
        assert result[5, 8] == 1
        assert result[5, 12] == 2

    def test_mask_restricts_output(self):
        """Pixels outside mask are labeled 0."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:8, 2:8] = True
        markers = np.zeros((10, 10), dtype=np.int32)
        markers[5, 5] = 1

        result = euclidean_voronoi_assign(markers, mask)

        assert result[0, 0] == 0
        assert result[5, 5] == 1

    def test_no_seeds_returns_zeros(self):
        mask = np.ones((5, 5), dtype=bool)
        markers = np.zeros((5, 5), dtype=np.int32)

        result = euclidean_voronoi_assign(markers, mask)

        assert result.max() == 0

    def test_single_seed_labels_entire_mask(self):
        mask = np.ones((10, 10), dtype=bool)
        markers = np.zeros((10, 10), dtype=np.int32)
        markers[5, 5] = 3

        result = euclidean_voronoi_assign(markers, mask)

        assert np.all(result[mask] == 3)

    def test_seed_outside_mask_ignored(self):
        """Markers outside mask should not influence partition."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[3:7, 8:15] = True
        markers = np.zeros((10, 20), dtype=np.int32)
        markers[5, 0] = 1   # Outside mask
        markers[5, 12] = 2  # Inside mask

        result = euclidean_voronoi_assign(markers, mask)

        # Only seed 2 is inside mask; entire mask region gets label 2
        assert np.all(result[mask] == 2)

    def test_seedless_cc_stays_zero(self):
        """Disconnected mask region with no seed stays unlabeled."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[3:7, 1:5] = True    # Blob with seed
        mask[3:7, 15:19] = True  # Isolated blob (no seed)

        markers = np.zeros((10, 20), dtype=np.int32)
        markers[5, 3] = 1

        result = euclidean_voronoi_assign(markers, mask)

        assert result[5, 3] == 1
        assert result[5, 17] == 0  # Seedless CC stays 0

    def test_output_dtype(self):
        mask = np.ones((5, 5), dtype=bool)
        markers = np.zeros((5, 5), dtype=np.int32)
        markers[2, 2] = 1

        result = euclidean_voronoi_assign(markers, mask)

        assert result.dtype == np.int32

    def test_unrestricted_assigns_all_mask_pixels(self):
        """Seeds outside mask assign all mask pixels when unrestricted."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[3:7, 8:15] = True
        markers = np.zeros((10, 20), dtype=np.int32)
        markers[5, 0] = 1   # Outside mask
        markers[5, 19] = 2  # Outside mask

        result = euclidean_voronoi_assign(
                markers, mask, restrict_to_seeded_cc=False,
        )
        # All mask pixels should have a label
        assert np.all(result[mask] > 0)

    def test_unrestricted_seedless_cc_gets_label(self):
        """Disconnected CC gets nearest-seed label when unrestricted."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[3:7, 1:5] = True    # Left blob
        mask[3:7, 15:19] = True  # Right blob (no seed nearby)
        markers = np.zeros((10, 20), dtype=np.int32)
        markers[5, 3] = 1  # Inside left blob

        result = euclidean_voronoi_assign(
                markers, mask, restrict_to_seeded_cc=False,
        )
        # Both blobs get label 1 (only seed)
        assert result[5, 3] == 1
        assert result[5, 17] == 1


class TestConnectivityCorrectLabels:
    def test_case_a_reassign_to_connected_seed(self):
        """Fragment connected only to seed 2 but Voronoi-labeled 1 -> reassign."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[3:7, 1:5] = True    # Left blob (seed 1)
        mask[3:7, 8:12] = True   # Middle blob (no seed, Euclidean-closer to 1)
        mask[3:7, 15:19] = True  # Right blob (seed 2)
        mask[5, 12:15] = True    # Bridge: middle <-> right only

        markers = np.zeros((10, 20), dtype=np.int32)
        markers[5, 3] = 1
        markers[5, 17] = 2

        voronoi = euclidean_voronoi_assign(markers, mask)
        result = connectivity_correct_labels(voronoi, mask, markers)

        # Middle blob physically connected only to seed 2 -> Case A
        assert result[5, 10] == 2

    def test_case_b_multiple_seeds_keeps_voronoi(self):
        """Single connected blob with two seeds keeps Voronoi labels."""
        mask = np.ones((10, 20), dtype=bool)
        markers = np.zeros((10, 20), dtype=np.int32)
        markers[5, 3] = 1
        markers[5, 17] = 2

        voronoi = euclidean_voronoi_assign(markers, mask)
        result = connectivity_correct_labels(voronoi, mask, markers)

        np.testing.assert_array_equal(result, voronoi)

    def test_case_b_fragment_reassigned_to_majority_neighbor(self):
        """Small fragment of label 1 in label 2 territory gets reassigned."""
        mask = np.ones((10, 20), dtype=bool)
        markers = np.zeros((10, 20), dtype=np.int32)
        markers[5, 3] = 1
        markers[5, 17] = 2

        voronoi = euclidean_voronoi_assign(markers, mask)

        # Inject a small fragment: set a few pixels deep in label 2 territory
        # to label 1 (simulating a Voronoi misassignment)
        fragment_coords = [(4, 15), (5, 15), (5, 16)]
        for r, c in fragment_coords:
            voronoi[r, c] = 1

        result = connectivity_correct_labels(voronoi, mask, markers)

        # Fragment should be reassigned to label 2 (majority neighbor)
        for r, c in fragment_coords:
            assert result[r, c] == 2, (
                f"pixel ({r},{c}) should be 2 but got {result[r, c]}"
            )

    def test_case_c_isolated_fragment_stays_zero(self):
        """Fragment with no reachable seed stays 0 (Voronoi zeros seedless CCs)."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[3:7, 1:5] = True    # Blob with seed
        mask[3:7, 15:19] = True  # Isolated blob (no seed, no bridge)

        markers = np.zeros((10, 20), dtype=np.int32)
        markers[5, 3] = 1

        voronoi = euclidean_voronoi_assign(markers, mask)
        result = connectivity_correct_labels(voronoi, mask, markers)

        # Isolated blob is already 0 from Voronoi; connectivity keeps it 0
        assert result[5, 17] == 0
        assert result[5, 3] == 1

    def test_preserves_background(self):
        """Pixels outside mask stay 0."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        markers = np.zeros((10, 10), dtype=np.int32)
        markers[5, 5] = 1

        voronoi = euclidean_voronoi_assign(markers, mask)
        result = connectivity_correct_labels(voronoi, mask, markers)

        assert result[0, 0] == 0
        assert result[5, 5] == 1

    def test_multiple_single_seed_components(self):
        """Three disconnected blobs, each with one seed, all correctly labeled."""
        mask = np.zeros((10, 30), dtype=bool)
        mask[3:7, 1:5] = True
        mask[3:7, 11:15] = True
        mask[3:7, 21:25] = True

        markers = np.zeros((10, 30), dtype=np.int32)
        markers[5, 3] = 1
        markers[5, 13] = 2
        markers[5, 23] = 3

        voronoi = euclidean_voronoi_assign(markers, mask)
        result = connectivity_correct_labels(voronoi, mask, markers)

        assert np.all(result[3:7, 1:5][mask[3:7, 1:5]] == 1)
        assert np.all(result[3:7, 11:15][mask[3:7, 11:15]] == 2)
        assert np.all(result[3:7, 21:25][mask[3:7, 21:25]] == 3)

    def test_seeds_outside_mask_within_expansion(self):
        """Fragment finds seed 1px outside mask via bbox expansion."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[3:7, 5:15] = True

        markers = np.zeros((10, 20), dtype=np.int32)
        markers[5, 4] = 1   # 1px outside mask (left)
        markers[5, 15] = 2  # 1px outside mask (right)

        voronoi = euclidean_voronoi_assign(
                markers, mask, restrict_to_seeded_cc=False,
        )
        result = connectivity_correct_labels(voronoi, mask, markers)

        # Left side of mask should get label 1, right side label 2
        assert result[5, 6] == 1
        assert result[5, 13] == 2
        # Seeds outside mask remain in seeded_voronoi but outside mask → 0
        assert result[5, 4] == 0
        assert result[5, 15] == 0
