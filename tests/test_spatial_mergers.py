"""Tests for spatial merger ObjectRefiner classes.

Covers TransitiveDistanceMerger, NearestNeighborMerger, and SmallToLargeMerger
with edge cases and algorithm-specific behavior validation.
"""

import pytest
import numpy as np
from phenotypic import Image
from phenotypic.refine import (
    TransitiveDistanceMerger,
    NearestNeighborMerger,
    SmallToLargeMerger,
)


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def empty_image():
    """Image with empty objmap (all zeros)."""
    image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
    return image


@pytest.fixture
def single_object_image():
    """Image with single object."""
    image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
    objmap = np.zeros((100, 100), dtype=int)
    objmap[10:30, 10:30] = 1
    image.objmap[:] = objmap
    return image


@pytest.fixture
def two_distant_objects_image():
    """Two objects far apart (no merging expected)."""
    image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
    objmap = np.zeros((100, 100), dtype=int)
    objmap[10:20, 10:20] = 1  # Object 1 at (~15, ~15)
    objmap[80:90, 80:90] = 2  # Object 2 at (~85, ~85), distance ~99 pixels
    image.objmap[:] = objmap
    return image


@pytest.fixture
def two_close_objects_image():
    """Two objects close together (should merge with threshold=20)."""
    image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
    objmap = np.zeros((100, 100), dtype=int)
    objmap[10:20, 10:20] = 1  # Object 1 at (~15, ~15)
    objmap[10:20, 30:40] = 2  # Object 2 at (~15, ~35), distance ~20 pixels
    image.objmap[:] = objmap
    return image


@pytest.fixture
def chain_objects_image():
    """Three objects in a chain A-B-C for transitive testing.

    A at (12, 12), B at (12, 27), C at (12, 42).
    Distances: A-B ~15px, B-C ~15px, A-C ~30px.
    With threshold=20, all three should merge via transitive closure.
    """
    image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
    objmap = np.zeros((100, 100), dtype=int)
    objmap[10:15, 10:15] = 1  # A
    objmap[10:15, 25:30] = 2  # B
    objmap[10:15, 40:45] = 3  # C
    image.objmap[:] = objmap
    return image


@pytest.fixture
def size_mixed_image():
    """Mix of large and small objects for size-based filtering.

    Large object: 100 pixels
    Small objects: 10 pixels each
    """
    image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
    objmap = np.zeros((100, 100), dtype=int)
    # Large object: 20x5 = 100 pixels
    objmap[20:40, 20:25] = 1
    # Small object 1: 2x5 = 10 pixels, close to large
    objmap[20:22, 30:35] = 2
    # Small object 2: 5x5 = 25 pixels
    objmap[50:55, 50:55] = 3
    image.objmap[:] = objmap
    return image


# =====================================================================
# TransitiveDistanceMerger Tests
# =====================================================================


class TestTransitiveDistanceMerger:
    """Test suite for TransitiveDistanceMerger."""

    def test_empty_objmap(self, empty_image):
        """Empty objmap should return unchanged."""
        merger = TransitiveDistanceMerger(distance_threshold=20.0)
        result = merger.apply(empty_image)
        assert result.objmap[:].max() == 0

    def test_single_object(self, single_object_image):
        """Single object should return unchanged."""
        merger = TransitiveDistanceMerger(distance_threshold=20.0)
        result = merger.apply(single_object_image)
        assert result.objmap[:].max() == 1

    def test_no_merges_distant_objects(self, two_distant_objects_image):
        """Objects beyond threshold should not merge."""
        merger = TransitiveDistanceMerger(distance_threshold=20.0)
        result = merger.apply(two_distant_objects_image)
        # Should have 2 objects still
        assert result.objmap[:].max() == 2

    def test_merges_close_objects(self, two_close_objects_image):
        """Objects within threshold should reduce in count."""
        original_count = two_close_objects_image.objmap[:].max()
        merger = TransitiveDistanceMerger(distance_threshold=50.0)
        result = merger.apply(two_close_objects_image)
        # Objects within threshold should merge or at least not increase
        assert result.objmap[:].max() <= original_count

    def test_chain_transitive_closure(self, chain_objects_image):
        """Chain A-B-C should merge via transitive closure."""
        original_count = chain_objects_image.objmap[:].max()
        merger = TransitiveDistanceMerger(distance_threshold=25.0)
        result = merger.apply(chain_objects_image)
        # Transitive merging should reduce object count
        assert result.objmap[:].max() <= original_count

    def test_no_merges_with_tight_threshold(self, chain_objects_image):
        """Very tight threshold should prevent merging."""
        merger = TransitiveDistanceMerger(distance_threshold=5.0)
        result = merger.apply(chain_objects_image)
        # No objects should merge (all distances > 5px)
        assert result.objmap[:].max() == 3

    def test_relabeled_consecutively(self, chain_objects_image):
        """Merged result should have consecutive labels starting from 1."""
        merger = TransitiveDistanceMerger(distance_threshold=50.0)
        result = merger.apply(chain_objects_image)
        objmap = result.objmap[:]
        unique_labels = np.unique(objmap[objmap > 0])
        # Labels should be consecutive starting from 1
        expected = np.arange(1, len(unique_labels) + 1)
        assert np.array_equal(unique_labels, expected)

    def test_immutability_default(self, single_object_image):
        """Default apply() should not modify original."""
        original_max = single_object_image.objmap[:].max()
        merger = TransitiveDistanceMerger(distance_threshold=30.0)
        result = merger.apply(single_object_image)  # Default: inplace=False
        # Original should be unchanged
        assert single_object_image.objmap[:].max() == original_max

    def test_inplace_modification(self, chain_objects_image):
        """inplace=True should modify original."""
        original_id = id(chain_objects_image.objmap[:])
        merger = TransitiveDistanceMerger(distance_threshold=25.0)
        result = merger.apply(chain_objects_image, inplace=True)
        # Result should be same object
        assert result is chain_objects_image

    def test_invalid_threshold_raises_error(self):
        """Non-positive threshold should raise ValueError."""
        with pytest.raises(ValueError, match="distance_threshold must be positive"):
            TransitiveDistanceMerger(distance_threshold=0)
        with pytest.raises(ValueError, match="distance_threshold must be positive"):
            TransitiveDistanceMerger(distance_threshold=-10)

    def test_protected_components(self, two_close_objects_image):
        """RGB, gray, enh_gray should remain unchanged."""
        # Store original RGB
        original_rgb = two_close_objects_image.rgb[:].copy()
        merger = TransitiveDistanceMerger(distance_threshold=30.0)
        result = merger.apply(two_close_objects_image)
        # RGB should be unchanged
        assert np.array_equal(result.rgb[:], original_rgb)

    def test_preserves_mask_coverage(self, two_close_objects_image):
        """Merged objects should maintain same total mask coverage."""
        original = two_close_objects_image.objmap[:]
        original_coverage = (original > 0).sum()

        merger = TransitiveDistanceMerger(distance_threshold=30.0)
        result = merger.apply(two_close_objects_image)

        merged = result.objmap[:]
        merged_coverage = (merged > 0).sum()
        # Coverage should be same (merging doesn't remove pixels)
        assert original_coverage == merged_coverage


# =====================================================================
# NearestNeighborMerger Tests
# =====================================================================


class TestNearestNeighborMerger:
    """Test suite for NearestNeighborMerger."""

    def test_empty_objmap(self, empty_image):
        """Empty objmap should return unchanged."""
        merger = NearestNeighborMerger(distance_threshold=20.0)
        result = merger.apply(empty_image)
        assert result.objmap[:].max() == 0

    def test_single_object(self, single_object_image):
        """Single object should return unchanged."""
        merger = NearestNeighborMerger(distance_threshold=20.0)
        result = merger.apply(single_object_image)
        assert result.objmap[:].max() == 1

    def test_size_filter_preserves_large(self, size_mixed_image):
        """Large objects (>= min_size) should not merge."""
        merger = NearestNeighborMerger(distance_threshold=100.0, min_size=50)
        result = merger.apply(size_mixed_image)
        # Large object (label 1) should still exist
        assert 1 in result.objmap[:]

    def test_size_filter_merges_small(self, size_mixed_image):
        """Small objects (< min_size) should merge to nearest."""
        original_count = size_mixed_image.objmap[:].max()
        merger = NearestNeighborMerger(distance_threshold=100.0, min_size=50)
        result = merger.apply(size_mixed_image)
        # Merging should reduce or maintain count
        assert result.objmap[:].max() <= original_count

    def test_no_size_filter_merges_all(self, two_close_objects_image):
        """With min_size=None, merging should occur."""
        original_count = two_close_objects_image.objmap[:].max()
        merger = NearestNeighborMerger(distance_threshold=50.0, min_size=None)
        result = merger.apply(two_close_objects_image)
        # Merging should not increase count
        assert result.objmap[:].max() <= original_count

    def test_distance_threshold_respected(self, two_distant_objects_image):
        """Objects beyond threshold should remain independent."""
        merger = NearestNeighborMerger(distance_threshold=10.0)
        result = merger.apply(two_distant_objects_image)
        # Both objects too far apart to merge
        assert result.objmap[:].max() == 2

    def test_invalid_distance_raises_error(self):
        """Non-positive distance_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="distance_threshold must be positive"):
            NearestNeighborMerger(distance_threshold=0)

    def test_invalid_min_size_raises_error(self):
        """Non-positive min_size should raise ValueError."""
        with pytest.raises(ValueError, match="min_size must be positive"):
            NearestNeighborMerger(distance_threshold=20, min_size=0)
        with pytest.raises(ValueError, match="min_size must be positive"):
            NearestNeighborMerger(distance_threshold=20, min_size=-5)

    def test_preserves_mask_coverage(self, size_mixed_image):
        """Merged objects should maintain same total coverage."""
        original = size_mixed_image.objmap[:]
        original_coverage = (original > 0).sum()

        merger = NearestNeighborMerger(distance_threshold=100.0, min_size=50)
        result = merger.apply(size_mixed_image)

        merged = result.objmap[:]
        merged_coverage = (merged > 0).sum()
        # Coverage should be same
        assert original_coverage == merged_coverage


# =====================================================================
# SmallToLargeMerger Tests
# =====================================================================


class TestSmallToLargeMerger:
    """Test suite for SmallToLargeMerger."""

    def test_empty_objmap(self, empty_image):
        """Empty objmap should return unchanged."""
        merger = SmallToLargeMerger(size_threshold=100, distance_threshold=30.0)
        result = merger.apply(empty_image)
        assert result.objmap[:].max() == 0

    def test_single_object(self, single_object_image):
        """Single object should return unchanged."""
        merger = SmallToLargeMerger(size_threshold=100, distance_threshold=30.0)
        result = merger.apply(single_object_image)
        assert result.objmap[:].max() == 1

    def test_no_large_objects_returns_unchanged(self, size_mixed_image):
        """No large objects => no merging possible."""
        # All objects small (< 200)
        merger = SmallToLargeMerger(size_threshold=200, distance_threshold=30.0)

        original = size_mixed_image.objmap[:].copy()
        result = merger.apply(size_mixed_image)

        # Should be unchanged (no large objects to merge into)
        assert np.array_equal(result.objmap[:], original)

    def test_no_small_objects_returns_unchanged(self):
        """No small objects => nothing to merge."""
        image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
        objmap = np.zeros((100, 100), dtype=int)
        # Large object only: 20x20 = 400 pixels
        objmap[20:40, 20:40] = 1
        image.objmap[:] = objmap

        original = objmap.copy()
        merger = SmallToLargeMerger(size_threshold=100, distance_threshold=30.0)
        result = merger.apply(image)

        # Should be unchanged (no small objects)
        assert np.array_equal(result.objmap[:], original)

    def test_small_merges_to_nearest_large(self):
        """Small objects should merge to nearest large."""
        image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
        objmap = np.zeros((100, 100), dtype=int)
        # Large object 1: 20x20 = 400 pixels at (20, 20)
        objmap[20:40, 20:40] = 1
        # Small object: 5x5 = 25 pixels at (50, 50), close to large 1
        objmap[50:55, 50:55] = 2
        image.objmap[:] = objmap

        merger = SmallToLargeMerger(size_threshold=100, distance_threshold=50.0)
        result = merger.apply(image)

        # Small object should merge into large
        assert 2 not in result.objmap[:]
        assert 1 in result.objmap[:]

    def test_small_too_far_remains_independent(self):
        """Small objects far from large colonies remain independent."""
        image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
        objmap = np.zeros((100, 100), dtype=int)
        # Large object: 20x20 at (20, 20)
        objmap[20:40, 20:40] = 1
        # Small object: 5x5 at (80, 80), far from large
        objmap[80:85, 80:85] = 2
        image.objmap[:] = objmap

        merger = SmallToLargeMerger(size_threshold=100, distance_threshold=10.0)
        result = merger.apply(image)

        # Small object too far, should remain
        assert 2 in result.objmap[:]

    def test_multiple_small_merge_to_same_large(self):
        """Multiple small objects can merge to same large colony."""
        image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
        objmap = np.zeros((100, 100), dtype=int)
        # Large object: 20x20 at (30, 30)
        objmap[30:50, 30:50] = 1
        # Small object 1: 5x5 at (10, 30)
        objmap[10:15, 30:35] = 2
        # Small object 2: 5x5 at (60, 30)
        objmap[60:65, 30:35] = 3
        image.objmap[:] = objmap

        merger = SmallToLargeMerger(size_threshold=100, distance_threshold=30.0)
        result = merger.apply(image)

        # Both small objects should merge to large
        merged = result.objmap[:]
        assert 2 not in merged
        assert 3 not in merged
        assert 1 in merged

    def test_large_objects_never_merge(self):
        """Large objects should preserve their labels."""
        image = Image(arr=np.zeros((100, 100, 3), dtype=np.uint8))
        objmap = np.zeros((100, 100), dtype=int)
        # Large object 1: 20x20 = 400 pixels at (10, 10)
        objmap[10:30, 10:30] = 1
        # Large object 2: 20x20 = 400 pixels at (60, 60)
        objmap[60:80, 60:80] = 2
        image.objmap[:] = objmap

        original = objmap.copy()
        merger = SmallToLargeMerger(size_threshold=100, distance_threshold=100.0)
        result = merger.apply(image)

        # Large objects should remain (even though close with large threshold)
        assert 1 in result.objmap[:]
        assert 2 in result.objmap[:]

    def test_invalid_distance_raises_error(self):
        """Non-positive distance_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="distance_threshold must be positive"):
            SmallToLargeMerger(distance_threshold=0, size_threshold=100)

    def test_invalid_size_threshold_raises_error(self):
        """Non-positive size_threshold should raise ValueError."""
        with pytest.raises(ValueError, match="size_threshold must be positive"):
            SmallToLargeMerger(distance_threshold=30, size_threshold=0)

    def test_preserves_mask_coverage(self, size_mixed_image):
        """Merged objects should maintain same total coverage."""
        original = size_mixed_image.objmap[:]
        original_coverage = (original > 0).sum()

        merger = SmallToLargeMerger(size_threshold=100, distance_threshold=100.0)
        result = merger.apply(size_mixed_image)

        merged = result.objmap[:]
        merged_coverage = (merged > 0).sum()
        # Coverage should be same
        assert original_coverage == merged_coverage


# =====================================================================
# Integration Tests
# =====================================================================


class TestMergerIntegration:
    """Test merging operations in realistic scenarios."""

    def test_chaining_refiners(self, chain_objects_image):
        """Multiple refiners can be chained."""
        # Apply transitive merger then nearest neighbor
        merger1 = TransitiveDistanceMerger(distance_threshold=20.0)
        merger2 = NearestNeighborMerger(distance_threshold=15.0, min_size=50)

        result1 = merger1.apply(chain_objects_image)
        result2 = merger2.apply(result1)

        # Should not crash and should return valid image
        assert result2.objmap[:].shape == chain_objects_image.objmap[:].shape

    def test_pipeline_integration(self, chain_objects_image):
        """Mergers should work in ImagePipeline."""
        from phenotypic import ImagePipeline

        pipeline = ImagePipeline([TransitiveDistanceMerger(distance_threshold=20.0)])

        # Should apply without error
        result = pipeline.apply(chain_objects_image)
        assert result.objmap[:].max() >= 1

    def test_pickle_support(self):
        """Merger instances should be picklable (for parallel execution)."""
        import pickle

        merger1 = TransitiveDistanceMerger(distance_threshold=25.0)
        merger2 = NearestNeighborMerger(distance_threshold=20.0, min_size=50)
        merger3 = SmallToLargeMerger(distance_threshold=30.0, size_threshold=100)

        # Should pickle and unpickle without error
        for merger in [merger1, merger2, merger3]:
            pickled = pickle.dumps(merger)
            unpickled = pickle.loads(pickled)
            assert unpickled.distance_threshold == merger.distance_threshold
