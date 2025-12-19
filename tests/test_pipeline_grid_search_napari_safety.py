"""Integration tests for napari data integrity in pipeline grid search.

Tests for memory safety when napari receives data from Image objects that are
subsequently deleted. Regression tests for issue where non-optimized path passed
Image objects to napari, then deleted them while napari held references.
"""
import gc
import numpy as np
import pytest

from phenotypic import Image
from phenotypic.enhance import GaussianBlur, MedianFilter
from phenotypic.detect import OtsuDetector
from phenotypic.util import PipelineGridSearch, MultiPipelineGridSearch


class TestNapariDataIntegrityOptimizedPath:
    """Test napari data integrity in optimized (trie) path."""

    def test_optimized_napari_data_valid_after_image_deletion(self):
        """Test that optimized path produces valid napari data after Image deletion.

        In the optimized path, arrays are extracted BEFORE napari receives them.
        This test verifies that napari layers remain valid after source Images are deleted.
        """
        # Create test image
        image = Image(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))

        # Run grid search with optimized path (trie)
        pipeline_configs = [
            {
                "name": "Pipeline1",
                "ops": [(GaussianBlur(sigma=1), {"sigma": [1, 2]})],
            },
            {
                "name": "Pipeline2",
                "ops": [(MedianFilter(width=3), {"width": [3, 5]})],
            },
        ]

        viewer, configs = MultiPipelineGridSearch(
            image=image,
            pipeline_configs=pipeline_configs,
            optimize_shared_prefixes=True,  # Optimized path
            data_layers=["rgb", "gray"],
            n_jobs=1,
        )

        # Force garbage collection
        gc.collect()

        # Verify napari layers are still accessible and valid
        layer_count = 0
        for layer in viewer.layers:
            if layer.name.startswith("Pipeline"):
                layer_count += 1

                # Access layer data - should not segfault or return garbage
                data = layer.data
                assert data is not None, f"Layer {layer.name} has None data"
                assert data.size > 0, f"Layer {layer.name} has empty data"

                # Verify data is not corrupted (has expected shape)
                if "_rgb" in layer.name:
                    assert len(data.shape) in [2, 3], f"RGB layer {layer.name} has unexpected shape {data.shape}"
                    assert data.shape[0] == 100, f"Layer {layer.name} height mismatch"
                    assert data.shape[1] == 100, f"Layer {layer.name} width mismatch"
                elif "_gray" in layer.name:
                    assert len(data.shape) == 2, f"Gray layer {layer.name} has unexpected shape {data.shape}"
                    assert data.shape[0] == 100, f"Layer {layer.name} height mismatch"
                    assert data.shape[1] == 100, f"Layer {layer.name} width mismatch"

                # Verify data is not all zeros (would indicate freed memory)
                assert data.sum() > 0, f"Layer {layer.name} is all zeros (possible freed memory)"

        assert layer_count > 0, "No pipeline layers found in viewer"
        viewer.close()

    def test_optimized_path_multiple_pipelines_memory_safe(self):
        """Test that optimized path handles multiple pipelines safely without memory errors."""
        image = Image(np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8))

        # Multiple pipelines with parameter sweeps
        from phenotypic.detect import OtsuDetector

        pipeline_configs = [
            {
                "name": "Blur",
                "ops": [(GaussianBlur(sigma=2), {"sigma": [2]})],
            },
            {
                "name": "Detection",
                "ops": [(GaussianBlur(sigma=2), {"sigma": [2]}), (OtsuDetector(), {})],
            },
        ]

        # This should run without memory errors
        viewer, configs = MultiPipelineGridSearch(
            image=image,
            pipeline_configs=pipeline_configs,
            optimize_shared_prefixes=True,
            data_layers=["gray"],
            n_jobs=1,
        )

        # Force garbage collection to simulate aggressive cleanup
        gc.collect()

        # Verify viewer has layers (regardless of exact data values)
        assert len(viewer.layers) > 0, "No layers created in viewer"

        # All layers should be accessible without memory errors
        for layer in viewer.layers:
            if not layer.name.startswith("Original"):
                # Access layer data - should not segfault or return garbage
                data = layer.data
                assert data is not None, f"Layer {layer.name} has None data"
                assert data.size > 0, f"Layer {layer.name} has empty data"

        viewer.close()


class TestNapariDataIntegrityNonOptimizedPath:
    """Test napari data integrity in non-optimized path (now fixed).

    This tests the CRITICAL fix where non-optimized path now extracts arrays
    BEFORE passing to napari (matching optimized path behavior).
    """

    def test_non_optimized_napari_data_valid_after_image_deletion(self):
        """Test that non-optimized path produces valid napari data after Image deletion.

        Regression test for issue where non-optimized path passed Image objects
        to napari, then deleted them while napari held references to their arrays.

        After the fix, non-optimized path also extracts arrays before napari,
        so this should pass safely.
        """
        # Create test image
        image = Image(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))

        # Run grid search with non-optimized path (optimize_shared_prefixes=False)
        pipeline_configs = [
            {
                "name": "Pipeline1",
                "ops": [(GaussianBlur(sigma=1), {"sigma": [1, 2]})],
            },
        ]

        viewer, configs = MultiPipelineGridSearch(
            image=image,
            pipeline_configs=pipeline_configs,
            optimize_shared_prefixes=False,  # Non-optimized path
            data_layers=["rgb", "gray"],
            n_jobs=1,
        )

        # Force garbage collection to ensure Image objects are freed
        gc.collect()

        # Verify napari layers are still accessible and valid
        layer_count = 0
        for layer in viewer.layers:
            if layer.name.startswith("Pipeline1_"):
                layer_count += 1

                # Access layer data - should not segfault or return garbage
                data = layer.data
                assert data is not None, f"Layer {layer.name} has None data"
                assert data.size > 0, f"Layer {layer.name} has empty data"

                # Verify data is not corrupted
                assert data.shape[0] == 100, f"Layer {layer.name} height mismatch"
                assert data.shape[1] == 100, f"Layer {layer.name} width mismatch"

                # Verify data is not all zeros (would indicate freed memory)
                assert data.sum() > 0, f"Layer {layer.name} is all zeros (possible freed memory)"

        assert layer_count > 0, "No Pipeline1 layers found in viewer"
        viewer.close()

    def test_non_optimized_parameter_sweep_memory_safe(self):
        """Test that non-optimized path handles parameter sweeps safely without memory errors."""
        image = Image(np.random.randint(50, 200, (50, 50, 3), dtype=np.uint8))

        # Parameter sweep with different sigma values
        pipeline_configs = [
            {
                "name": "GaussBlur",
                "ops": [(GaussianBlur(sigma=1), {"sigma": [1, 3, 5]})],
            },
        ]

        # This should run without memory errors
        viewer, configs = MultiPipelineGridSearch(
            image=image,
            pipeline_configs=pipeline_configs,
            optimize_shared_prefixes=False,
            data_layers=["gray"],
            n_jobs=1,
        )

        # Force garbage collection to simulate aggressive cleanup
        gc.collect()

        # Extract gray layers from different parameter values
        layers = [layer for layer in viewer.layers if "GaussBlur" in layer.name and "gray" in layer.name]

        # Should have 3 layers (one for each sigma value)
        assert len(layers) == 3, f"Expected 3 parameter sweeps, got {len(layers)}"

        # Verify all have valid data and can be accessed without memory errors
        for layer in layers:
            data = layer.data
            assert data is not None, f"Layer {layer.name} has None data"
            assert data.size > 0, f"Layer {layer.name} has empty data"
            assert not np.all(np.isnan(data)), f"Layer {layer.name} contains NaN values"
            assert not np.all(np.isinf(data)), f"Layer {layer.name} contains Inf values"

        viewer.close()


class TestPipelineGridSearchNapariSafety:
    """Test napari safety in the simple PipelineGridSearch function."""

    def test_pipeline_grid_search_napari_data_valid(self):
        """Test that PipelineGridSearch produces valid napari data."""
        image = Image(np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8))

        # Simple parameter sweep
        ops = [(GaussianBlur(sigma=1), {"sigma": [1, 2, 3]})]

        viewer, configs = PipelineGridSearch(
            image=image,
            ops=ops,
            data_layers=["rgb", "gray"],
            n_jobs=1,
        )

        # Force garbage collection
        gc.collect()

        # Verify layers are valid
        layer_count = 0
        for layer in viewer.layers:
            if "_" in layer.name and not layer.name.startswith("Original"):
                layer_count += 1
                data = layer.data
                assert data is not None
                assert data.size > 0
                assert data.sum() > 0  # Not all zeros

        assert layer_count > 0, "No result layers found"
        viewer.close()
