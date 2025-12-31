"""Unit tests for core PipelineGridSearch functionality.

Tests for trie logic, memory estimation, and parameter handling.
"""
import numpy as np
import pytest

from phenotypic.enhance import GaussianBlur, MedianFilter
from phenotypic.detect import OtsuDetector, CannyDetector
from phenotypic.util._pipeline_grid_search._shared import (
    _build_pipeline_trie,
    _group_pipelines_by_longest_prefix,
    _analyze_trie_structure,
    _ops_key,
    _estimate_pipeline_memory,
    _calculate_optimal_batch_size,
)


class TestTriePathCounting:
    """Test that trie path counting uses summation, not multiplication."""
    
    def test_trie_path_counting_with_branches(self):
        """Verify multiple leaf pipelines are counted correctly (not multiplied)."""
        configs = [
            {"name": "p1", "ops": [(GaussianBlur(sigma=1), {}), (OtsuDetector(), {})]},
            {"name": "p2", "ops": [(GaussianBlur(sigma=1), {}), (CannyDetector(), {})]},
            {"name": "p3", "ops": [(GaussianBlur(sigma=1), {}), (MedianFilter(width=3), {})]},
        ]
        root = _build_pipeline_trie(configs)
        stats = _analyze_trie_structure(root)
        
        # With 3 pipelines diverging at detector, we should have:
        # - Shared GaussianBlur prefix
        # - 3 branches at detector level
        # - Total paths = 3 (one for each detector), not 3x3=9 (multiplication)
        assert stats["total_leaf_paths"] == 3, \
            f"Expected 3 paths (sum), got {stats['total_leaf_paths']}"


class TestTrieGrouping:
    """Test pipeline grouping logic."""
    
    def test_parameter_sweep_grouping(self):
        """Verify parameter sweeps are grouped together."""
        # These are CONCRETE configs (already expanded from parameter sweep)
        # Both pipelines diverge at GaussianBlur (sigma differs), then share OtsuDetector
        configs = [
            {"name": "p1_sigma=1", "ops": [(GaussianBlur(sigma=1), {"sigma": 1}), (OtsuDetector(), {})]},
            {"name": "p2_sigma=2", "ops": [(GaussianBlur(sigma=2), {"sigma": 2}), (OtsuDetector(), {})]},
        ]
        groups = _group_pipelines_by_longest_prefix(configs)
        
        # Both should be in ONE group because they're both parameter sweep variants
        # of the same pipeline structure (just different GaussianBlur sigma values)
        assert len(groups) == 1, \
            f"Parameter sweep should create 1 group, got {len(groups)}"
        assert len(groups[0]) == 2, \
            f"Group should contain both pipelines, got {len(groups[0])}"
    
    def test_mixed_parameter_and_structural_branching(self):
        """Verify parameter sweeps AND structural divergence group together."""
        # Both diverge at first operation (different sigma AND different detector)
        configs = [
            {"name": "p1_sigma=1_otsu", "ops": [(GaussianBlur(sigma=1), {"sigma": 1}), (OtsuDetector(), {})]},
            {"name": "p2_sigma=2_canny", "ops": [(GaussianBlur(sigma=2), {"sigma": 2}), (CannyDetector(), {})]},
        ]
        groups = _group_pipelines_by_longest_prefix(configs)
        
        # BOTH should be in SAME group (branch at root)
        # Even though detectors differ, they're part of a mixed parameter sweep + structural divergence
        assert len(groups) == 1, \
            f"Mixed branching should create 1 group, got {len(groups)}"
        assert len(groups[0]) == 2, \
            f"Group should contain both pipelines, got {len(groups[0])}"
    
    def test_structural_divergence_after_shared_prefix(self):
        """Verify structural divergence is handled correctly after shared prefix."""
        configs = [
            {"name": "p1", "ops": [
                (GaussianBlur(sigma=1), {"sigma": 1}),
                (MedianFilter(width=3), {"width": 3}),
                (OtsuDetector(), {})
            ]},
            {"name": "p2", "ops": [
                (GaussianBlur(sigma=1), {"sigma": 1}),
                (MedianFilter(width=3), {"width": 3}),
                (CannyDetector(), {})
            ]},
        ]
        groups = _group_pipelines_by_longest_prefix(configs)
        
        # Should group together (shared prefix up to MedianFilter, then branch at detector level)
        assert len(groups) == 1, \
            f"Should create 1 group with shared prefix, got {len(groups)}"


class TestOpsKeyHandling:
    """Test _ops_key with unhashable parameter types."""
    
    def test_ops_key_with_list_params(self):
        """Verify _ops_key handles list parameters."""
        op = GaussianBlur(sigma=1)
        params = {"kernel_size": [3, 3]}  # List (unhashable)
        
        key = _ops_key(op, params)  # Should not crash
        
        assert isinstance(key, tuple), f"Key should be tuple, got {type(key)}"
        assert key[0] == "GaussianBlur", f"First element should be operation name"
    
    def test_ops_key_with_dict_params(self):
        """Verify _ops_key handles dict parameters."""
        op = MedianFilter(width=3)
        params = {"range": {"min": 0, "max": 1}}  # Dict (unhashable)
        
        key = _ops_key(op, params)  # Should not crash
        
        assert isinstance(key, tuple), f"Key should be tuple, got {type(key)}"
    
    def test_ops_key_with_numpy_array_params(self):
        """Verify _ops_key handles numpy array parameters."""
        op = GaussianBlur(sigma=1)
        params = {"values": np.array([1, 2, 3])}  # Array (unhashable)
        
        key = _ops_key(op, params)  # Should not crash
        
        assert isinstance(key, tuple), f"Key should be tuple, got {type(key)}"
    
    def test_ops_key_consistency(self):
        """Verify identical params produce identical keys."""
        op = GaussianBlur(sigma=1)
        params = {"kernel": [3, 3], "range": {"min": 0, "max": 1}}
        
        key1 = _ops_key(op, params)
        key2 = _ops_key(op, params)
        
        assert key1 == key2, "Identical params should produce identical keys"


class TestMemoryEstimation:
    """Test memory estimation without unnecessary copies."""
    
    def test_memory_estimation_returns_positive(self):
        """Verify memory estimation returns a positive value."""
        from phenotypic import Image
        from pathlib import Path
        
        # Use test image if available
        test_image_path = Path(__file__).parent / "resources" / "test_plate.jpg"
        if test_image_path.exists():
            image = Image.imread(str(test_image_path))
            mem_est = _estimate_pipeline_memory(image, num_operations=5, data_layers=["rgb"])
            
            assert mem_est > 0, f"Memory estimation should be positive, got {mem_est}"
    
    def test_batch_size_calculation_with_valid_memory(self):
        """Verify batch size calculation with valid memory estimates."""
        batch_size, jobs = _calculate_optimal_batch_size(
            total_pipelines=10,
            memory_per_pipeline=50_000_000,  # 50 MB
            memory_limit_gb=4.0,
            n_jobs=-1
        )
        
        assert batch_size > 0, f"Batch size should be positive, got {batch_size}"
        assert jobs > 0, f"Jobs should be positive, got {jobs}"
        assert batch_size <= 10, f"Batch size should not exceed total pipelines"
    
    def test_batch_size_calculation_fallback_on_zero_memory(self):
        """Verify fallback when memory estimation returns 0."""
        batch_size, jobs = _calculate_optimal_batch_size(
            total_pipelines=10,
            memory_per_pipeline=0,  # Invalid estimate
            memory_limit_gb=4.0,
            n_jobs=-1
        )
        
        # Should not crash and should return positive values
        assert batch_size > 0, f"Batch size should be positive (fallback), got {batch_size}"
        assert jobs > 0, f"Jobs should be positive (fallback), got {jobs}"
    
    def test_batch_size_max_configurable(self):
        """Verify max_batch_size parameter is respected."""
        batch_size_default, _ = _calculate_optimal_batch_size(
            total_pipelines=100,
            memory_per_pipeline=10_000_000,  # 10 MB
            memory_limit_gb=8.0,
            n_jobs=-1
        )
        
        batch_size_large, _ = _calculate_optimal_batch_size(
            total_pipelines=100,
            memory_per_pipeline=10_000_000,  # 10 MB
            memory_limit_gb=8.0,
            n_jobs=-1,
            max_batch_size=32  # Allow larger batches
        )
        
        # With larger max_batch_size, should allow bigger batches
        assert batch_size_large >= batch_size_default or batch_size_large == 100, \
            f"Larger max_batch_size should allow >= default, got {batch_size_large} vs {batch_size_default}"
