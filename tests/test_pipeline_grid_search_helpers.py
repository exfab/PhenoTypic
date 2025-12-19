"""Comprehensive tests for _pipeline_grid_search helper functions.

Tests cover all internal helper functions to ensure expected behavior,
error handling, and edge cases.
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from phenotypic.util._pipeline_grid_search import (
    _ops_key,
    _TrieNode,
    _unpack_ops_tuples,
    _generate_param_combinations,
    _create_param_name_string,
    _expand_pipeline_configs_to_concrete,
    _build_pipeline_trie,
    _build_results_dict,
    _build_configs_dict,
    _extract_data_layers,
    _estimate_pipeline_memory,
)


# ============================================================================
# Mock Classes for Testing
# ============================================================================

class MockOperation:
    """Mock ImageOperation for testing."""

    def __init__(self, name: str = "MockOp"):
        self.name = name
        self.sigma = 1.0
        self.threshold = 100
        self.size = 3


class MockOperationA(MockOperation):
    """First variant of mock operation."""
    pass


class MockOperationB(MockOperation):
    """Second variant of mock operation."""
    pass


class MockImage:
    """Mock Image for testing."""

    def __init__(self, name: str = "test_image"):
        self.name = name
        self.data = None

    def copy(self):
        """Return a copy of the image."""
        return MockImage(self.name)


# ============================================================================
# Tests for _ops_key
# ============================================================================

class TestOpsKey:
    """Tests for _ops_key helper function."""

    def test_ops_key_basic(self):
        """Test basic operation key generation."""
        op = MockOperation("Op1")
        params = {"sigma": 2.0}
        key = _ops_key(op, params)

        # Key should be tuple of (class_name, sorted_params)
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert key[0] == "MockOperation"
        assert key[1] == (("sigma", 2.0),)

    def test_ops_key_empty_params(self):
        """Test key generation with empty parameters."""
        op = MockOperation()
        key = _ops_key(op, {})

        assert key[0] == "MockOperation"
        assert key[1] == ()

    def test_ops_key_multiple_params(self):
        """Test key generation with multiple parameters."""
        op = MockOperation()
        params = {"sigma": 1.5, "threshold": 200, "size": 5}
        key = _ops_key(op, params)

        # Params should be sorted
        assert key[1] == (("sigma", 1.5), ("size", 5), ("threshold", 200))

    def test_ops_key_sorted_params(self):
        """Test that parameter order doesn't matter."""
        op = MockOperation()
        params1 = {"sigma": 2.0, "threshold": 100}
        params2 = {"threshold": 100, "sigma": 2.0}

        key1 = _ops_key(op, params1)
        key2 = _ops_key(op, params2)

        assert key1 == key2

    def test_ops_key_different_classes_same_params(self):
        """Test that different operation classes produce different keys."""
        op1 = MockOperationA()
        op2 = MockOperationB()

        params = {"sigma": 1.0}

        key1 = _ops_key(op1, params)
        key2 = _ops_key(op2, params)

        assert key1[0] != key2[0]
        assert key1[1] == key2[1]

    def test_ops_key_hashable(self):
        """Test that ops_key returns hashable value."""
        op = MockOperation()
        params = {"sigma": 1.0, "threshold": 100}
        key = _ops_key(op, params)

        # Should be able to use as dict key
        d = {key: "value"}
        assert d[key] == "value"


# ============================================================================
# Tests for _TrieNode
# ============================================================================

class TestTrieNode:
    """Tests for _TrieNode dataclass."""

    def test_trie_node_creation(self):
        """Test basic trie node creation."""
        op = MockOperation()
        params = {"sigma": 1.0}

        node = _TrieNode(op=op, params=params)

        assert node.op is op
        assert node.params == params
        assert node.children == {}
        assert node.pipeline_names == []

    def test_trie_node_root(self):
        """Test root node creation (no op/params)."""
        root = _TrieNode()

        assert root.op is None
        assert root.params is None
        assert root.children == {}
        assert root.pipeline_names == []

    def test_trie_node_add_child(self):
        """Test adding children to trie node."""
        root = _TrieNode()
        op = MockOperation()
        key = _ops_key(op, {})

        child = _TrieNode(op=op, params={})
        root.children[key] = child

        assert key in root.children
        assert root.children[key] is child

    def test_trie_node_add_pipeline_name(self):
        """Test adding pipeline names to node."""
        node = _TrieNode()

        node.pipeline_names.append("Pipeline1")
        node.pipeline_names.append("Pipeline2")

        assert len(node.pipeline_names) == 2
        assert "Pipeline1" in node.pipeline_names
        assert "Pipeline2" in node.pipeline_names


# ============================================================================
# Tests for _unpack_ops_tuples
# ============================================================================

class TestUnpackOpsTuples:
    """Tests for _unpack_ops_tuples helper function."""

    def test_unpack_single_operation(self):
        """Test unpacking single operation tuple."""
        op = MockOperation()
        params = {"sigma": [1.0, 2.0]}
        ops = [(op, params)]

        operations, parameters = _unpack_ops_tuples(ops)

        assert len(operations) == 1
        assert len(parameters) == 1
        assert operations[0] is op
        assert parameters[0] == params

    def test_unpack_multiple_operations(self):
        """Test unpacking multiple operation tuples."""
        op1 = MockOperation("Op1")
        op2 = MockOperation("Op2")
        params1 = {"sigma": [1.0, 2.0]}
        params2 = {"threshold": [50, 100]}

        ops = [(op1, params1), (op2, params2)]

        operations, parameters = _unpack_ops_tuples(ops)

        assert len(operations) == 2
        assert operations[0] is op1
        assert operations[1] is op2
        assert parameters[0] == params1
        assert parameters[1] == params2

    def test_unpack_empty_params(self):
        """Test unpacking with empty parameter dict."""
        op = MockOperation()
        ops = [(op, {})]

        operations, parameters = _unpack_ops_tuples(ops)

        assert operations[0] is op
        assert parameters[0] == {}


# ============================================================================
# Tests for _generate_param_combinations
# ============================================================================

class TestGenerateParamCombinations:
    """Tests for _generate_param_combinations helper function."""

    def test_single_operation_single_param(self):
        """Test generating combinations for single operation with one parameter."""
        params = [{"sigma": [1.0, 2.0, 3.0]}]
        combos = _generate_param_combinations(params)

        assert len(combos) == 3
        assert combos[0] == ({"sigma": 1.0},)
        assert combos[1] == ({"sigma": 2.0},)
        assert combos[2] == ({"sigma": 3.0},)

    def test_single_operation_multiple_params(self):
        """Test combinations for single operation with multiple parameters."""
        params = [{"sigma": [1.0, 2.0], "threshold": [50, 100]}]
        combos = _generate_param_combinations(params)

        # Should generate 2x2 = 4 combinations
        assert len(combos) == 4

    def test_multiple_operations(self):
        """Test combinations across multiple operations."""
        params = [
            {"sigma": [1.0, 2.0]},  # 2 values
            {"threshold": [50, 100, 150]},  # 3 values
        ]
        combos = _generate_param_combinations(params)

        # Should generate 2x3 = 6 combinations
        assert len(combos) == 6
        assert combos[0] == ({"sigma": 1.0}, {"threshold": 50})
        assert combos[-1] == ({"sigma": 2.0}, {"threshold": 150})

    def test_empty_params_dict(self):
        """Test with empty parameter dict (no params to vary)."""
        params = [{}]
        combos = _generate_param_combinations(params)

        assert len(combos) == 1
        assert combos[0] == ({},)

    def test_mixed_empty_and_params(self):
        """Test mix of operations with and without parameters."""
        params = [
            {"sigma": [1.0, 2.0]},
            {},  # No parameters
            {"threshold": [50, 100]},
        ]
        combos = _generate_param_combinations(params)

        # 2 x 1 x 2 = 4 combinations
        assert len(combos) == 4

    def test_single_value_per_param(self):
        """Test with single value per parameter."""
        params = [{"sigma": [1.0]}, {"threshold": [100]}]
        combos = _generate_param_combinations(params)

        assert len(combos) == 1
        assert combos[0] == ({"sigma": 1.0}, {"threshold": 100})


# ============================================================================
# Tests for _create_param_name_string
# ============================================================================

class TestCreateParamNameString:
    """Tests for _create_param_name_string helper function."""

    def test_single_param_single_operation(self):
        """Test creating name string for single parameter."""
        param_config = ({"sigma": 1.0},)
        name = _create_param_name_string(param_config)

        assert name == "sigma=1.0"

    def test_multiple_params_single_operation(self):
        """Test with multiple parameters in single operation."""
        param_config = ({"sigma": 1.0, "threshold": 100},)
        name = _create_param_name_string(param_config)

        # Order may vary, check components exist
        assert "sigma=1.0" in name
        assert "threshold=100" in name
        assert "_" in name

    def test_multiple_operations(self):
        """Test with multiple operations."""
        param_config = ({"sigma": 1.0}, {"threshold": 100})
        name = _create_param_name_string(param_config)

        assert "sigma=1.0" in name
        assert "threshold=100" in name

    def test_empty_params(self):
        """Test with empty parameters."""
        param_config = ({},)
        name = _create_param_name_string(param_config)

        assert name == "default"

    def test_multiple_empty_params(self):
        """Test with multiple empty parameter dicts."""
        param_config = ({}, {}, {})
        name = _create_param_name_string(param_config)

        assert name == "default"

    def test_string_and_numeric_params(self):
        """Test with various parameter types."""
        param_config = ({"name": "test", "sigma": 2.5, "size": 3},)
        name = _create_param_name_string(param_config)

        assert "name=test" in name
        assert "sigma=2.5" in name
        assert "size=3" in name


# ============================================================================
# Tests for _expand_pipeline_configs_to_concrete
# ============================================================================

class TestExpandPipelineConfigsToConcrete:
    """Tests for _expand_pipeline_configs_to_concrete helper function."""

    def test_single_pipeline_no_params(self):
        """Test expansion with pipeline that has no parameters to vary."""
        op = MockOperation()
        config = {"name": "Pipeline1", "ops": [(op, {})]}

        concrete = _expand_pipeline_configs_to_concrete([config])

        assert len(concrete) == 1
        assert concrete[0]["name"] == "Pipeline1"
        assert len(concrete[0]["ops"]) == 1

    def test_single_pipeline_single_param_values(self):
        """Test expansion with single parameter having multiple values."""
        op = MockOperation()
        config = {"name": "Pipeline1", "ops": [(op, {"sigma": [1.0, 2.0, 3.0]})]}

        concrete = _expand_pipeline_configs_to_concrete([config])

        # Should create 3 concrete pipelines
        assert len(concrete) == 3
        assert concrete[0]["name"] == "Pipeline1_sigma=1.0"
        assert concrete[1]["name"] == "Pipeline1_sigma=2.0"
        assert concrete[2]["name"] == "Pipeline1_sigma=3.0"

        # Each should have scalar param value
        assert concrete[0]["ops"][0][1] == {"sigma": 1.0}
        assert concrete[1]["ops"][0][1] == {"sigma": 2.0}
        assert concrete[2]["ops"][0][1] == {"sigma": 3.0}

    def test_single_pipeline_multiple_params(self):
        """Test expansion with multiple parameters (Cartesian product)."""
        op = MockOperation()
        config = {"name": "Pipeline1", "ops": [(op, {"sigma": [1.0, 2.0], "threshold": [50, 100]})]}

        concrete = _expand_pipeline_configs_to_concrete([config])

        # Should create 2x2 = 4 concrete pipelines
        assert len(concrete) == 4

    def test_multiple_pipelines(self):
        """Test expansion with multiple pipeline configs."""
        op1 = MockOperation()
        op2 = MockOperation()
        configs = [
            {"name": "Pipeline1", "ops": [(op1, {"sigma": [1.0, 2.0]})]},
            {"name": "Pipeline2", "ops": [(op2, {"size": [3, 5]})]},
        ]

        concrete = _expand_pipeline_configs_to_concrete(configs)

        # Pipeline1: 2 combos, Pipeline2: 2 combos = 4 total
        assert len(concrete) == 4

    def test_multiple_operations_per_pipeline(self):
        """Test expansion with pipeline having multiple operations."""
        op1 = MockOperation()
        op2 = MockOperation()
        config = {
            "name": "Pipeline1",
            "ops": [
                (op1, {"sigma": [1.0, 2.0]}),
                (op2, {"threshold": [50, 100]}),
            ],
        }

        concrete = _expand_pipeline_configs_to_concrete([config])

        # 2 x 2 = 4 combinations
        assert len(concrete) == 4

        # Check first combo has correct structure
        first = concrete[0]
        assert len(first["ops"]) == 2
        assert isinstance(first["ops"][0][1]["sigma"], float)
        assert isinstance(first["ops"][1][1]["threshold"], int)

    def test_params_are_scalar_not_lists(self):
        """Test that expanded configs have scalar values, not lists."""
        op = MockOperation()
        config = {"name": "Pipeline1", "ops": [(op, {"sigma": [1.0, 2.0]})]}

        concrete = _expand_pipeline_configs_to_concrete([config])

        for c in concrete:
            for op, params in c["ops"]:
                for key, value in params.items():
                    # Value should not be a list
                    assert not isinstance(value, list)


# ============================================================================
# Tests for _build_pipeline_trie
# ============================================================================

class TestBuildPipelineTrie:
    """Tests for _build_pipeline_trie helper function."""

    def test_single_pipeline_single_op(self):
        """Test building trie with single pipeline, single operation."""
        op = MockOperation()
        config = {"name": "Pipeline1", "ops": [(op, {})]}

        trie = _build_pipeline_trie([config])

        # Root should have one child
        assert len(trie.children) == 1
        # Root should not be an endpoint
        assert len(trie.pipeline_names) == 0

    def test_single_pipeline_multiple_ops(self):
        """Test building trie with single pipeline, multiple operations."""
        op1 = MockOperation("Op1")
        op2 = MockOperation("Op2")
        config = {"name": "Pipeline1", "ops": [(op1, {}), (op2, {})]}

        trie = _build_pipeline_trie([config])

        # Root has one child
        assert len(trie.children) == 1
        # First child has one child
        first_child = list(trie.children.values())[0]
        assert len(first_child.children) == 1

    def test_shared_prefix_two_pipelines(self):
        """Test trie with two pipelines sharing prefix."""
        op1 = MockOperationA()
        op2 = MockOperationB()
        op3 = MockOperation("Op3")

        config1 = {"name": "Pipeline1", "ops": [(op1, {}), (op2, {})]}
        config2 = {"name": "Pipeline2", "ops": [(op1, {}), (op3, {})]}

        trie = _build_pipeline_trie([config1, config2])

        # Root should have one child (shared op1)
        assert len(trie.children) == 1

        # That child should have two children (op2 and op3)
        shared_child = list(trie.children.values())[0]
        assert len(shared_child.children) == 2

    def test_pipeline_endpoint_names(self):
        """Test that pipeline names are stored at endpoints."""
        op = MockOperation()
        config = {"name": "TestPipeline", "ops": [(op, {})]}

        trie = _build_pipeline_trie([config])

        # Find the endpoint
        first_child = list(trie.children.values())[0]
        assert "TestPipeline" in first_child.pipeline_names

    def test_multiple_pipelines_same_endpoint(self):
        """Test multiple pipelines ending at same point."""
        op = MockOperation()
        config1 = {"name": "Pipeline1", "ops": [(op, {})]}
        config2 = {"name": "Pipeline2", "ops": [(op, {})]}

        trie = _build_pipeline_trie([config1, config2])

        # Should have one child with two pipeline names
        child = list(trie.children.values())[0]
        assert len(child.pipeline_names) == 2
        assert "Pipeline1" in child.pipeline_names
        assert "Pipeline2" in child.pipeline_names

    def test_no_shared_prefixes(self):
        """Test pipelines with no shared operations."""
        op1 = MockOperationA()
        op2 = MockOperationB()

        config1 = {"name": "Pipeline1", "ops": [(op1, {})]}
        config2 = {"name": "Pipeline2", "ops": [(op2, {})]}

        trie = _build_pipeline_trie([config1, config2])

        # Root should have two children
        assert len(trie.children) == 2

    def test_shared_operation_same_params(self):
        """Test pipelines sharing operation with same parameter values."""
        op1 = MockOperationA()
        op2 = MockOperationB()
        op3 = MockOperation("Op3")

        # Both use op1 with sigma=1.0
        config1 = {"name": "Pipeline1", "ops": [(op1, {"sigma": 1.0}), (op2, {})]}
        config2 = {"name": "Pipeline2", "ops": [(op1, {"sigma": 1.0}), (op3, {})]}

        trie = _build_pipeline_trie([config1, config2])

        # Root should have one child (shared op1 with sigma=1.0)
        assert len(trie.children) == 1

        # That child should have two children (op2 and op3)
        shared_child = list(trie.children.values())[0]
        assert len(shared_child.children) == 2

    def test_same_operation_different_params(self):
        """Test pipelines with same operation but different parameter values."""
        op1 = MockOperationA()
        op2 = MockOperationA()

        # Same operation class but different params
        config1 = {"name": "Pipeline1", "ops": [(op1, {"sigma": 1.0})]}
        config2 = {"name": "Pipeline2", "ops": [(op2, {"sigma": 2.0})]}

        trie = _build_pipeline_trie([config1, config2])

        # Root should have two children (different params = different nodes)
        assert len(trie.children) == 2


# ============================================================================
# Tests for _build_results_dict
# ============================================================================

class TestBuildResultsDict:
    """Tests for _build_results_dict helper function."""

    def test_single_result(self):
        """Test building results dict with single result."""
        image = MockImage("result1")
        param_config = ({"sigma": 1.0},)

        results = [(image, param_config)]
        results_dict = _build_results_dict(results)

        assert len(results_dict) == 1
        # Key should be tuple of sorted params for each operation
        # So ({"sigma": 1.0},) becomes ((("sigma", 1.0),),)
        key = ((("sigma", 1.0),),)
        assert key in results_dict
        assert results_dict[key] is image

    def test_multiple_results(self):
        """Test building results dict with multiple results."""
        image1 = MockImage("result1")
        image2 = MockImage("result2")

        param_config1 = ({"sigma": 1.0},)
        param_config2 = ({"sigma": 2.0},)

        results = [(image1, param_config1), (image2, param_config2)]
        results_dict = _build_results_dict(results)

        assert len(results_dict) == 2

    def test_multiple_operations_params(self):
        """Test key generation with multiple operation parameters."""
        image = MockImage("result")
        param_config = ({"sigma": 1.0}, {"threshold": 100})

        results = [(image, param_config)]
        results_dict = _build_results_dict(results)

        # Key should have tuples for each operation
        assert len(results_dict) == 1

    def test_params_ordering_independence(self):
        """Test that parameter order doesn't affect result lookup."""
        image = MockImage("result")
        # Different order of parameters in dict
        param_config1 = ({"sigma": 1.0, "size": 3},)
        param_config2 = ({"size": 3, "sigma": 1.0},)

        results1 = [(image, param_config1)]
        results2 = [(image, param_config2)]

        dict1 = _build_results_dict(results1)
        dict2 = _build_results_dict(results2)

        # Should have same keys
        assert list(dict1.keys()) == list(dict2.keys())


# ============================================================================
# Tests for _build_configs_dict
# ============================================================================

class TestBuildConfigsDict:
    """Tests for _build_configs_dict helper function."""

    def test_single_config_no_prefix(self):
        """Test building configs dict without pipeline prefix."""
        image = MockImage()
        param_config = ({"sigma": 1.0},)
        json_config = '{"pipeline": "config"}'

        results = [(image, param_config, json_config)]
        configs_dict = _build_configs_dict(results)

        assert len(configs_dict) == 1
        # Should have entry with format "000_sigma=1.0"
        assert "000_sigma=1.0" in configs_dict
        assert configs_dict["000_sigma=1.0"] == json_config

    def test_single_config_with_prefix(self):
        """Test building configs dict with pipeline prefix."""
        image = MockImage()
        param_config = ({"sigma": 1.0},)
        json_config = '{"pipeline": "config"}'

        results = [(image, param_config, json_config)]
        configs_dict = _build_configs_dict(results, pipeline_name="TestPipeline")

        assert len(configs_dict) == 1
        # Should have entry with format "PipelineName_000_sigma=1.0"
        assert "TestPipeline_000_sigma=1.0" in configs_dict

    def test_multiple_configs(self):
        """Test building configs dict with multiple configurations."""
        image1 = MockImage()
        image2 = MockImage()
        param_config1 = ({"sigma": 1.0},)
        param_config2 = ({"sigma": 2.0},)
        json_config1 = '{"sigma": 1.0}'
        json_config2 = '{"sigma": 2.0}'

        results = [
            (image1, param_config1, json_config1),
            (image2, param_config2, json_config2),
        ]
        configs_dict = _build_configs_dict(results)

        assert len(configs_dict) == 2
        assert "000_sigma=1.0" in configs_dict
        assert "001_sigma=2.0" in configs_dict

    def test_configs_indexed_sequentially(self):
        """Test that configs are indexed 000, 001, 002, etc."""
        configs_list = []
        for i in range(5):
            image = MockImage()
            param_config = ({"param": i},)
            json_config = f'{{"value": {i}}}'
            configs_list.append((image, param_config, json_config))

        configs_dict = _build_configs_dict(configs_list)

        for i in range(5):
            key = f"{i:03d}_param={i}"
            assert key in configs_dict

    def test_configs_preserved_as_strings(self):
        """Test that JSON configs are preserved exactly."""
        image = MockImage()
        param_config = ({},)
        json_config = '{"exact": "json", "preserved": true}'

        results = [(image, param_config, json_config)]
        configs_dict = _build_configs_dict(results)

        stored_config = list(configs_dict.values())[0]
        assert stored_config == json_config


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple helper functions."""

    def test_full_trie_workflow(self):
        """Test complete trie building and inspection workflow."""
        op1 = MockOperationA()
        op2 = MockOperationB()
        op3 = MockOperation("Contrast")

        configs = [
            {"name": "DenoiseContrast", "ops": [(op1, {}), (op2, {}), (op3, {})]},
            {"name": "DenoiseOnly", "ops": [(op1, {}), (op2, {})]},
            {"name": "ContrastOnly", "ops": [(op3, {})]},
        ]

        trie = _build_pipeline_trie(configs)

        # Should have appropriate structure
        assert trie is not None
        # Should have at least 2 children (op1 and op3) at root level
        assert len(trie.children) >= 2

    def test_param_combinations_and_naming(self):
        """Test parameter combinations generation and naming."""
        params = [
            {"sigma": [1.0, 2.0]},
            {"threshold": [50, 100]},
        ]

        combos = _generate_param_combinations(params)
        assert len(combos) == 4

        # Each combination should produce valid name string
        for combo in combos:
            name = _create_param_name_string(combo)
            assert len(name) > 0
            assert "=" in name

    def test_results_and_configs_consistency(self):
        """Test that results and configs dicts are consistent."""
        images = [MockImage(f"img{i}") for i in range(3)]
        param_configs = [
            ({"sigma": 1.0},),
            ({"sigma": 2.0},),
            ({"sigma": 3.0},),
        ]
        json_configs = [f'{{"sigma": {i}}}' for i in range(3)]

        results = list(zip(images, param_configs, json_configs))

        results_dict = _build_results_dict([(img, cfg) for img, cfg, _ in results])
        configs_dict = _build_configs_dict(results)

        # Should have same number of entries
        assert len(results_dict) == len(configs_dict)


# ============================================================================
# Test _extract_data_layers
# ============================================================================

class TestExtractDataLayers:
    """Tests for _extract_data_layers helper function."""

    @pytest.fixture
    def mock_image_with_data(self):
        """Create a mock Image with realistic data arrays."""
        import numpy as np

        class MockImageWithData:
            def __init__(self):
                self._rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
                self._gray = np.random.rand(100, 100).astype(np.float32)
                self._enh_gray = np.random.rand(100, 100).astype(np.float32)
                self._objmask = np.random.randint(0, 2, (100, 100), dtype=np.uint8)
                self._objmap = np.random.randint(0, 10, (100, 100), dtype=np.uint16)

                # Create accessor-like objects
                self.rgb = type('obj', (object,), {'__getitem__': lambda self, key: self._arr[key], '_arr': self._rgb})()
                self.gray = type('obj', (object,), {'__getitem__': lambda self, key: self._arr[key], '_arr': self._gray})()
                self.enh_gray = type('obj', (object,), {'__getitem__': lambda self, key: self._arr[key], '_arr': self._enh_gray})()
                self.objmask = type('obj', (object,), {'__getitem__': lambda self, key: self._arr[key], '_arr': self._objmask})()
                self.objmap = type('obj', (object,), {'__getitem__': lambda self, key: self._arr[key], '_arr': self._objmap})()

        return MockImageWithData()

    def test_extract_single_layer(self, mock_image_with_data):
        """Test extracting a single data layer."""
        result = _extract_data_layers(mock_image_with_data, ["rgb"])

        assert "rgb" in result
        assert len(result) == 1
        assert result["rgb"].shape == (100, 100, 3)
        assert result["rgb"].dtype == np.uint8

    def test_extract_multiple_layers(self, mock_image_with_data):
        """Test extracting multiple data layers."""
        result = _extract_data_layers(
            mock_image_with_data,
            ["rgb", "gray", "objmask"]
        )

        assert "rgb" in result
        assert "gray" in result
        assert "objmask" in result
        assert len(result) == 3

    def test_extract_all_layers(self, mock_image_with_data):
        """Test extracting all available data layers."""
        result = _extract_data_layers(
            mock_image_with_data,
            ["rgb", "gray", "enh_gray", "objmask", "objmap"]
        )

        assert len(result) == 5
        for layer in ["rgb", "gray", "enh_gray", "objmask", "objmap"]:
            assert layer in result

    def test_extracted_arrays_are_copies(self, mock_image_with_data):
        """Test that extracted arrays are independent copies, not views."""
        result = _extract_data_layers(mock_image_with_data, ["rgb"])

        # Modify extracted array
        original_value = result["rgb"][0, 0, 0]
        result["rgb"][0, 0, 0] = 255

        # Original should be unchanged
        assert mock_image_with_data.rgb[:][0, 0, 0] == original_value

    def test_handles_empty_layer_list(self, mock_image_with_data):
        """Test behavior with empty layer list."""
        result = _extract_data_layers(mock_image_with_data, [])

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_memory_reduction(self, mock_image_with_data):
        """Test that extracted arrays use less memory than full Image."""
        import sys

        # Extract only one layer
        result_single = _extract_data_layers(mock_image_with_data, ["objmask"])

        # Extract all layers
        result_all = _extract_data_layers(
            mock_image_with_data,
            ["rgb", "gray", "enh_gray", "objmask", "objmap"]
        )

        # Single layer should be smaller
        single_size = sys.getsizeof(result_single["objmask"])
        all_size = sum(sys.getsizeof(arr) for arr in result_all.values())

        assert single_size < all_size


# ============================================================================
# Tests for _estimate_pipeline_memory
# ============================================================================

class TestEstimatePipelineMemory:
    """Test memory estimation for pipeline execution."""

    @pytest.fixture
    def mock_image_for_memory(self):
        """Create a mock image with controllable dimensions for memory testing."""
        class MockImageForMemory:
            def __init__(self, height=100, width=100, has_rgb=True, has_gray=True, has_enh_gray=True):
                self.height = height
                self.width = width
                self._rgb = np.zeros((height, width, 3), dtype=np.uint8) if has_rgb else None
                self._gray = np.zeros((height, width), dtype=np.uint8) if has_gray else None
                self._enh_gray = np.zeros((height, width), dtype=np.uint8) if has_enh_gray else None

            @property
            def rgb(self):
                class Accessor:
                    def __init__(self, data):
                        self.data = data
                    def __getitem__(self, key):
                        return self.data
                return Accessor(self._rgb)

            @property
            def gray(self):
                class Accessor:
                    def __init__(self, data):
                        self.data = data
                    def __getitem__(self, key):
                        return self.data
                return Accessor(self._gray)

            @property
            def enh_gray(self):
                class Accessor:
                    def __init__(self, data):
                        self.data = data
                    def __getitem__(self, key):
                        return self.data
                return Accessor(self._enh_gray)

        return MockImageForMemory()

    def test_memory_estimate_rgb_layer(self, mock_image_for_memory):
        """Test memory estimation includes RGB layer correctly."""
        # 100x100 RGB image = 100*100*3 bytes
        estimated = _estimate_pipeline_memory(
            mock_image_for_memory,
            num_operations=1,
            data_layers=["rgb"],
            extract_arrays=True
        )

        # Should include: RGB (30KB) + gray (10KB) + enh_gray (10KB) + overhead
        # sys.getsizeof() includes Python object overhead
        assert estimated > 40000  # At least base sizes
        assert estimated < 150000  # But reasonable upper bound

    def test_memory_estimate_uint16_objmask(self, mock_image_for_memory):
        """Test that objmask/objmap use correct uint16 itemsize (2 bytes)."""
        # 100x100 uint16 label map = 100*100*2 bytes = 20KB
        estimated = _estimate_pipeline_memory(
            mock_image_for_memory,
            num_operations=1,
            data_layers=["objmask"],
            extract_arrays=True
        )

        # Should include grayscale + objmask uint16 + enh_gray + overhead
        # sys.getsizeof() includes Python object overhead, so actual is higher
        assert estimated >= 60000  # At least gray + objmask + enh_gray
        assert estimated < 150000  # Reasonable upper bound with overhead

    def test_memory_estimate_all_layers(self, mock_image_for_memory):
        """Test memory estimation with all layer types."""
        estimated = _estimate_pipeline_memory(
            mock_image_for_memory,
            num_operations=1,
            data_layers=["rgb", "gray", "enh_gray", "objmask", "objmap"],
            extract_arrays=True
        )

        # RGB: 100*100*3 = 30KB
        # Gray: 100*100 = 10KB
        # Enh_gray: 100*100 = 10KB
        # Objmask: 100*100*2 = 20KB (uint16)
        # Objmap: 100*100*2 = 20KB (uint16)
        # Total: ~90KB + sys.getsizeof() overhead + 1.5x factor = ~135KB
        # With Python overhead, typically 60-200 KB
        assert estimated > 50000
        assert estimated < 250000

    def test_memory_estimate_proportional_to_image_size(self, mock_image_for_memory):
        """Test that memory estimate scales with image size."""
        # Small image
        small_image = type(mock_image_for_memory)(height=50, width=50)
        small_estimate = _estimate_pipeline_memory(
            small_image,
            num_operations=1,
            data_layers=["objmask"],
            extract_arrays=True
        )

        # Large image (2x larger in each dimension = 4x larger in area)
        large_image = type(mock_image_for_memory)(height=100, width=100)
        large_estimate = _estimate_pipeline_memory(
            large_image,
            num_operations=1,
            data_layers=["objmask"],
            extract_arrays=True
        )

        # Large image should require ~4x more memory (area scales quadratically)
        # With sys.getsizeof() overhead, the ratio may vary, but should still be significant
        assert large_estimate > small_estimate * 3

    def test_memory_estimate_without_extraction(self, mock_image_for_memory):
        """Test memory estimation for non-extracted arrays mode."""
        # Without extraction, scales with number of operations
        estimate_1_op = _estimate_pipeline_memory(
            mock_image_for_memory,
            num_operations=1,
            data_layers=["rgb"],
            extract_arrays=False
        )

        estimate_5_ops = _estimate_pipeline_memory(
            mock_image_for_memory,
            num_operations=5,
            data_layers=["rgb"],
            extract_arrays=False
        )

        # 5 operations should require more memory
        # (base * (5 + 1) * 1.2) vs (base * (1 + 1) * 1.2)
        assert estimate_5_ops > estimate_1_op * 2

    def test_memory_estimate_returns_bytes(self, mock_image_for_memory):
        """Test that memory estimate is returned in bytes."""
        estimated = _estimate_pipeline_memory(
            mock_image_for_memory,
            num_operations=1,
            data_layers=["gray"],
            extract_arrays=True
        )

        # Should be an integer (bytes)
        assert isinstance(estimated, int)
        assert estimated > 0
        assert estimated < 1_000_000_000  # Less than 1GB (sanity check)

    def test_objmask_itemsize_is_two_bytes(self, mock_image_for_memory):
        """Regression test: verify objmask uses 2-byte uint16 not 1-byte uint8."""
        # This is the critical bug fix test
        # 100x100 objmask should be 20000 bytes (2 bytes per pixel), not 10000 bytes
        estimated = _estimate_pipeline_memory(
            mock_image_for_memory,
            num_operations=1,
            data_layers=["objmask"],
            extract_arrays=True
        )

        # Expected: base grayscale + enh_gray + objmask (uint16) + overhead
        # With sys.getsizeof() overhead, this is higher
        # Key: objmask should contribute 100*100*2 = 20000 bytes, NOT 10000
        # So estimated should be > 60000 (all layers including overhead)
        # If it was using uint8 (1 byte) instead of uint16 (2 bytes), it would be < 60000
        assert estimated >= 60000, f"Objmask itemsize bug: estimate {estimated} is too low (should use 2-byte uint16)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

