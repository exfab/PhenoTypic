"""Tests for PipelineGraph class."""

import tempfile
from pathlib import Path


from phenotypic.gui.explorer import PipelineGraph, SweepSpec, GraphNode
from phenotypic.enhance import GaussianBlur, CLAHE
from phenotypic.detect import OtsuDetector


class TestPipelineGraphBasics:
    """Test basic graph operations."""

    def test_create_empty_graph(self):
        """Test creating an empty graph."""
        graph = PipelineGraph()

        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.path_count == 0
        assert graph.variant_count == 0

    def test_add_operation_node(self):
        """Test adding an operation node."""
        graph = PipelineGraph()
        node_id = graph.add_operation(GaussianBlur, sigma=1.5)

        assert len(graph.nodes) == 1
        node = graph.get_node(node_id)
        assert node.operation_params["sigma"] == 1.5
        assert "GaussianBlur" in node.operation_class

    def test_add_output_node(self):
        """Test adding an output node."""
        graph = PipelineGraph()
        output_id = graph.add_output()

        assert len(graph.nodes) == 1
        assert output_id in graph.output_ids
        node = graph.get_node(output_id)
        assert node.is_output

    def test_connect_nodes(self):
        """Test connecting nodes."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector)
        graph.connect(gauss, otsu)

        assert len(graph.edges) == 1
        assert (gauss, otsu) in graph.edges

    def test_connect_chaining(self):
        """Test that connect returns self for chaining."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector)
        output = graph.add_output()

        result = graph.connect(gauss, otsu).connect(otsu, output)

        assert result is graph
        assert len(graph.edges) == 2

    def test_remove_node(self):
        """Test removing a node."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector)
        graph.connect(gauss, otsu)
        graph.remove_node(otsu)

        assert len(graph.nodes) == 1
        assert len(graph.edges) == 0

    def test_update_node_params(self):
        """Test updating node parameters."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        graph.update_node_params(gauss, sigma=2.5, mode="reflect")

        node = graph.get_node(gauss)
        assert node.operation_params["sigma"] == 2.5
        assert node.operation_params["mode"] == "reflect"


class TestPipelineGraphPaths:
    """Test path enumeration."""

    def test_linear_path(self):
        """Test enumerating a simple linear path."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector)
        output = graph.add_output()
        graph.connect(gauss, otsu).connect(otsu, output)

        paths = graph.enumerate_paths()
        assert len(paths) == 1
        assert paths[0] == [gauss, otsu, output]

    def test_branching_paths(self):
        """Test enumerating branching paths."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector)
        clahe = graph.add_operation(CLAHE)
        output = graph.add_output()

        # Gauss -> Otsu -> Output
        #       -> CLAHE -> Output (branch)
        graph.connect(gauss, otsu).connect(gauss, clahe)
        graph.connect(otsu, output).connect(clahe, output)

        paths = graph.enumerate_paths()
        assert len(paths) == 2
        assert graph.path_count == 2

    def test_source_ids(self):
        """Test identifying source nodes."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector)
        output = graph.add_output()
        graph.connect(gauss, otsu).connect(otsu, output)

        assert graph.source_ids == [gauss]


class TestPipelineGraphSweeps:
    """Test sweep configuration."""

    def test_add_sweep(self):
        """Test adding a sweep to a node."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        sweep = SweepSpec.from_range("sigma", 1.0, 3.0, 1.0)
        graph.add_sweep(gauss, sweep)

        sweeps = graph.get_sweeps(gauss)
        assert len(sweeps) == 1
        assert sweeps[0].param == "sigma"

    def test_variant_count_with_sweep(self):
        """Test variant count includes sweeps."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector)
        output = graph.add_output()
        graph.connect(gauss, otsu).connect(otsu, output)

        # No sweep yet
        assert graph.variant_count == 1

        # Add sweep with 5 values
        graph.add_sweep(gauss, SweepSpec.from_range("sigma", 1.0, 3.0, 0.5))
        assert graph.variant_count == 5

    def test_variant_count_multiple_sweeps(self):
        """Test variant count with multiple sweeps."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector, offset=0)
        output = graph.add_output()
        graph.connect(gauss, otsu).connect(otsu, output)

        graph.add_sweep(gauss, SweepSpec("sigma", [1.0, 2.0]))  # 2 values
        graph.add_sweep(otsu, SweepSpec("offset", [-5, 0, 5]))  # 3 values

        assert graph.variant_count == 6  # 2 * 3

    def test_variant_count_branching_with_sweep(self):
        """Test variant count with branching and sweeps."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector)
        clahe = graph.add_operation(CLAHE)
        output = graph.add_output()

        graph.connect(gauss, otsu).connect(gauss, clahe)
        graph.connect(otsu, output).connect(clahe, output)

        # 2 paths, sigma sweep with 3 values
        graph.add_sweep(gauss, SweepSpec("sigma", [1.0, 1.5, 2.0]))

        # Each path gets 3 variants from sigma sweep
        assert graph.variant_count == 6  # 2 paths * 3 sigma values

    def test_remove_sweeps(self):
        """Test removing sweeps from a node."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        graph.add_sweep(gauss, SweepSpec("sigma", [1.0, 2.0]))
        graph.remove_sweeps(gauss)

        assert len(graph.get_sweeps(gauss)) == 0


class TestPipelineGraphEnumeration:
    """Test pipeline enumeration."""

    def test_enumerate_single_pipeline(self):
        """Test enumerating a single pipeline without sweeps."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector)
        output = graph.add_output()
        graph.connect(gauss, otsu).connect(otsu, output)

        pipelines = list(graph.enumerate_pipelines())
        assert len(pipelines) == 1

        variant_id, pipeline, config = pipelines[0]
        assert "path0" in variant_id
        assert len(pipeline.get_ops()) == 2
        assert config == {}

    def test_enumerate_with_sweep(self):
        """Test enumerating pipelines with parameter sweep."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        output = graph.add_output()
        graph.connect(gauss, output)
        graph.add_sweep(gauss, SweepSpec("sigma", [1.0, 2.0]))

        pipelines = list(graph.enumerate_pipelines())
        assert len(pipelines) == 2

        # Check that configs have different sigma values
        sigmas = []
        for _, pipeline, config in pipelines:
            # Config is {node_id: {param: value}}
            for node_config in config.values():
                if "sigma" in node_config:
                    sigmas.append(node_config["sigma"])

        assert 1.0 in sigmas
        assert 2.0 in sigmas


class TestPipelineGraphSerialization:
    """Test graph serialization."""

    def test_to_dict(self):
        """Test converting graph to dictionary."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector)
        output = graph.add_output()
        graph.connect(gauss, otsu).connect(otsu, output)
        graph.add_sweep(gauss, SweepSpec("sigma", [1.0, 2.0]))

        data = graph.to_dict()

        assert "version" in data
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2
        assert gauss in data["sweeps"]

    def test_from_dict(self):
        """Test creating graph from dictionary."""
        original = PipelineGraph()
        gauss = original.add_operation(GaussianBlur, sigma=1.5)
        otsu = original.add_operation(OtsuDetector)
        output = original.add_output()
        original.connect(gauss, otsu).connect(otsu, output)
        original.add_sweep(gauss, SweepSpec("sigma", [1.0, 2.0]))

        data = original.to_dict()
        restored = PipelineGraph.from_dict(data)

        assert len(restored.nodes) == 3
        assert len(restored.edges) == 2
        assert restored.path_count == 1
        assert restored.variant_count == 2

    def test_json_roundtrip(self):
        """Test saving and loading from JSON file."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        output = graph.add_output()
        graph.connect(gauss, output)
        graph.add_sweep(gauss, SweepSpec("sigma", [1.0, 2.0, 3.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_graph.json"
            graph.to_json(path)

            assert path.exists()

            loaded = PipelineGraph.from_json(path)
            assert loaded.variant_count == 3


class TestPipelineGraphConvenience:
    """Test convenience constructors."""

    def test_linear_constructor(self):
        """Test creating linear graph from operations."""
        graph = PipelineGraph.linear(
            GaussianBlur(sigma=1.5),
            OtsuDetector(),
        )

        assert len(graph.nodes) == 3  # 2 ops + 1 output
        assert len(graph.edges) == 2
        assert graph.path_count == 1

    def test_from_pipeline(self):
        """Test creating graph from existing ImagePipeline."""
        from phenotypic import ImagePipeline

        pipeline = ImagePipeline([
            GaussianBlur(sigma=1.5),
            OtsuDetector(),
        ])
        graph = PipelineGraph.from_pipeline(pipeline)

        assert len(graph.nodes) == 3  # 2 ops + 1 output
        assert graph.path_count == 1


class TestPipelineGraphValidation:
    """Test graph validation."""

    def test_valid_graph(self):
        """Test validation of valid graph."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        output = graph.add_output()
        graph.connect(gauss, output)

        issues = graph.validate()
        assert len(issues) == 0

    def test_no_output_nodes(self):
        """Test validation catches missing output nodes."""
        graph = PipelineGraph()
        graph.add_operation(GaussianBlur, sigma=1.5)

        issues = graph.validate()
        assert any("output" in issue.lower() for issue in issues)

    def test_disconnected_nodes(self):
        """Test validation catches disconnected nodes."""
        graph = PipelineGraph()
        gauss = graph.add_operation(GaussianBlur, sigma=1.5)
        otsu = graph.add_operation(OtsuDetector)  # Disconnected
        output = graph.add_output()
        graph.connect(gauss, output)

        issues = graph.validate()
        # Should have issues about disconnected node or unreachable output
        # The disconnected OtsuDetector is a "source" with no path to output
        assert len(issues) > 0  # Some validation issue should be caught


class TestGraphNode:
    """Test GraphNode class."""

    def test_instantiate_operation(self):
        """Test instantiating operation from node."""
        node = GraphNode(
            id="test-id",
            operation_class="phenotypic.enhance.GaussianBlur",
            operation_params={"sigma": 2.0},
        )

        op = node.instantiate()
        assert isinstance(op, GaussianBlur)
        assert op.sigma == 2.0

    def test_instantiate_with_overrides(self):
        """Test instantiating with parameter overrides."""
        node = GraphNode(
            id="test-id",
            operation_class="phenotypic.enhance.GaussianBlur",
            operation_params={"sigma": 2.0},
        )

        op = node.instantiate({"sigma": 3.0})
        assert op.sigma == 3.0

    def test_is_output(self):
        """Test output node detection."""
        regular = GraphNode(
            id="test-id",
            operation_class="phenotypic.enhance.GaussianBlur",
            operation_params={},
        )
        output = GraphNode(
            id="test-id",
            operation_class="__output__",
            operation_params={},
        )

        assert regular.is_output is False
        assert output.is_output is True

    def test_class_name(self):
        """Test extracting class name."""
        node = GraphNode(
            id="test-id",
            operation_class="phenotypic.enhance.GaussianBlur",
            operation_params={},
        )

        assert node.class_name == "GaussianBlur"

    def test_serialization(self):
        """Test node serialization roundtrip."""
        original = GraphNode(
            id="test-id",
            operation_class="phenotypic.enhance.GaussianBlur",
            operation_params={"sigma": 1.5},
            position=(100, 200),
        )

        data = original.to_dict()
        restored = GraphNode.from_dict(data)

        assert restored.id == original.id
        assert restored.operation_class == original.operation_class
        assert restored.operation_params == original.operation_params
        assert restored.position == original.position
