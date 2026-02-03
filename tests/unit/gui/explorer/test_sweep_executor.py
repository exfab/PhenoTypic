"""Tests for SweepExecutor class."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest

from phenotypic.gui.explorer import (
    PipelineGraph,
    SweepSpec,
    SweepExecutor,
    SweepResult,
    SweepResults,
)
from phenotypic.gui.explorer._sweep_executor import ExecutionTask
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector


# =============================================================================
# Helper Functions for Testing
# =============================================================================


def create_mock_image(
    shape: tuple = (100, 100),
    has_objects: bool = True,
    object_count: int = 5,
):
    """Create a mock Image object for testing.

    Args:
        shape: Image shape (height, width).
        has_objects: Whether to simulate detected objects.
        object_count: Number of simulated objects.

    Returns:
        MagicMock configured as an Image.
    """
    mock_img = MagicMock()

    # Basic image properties
    mock_img.shape = shape
    mock_img.gray.__getitem__ = MagicMock(
        return_value=np.random.rand(*shape).astype(np.float32)
    )
    mock_img.rgb.__getitem__ = MagicMock(
        return_value=np.random.randint(0, 256, (*shape, 3), dtype=np.uint8)
    )
    mock_img.rgb.isempty = MagicMock(return_value=False)
    mock_img.detect_mat.__getitem__ = MagicMock(
        return_value=np.random.rand(*shape).astype(np.float32)
    )

    if has_objects:
        # Create a simple binary mask
        objmask = np.zeros(shape, dtype=bool)
        objmask[20:40, 20:40] = True
        objmask[50:70, 50:70] = True
        mock_img.objmask.__getitem__ = MagicMock(return_value=objmask)
        mock_img.objmask = MagicMock()
        mock_img.objmask.__getitem__ = MagicMock(return_value=objmask)

        # Create labeled object map
        objmap = np.zeros(shape, dtype=np.int32)
        objmap[20:40, 20:40] = 1
        objmap[50:70, 50:70] = 2
        mock_img.objmap.__getitem__ = MagicMock(return_value=objmap)
        mock_img.objmap = MagicMock()
        mock_img.objmap.__getitem__ = MagicMock(return_value=objmap)

        # Object count
        mock_img.objects = MagicMock()
        mock_img.objects.count = object_count
    else:
        mock_img.objmask = None
        mock_img.objmap = None

    # RGB accessor for save_overlay
    mock_img.rgb.save_overlay = MagicMock()

    return mock_img


def create_test_graph(with_sweep: bool = False):
    """Create a simple test graph.

    Args:
        with_sweep: Whether to add a parameter sweep.

    Returns:
        PipelineGraph for testing.
    """
    graph = PipelineGraph()
    gauss = graph.add_operation(GaussianBlur, sigma=1.5)
    otsu = graph.add_operation(OtsuDetector)
    output = graph.add_output()
    graph.connect(gauss, otsu).connect(otsu, output)

    if with_sweep:
        graph.add_sweep(gauss, SweepSpec("sigma", [1.0, 2.0]))

    return graph


def create_test_image_file(tmpdir: Path, name: str = "test.png"):
    """Create a test image file.

    Args:
        tmpdir: Directory to create file in.
        name: Image filename.

    Returns:
        Path to created image file.
    """
    from skimage import io as skio

    img_path = tmpdir / name
    img_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    skio.imsave(str(img_path), img_data)
    return img_path


# =============================================================================
# Tests: ExecutionTask
# =============================================================================


class TestExecutionTask:
    """Test ExecutionTask dataclass."""

    def test_create_task(self):
        """Test creating an ExecutionTask."""
        graph = create_test_graph()
        pipelines = list(graph.enumerate_pipelines())
        variant_id, pipeline, config = pipelines[0]

        task = ExecutionTask(
            variant_id=variant_id,
            pipeline=pipeline,
            config=config,
            image_path=Path("/fake/image.png"),
        )

        assert task.variant_id == variant_id
        assert task.image_path == Path("/fake/image.png")


# =============================================================================
# Tests: SweepExecutor Initialization
# =============================================================================


class TestSweepExecutorInit:
    """Test SweepExecutor initialization."""

    def test_create_executor(self):
        """Test creating a SweepExecutor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir,
            )

            assert executor.graph is graph
            assert executor.output_dir == Path(tmpdir)
            assert executor.data2save == {"overlay", "objmask"}
            assert executor.njobs == -1

    def test_create_executor_with_options(self):
        """Test creating executor with custom options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir,
                data2save={"overlay", "objmap", "detect_mat"},
                njobs=4,
            )

            assert executor.data2save == {"overlay", "objmap", "detect_mat"}
            assert executor.njobs == 4

    def test_creates_output_directories(self):
        """Test that executor creates output directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "sweep_results"
            graph = create_test_graph()

            executor = SweepExecutor(
                graph=graph,
                output_dir=output_dir,
            )

            assert output_dir.exists()
            assert (output_dir / "images").exists()
            assert (output_dir / "pipelines").exists()

    def test_ground_truth_dir(self):
        """Test setting ground truth directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gt_dir = Path(tmpdir) / "ground_truth"
            gt_dir.mkdir()

            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir,
                ground_truth_dir=gt_dir,
            )

            assert executor.ground_truth_dir == gt_dir


# =============================================================================
# Tests: Image Path Resolution
# =============================================================================


class TestImagePathResolution:
    """Test _resolve_image_paths helper."""

    def test_resolve_single_file(self):
        """Test resolving a single image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img_path = create_test_image_file(tmpdir)

            graph = create_test_graph()
            executor = SweepExecutor(graph=graph, output_dir=tmpdir / "out")

            paths = executor._resolve_image_paths(img_path)

            assert len(paths) == 1
            assert paths[0] == img_path

    def test_resolve_directory(self):
        """Test resolving all images in a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img_dir = tmpdir / "images"
            img_dir.mkdir()

            # Create multiple images
            create_test_image_file(img_dir, "img1.png")
            create_test_image_file(img_dir, "img2.png")
            create_test_image_file(img_dir, "img3.tif")

            graph = create_test_graph()
            executor = SweepExecutor(graph=graph, output_dir=tmpdir / "out")

            paths = executor._resolve_image_paths(img_dir)

            assert len(paths) == 3

    def test_resolve_path_list(self):
        """Test resolving list of paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img1 = create_test_image_file(tmpdir, "img1.png")
            img2 = create_test_image_file(tmpdir, "img2.png")

            graph = create_test_graph()
            executor = SweepExecutor(graph=graph, output_dir=tmpdir / "out")

            paths = executor._resolve_image_paths([img1, img2])

            assert len(paths) == 2

    def test_resolve_nonexistent_raises(self):
        """Test that nonexistent path raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_test_graph()
            executor = SweepExecutor(graph=graph, output_dir=tmpdir)

            with pytest.raises(FileNotFoundError):
                executor._resolve_image_paths("/nonexistent/path.png")


# =============================================================================
# Tests: Task Building
# =============================================================================


class TestTaskBuilding:
    """Test _build_tasks helper."""

    def test_build_tasks_single_image_no_sweep(self):
        """Test building tasks for single image without sweep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img_path = create_test_image_file(tmpdir)

            graph = create_test_graph(with_sweep=False)
            executor = SweepExecutor(graph=graph, output_dir=tmpdir / "out")

            tasks = executor._build_tasks([img_path])

            assert len(tasks) == 1
            assert tasks[0].image_path == img_path

    def test_build_tasks_single_image_with_sweep(self):
        """Test building tasks for single image with sweep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img_path = create_test_image_file(tmpdir)

            graph = create_test_graph(with_sweep=True)  # 2 sweep values
            executor = SweepExecutor(graph=graph, output_dir=tmpdir / "out")

            tasks = executor._build_tasks([img_path])

            assert len(tasks) == 2  # 2 variants

    def test_build_tasks_multiple_images(self):
        """Test building tasks for multiple images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img1 = create_test_image_file(tmpdir, "img1.png")
            img2 = create_test_image_file(tmpdir, "img2.png")

            graph = create_test_graph(with_sweep=True)  # 2 variants
            executor = SweepExecutor(graph=graph, output_dir=tmpdir / "out")

            tasks = executor._build_tasks([img1, img2])

            # 2 images × 2 variants = 4 tasks
            assert len(tasks) == 4


# =============================================================================
# Tests: Config Flattening
# =============================================================================


class TestConfigFlattening:
    """Test _flatten_config helper."""

    def test_flatten_empty_config(self):
        """Test flattening empty config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_test_graph()
            executor = SweepExecutor(graph=graph, output_dir=tmpdir)

            flat = executor._flatten_config({})

            assert flat == {}

    def test_flatten_nested_config(self):
        """Test flattening nested config dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_test_graph()
            executor = SweepExecutor(graph=graph, output_dir=tmpdir)

            config = {
                "node-1234-5678": {"sigma": 1.5, "mode": "reflect"},
                "node-abcd-efgh": {"offset": 5},
            }
            flat = executor._flatten_config(config)

            assert "node-123.sigma" in flat
            assert "node-123.mode" in flat
            assert "node-abc.offset" in flat


# =============================================================================
# Tests: Task Execution
# =============================================================================


class TestTaskExecution:
    """Test _execute_task method."""

    @patch("phenotypic.gui.explorer._sweep_executor.Image")
    def test_execute_task_success(self, mock_image_class):
        """Test successful task execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img_path = create_test_image_file(tmpdir)

            # Setup mock
            mock_result_image = create_mock_image()
            mock_image_class.imread.return_value = create_mock_image()

            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir / "out",
                data2save=set(),  # Don't save anything for speed
            )

            # Create a task
            pipelines = list(graph.enumerate_pipelines())
            variant_id, pipeline, config = pipelines[0]

            # Mock the pipeline apply
            with patch.object(pipeline, 'apply', return_value=mock_result_image):
                task = ExecutionTask(
                    variant_id=variant_id,
                    pipeline=pipeline,
                    config=config,
                    image_path=img_path,
                )

                result = executor._execute_task(task)

            assert result.success is True
            assert result.variant_id == variant_id
            assert result.image_name == img_path.name
            assert result.error is None
            assert result.execution_time > 0

    @patch("phenotypic.gui.explorer._sweep_executor.Image")
    def test_execute_task_failure(self, mock_image_class):
        """Test task execution with error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img_path = create_test_image_file(tmpdir)

            # Setup mock to raise error
            mock_image_class.imread.side_effect = ValueError("Test error")

            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir / "out",
            )

            pipelines = list(graph.enumerate_pipelines())
            variant_id, pipeline, config = pipelines[0]

            task = ExecutionTask(
                variant_id=variant_id,
                pipeline=pipeline,
                config=config,
                image_path=img_path,
            )

            result = executor._execute_task(task)

            assert result.success is False
            assert "Test error" in result.error


# =============================================================================
# Tests: Metrics Computation
# =============================================================================


class TestMetricsComputation:
    """Test _compute_metrics and related methods."""

    def test_compute_metrics_with_objects(self):
        """Test computing metrics when objects are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_test_graph()
            executor = SweepExecutor(graph=graph, output_dir=tmpdir)

            mock_image = create_mock_image(has_objects=True, object_count=10)
            metrics = executor._compute_metrics(mock_image, Path("test.png"))

            assert "object_count" in metrics
            assert metrics["object_count"] == 10

    def test_compute_metrics_no_objects(self):
        """Test computing metrics when no objects detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_test_graph()
            executor = SweepExecutor(graph=graph, output_dir=tmpdir)

            mock_image = create_mock_image(has_objects=False)
            metrics = executor._compute_metrics(mock_image, Path("test.png"))

            assert "object_count" not in metrics


class TestGroundTruthMetrics:
    """Test ground truth comparison metrics."""

    def test_compute_gt_metrics_no_gt_dir(self):
        """Test GT metrics when no GT directory configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir,
                ground_truth_dir=None,
            )

            mock_image = create_mock_image()
            metrics = executor._compute_metrics(mock_image, Path("test.png"))

            # No GT metrics should be present
            assert "iou" not in metrics

    def test_compute_gt_metrics_with_gt(self):
        """Test GT metrics computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt"
            gt_dir.mkdir()

            # Create GT mask file
            from skimage import io as skio
            gt_mask = np.zeros((100, 100), dtype=np.uint8)
            gt_mask[20:40, 20:40] = 1
            gt_mask[50:70, 50:70] = 2
            skio.imsave(str(gt_dir / "test.png"), gt_mask)

            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir / "out",
                ground_truth_dir=gt_dir,
            )

            # Create mock image with matching objects
            mock_image = create_mock_image()

            metrics = executor._compute_gt_metrics(mock_image, Path("test.png"))

            assert "iou" in metrics
            assert "precision" in metrics
            assert "recall" in metrics
            assert "f1" in metrics

    def test_compute_gt_metrics_missing_gt_file(self):
        """Test GT metrics when GT file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt"
            gt_dir.mkdir()

            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir / "out",
                ground_truth_dir=gt_dir,
            )

            mock_image = create_mock_image()
            metrics = executor._compute_gt_metrics(
                mock_image, Path("nonexistent.png")
            )

            # No metrics since GT file doesn't exist
            assert metrics == {}


# =============================================================================
# Tests: Full Sweep Execution
# =============================================================================


class TestFullSweepExecution:
    """Test complete sweep execution workflow."""

    @patch("phenotypic.gui.explorer._sweep_executor.Image")
    def test_run_sweep_single_image(self, mock_image_class):
        """Test running sweep on single image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img_path = create_test_image_file(tmpdir)

            mock_result = create_mock_image()
            mock_image_class.imread.return_value = create_mock_image()

            graph = create_test_graph(with_sweep=True)  # 2 variants
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir / "out",
                data2save=set(),  # Skip saving for speed
                njobs=1,  # Sequential for testing
            )

            # Mock pipeline apply
            with patch(
                "phenotypic.ImagePipeline.apply",
                return_value=mock_result
            ):
                results = executor.run(images=img_path)

            assert isinstance(results, SweepResults)
            assert len(results.results) == 2  # 2 variants
            assert results.sweep_dir == tmpdir / "out"

    @patch("phenotypic.gui.explorer._sweep_executor.Image")
    def test_run_sweep_creates_manifest(self, mock_image_class):
        """Test that sweep creates manifest file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img_path = create_test_image_file(tmpdir)

            mock_result = create_mock_image()
            mock_image_class.imread.return_value = create_mock_image()

            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir / "out",
                data2save=set(),
                njobs=1,
            )

            with patch(
                "phenotypic.ImagePipeline.apply",
                return_value=mock_result
            ):
                results = executor.run(images=img_path)

            manifest_path = tmpdir / "out" / "manifest.json"
            assert manifest_path.exists()

    @patch("phenotypic.gui.explorer._sweep_executor.Image")
    def test_run_sweep_with_progress_callback(self, mock_image_class):
        """Test progress callback is called during sweep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            img_path = create_test_image_file(tmpdir)

            mock_result = create_mock_image()
            mock_image_class.imread.return_value = create_mock_image()

            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir / "out",
                data2save=set(),
                njobs=1,
            )

            progress_calls = []

            def progress_callback(current, total, message):
                progress_calls.append((current, total, message))

            with patch(
                "phenotypic.ImagePipeline.apply",
                return_value=mock_result
            ):
                executor.run(images=img_path, progress_callback=progress_callback)

            assert len(progress_calls) > 0
            # Last call should be completion
            assert progress_calls[-1][0] == progress_calls[-1][1]
            assert "Complete" in progress_calls[-1][2]

    def test_run_no_images_raises(self):
        """Test that empty image list raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            graph = create_test_graph()
            executor = SweepExecutor(graph=graph, output_dir=tmpdir)

            with pytest.raises(ValueError, match="No images found"):
                executor.run(images=[])


# =============================================================================
# Tests: Output Saving
# =============================================================================


class TestOutputSaving:
    """Test _save_outputs method."""

    def test_save_overlay(self):
        """Test saving overlay output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir,
                data2save={"overlay"},
            )

            mock_image = create_mock_image()
            outputs = executor._save_outputs(
                "test_variant",
                Path("test_image.png"),
                mock_image,
            )

            # Verify save_overlay was called
            mock_image.rgb.save_overlay.assert_called_once()

    def test_save_objmask(self):
        """Test saving objmask output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            graph = create_test_graph()
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir,
                data2save={"objmask"},
            )

            mock_image = create_mock_image()
            outputs = executor._save_outputs(
                "test_variant",
                Path("test_image.png"),
                mock_image,
            )

            assert "objmask" in outputs
            assert outputs["objmask"].exists()


# =============================================================================
# Tests: Pipeline Export
# =============================================================================


class TestPipelineExport:
    """Test pipeline JSON export."""

    def test_save_pipelines(self):
        """Test saving pipeline JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            graph = create_test_graph(with_sweep=True)  # 2 variants
            executor = SweepExecutor(
                graph=graph,
                output_dir=tmpdir,
            )

            executor._save_pipelines()

            pipelines_dir = tmpdir / "pipelines"
            json_files = list(pipelines_dir.glob("*.json"))

            assert len(json_files) == 2


# =============================================================================
# Tests: SweepResults Integration
# =============================================================================


class TestSweepResultsIntegration:
    """Test SweepResults data structure integration."""

    def test_results_to_dataframe(self):
        """Test converting results to DataFrame."""
        results = SweepResults(
            sweep_dir=Path("/fake/dir"),
            results=[
                SweepResult(
                    variant_id="path0_combo0",
                    pipeline_config={"sigma": 1.0},
                    image_name="test.png",
                    success=True,
                    metrics={"object_count": 10},
                    execution_time=0.5,
                ),
                SweepResult(
                    variant_id="path0_combo1",
                    pipeline_config={"sigma": 2.0},
                    image_name="test.png",
                    success=True,
                    metrics={"object_count": 15},
                    execution_time=0.6,
                ),
            ],
        )

        df = results.to_dataframe()

        assert len(df) == 2
        assert "variant_id" in df.columns
        assert "object_count" in df.columns

    def test_results_best_by_metric(self):
        """Test finding best result by metric."""
        results = SweepResults(
            sweep_dir=Path("/fake/dir"),
            results=[
                SweepResult(
                    variant_id="v0",
                    pipeline_config={},
                    image_name="test.png",
                    success=True,
                    metrics={"object_count": 10},
                ),
                SweepResult(
                    variant_id="v1",
                    pipeline_config={},
                    image_name="test.png",
                    success=True,
                    metrics={"object_count": 20},
                ),
            ],
        )

        best = results.best_by_metric("object_count")

        assert best.variant_id == "v1"
        assert best.metrics["object_count"] == 20

    def test_results_filter_by_metric(self):
        """Test filtering results by metric range."""
        results = SweepResults(
            sweep_dir=Path("/fake/dir"),
            results=[
                SweepResult(
                    variant_id="v0",
                    pipeline_config={},
                    image_name="test.png",
                    success=True,
                    metrics={"object_count": 5},
                ),
                SweepResult(
                    variant_id="v1",
                    pipeline_config={},
                    image_name="test.png",
                    success=True,
                    metrics={"object_count": 15},
                ),
                SweepResult(
                    variant_id="v2",
                    pipeline_config={},
                    image_name="test.png",
                    success=True,
                    metrics={"object_count": 25},
                ),
            ],
        )

        filtered = results.filter_by_metric("object_count", min_value=10, max_value=20)

        assert len(filtered) == 1
        assert filtered[0].variant_id == "v1"

    def test_results_manifest_roundtrip(self):
        """Test saving and loading manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            results = SweepResults(
                sweep_dir=tmpdir,
                results=[
                    SweepResult(
                        variant_id="v0",
                        pipeline_config={"sigma": 1.5},
                        image_name="test.png",
                        success=True,
                        metrics={"object_count": 10},
                    ),
                ],
            )

            manifest_path = results.save_manifest()
            loaded = SweepResults.load_manifest(manifest_path)

            assert len(loaded.results) == 1
            assert loaded.results[0].variant_id == "v0"
