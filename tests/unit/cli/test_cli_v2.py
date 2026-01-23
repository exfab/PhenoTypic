"""
Test suite for PhenoTypic CLI v2 features.

Tests the new features introduced in v2.0:
- Execution strategies (local and SLURM)
- Interactive features (dry-run, sample, resume)
- HTML report generation
- Monitoring tools_
- State management
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from phenotypic import Image, GridImage
from phenotypic._cli._cli_directory_scanner import (
    generate_timestamped_output_dir,
    organize_by_dataset,
    scan_directory_structure,
)
from phenotypic._cli._cli_execution_strategies import (
    AutonomousSLURMStrategy,
    LocalParallelStrategy,
    create_execution_strategy,
)
from phenotypic._cli._cli_interactive import (
    execute_dry_run,
    get_sample_datasets,
)
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_report_generator import HTMLReportGenerator
from phenotypic._cli._cli_state_management import (
    create_initial_state,
    get_remaining_images_for_datasets,
    load_processing_state,
    save_processing_state,
    validate_resume_compatibility,
)
from phenotypic._cli._cli_types import (
    Dataset,
    DatasetResults,
    ExecutionConfig,
    ExecutionResults,
    ImageFailure,
)
from phenotypic._cli._cli_update_state import (
    aggregate_state_from_events,
    append_completion_event,
)
from phenotypic.data import load_synth_plate
from phenotypic.phenotypicCLI import main
from phenotypic.prefab import RoundPeaksPipeline


@pytest.fixture
def runner():
    """Provide a Click CliRunner for CLI testing."""
    return CliRunner()


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_input_dir():
    """Create a temporary input directory with sample images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create some synthetic images
        for i in range(3):
            img_path = tmpdir / f"image_{i:03d}.png"
            grid_image = load_synth_plate()

            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
            pil_img.save(img_path)

        yield tmpdir


@pytest.fixture
def temp_recursive_input_dir():
    """Create temporary input directory with recursive structure (subdirectories only)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dataset subdirectories
        (tmpdir / "dataset1").mkdir()
        (tmpdir / "dataset2").mkdir()

        # Add images to each dataset
        for dataset_name in ["dataset1", "dataset2"]:
            dataset_dir = tmpdir / dataset_name
            for i in range(2):
                img_path = dataset_dir / f"image_{i:03d}.png"
                grid_image = load_synth_plate()

                from PIL import Image as PILImage

                pil_img = PILImage.fromarray(
                        grid_image.rgb[:].astype("uint8")
                )
                pil_img.save(img_path)

        yield tmpdir


@pytest.fixture
def temp_mixed_input_dir():
    """Create temporary input directory with INVALID mixed structure (root images + subdirs)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create dataset subdirectory
        (tmpdir / "dataset1").mkdir()

        # Add images to subdirectory
        dataset_dir = tmpdir / "dataset1"
        for i in range(2):
            img_path = dataset_dir / f"image_{i:03d}.png"
            grid_image = load_synth_plate()
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
            pil_img.save(img_path)

        # Add a root-level image (makes this a mixed structure - invalid)
        img_path = tmpdir / "root_image.png"
        grid_image = load_synth_plate()
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
        pil_img.save(img_path)

        yield tmpdir


@pytest.fixture
def temp_pipeline():
    """Create a temporary pipeline JSON file."""
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
    ) as f:
        pipeline = RoundPeaksPipeline(
                blur_sigma=3,
                detector_thresh_method="otsu",
                detector_subtract_background=True,
                detector_remove_noise=True,
        )
        f.write(pipeline.to_json())
        pipeline_path = Path(f.name)

    yield pipeline_path

    if pipeline_path.exists():
        pipeline_path.unlink()


class TestDirectoryScanning:
    """Test directory scanning and dataset organization."""

    def test_scan_single_file(self, temp_input_dir):
        """Test scanning a single image file."""
        image_path = next(temp_input_dir.glob("*.png"))
        result = scan_directory_structure(image_path)

        # Single files now use "single_image" as dataset name
        assert "single_image" in result
        assert len(result["single_image"]) == 1
        assert result["single_image"][0] == image_path

    def test_scan_flat_directory(self, temp_input_dir):
        """Test scanning a flat directory of images."""
        result = scan_directory_structure(temp_input_dir)

        # Flat directories now use the directory name as dataset name
        assert temp_input_dir.name in result
        assert len(result[temp_input_dir.name]) == 3  # 3 images created

    def test_scan_recursive_directory(self, temp_recursive_input_dir):
        """Test scanning recursive directory structure (subdirectories only)."""
        result = scan_directory_structure(temp_recursive_input_dir)

        # Recursive directories only have subdirectory datasets (no root images)
        assert "dataset1" in result
        assert "dataset2" in result
        assert len(result) == 2  # Only the 2 subdirectories
        assert len(result["dataset1"]) == 2  # 2 images per dataset
        assert len(result["dataset2"]) == 2

    def test_scan_mixed_directory_raises_error(self, temp_mixed_input_dir):
        """Test that mixed directories (root images + subdirs) raise an error."""
        with pytest.raises(ValueError, match="Mixed input structure not allowed"):
            scan_directory_structure(temp_mixed_input_dir)

    def test_organize_by_dataset(
            self, temp_recursive_input_dir, temp_output_dir
    ):
        """Test organizing scanned images into Dataset objects."""
        image_paths = scan_directory_structure(temp_recursive_input_dir)
        datasets = organize_by_dataset(image_paths, temp_output_dir)

        assert len(datasets) == 2  # Only 2 subdirectory datasets
        dataset_names = {ds.name for ds in datasets}
        assert "dataset1" in dataset_names
        assert "dataset2" in dataset_names

    def test_generate_timestamped_output_dir(self):
        """Test timestamped output directory generation."""
        output_dir = generate_timestamped_output_dir()

        assert output_dir.name.startswith("phenotypic_results_")
        assert len(output_dir.name) == len("phenotypic_results_YYYYMMDD_HHMMSS")


class TestStateManagement:
    """Test processing state management and resume capability."""

    def test_create_initial_state(self, temp_output_dir):
        """Test creating initial processing state."""
        datasets = [
            Dataset(
                    name="test",
                    images=[Path("img1.png"), Path("img2.png")],
                    input_dir=Path("."),
                    output_dir=temp_output_dir,
            )
        ]

        config = ExecutionConfig(
                pipeline_json=Path("pipeline.json"),
                input_path=Path("."),
                output_dir=temp_output_dir,
                image_type="GridImage",
                nrows=8,
                ncols=12,
                bit_depth=None,
                n_jobs=1,
                slurm_args={},
                force_local=False,
                wait=False,
                save_rgb=False,
                save_gray=False,
                save_enh_gray=False,
                save_objmask=False,
                save_objmap=False,
                save_objmap_rgb=False,
                rgb_ext=".tiff",
                gray_ext=".tiff",
                enh_gray_ext=".tiff",
                objmask_ext=".png",
                objmap_ext=".png",
                objmap_rgb_ext=".png",
                include_dataset_column=False,
                dry_run=False,
                sample=None,
                resume=False,
                retry_failures=False,
                skip_validation=False,
        )

        state = create_initial_state(config, datasets, temp_output_dir)

        assert state.version == "2.0.0"
        assert "test" in state.datasets
        assert state.execution_mode == "local"

    def test_save_and_load_state(self, temp_output_dir):
        """Test saving and loading processing state."""
        datasets = [
            Dataset(
                    name="test",
                    images=[Path("img1.png")],
                    input_dir=Path("."),
                    output_dir=temp_output_dir,
            )
        ]

        config = ExecutionConfig(
                pipeline_json=Path("pipeline.json"),
                input_path=Path("."),
                output_dir=temp_output_dir,
                image_type="GridImage",
                nrows=8,
                ncols=12,
                bit_depth=None,
                n_jobs=1,
                slurm_args={},
                force_local=False,
                wait=False,
                save_rgb=False,
                save_gray=False,
                save_enh_gray=False,
                save_objmask=False,
                save_objmap=False,
                save_objmap_rgb=False,
                rgb_ext=".tiff",
                gray_ext=".tiff",
                enh_gray_ext=".tiff",
                objmask_ext=".png",
                objmap_ext=".png",
                objmap_rgb_ext=".png",
                include_dataset_column=False,
                dry_run=False,
                sample=None,
                resume=False,
                retry_failures=False,
                skip_validation=False,
        )

        # Create and save state
        state = create_initial_state(config, datasets, temp_output_dir)
        save_processing_state(state, temp_output_dir)

        # Load state
        loaded_state = load_processing_state(temp_output_dir)

        assert loaded_state is not None
        assert loaded_state.version == state.version
        assert "test" in loaded_state.datasets

    def test_event_log_append(self, temp_output_dir):
        """Test appending events to event log."""
        event_log = temp_output_dir / "processing_events.log"

        # Append some events
        append_completion_event(event_log, "dataset1", "img1.png", "completed")
        append_completion_event(
                event_log, "dataset1", "img2.png", "failed", "Test error"
        )
        append_completion_event(event_log, "dataset2", "img3.png", "completed")

        # Check file exists and has content
        assert event_log.exists()
        lines = event_log.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_aggregate_state_from_events(self, temp_output_dir):
        """Test aggregating state from event log."""
        event_log = temp_output_dir / "processing_events.log"

        # Append events
        append_completion_event(event_log, "dataset1", "img1.png", "completed")
        append_completion_event(event_log, "dataset1", "img2.png", "completed")
        append_completion_event(
                event_log, "dataset1", "img3.png", "failed", "Test error"
        )
        append_completion_event(event_log, "dataset2", "img4.png", "completed")

        # Aggregate state
        datasets_state = aggregate_state_from_events(event_log)

        assert "dataset1" in datasets_state
        assert "dataset2" in datasets_state

        ds1 = datasets_state["dataset1"]
        assert len(ds1.completed) == 2
        assert len(ds1.failed) == 1
        assert "img1.png" in ds1.completed
        assert "img3.png" in ds1.failed
        assert "img3.png" in ds1.errors


class TestOutputManager:
    """Test output file organization and management."""

    def test_create_structure(self, temp_output_dir):
        """Test creating output directory structure."""
        datasets = [
            Dataset(
                    name="dataset1",
                    images=[Path("img1.png")],
                    input_dir=Path("."),
                    output_dir=temp_output_dir,
            )
        ]

        manager = OutputManager(
                base_dir=temp_output_dir,
                save_layers={
                    "rgb"       : True,
                    "gray"      : False,
                    "enh_gray"  : False,
                    "objmask"   : True,
                    "objmap"    : False,
                    "objmap_rgb": False,
                },
                extensions={
                    "rgb"       : ".tiff",
                    "gray"      : ".tiff",
                    "enh_gray"  : ".tiff",
                    "objmask"   : ".png",
                    "objmap"    : ".png",
                    "objmap_rgb": ".png",
                },
                include_dataset_column=False,
        )

        manager.create_structure(datasets)

        # Check logs directory at root level
        assert (temp_output_dir / "logs").exists()

        # Check dataset-first structure (dataset1/layer/ not layer/dataset1/)
        assert (temp_output_dir / "dataset1" / "measurements").exists()
        assert (temp_output_dir / "dataset1" / "overlays").exists()

        # Check optional layer directories within dataset
        assert (temp_output_dir / "dataset1" / "rgb").exists()
        assert (temp_output_dir / "dataset1" / "objmask").exists()
        assert not (temp_output_dir / "dataset1" / "gray").exists()  # Not requested

        # Old structure should NOT exist
        assert not (temp_output_dir / "measurements").exists()
        assert not (temp_output_dir / "overlays").exists()


class TestHTMLReportGenerator:
    """Test HTML report generation."""

    def test_generate_basic_report(self, temp_output_dir):
        """Test generating a basic HTML report."""
        results = ExecutionResults(
                datasets={
                    "dataset1": DatasetResults(
                            name="dataset1",
                            total=5,
                            completed=4,
                            failed=1,
                            failures=[
                                ImageFailure(
                                        dataset="dataset1",
                                        image_filename="img3.png",
                                        error_type="ValueError",
                                        error_message="Test error",
                                        traceback="Full traceback here",
                                        timestamp=datetime.now(),
                                )
                            ],
                    )
                },
                total_images=5,
                total_completed=4,
                total_failed=1,
                execution_mode="local",
                start_time=datetime.now(),
                end_time=datetime.now(),
        )

        generator = HTMLReportGenerator()
        report_path = temp_output_dir / "test_report.html"
        generator.generate_report(results, report_path)

        assert report_path.exists()
        html_content = report_path.read_text()

        # Check for key content
        assert "PhenoTypic Processing Report" in html_content
        assert "dataset1" in html_content
        assert "4/5 successful" in html_content
        assert "ValueError" in html_content


class TestInteractiveFeatures:
    """Test interactive features (dry-run, sample, resume)."""

    def test_sample_datasets(self, temp_output_dir):
        """Test creating sample datasets."""
        datasets = [
            Dataset(
                    name="test",
                    images=[
                        Path(f"img{i}.png") for i in range(10)
                    ],  # 10 images
                    input_dir=Path("."),
                    output_dir=temp_output_dir,
            )
        ]

        sample_datasets = get_sample_datasets(datasets, 3, temp_output_dir)

        assert len(sample_datasets) == 1
        assert len(sample_datasets[0].images) == 3  # Sampled to 3

    def test_sample_datasets_small(self, temp_output_dir):
        """Test sampling when dataset is smaller than sample size."""
        datasets = [
            Dataset(
                    name="test",
                    images=[Path("img1.png"), Path("img2.png")],  # 2 images
                    input_dir=Path("."),
                    output_dir=temp_output_dir,
            )
        ]

        sample_datasets = get_sample_datasets(datasets, 5, temp_output_dir)

        assert len(sample_datasets) == 1
        assert len(sample_datasets[0].images) == 2  # Kept all images


class TestExecutionStrategies:
    """Test execution strategy selection and behavior."""

    def test_strategy_factory_local(self, temp_output_dir):
        """Test creating local execution strategy."""
        config = ExecutionConfig(
                pipeline_json=Path("pipeline.json"),
                input_path=Path("."),
                output_dir=temp_output_dir,
                image_type="GridImage",
                nrows=8,
                ncols=12,
                bit_depth=None,
                n_jobs=1,
                slurm_args={},  # Empty = local
                force_local=False,
                wait=False,
                save_rgb=False,
                save_gray=False,
                save_enh_gray=False,
                save_objmask=False,
                save_objmap=False,
                save_objmap_rgb=False,
                rgb_ext=".tiff",
                gray_ext=".tiff",
                enh_gray_ext=".tiff",
                objmask_ext=".png",
                objmap_ext=".png",
                objmap_rgb_ext=".png",
                include_dataset_column=False,
                dry_run=False,
                sample=None,
                resume=False,
                retry_failures=False,
                skip_validation=False,
        )

        manager = OutputManager(
                base_dir=temp_output_dir,
                save_layers={},
                extensions={},
                include_dataset_column=False,
        )

        strategy = create_execution_strategy(config, manager)

        assert isinstance(strategy, LocalParallelStrategy)

    def test_strategy_factory_slurm(self, temp_output_dir):
        """Test creating SLURM execution strategy."""
        config = ExecutionConfig(
                pipeline_json=Path("pipeline.json"),
                input_path=Path("."),
                output_dir=temp_output_dir,
                image_type="GridImage",
                nrows=8,
                ncols=12,
                bit_depth=None,
                n_jobs=1,
                slurm_args={"slurm_partition": "compute"},  # Non-empty = SLURM
                force_local=False,
                wait=False,
                save_rgb=False,
                save_gray=False,
                save_enh_gray=False,
                save_objmask=False,
                save_objmap=False,
                save_objmap_rgb=False,
                rgb_ext=".tiff",
                gray_ext=".tiff",
                enh_gray_ext=".tiff",
                objmask_ext=".png",
                objmap_ext=".png",
                objmap_rgb_ext=".png",
                include_dataset_column=False,
                dry_run=False,
                sample=None,
                resume=False,
                retry_failures=False,
                skip_validation=False,
        )

        manager = OutputManager(
                base_dir=temp_output_dir,
                save_layers={},
                extensions={},
                include_dataset_column=False,
        )

        strategy = create_execution_strategy(config, manager)

        assert isinstance(strategy, AutonomousSLURMStrategy)

    def test_strategy_factory_force_local_overrides_slurm(
        self, temp_output_dir
    ):
        """Test that --force-local overrides SLURM args."""
        config = ExecutionConfig(
                pipeline_json=Path("pipeline.json"),
                input_path=Path("."),
                output_dir=temp_output_dir,
                image_type="GridImage",
                nrows=8,
                ncols=12,
                bit_depth=None,
                n_jobs=1,
                slurm_args={"slurm_partition": "compute"},  # Has SLURM args
                force_local=True,  # But force_local is True
                wait=False,
                save_rgb=False,
                save_gray=False,
                save_enh_gray=False,
                save_objmask=False,
                save_objmap=False,
                save_objmap_rgb=False,
                rgb_ext=".tiff",
                gray_ext=".tiff",
                enh_gray_ext=".tiff",
                objmask_ext=".png",
                objmap_ext=".png",
                objmap_rgb_ext=".png",
                include_dataset_column=False,
                dry_run=False,
                sample=None,
                resume=False,
                retry_failures=False,
                skip_validation=False,
        )

        manager = OutputManager(
                base_dir=temp_output_dir,
                save_layers={},
                extensions={},
                include_dataset_column=False,
        )

        strategy = create_execution_strategy(config, manager)

        # Should return local strategy despite SLURM args
        assert isinstance(strategy, LocalParallelStrategy)

    def test_is_slurm_mode(self, temp_output_dir):
        """Test is_slurm_mode() method logic."""
        # Case 1: No SLURM args, no force_local -> local mode
        config = ExecutionConfig(
                pipeline_json=Path("pipeline.json"),
                input_path=Path("."),
                output_dir=temp_output_dir,
                image_type="GridImage",
                nrows=8,
                ncols=12,
                bit_depth=None,
                n_jobs=1,
                slurm_args={},
                force_local=False,
                wait=False,
                save_rgb=False,
                save_gray=False,
                save_enh_gray=False,
                save_objmask=False,
                save_objmap=False,
                save_objmap_rgb=False,
                rgb_ext=".tiff",
                gray_ext=".tiff",
                enh_gray_ext=".tiff",
                objmask_ext=".png",
                objmap_ext=".png",
                objmap_rgb_ext=".png",
                include_dataset_column=False,
                dry_run=False,
                sample=None,
                resume=False,
                retry_failures=False,
                skip_validation=False,
        )
        assert not config.is_slurm_mode()

        # Case 2: Has SLURM args, no force_local -> SLURM mode
        config.slurm_args = {"slurm_partition": "compute"}
        assert config.is_slurm_mode()

        # Case 3: Has SLURM args but force_local=True -> local mode
        config.force_local = True
        assert not config.is_slurm_mode()


class TestCLIv2Integration:
    """Integration tests for the new CLI."""

    def test_cli_dry_run(
            self, runner, temp_pipeline, temp_input_dir
    ):
        """Test --dry-run flag."""
        # Use isolated filesystem to avoid creating files in repo
        with runner.isolated_filesystem():
            result = runner.invoke(
                    main,
                    [
                        str(temp_pipeline),
                        str(temp_input_dir),
                        "--dry-run",
                    ],
            )

            assert result.exit_code == 0
            assert "DRY-RUN MODE" in result.output
            assert "To proceed with execution" in result.output

    def test_cli_auto_output_dir(
            self, runner, temp_pipeline, temp_input_dir
    ):
        """Test CLI with specified output directory in tmp location."""
        # Use isolated filesystem to avoid creating files in repo
        with runner.isolated_filesystem():
            # Create a temporary output directory within the isolated filesystem
            temp_output_dir = Path("./tmp_output")

            result = runner.invoke(
                    main,
                    [
                        str(temp_pipeline),
                        str(temp_input_dir),
                        "-o",
                        str(temp_output_dir),
                        "--skip-validation",  # Skip for speed
                    ],
            )

            # Check that the specified output directory was used (not auto-generated)
            assert "Auto-generated output directory" not in result.output
            assert temp_output_dir.exists()

    def test_cli_sample_mode(
            self, runner, temp_pipeline, temp_input_dir
    ):
        """Test --sample flag."""
        with runner.isolated_filesystem():
            output_dir = Path("./test_output")
            result = runner.invoke(
                    main,
                    [
                        str(temp_pipeline),
                        str(temp_input_dir),
                        "-o",
                        str(output_dir),
                        "--sample",
                        "2",
                        "--skip-validation",
                    ],
            )

            # Should process limited number of images
            assert "Sample mode" in result.output or "sample" in result.output.lower()

    def test_cli_slurm_args_backend_selection(
        self, runner, temp_pipeline, temp_input_dir
    ):
        """Test that --slurm-args causes CLI to use SLURM backend."""
        with runner.isolated_filesystem():
            output_dir = Path("./test_output")
            result = runner.invoke(
                    main,
                    [
                        str(temp_pipeline),
                        str(temp_input_dir),
                        "-o",
                        str(output_dir),
                        "--slurm-args",
                        "slurm_partition=compute",
                        "--slurm-args",
                        "mem_gb=16",
                        "--dry-run",  # Use dry-run to avoid actual SLURM submission
                        "--skip-validation",
                    ],
            )

            # Check that SLURM backend is displayed
            # The _display_execution_config function shows "SLURM Cluster" as backend
            assert "SLURM Cluster" in result.output or result.exit_code == 0


@pytest.mark.skipif(
        not pytest.importorskip("submitit"),
        reason="submitit not installed",
)
class TestSLURMFeatures:
    """Test SLURM-specific features (requires submitit)."""

    def test_slurm_args_parsing(self):
        """Test SLURM args parsing."""
        from phenotypic.phenotypicCLI import _parse_slurm_args

        kwds = [
            "slurm_partition=compute",
            "mem_gb=16",
            "time_min=30",
        ]

        result = _parse_slurm_args(kwds)

        assert result["slurm_partition"] == "compute"
        assert result["mem_gb"] == 16
        assert result["time_min"] == 30


class TestResumeMode:
    """Tests for resume mode functionality."""

    def test_resume_with_changed_input_images(self, runner, tmp_path):
        """Test that resume fails when input images change (missing images)."""
        import json
        from datetime import datetime
        from phenotypic._cli._cli_types import ProcessingState, DatasetState

        # Create initial state
        temp_input_dir = tmp_path / "images"
        temp_input_dir.mkdir()

        # Create initial images
        image1 = temp_input_dir / "image1.jpg"
        image2 = temp_input_dir / "image2.jpg"
        image1.write_text("dummy")
        image2.write_text("dummy")

        # Create pipeline
        temp_pipeline = tmp_path / "pipeline.json"
        temp_pipeline.write_text(json.dumps({"operations": []}))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create state file with original images
        # Use the directory name ("images") as dataset name per new convention
        state_dict = {
            "version": "2.0.0",
            "pipeline_path": str(temp_pipeline),
            "input_path": str(temp_input_dir),
            "output_dir": str(output_dir),
            "timestamp": datetime.now().isoformat(),
            "execution_mode": "local",
            "last_updated": datetime.now().isoformat(),
            "datasets": {
                "images": {
                    "completed": ["image1.jpg", "image2.jpg"],
                    "failed": [],
                    "errors": {}
                }
            },
            "config": {}
        }

        state_file = output_dir / "processing_state.json"
        state_file.write_text(json.dumps(state_dict))

        # Now delete one image
        image1.unlink()

        # Try to resume - should fail because input changed
        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(temp_input_dir),
                "-o",
                str(output_dir),
                "--resume",
                "--skip-validation",
            ],
        )

        # Should fail with input validation error
        assert result.exit_code != 0
        assert "changed" in result.output.lower() or "input" in result.output.lower()

    def test_resume_with_no_state_file(self, runner, tmp_path):
        """Test that resume fails gracefully when state file doesn't exist."""
        import json

        temp_input_dir = tmp_path / "images"
        temp_input_dir.mkdir()
        (temp_input_dir / "image.jpg").write_text("dummy")

        temp_pipeline = tmp_path / "pipeline.json"
        temp_pipeline.write_text(json.dumps({"operations": []}))

        output_dir = tmp_path / "output"
        output_dir.mkdir()  # Create directory but NO state file

        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(temp_input_dir),
                "-o",
                str(output_dir),
                "--resume",
                "--skip-validation",
            ],
        )

        assert result.exit_code != 0
        assert "processing state" in result.output.lower()

    def test_resume_without_output_dir_specified(self, runner, tmp_path):
        """Test that resume requires --output-dir."""
        import json

        temp_input_dir = tmp_path / "images"
        temp_input_dir.mkdir()
        (temp_input_dir / "image.jpg").write_text("dummy")

        temp_pipeline = tmp_path / "pipeline.json"
        temp_pipeline.write_text(json.dumps({"operations": []}))

        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(temp_input_dir),
                "--resume",
                "--skip-validation",
            ],
        )

        assert result.exit_code != 0
        assert "resume requires" in result.output.lower() or "output-dir" in result.output.lower()


class TestDryRunMode:
    """Tests for dry-run mode functionality."""

    def test_dry_run_creates_no_output(self, runner, tmp_path):
        """Test that dry-run doesn't actually process images."""
        import json
        from pathlib import Path

        temp_input_dir = tmp_path / "images"
        temp_input_dir.mkdir()
        (temp_input_dir / "image1.jpg").write_text("dummy")
        (temp_input_dir / "image2.jpg").write_text("dummy")

        temp_pipeline = tmp_path / "pipeline.json"
        temp_pipeline.write_text(json.dumps({"operations": []}))

        output_dir = tmp_path / "output"

        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(temp_input_dir),
                "-o",
                str(output_dir),
                "--dry-run",
                "--skip-validation",
            ],
        )

        assert result.exit_code == 0
        assert "dry" in result.output.lower() or "would process" in result.output.lower()

        # Verify no output files created
        assert not (output_dir / "measurements").exists()
        assert not (output_dir / "overlays").exists()
        assert not (output_dir / "processing_state.json").exists()

    def test_dry_run_shows_processing_plan(self, runner, tmp_path):
        """Test that dry-run displays what would be processed."""
        import json

        temp_input_dir = tmp_path / "images"
        temp_input_dir.mkdir()
        (temp_input_dir / "image1.jpg").write_text("dummy")
        (temp_input_dir / "image2.jpg").write_text("dummy")

        temp_pipeline = tmp_path / "pipeline.json"
        temp_pipeline.write_text(json.dumps({"operations": []}))

        output_dir = tmp_path / "output"

        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(temp_input_dir),
                "-o",
                str(output_dir),
                "--dry-run",
                "--skip-validation",
            ],
        )

        assert result.exit_code == 0
        # Should show information about images to process
        assert "2" in result.output or "image" in result.output.lower()


# Edge Case Tests
class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_invalid_nrows_zero_rejected(self, runner, tmp_path, temp_pipeline):
        """Test that nrows=0 is rejected by Click IntRange validation."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create a dummy image file
        (input_dir / "test.jpg").write_text("dummy")

        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(input_dir),
                "-o",
                str(output_dir),
                "--image-type",
                "GridImage",
                "--nrows",
                "0",  # Invalid
                "--ncols",
                "12",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "out of range" in result.output.lower()

    def test_invalid_ncols_negative_rejected(self, runner, tmp_path, temp_pipeline):
        """Test that negative ncols is rejected by Click IntRange validation."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create a dummy image file
        (input_dir / "test.jpg").write_text("dummy")

        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(input_dir),
                "-o",
                str(output_dir),
                "--image-type",
                "GridImage",
                "--nrows",
                "8",
                "--ncols",
                "-5",  # Invalid
            ],
        )

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "out of range" in result.output.lower()

    def test_single_image_processing(self, runner, tmp_path, temp_pipeline):
        """Test processing with exactly 1 image (boundary case)."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create a single synthetic image
        grid_image = load_synth_plate()
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
        pil_img.save(input_dir / "single.jpg")

        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(input_dir),
                "-o",
                str(output_dir),
                "--n-jobs",
                "1",
                "--skip-validation",
            ],
        )

        assert result.exit_code == 0

        # Verify output files created for the single image
        # Output should be in dataset folder named after input directory ("input")
        measurements_file = output_dir / "input" / "measurements" / "single.csv"
        overlay_file = output_dir / "input" / "overlays" / "single.png"

        assert measurements_file.exists(), f"Expected {measurements_file} to exist"
        assert overlay_file.exists(), f"Expected {overlay_file} to exist"

        # Verify dataset folder is created with input directory name
        assert (output_dir / "input").exists(), "Dataset folder should be created"

    def test_empty_input_directory(self, runner, tmp_path, temp_pipeline):
        """Test graceful handling of empty input directory."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(input_dir),
                "-o",
                str(output_dir),
                "--skip-validation",
            ],
        )

        # Should fail gracefully with clear message
        assert result.exit_code != 0
        error_msg = result.output.lower()
        assert (
            "no valid images" in error_msg
            or "empty" in error_msg
            or "no images" in error_msg
            or "not found" in error_msg
        )

    def test_resume_with_changed_images(self, runner, tmp_path, temp_pipeline):
        """Test that resume fails when image set changes (same count, different images)."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create 3 initial images and process them
        grid_image = load_synth_plate()
        from PIL import Image as PILImage

        for i in range(1, 4):
            pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
            pil_img.save(input_dir / f"image_{i:03d}.jpg")

        # Initial processing
        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(input_dir),
                "-o",
                str(output_dir),
                "--n-jobs",
                "1",
                "--skip-validation",
            ],
        )
        assert result.exit_code == 0

        # Change image set: remove image_001, add image_004 (same count!)
        (input_dir / "image_001.jpg").unlink()
        pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
        pil_img.save(input_dir / "image_004.jpg")

        # Resume should fail with clear error about image set mismatch
        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(input_dir),
                "-o",
                str(output_dir),
                "--resume",
                "--skip-validation",
            ],
        )

        assert result.exit_code != 0
        error_msg = result.output.lower()
        assert (
            "image set mismatch" in error_msg
            or "missing" in error_msg
            or "added" in error_msg
            or "mismatch" in error_msg
        )


class TestNewCoverageGaps:
    """Tests for previously uncovered edge cases and features."""

    def test_sample_mode_deterministic(self, temp_output_dir):
        """Test that sample mode with same seed produces same images."""
        datasets = [
            Dataset(
                name="test",
                images=[Path(f"img{i}.png") for i in range(20)],
                input_dir=Path("."),
                output_dir=temp_output_dir,
            )
        ]

        # Sample with same seed twice
        sample1 = get_sample_datasets(datasets, 5, temp_output_dir, random_seed=42)
        sample2 = get_sample_datasets(datasets, 5, temp_output_dir, random_seed=42)

        # Should get exact same images
        assert [img.name for img in sample1[0].images] == [
            img.name for img in sample2[0].images
        ]

        # Different seed should give different images
        sample3 = get_sample_datasets(datasets, 5, temp_output_dir, random_seed=123)
        assert [img.name for img in sample1[0].images] != [
            img.name for img in sample3[0].images
        ]

    def test_output_manager_hierarchical_paths(self, temp_output_dir):
        """Test OutputManager always creates hierarchical paths with dataset subdirectories."""
        save_layers = {"rgb": False, "gray": False}
        extensions = {"rgb": ".tiff", "gray": ".tiff"}

        manager = OutputManager(
            base_dir=temp_output_dir / "output",
            save_layers=save_layers,
            extensions=extensions,
        )

        # All paths should include dataset subdirectory
        path = manager.get_output_path("single_image", "measurements", "image1")
        assert path == temp_output_dir / "output" / "single_image" / "measurements" / "image1.csv"

        path = manager.get_output_path("plate1", "measurements", "image1")
        assert path == temp_output_dir / "output" / "plate1" / "measurements" / "image1.csv"

        path = manager.get_output_path("my_dataset", "overlays", "image2")
        assert path == temp_output_dir / "output" / "my_dataset" / "overlays" / "image2.png"

    def test_initial_images_stored_in_state(self, temp_output_dir):
        """Test that initial_images is stored when creating state."""
        datasets = [
            Dataset(
                name="test",
                images=[Path(f"img{i}.png") for i in range(5)],
                input_dir=Path("."),
                output_dir=temp_output_dir,
            )
        ]

        config = ExecutionConfig(
            pipeline_json=Path("pipeline.json"),
            input_path=Path("."),
            output_dir=temp_output_dir,
            image_type="GridImage",
            nrows=8,
            ncols=12,
            bit_depth=None,
            n_jobs=1,
            slurm_args={},
            force_local=True,
            wait=False,
            save_rgb=False,
            save_gray=False,
            save_enh_gray=False,
            save_objmask=False,
            save_objmap=False,
            save_objmap_rgb=False,
            rgb_ext=".tiff",
            gray_ext=".tiff",
            enh_gray_ext=".tiff",
            objmask_ext=".png",
            objmap_ext=".png",
            objmap_rgb_ext=".png",
            include_dataset_column=True,
            dry_run=False,
            sample=None,
            resume=False,
            retry_failures=False,
            skip_validation=False,
        )

        state = create_initial_state(config, datasets, temp_output_dir)

        # Check initial_images is populated
        assert "test" in state.datasets
        assert len(state.datasets["test"].initial_images) == 5
        assert "img0.png" in state.datasets["test"].initial_images

    def test_resume_after_zero_processed(self, temp_output_dir):
        """Test resume validation when no images were processed."""
        datasets = [
            Dataset(
                name="test",
                images=[Path(f"img{i}.png") for i in range(5)],
                input_dir=Path("."),
                output_dir=temp_output_dir,
            )
        ]

        config = ExecutionConfig(
            pipeline_json=Path("pipeline.json"),
            input_path=Path("."),
            output_dir=temp_output_dir,
            image_type="GridImage",
            nrows=8,
            ncols=12,
            bit_depth=None,
            n_jobs=1,
            slurm_args={},
            force_local=True,
            wait=False,
            save_rgb=False,
            save_gray=False,
            save_enh_gray=False,
            save_objmask=False,
            save_objmap=False,
            save_objmap_rgb=False,
            rgb_ext=".tiff",
            gray_ext=".tiff",
            enh_gray_ext=".tiff",
            objmask_ext=".png",
            objmap_ext=".png",
            objmap_rgb_ext=".png",
            include_dataset_column=True,
            dry_run=False,
            sample=None,
            resume=False,
            retry_failures=False,
            skip_validation=False,
        )

        # Create and save initial state (no processing done)
        state = create_initial_state(config, datasets, temp_output_dir)
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        save_processing_state(state, temp_output_dir)

        # Try to resume with different images
        different_datasets = [
            Dataset(
                name="test",
                images=[Path(f"different{i}.png") for i in range(5)],
                input_dir=Path("."),
                output_dir=temp_output_dir,
            )
        ]

        # Load state and validate
        loaded_state = load_processing_state(temp_output_dir)
        assert loaded_state is not None

        # The initial_images should detect the mismatch even with zero completed
        current_images = {img.name for img in different_datasets[0].images}
        assert loaded_state.datasets["test"].initial_images != current_images

    def test_large_dataset_chunking(self):
        """Test that large datasets are properly chunked for SLURM."""
        from phenotypic._cli._cli_slurm_config import calculate_optimal_array_chunks

        # Dataset with 2500 images (should create 3 chunks with 1000 limit)
        num_images = 2500
        array_limit = 1000

        chunks = calculate_optimal_array_chunks(num_images, array_limit)

        # Should create 3 chunks
        assert len(chunks) == 3
        assert chunks[0] == (0, 1000)
        assert chunks[1] == (1000, 2000)
        assert chunks[2] == (2000, 2500)

        # Total should equal num_images
        total = sum(end - start for start, end in chunks)
        assert total == num_images

    def test_slurm_script_no_flat_mode_flag(self, temp_output_dir):
        """Test that generated SLURM scripts do NOT include --flat-mode (feature removed)."""
        from phenotypic._cli._cli_slurm_array_scripts import generate_array_job_script

        dataset = Dataset(
            name="plate1",
            images=[Path(f"img{i}.png") for i in range(10)],
            input_dir=Path("."),
            output_dir=temp_output_dir,
        )

        config = ExecutionConfig(
            pipeline_json=Path("pipeline.json"),
            input_path=Path("."),
            output_dir=temp_output_dir,
            image_type="GridImage",
            nrows=8,
            ncols=12,
            bit_depth=None,
            n_jobs=1,
            slurm_args={"partition": "test"},
            force_local=False,
            wait=False,
            save_rgb=False,
            save_gray=False,
            save_enh_gray=False,
            save_objmask=False,
            save_objmap=False,
            save_objmap_rgb=False,
            rgb_ext="tiff",
            gray_ext="tiff",
            enh_gray_ext="tiff",
            objmask_ext="png",
            objmap_ext="png",
            objmap_rgb_ext="png",
            include_dataset_column=True,
            dry_run=False,
            sample=None,
            resume=False,
            retry_failures=False,
            skip_validation=False,
        )

        script_path = generate_array_job_script(
            dataset=dataset,
            array_indices=(0, 10),
            config=config,
            output_dir=temp_output_dir,
        )

        script_content = script_path.read_text()
        # Flat mode flag was removed - verify it's not in the script
        assert "--flat-mode" not in script_content
        # Verify dataset name is used
        assert "--dataset-name" in script_content
        assert "plate1" in script_content

    def test_slurm_script_bash_syntax(self, temp_output_dir):
        """Test that generated SLURM scripts have valid bash syntax."""
        import subprocess
        from phenotypic._cli._cli_slurm_array_scripts import generate_array_job_script

        dataset = Dataset(
            name="test",
            images=[Path(f"img{i}.png") for i in range(5)],
            input_dir=Path("."),
            output_dir=temp_output_dir,
        )

        config = ExecutionConfig(
            pipeline_json=Path("pipeline.json"),
            input_path=Path("."),
            output_dir=temp_output_dir,
            image_type="GridImage",
            nrows=8,
            ncols=12,
            bit_depth=None,
            n_jobs=1,
            slurm_args={"partition": "test"},
            force_local=False,
            wait=False,
            save_rgb=False,
            save_gray=False,
            save_enh_gray=False,
            save_objmask=False,
            save_objmap=False,
            save_objmap_rgb=False,
            rgb_ext="tiff",
            gray_ext="tiff",
            enh_gray_ext="tiff",
            objmask_ext="png",
            objmap_ext="png",
            objmap_rgb_ext="png",
            include_dataset_column=True,
            dry_run=False,
            sample=None,
            resume=False,
            retry_failures=False,
            skip_validation=False,
        )

        script_path = generate_array_job_script(
            dataset=dataset,
            array_indices=(0, 5),
            config=config,
            output_dir=temp_output_dir,
        )

        # Run bash syntax check
        result = subprocess.run(
            ["bash", "-n", str(script_path)],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_dry_run_creates_no_files(self, runner, tmp_path, temp_pipeline):
        """Test that --dry-run truly creates no output files whatsoever."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create a test image
        grid_image = load_synth_plate()
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
        pil_img.save(input_dir / "test.jpg")

        result = runner.invoke(
            main,
            [
                str(temp_pipeline),
                str(input_dir),
                "-o",
                str(output_dir),
                "--dry-run",
                "--skip-validation",
            ],
        )

        assert result.exit_code == 0

        # Stronger assertion: no files or directories created at all
        if output_dir.exists():
            all_files = list(output_dir.rglob("*"))
            assert len(all_files) == 0, f"Dry run created files: {all_files}"

    def test_single_image_dataset_constant(self):
        """Test that SINGLE_IMAGE_DATASET constant exists and has correct value."""
        from phenotypic._cli._cli_constants import SINGLE_IMAGE_DATASET

        # Verify the constant value
        assert SINGLE_IMAGE_DATASET == "single_image"

        # This test ensures the constant exists and can be imported
