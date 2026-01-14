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
    """Create temporary input directory with recursive structure."""
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

        # Add a root-level image
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

        assert "_root" in result
        assert len(result["_root"]) == 1
        assert result["_root"][0] == image_path

    def test_scan_flat_directory(self, temp_input_dir):
        """Test scanning a flat directory of images."""
        result = scan_directory_structure(temp_input_dir)

        assert "_root" in result
        assert len(result["_root"]) == 3  # 3 images created

    def test_scan_recursive_directory(self, temp_recursive_input_dir):
        """Test scanning recursive directory structure."""
        result = scan_directory_structure(temp_recursive_input_dir)

        assert "_root" in result
        assert "dataset1" in result
        assert "dataset2" in result
        assert len(result["_root"]) == 1  # 1 root image
        assert len(result["dataset1"]) == 2  # 2 images per dataset
        assert len(result["dataset2"]) == 2

    def test_organize_by_dataset(
            self, temp_recursive_input_dir, temp_output_dir
    ):
        """Test organizing scanned images into Dataset objects."""
        image_paths = scan_directory_structure(temp_recursive_input_dir)
        datasets = organize_by_dataset(image_paths, temp_output_dir)

        assert len(datasets) == 3  # _root + 2 datasets
        dataset_names = {ds.name for ds in datasets}
        assert "_root" in dataset_names
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
                slurm_kwds={},
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
                slurm_kwds={},
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

        # Check core directories
        assert (temp_output_dir / "measurements").exists()
        assert (temp_output_dir / "overlays").exists()
        assert (temp_output_dir / "logs").exists()

        # Check optional layer directories
        assert (temp_output_dir / "rgb").exists()
        assert (temp_output_dir / "objmask").exists()
        assert not (temp_output_dir / "gray").exists()  # Not requested

        # Check dataset subdirectories
        assert (temp_output_dir / "measurements" / "dataset1").exists()


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
                slurm_kwds={},  # Empty = local
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
                slurm_kwds={"slurm_partition": "compute"},  # Non-empty = SLURM
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

        # Should create SLURM strategy (if submitit available)
        # Type will depend on whether submitit is installed
        assert strategy is not None


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
            assert "DRY RUN SUMMARY" in result.output
            assert "To proceed" in result.output

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


@pytest.mark.skipif(
        not pytest.importorskip("submitit"),
        reason="submitit not installed",
)
class TestSLURMFeatures:
    """Test SLURM-specific features (requires submitit)."""

    def test_slurm_kwds_parsing(self):
        """Test SLURM kwds parsing."""
        from phenotypic.phenotypicCLI import _parse_slurm_kwds

        kwds = [
            "slurm_partition=compute",
            "mem_gb=16",
            "time_min=30",
        ]

        result = _parse_slurm_kwds(kwds)

        assert result["slurm_partition"] == "compute"
        assert result["mem_gb"] == 16
        assert result["time_min"] == 30
