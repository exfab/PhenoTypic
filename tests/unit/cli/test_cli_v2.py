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
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

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
    get_sample_datasets,
)
from phenotypic._cli._cli_output_manager import (
    OutputManager,
    aggregate_measurements,
    finalize_post_master_outputs,
)
from phenotypic._cli._cli_report_generator import HTMLReportGenerator
from phenotypic._cli._cli_state_management import (
    create_initial_state,
    load_processing_state,
    save_processing_state,
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
from phenotypic.data import load_synth_yeast_plate
from phenotypic.phenotypicCLI import _copy_pipeline_to_output, phenotypic_cli
from phenotypic.prefab import RoundPeaksPipeline
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    deliverables_dir,
    master_measurements_csv_path,
    master_measurements_parquet_path,
    measurements_csv_path,
    measurements_parquet_path,
    order_measurement_columns,
    pipeline_json_path,
    zarr_store_path,
)
from phenotypic.schema import (
    CONDITION,
    EXPERIMENT,
    IMAGE,
    METADATA_MATCH,
)


DATASET_HEADER = str(EXPERIMENT.DATASET)
IMAGE_NAME_HEADER = str(IMAGE.IMAGE_NAME)
TREATMENT_LABEL = CONDITION.TREATMENT.label
TREATMENT_HEADER = str(CONDITION.TREATMENT)
METADATA_ONLY_HEADER = str(METADATA_MATCH.METADATA_ONLY)


@pytest.fixture
def runner():
    """Provide a Click CliRunner for CLI testing."""
    return CliRunner()


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="module")
def temp_input_dir():
    """Create a temporary input directory with sample images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create some synthetic images
        for i in range(3):
            img_path = tmpdir / f"image_{i:03d}.png"
            grid_image = load_synth_yeast_plate()

            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
            pil_img.save(img_path)

        yield tmpdir


@pytest.fixture(scope="module")
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
                grid_image = load_synth_yeast_plate()

                from PIL import Image as PILImage

                pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
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
            grid_image = load_synth_yeast_plate()
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
            pil_img.save(img_path)

        # Add a root-level image (makes this a mixed structure - invalid)
        img_path = tmpdir / "root_image.png"
        grid_image = load_synth_yeast_plate()
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
        with pytest.raises(
            ValueError, match="Mixed input structure not allowed"
        ):
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
        assert len(output_dir.name) == len(
            "phenotypic_results_YYYYMMDD_HHMMSS"
        )


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
            ext=".tiff",
            overlay_alpha=0.3,
            include_dataset_column=False,
            dry_run=False,
            sample=None,
            resume=False,
            retry_failures=False,
            skip_validation=False,
        )

        state = create_initial_state(config, datasets, temp_output_dir)

        assert state.version == "3.0.0"
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
            ext=".tiff",
            overlay_alpha=0.3,
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
            save_layers={"hdf": True},
            extensions={"hdf": ".h5"},
            include_dataset_column=False,
            save_overlays=True,
        )

        manager.create_structure(datasets)

        from phenotypic.sdk_ import logs_dir

        # Check logs directory in the hidden machine-state cache
        assert logs_dir(temp_output_dir).exists()

        # Check results/ directory exists
        assert (temp_output_dir / "results").exists()

        # Current runs create no external per-image measurement authority.
        assert not (
            temp_output_dir / "results" / "dataset1" / "measurements"
        ).exists()
        # overlays dir created only because save_overlays=True
        assert (
            temp_output_dir / "deliverables" / "overlays" / "dataset1"
        ).exists()
        # The per-image image directory is `zarr/`: forward runs write one
        # OME-Zarr store per image, and nothing writes an `.h5` any more,
        # so provisioning `hdf/` would leave an empty directory behind.
        assert (temp_output_dir / "results" / "dataset1" / "zarr").exists()
        assert not (temp_output_dir / "results" / "dataset1" / "hdf").exists()

        # Old structure should NOT exist (datasets at root level)
        assert not (temp_output_dir / "dataset1").exists()
        assert not (temp_output_dir / "measurements").exists()
        assert not (temp_output_dir / "overlays").exists()
        assert not (
            temp_output_dir / "results" / "dataset1" / "overlays"
        ).exists()

    def test_create_structure_overlays_gated(self, temp_output_dir):
        """Without save_overlays=True, overlays/ must NOT be created."""
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
            save_layers={"hdf": True},
            extensions={"hdf": ".h5"},
            include_dataset_column=False,
            save_overlays=False,
        )

        manager.create_structure(datasets)

        assert (temp_output_dir / "results" / "dataset1" / "zarr").exists()
        assert not (
            temp_output_dir / "deliverables" / "overlays" / "dataset1"
        ).exists()

    def test_from_config_factory(self, temp_output_dir):
        """Test OutputManager.from_config() produces correct layers and extensions."""
        manager = OutputManager.from_config(
            base_dir=temp_output_dir,
            ext=".tiff",
            overlay_alpha=0.5,
        )

        # Forward runs now write a single HDF per image (Phase 2 change).
        assert manager.save_layers == {"hdf": True}
        assert manager.extensions == {"hdf": ".h5"}
        assert manager.overlay_alpha == 0.5
        assert manager.include_dataset_column is True
        # Overlays are always-on by default for forward runs.
        assert manager.save_overlays is True

    def test_from_config_factory_overlays_disabled(self, temp_output_dir):
        """from_config(save_overlays=False) disables overlay output (e.g. measure mode)."""
        manager = OutputManager.from_config(
            base_dir=temp_output_dir,
            ext=".tiff",
            overlay_alpha=0.5,
            save_overlays=False,
        )

        assert manager.save_overlays is False

    def test_pipeline_json_copied_to_output(self, temp_output_dir):
        """Test pipeline JSON is copied to output directory for reproducibility."""
        source_dir = temp_output_dir / "source"
        source_dir.mkdir()
        pipeline_path = source_dir / "my_pipeline.json"
        pipeline_content = '{"operations": []}'
        pipeline_path.write_text(pipeline_content)

        output_dir = temp_output_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        result = _copy_pipeline_to_output(pipeline_path, output_dir)

        assert result is not None
        assert result.exists()
        assert result.read_text() == pipeline_content

    def test_pipeline_json_not_overwritten_on_resume(self, temp_output_dir):
        """Test pipeline JSON is not overwritten if it already exists (resume)."""
        output_dir = temp_output_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-existing pipeline copy (from first run)
        pipeline_copy_path = output_dir / "pipeline.json"
        original_content = '{"operations": [{"name": "original"}]}'
        pipeline_copy_path.write_text(original_content)

        # New pipeline with different content
        source_dir = temp_output_dir / "source"
        source_dir.mkdir()
        new_pipeline = source_dir / "pipeline.json"
        new_pipeline.write_text('{"operations": [{"name": "modified"}]}')

        result = _copy_pipeline_to_output(new_pipeline, output_dir)

        assert result is None
        assert pipeline_copy_path.read_text() == original_content


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
                images=[Path(f"img{i}.png") for i in range(10)],  # 10 images
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

    def test_strategy_factory_local(
        self, temp_output_dir, simple_pipeline_json
    ):
        """Test creating local execution strategy."""
        config = ExecutionConfig(
            pipeline_json=simple_pipeline_json,
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
            ext=".tiff",
            overlay_alpha=0.3,
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

    def test_strategy_factory_slurm(
        self, temp_output_dir, simple_pipeline_json
    ):
        """Test creating SLURM execution strategy."""
        config = ExecutionConfig(
            pipeline_json=simple_pipeline_json,
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
            ext=".tiff",
            overlay_alpha=0.3,
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
        self, temp_output_dir, simple_pipeline_json
    ):
        """Test that --force-local overrides SLURM args."""
        config = ExecutionConfig(
            pipeline_json=simple_pipeline_json,
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
            ext=".tiff",
            overlay_alpha=0.3,
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
            ext=".tiff",
            overlay_alpha=0.3,
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

    def test_cli_accepts_pipeline_input_options(
        self, runner, temp_pipeline, temp_input_dir
    ):
        """Forward runs use explicit --pipeline and --input options."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                phenotypic_cli,
                [
                    "--pipeline",
                    str(temp_pipeline),
                    "--input",
                    str(temp_input_dir),
                    "--output",
                    "./out",
                    "--dry-run",
                ],
            )

            assert result.exit_code == 0, result.output
            assert "DRY-RUN MODE" in result.output

    def test_cli_accepts_short_pipeline_input_options(
        self, runner, temp_pipeline, temp_input_dir
    ):
        """Forward runs support -p and -i aliases."""
        with runner.isolated_filesystem():
            result = runner.invoke(
                phenotypic_cli,
                [
                    "-p",
                    str(temp_pipeline),
                    "-i",
                    str(temp_input_dir),
                    "--output",
                    "./out",
                    "--dry-run",
                ],
            )

            assert result.exit_code == 0, result.output
            assert "DRY-RUN MODE" in result.output

    def test_cli_rejects_old_positional_pipeline_input_style(
        self, runner, temp_pipeline, temp_input_dir
    ):
        """Old positional forward-run arguments fail with migration guidance."""
        result = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                str(temp_input_dir),
                "--output",
                "./out",
                "--dry-run",
            ],
        )

        assert result.exit_code != 0
        assert "--pipeline" in result.output
        assert "--input" in result.output
        assert "positional" in result.output.lower()

    def test_cli_rejects_stray_positional_argument_with_explicit_options(
        self, runner, temp_pipeline, temp_input_dir
    ):
        """Extra args captured by Click fail with the migration guidance."""
        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(temp_input_dir),
                "--output",
                "./out",
                "unexpected",
                "--dry-run",
            ],
        )

        assert result.exit_code != 0
        assert "Unexpected positional argument" in result.output
        assert "--pipeline" in result.output
        assert "--input" in result.output

    def test_cli_requires_pipeline_option_for_forward_run(
        self, runner, temp_input_dir
    ):
        """Normal runs fail clearly when --pipeline is omitted."""
        result = runner.invoke(
            phenotypic_cli,
            [
                "--input",
                str(temp_input_dir),
                "--output",
                "./out",
                "--dry-run",
            ],
        )

        assert result.exit_code != 0
        assert "--pipeline" in result.output

    def test_cli_requires_input_option_for_forward_run(
        self, runner, temp_pipeline
    ):
        """Normal runs fail clearly when --input is omitted."""
        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--output",
                "./out",
                "--dry-run",
            ],
        )

        assert result.exit_code != 0
        assert "--input" in result.output

    def test_cli_dry_run(self, runner, temp_pipeline, temp_input_dir):
        """Test --dry-run flag."""
        # Use isolated filesystem to avoid creating files in repo
        with runner.isolated_filesystem():
            result = runner.invoke(
                phenotypic_cli,
                [
                    "--pipeline",
                    str(temp_pipeline),
                    "--input",
                    str(temp_input_dir),
                    "--output",
                    "./out",
                    "--dry-run",
                ],
            )

            assert result.exit_code == 0
            assert "DRY-RUN MODE" in result.output
            assert "To proceed with execution" in result.output

    def test_cli_uses_explicit_output_dir(
        self, runner, temp_pipeline, temp_input_dir
    ):
        """Test CLI with specified output directory in tmp location."""
        # Use isolated filesystem to avoid creating files in repo
        with runner.isolated_filesystem():
            # Create a temporary output directory within the isolated filesystem
            temp_output_dir = Path("./tmp_output")

            result = runner.invoke(
                phenotypic_cli,
                [
                    "--pipeline",
                    str(temp_pipeline),
                    "--input",
                    str(temp_input_dir),
                    "-o",
                    str(temp_output_dir),
                    "--skip-validation",  # Skip for speed
                ],
            )

            # Check that the specified output directory was used.
            assert "Auto-generated output directory" not in result.output
            assert temp_output_dir.exists()

    def test_cli_sample_mode(self, runner, temp_pipeline, temp_input_dir):
        """Test --sample flag."""
        with runner.isolated_filesystem():
            output_dir = Path("./test_output")
            result = runner.invoke(
                phenotypic_cli,
                [
                    "--pipeline",
                    str(temp_pipeline),
                    "--input",
                    str(temp_input_dir),
                    "-o",
                    str(output_dir),
                    "--sample",
                    "2",
                    "--skip-validation",
                ],
            )

            # Should process limited number of images
            assert (
                "Sample mode" in result.output
                or "sample" in result.output.lower()
            )

    def test_cli_slurm_backend_selection(
        self, runner, temp_pipeline, temp_input_dir
    ):
        """Test that --slurm causes CLI to use SLURM backend."""
        with runner.isolated_filesystem():
            output_dir = Path("./test_output")
            result = runner.invoke(
                phenotypic_cli,
                [
                    "--pipeline",
                    str(temp_pipeline),
                    "--input",
                    str(temp_input_dir),
                    "-o",
                    str(output_dir),
                    "--slurm",
                    "slurm_partition=compute",
                    "--slurm",
                    "mem_gb=16",
                    "--dry-run",  # Use dry-run to avoid actual SLURM submission
                    "--skip-validation",
                ],
            )

            # Check that SLURM backend is displayed
            # The _display_execution_config function shows "SLURM Cluster" as backend
            assert "SLURM Cluster" in result.output or result.exit_code == 0


class TestSLURMFeatures:
    """Test SLURM-specific features."""

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


class TestAutomaticContinuation:
    """Tests for automatic continuation functionality."""

    def test_restart_and_overwrite_are_mutually_exclusive(
        self, runner, tmp_path
    ):
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        pipeline = tmp_path / "pipeline.json"
        pipeline.write_text(json.dumps({"operations": []}))

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(pipeline),
                "--input",
                str(input_dir),
                "--output",
                str(tmp_path / "output"),
                "--restart",
                "--overwrite",
            ],
        )

        assert result.exit_code != 0
        assert (
            "--restart and --overwrite are mutually exclusive" in result.output
        )

    def test_continuation_with_changed_input_images(self, runner, tmp_path):
        """Test that continuation fails when admitted input is missing."""
        from datetime import datetime

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
                    "errors": {},
                }
            },
            "config": {},
        }

        state_file = output_dir / "processing_state.json"
        state_file.write_text(json.dumps(state_dict))

        # Now delete one image
        image1.unlink()

        # Repeat the command. Automatic continuation rejects the missing input.
        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(temp_input_dir),
                "-o",
                str(output_dir),
                "--skip-validation",
            ],
        )

        # Should fail with input validation error
        assert result.exit_code != 0
        assert (
            "changed" in result.output.lower()
            or "input" in result.output.lower()
        )

    def test_continuation_command_without_output_specified(
        self, runner, tmp_path
    ):
        """Test that the public CLI requires --output."""

        temp_input_dir = tmp_path / "images"
        temp_input_dir.mkdir()
        (temp_input_dir / "image.jpg").write_text("dummy")

        temp_pipeline = tmp_path / "pipeline.json"
        temp_pipeline.write_text(json.dumps({"operations": []}))

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(temp_input_dir),
                "--skip-validation",
            ],
        )

        assert result.exit_code != 0
        assert "--output" in result.output

    def test_continuation_rejects_active_staged_jobs(
        self, runner, tmp_path, temp_pipeline, monkeypatch
    ):
        input_dir = tmp_path / "images"
        input_dir.mkdir()
        (input_dir / "image.tif").write_bytes(b"image")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        monkeypatch.setattr(
            "phenotypic._cli._cli_staged_orchestration.active_ledger_job_ids",
            lambda output: ["123", "456"],
        )

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-validation",
            ],
        )

        assert result.exit_code != 0
        assert "123, 456" in result.output

    def test_continuation_rejects_indeterminate_ledgered_job(
        self, runner, tmp_path, temp_pipeline, monkeypatch
    ):
        from phenotypic._cli._cli_staged_orchestration import append_job_ledger

        input_dir = tmp_path / "images"
        input_dir.mkdir()
        (input_dir / "image.tif").write_bytes(b"image")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        append_job_ledger(
            output_dir,
            epoch="epoch-1",
            token="controller",
            role="controller",
            round_index=0,
            status="submitted",
            job_id="789",
        )
        monkeypatch.setattr(
            "phenotypic._cli._cli_staged_orchestration.scheduler_job_is_active",
            lambda job_id: None,
        )

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--skip-validation",
            ],
        )

        assert result.exit_code != 0
        assert "jobs are active: 789" in result.output


class TestDryRunMode:
    """Tests for dry-run mode functionality."""

    def test_dry_run_creates_no_output(self, runner, tmp_path):
        """Test that dry-run doesn't actually process images."""

        temp_input_dir = tmp_path / "images"
        temp_input_dir.mkdir()
        (temp_input_dir / "image1.jpg").write_text("dummy")
        (temp_input_dir / "image2.jpg").write_text("dummy")

        temp_pipeline = tmp_path / "pipeline.json"
        temp_pipeline.write_text(json.dumps({"operations": []}))

        output_dir = tmp_path / "output"

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(temp_input_dir),
                "-o",
                str(output_dir),
                "--dry-run",
                "--skip-validation",
            ],
        )

        assert result.exit_code == 0
        assert (
            "dry" in result.output.lower()
            or "would process" in result.output.lower()
        )

        # Verify no output files created
        assert not (output_dir / "measurements").exists()
        assert not (output_dir / "overlays").exists()
        assert not (output_dir / "processing_state.json").exists()

    def test_dry_run_shows_processing_plan(self, runner, tmp_path):
        """Test that dry-run displays what would be processed."""

        temp_input_dir = tmp_path / "images"
        temp_input_dir.mkdir()
        (temp_input_dir / "image1.jpg").write_text("dummy")
        (temp_input_dir / "image2.jpg").write_text("dummy")

        temp_pipeline = tmp_path / "pipeline.json"
        temp_pipeline.write_text(json.dumps({"operations": []}))

        output_dir = tmp_path / "output"

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
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

    def test_invalid_nrows_zero_rejected(
        self, runner, tmp_path, temp_pipeline
    ):
        """Test that nrows=0 is rejected by Click IntRange validation."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create a dummy image file
        (input_dir / "test.jpg").write_text("dummy")

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
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
        assert (
            "Invalid value" in result.output
            or "out of range" in result.output.lower()
        )

    def test_invalid_ncols_negative_rejected(
        self, runner, tmp_path, temp_pipeline
    ):
        """Test that negative ncols is rejected by Click IntRange validation."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create a dummy image file
        (input_dir / "test.jpg").write_text("dummy")

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
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
        assert (
            "Invalid value" in result.output
            or "out of range" in result.output.lower()
        )

    def test_single_image_processing(self, runner, tmp_path, temp_pipeline):
        """Test processing with exactly 1 image (boundary case)."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create a single synthetic image
        grid_image = load_synth_yeast_plate()
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
        pil_img.save(input_dir / "single.jpg")

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(input_dir),
                "-o",
                str(output_dir),
                "--njobs",
                "1",
                "--skip-validation",
            ],
        )

        assert result.exit_code == 0, (
            f"CLI failed (exit_code={result.exit_code}):\n{result.output}"
        )

        # Verify output files created for the single image
        # Output should be in results/dataset folder named after input directory ("input")
        overlay_file = (
            output_dir / "deliverables" / "overlays" / "input" / "single.png"
        )
        # A forward run persists one OME-Zarr STORE per image -- a
        # directory, not a file. Resolve it through zarr_store_path so the
        # `.ome.zarr` double suffix is never hand-joined here.
        store = zarr_store_path(output_dir, "input", "single")

        import pandas as pd

        table = store / MEASUREMENT_TABLE_RELATIVE_PATH
        assert store.is_dir(), f"Expected {store} to exist"
        assert (store / "zarr.json").is_file()
        assert table.is_file(), f"Expected embedded measurements at {table}"
        assert not pd.read_parquet(table).empty
        assert not (output_dir / "results" / "input" / "measurements").exists()
        assert not list(output_dir.rglob("*.h5"))
        # Overlays are always written for forward runs.
        assert overlay_file.exists(), f"Expected {overlay_file} to exist"

        # Verify results and dataset folder is created
        assert (output_dir / "results").exists(), (
            "Results folder should be created"
        )
        assert (output_dir / "results" / "input").exists(), (
            "Dataset folder should be created under results/"
        )

    def test_empty_input_directory(self, runner, tmp_path, temp_pipeline):
        """Test graceful handling of empty input directory."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
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
        grid_image = load_synth_yeast_plate()
        from PIL import Image as PILImage

        for i in range(1, 4):
            pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
            pil_img.save(input_dir / f"image_{i:03d}.jpg")

        # Initial processing
        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(input_dir),
                "-o",
                str(output_dir),
                "--njobs",
                "1",
                "--skip-validation",
            ],
        )
        assert result.exit_code == 0, (
            f"CLI failed (exit_code={result.exit_code}):\n{result.output}"
        )

        # Change image set: remove image_001, add image_004 (same count!)
        (input_dir / "image_001.jpg").unlink()
        pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
        pil_img.save(input_dir / "image_004.jpg")

        # Automatic continuation should fail with an input-set mismatch.
        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(input_dir),
                "-o",
                str(output_dir),
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
        sample1 = get_sample_datasets(
            datasets, 5, temp_output_dir, random_seed=42
        )
        sample2 = get_sample_datasets(
            datasets, 5, temp_output_dir, random_seed=42
        )

        # Should get exact same images
        assert [img.name for img in sample1[0].images] == [
            img.name for img in sample2[0].images
        ]

        # Different seed should give different images
        sample3 = get_sample_datasets(
            datasets, 5, temp_output_dir, random_seed=123
        )
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

        # All paths should include results/ and dataset subdirectory
        path = manager.get_output_path(
            "single_image", "measurements", "image1"
        )
        assert (
            path
            == temp_output_dir
            / "output"
            / "results"
            / "single_image"
            / "measurements"
            / "image1.parquet"
        )

        path = manager.get_output_path("plate1", "measurements", "image1")
        assert (
            path
            == temp_output_dir
            / "output"
            / "results"
            / "plate1"
            / "measurements"
            / "image1.parquet"
        )

        path = manager.get_output_path("my_dataset", "overlays", "image2")
        assert (
            path
            == temp_output_dir
            / "output"
            / "deliverables"
            / "overlays"
            / "my_dataset"
            / "image2.png"
        )

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
            ext=".tiff",
            overlay_alpha=0.3,
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
            ext=".tiff",
            overlay_alpha=0.3,
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
        from phenotypic._cli._cli_slurm_config import (
            calculate_optimal_array_chunks,
        )

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
        from phenotypic._cli._cli_slurm_array_scripts import (
            generate_array_job_script,
        )

        pipeline_path = temp_output_dir / "pipeline.json"
        pipeline_path.write_text("{}", encoding="utf-8")
        images = [temp_output_dir / f"img{i}.png" for i in range(10)]
        for image in images:
            image.write_bytes(image.name.encode())
        dataset = Dataset(
            name="plate1",
            images=images,
            input_dir=Path("."),
            output_dir=temp_output_dir,
        )

        config = ExecutionConfig(
            pipeline_json=pipeline_path,
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
            ext=".tiff",
            overlay_alpha=0.3,
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

    @pytest.mark.skipif(
        sys.platform == "win32", reason="bash not available on Windows CI"
    )
    def test_slurm_script_bash_syntax(self, temp_output_dir):
        """Test that generated SLURM scripts have valid bash syntax."""
        import subprocess
        from phenotypic._cli._cli_slurm_array_scripts import (
            generate_array_job_script,
        )

        pipeline_path = temp_output_dir / "pipeline.json"
        pipeline_path.write_text("{}", encoding="utf-8")
        images = [temp_output_dir / f"img{i}.png" for i in range(5)]
        for image in images:
            image.write_bytes(image.name.encode())
        dataset = Dataset(
            name="test",
            images=images,
            input_dir=Path("."),
            output_dir=temp_output_dir,
        )

        config = ExecutionConfig(
            pipeline_json=pipeline_path,
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
            ext=".tiff",
            overlay_alpha=0.3,
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
        grid_image = load_synth_yeast_plate()
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
        pil_img.save(input_dir / "test.jpg")

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
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


class TestAggregateMeasurements:
    """Tests for the standalone aggregate_measurements() function."""

    def _create_measurement_csvs(self, output_dir, datasets):
        """Helper to create measurement Parquet files in the expected directory structure.

        Args:
            output_dir: Base output directory.
            datasets: Dict of {dataset_name: list of (image_stem, dataframe)} tuples.
        """
        import polars as pl

        for ds_name, images in datasets.items():
            meas_dir = output_dir / "results" / ds_name / "measurements"
            meas_dir.mkdir(parents=True, exist_ok=True)
            for stem, df in images:
                pl.from_pandas(df).write_parquet(
                    meas_dir / f"{stem}.parquet",
                    compression="zstd",
                    compression_level=3,
                )

    def test_aggregate_measurements_standalone(self, temp_output_dir):
        """Single dataset: master CSV has correct rows and Metadata_Dataset column."""
        import pandas as pd

        self._create_measurement_csvs(
            temp_output_dir,
            {
                "ds1": [
                    (
                        "img_001",
                        pd.DataFrame(
                            {"area": [10, 20], "circularity": [0.9, 0.8]}
                        ),
                    ),
                    (
                        "img_002",
                        pd.DataFrame({"area": [30], "circularity": [0.7]}),
                    ),
                ],
            },
        )

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=True,
        )

        assert result is not None
        assert result.name == "master_measurements.csv"
        master = pd.read_csv(result)
        assert len(master) == 3
        assert DATASET_HEADER in master.columns
        assert list(master[DATASET_HEADER].unique()) == ["ds1"]

        # Verify master Parquet is also written
        master_parquet = master_measurements_parquet_path(temp_output_dir)
        assert master_parquet.exists(), (
            "master_measurements.parquet should be written"
        )

    def test_aggregate_measurements_multi_dataset(self, temp_output_dir):
        """Two datasets: all rows present with correct dataset labels."""
        import pandas as pd

        self._create_measurement_csvs(
            temp_output_dir,
            {
                "plate_A": [
                    ("img_001", pd.DataFrame({"area": [10]})),
                ],
                "plate_B": [
                    ("img_001", pd.DataFrame({"area": [20]})),
                    ("img_002", pd.DataFrame({"area": [30]})),
                ],
            },
        )

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["plate_A", "plate_B"],
            include_dataset_column=True,
        )

        assert result is not None
        master = pd.read_csv(result)
        assert len(master) == 3
        assert set(master[DATASET_HEADER]) == {"plate_A", "plate_B"}
        assert (
            master.loc[master[DATASET_HEADER] == "plate_A", "area"].iloc[0]
            == 10
        )

    def test_aggregate_repairs_staged_uuid_when_using_scratch(
        self, temp_output_dir, tmp_path, monkeypatch
    ):
        """Scratch staging retains the original stem used by metadata joins."""
        import pandas as pd
        import polars as pl

        from phenotypic._cli import _cli_output_manager
        from phenotypic._cli._cli_parquet_agg import SOURCE_PATH_COLUMN

        aggregate_parquet_files = _cli_output_manager.aggregate_parquet_files

        def _aggregate_with_windows_source_paths(*args, **kwargs):
            frame = aggregate_parquet_files(*args, **kwargs)
            if frame is None:
                return None
            return frame.with_columns(
                pl.col(SOURCE_PATH_COLUMN).str.replace_all(
                    "/", "\\", literal=True
                )
            )

        monkeypatch.setattr(
            _cli_output_manager,
            "aggregate_parquet_files",
            _aggregate_with_windows_source_paths,
        )

        image_stem = "d000374_280_121_2026-04-11_16-55-18"
        self._create_measurement_csvs(
            temp_output_dir,
            {
                "plate": [
                    (
                        image_stem,
                        pd.DataFrame(
                            {
                                IMAGE_NAME_HEADER: [
                                    "4815d217-4afc-40dd-ab6c-bbf1521f4109"
                                ],
                                "area": [10],
                            }
                        ),
                    )
                ],
            },
        )
        metadata_path = temp_output_dir / "metadata.csv"
        pd.DataFrame(
            {
                IMAGE_NAME_HEADER: [image_stem],
                TREATMENT_LABEL: ["control"],
            }
        ).to_csv(metadata_path, index=False)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setenv("SCRATCH", str(scratch))

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["plate"],
            include_dataset_column=True,
            metadata_csv=metadata_path,
        )

        assert result is not None
        master = pd.read_csv(result)
        assert master[IMAGE_NAME_HEADER].tolist() == [image_stem]
        mirror = pd.read_csv(measurements_csv_path(temp_output_dir))
        assert mirror[IMAGE_NAME_HEADER].tolist() == [image_stem]
        assert mirror["area"].tolist() == [10]
        assert mirror[TREATMENT_HEADER].tolist() == ["control"]

    def test_aggregate_measurements_no_dataset_column(self, temp_output_dir):
        """include_dataset_column=False: no Metadata_Dataset column added."""
        import pandas as pd

        self._create_measurement_csvs(
            temp_output_dir,
            {
                "ds1": [
                    ("img_001", pd.DataFrame({"area": [10]})),
                ],
            },
        )

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=False,
        )

        assert result is not None
        master = pd.read_csv(result)
        assert DATASET_HEADER not in master.columns

    def test_aggregate_measurements_empty(self, temp_output_dir):
        """No CSVs found returns None."""
        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["nonexistent"],
            include_dataset_column=True,
        )
        assert result is None

    def test_aggregate_master_csv_delegates(self, temp_output_dir):
        """OutputManager.aggregate_master_csv() produces same result as standalone."""
        import pandas as pd

        self._create_measurement_csvs(
            temp_output_dir,
            {
                "ds1": [
                    ("img_001", pd.DataFrame({"area": [10, 20]})),
                ],
            },
        )

        om = OutputManager(
            base_dir=temp_output_dir,
            save_layers={},
            extensions={},
            include_dataset_column=True,
        )
        datasets = [
            Dataset(
                name="ds1",
                images=[],
                input_dir=temp_output_dir,
                output_dir=temp_output_dir,
            )
        ]
        result = om.aggregate_master_csv(datasets)

        assert result is not None
        master = pd.read_csv(result)
        assert len(master) == 2
        assert DATASET_HEADER in master.columns

    def test_aggregate_measurements_with_metadata(self, temp_output_dir):
        """Metadata CSV with shared column joins correctly, new columns appear."""
        import pandas as pd

        self._create_measurement_csvs(
            temp_output_dir,
            {
                "ds1": [
                    (
                        "img_001",
                        pd.DataFrame({"plate": ["A", "A"], "area": [10, 20]}),
                    ),
                    ("img_002", pd.DataFrame({"plate": ["B"], "area": [30]})),
                ],
            },
        )

        # Create metadata CSV with shared 'plate' column and schema-backed Treatment column.
        metadata_path = temp_output_dir / "metadata.csv"
        pd.DataFrame(
            {
                "plate": ["A", "B"],
                TREATMENT_LABEL: ["control", "drug_X"],
            }
        ).to_csv(metadata_path, index=False)

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=False,
            metadata_csv=metadata_path,
        )

        assert result is not None
        # Master archive stays clean: no external metadata columns join here.
        master = pd.read_csv(result)
        assert TREATMENT_HEADER not in master.columns
        assert len(master) == 3
        # Mirror carries the joined external metadata.
        mirror = pd.read_csv(measurements_csv_path(temp_output_dir))
        assert TREATMENT_HEADER in mirror.columns
        assert len(mirror) == 3
        assert list(
            mirror.loc[mirror["plate"] == "A", TREATMENT_HEADER].unique()
        ) == ["control"]
        assert list(
            mirror.loc[mirror["plate"] == "B", TREATMENT_HEADER].unique()
        ) == ["drug_X"]

    def test_aggregate_measurements_metadata_no_common_columns(
        self, temp_output_dir
    ):
        """No shared columns produces warning, master CSV unchanged."""
        import pandas as pd

        self._create_measurement_csvs(
            temp_output_dir,
            {
                "ds1": [
                    ("img_001", pd.DataFrame({"area": [10]})),
                ],
            },
        )

        metadata_path = temp_output_dir / "metadata.csv"
        pd.DataFrame(
            {
                "strain": ["WT"],
                "concentration": [0.5],
            }
        ).to_csv(metadata_path, index=False)

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=False,
            metadata_csv=metadata_path,
        )

        assert result is not None
        master = pd.read_csv(result)
        # No new columns added since there were no shared columns to join on
        assert "strain" not in master.columns
        assert "concentration" not in master.columns
        assert "area" in master.columns

    def test_aggregate_measurements_measurement_unmatched_is_dropped(
        self, temp_output_dir
    ):
        """Left join is asymmetric: *measurement* rows with no metadata are dropped.

        Measurements are the RIGHT frame, so plate C — measured but absent from
        the metadata CSV — does not survive, and no surviving row is a phantom.
        """
        import pandas as pd

        self._create_measurement_csvs(
            temp_output_dir,
            {
                "ds1": [
                    (
                        "img_001",
                        pd.DataFrame(
                            {"plate": ["A", "B", "C"], "area": [10, 20, 30]}
                        ),
                    ),
                ],
            },
        )

        # Only provide metadata for plates A and B, not C
        metadata_path = temp_output_dir / "metadata.csv"
        pd.DataFrame(
            {
                "plate": ["A", "B"],
                TREATMENT_LABEL: ["control", "drug_X"],
            }
        ).to_csv(metadata_path, index=False)

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=False,
            metadata_csv=metadata_path,
        )

        assert result is not None
        # Master archive is unfiltered: all source rows preserved, no metadata.
        master = pd.read_csv(result)
        assert TREATMENT_HEADER not in master.columns
        assert set(master["plate"].tolist()) == {"A", "B", "C"}
        # Mirror drops plate C (measurement-unmatched), treatment present.
        mirror = pd.read_csv(measurements_csv_path(temp_output_dir))
        assert len(mirror) == 2
        assert TREATMENT_HEADER in mirror.columns
        assert set(mirror["plate"].tolist()) == {"A", "B"}
        assert mirror[TREATMENT_HEADER].notna().all()
        # Every metadata row matched, so nothing is a phantom.
        assert not mirror[METADATA_ONLY_HEADER].any()

    def test_aggregate_measurements_metadata_duplicate_keys_warns(
        self, temp_output_dir, caplog
    ):
        """Duplicate keys in metadata CSV inflate rows and produce a warning.

        The signal is computed from the metadata frame's own key uniqueness, not
        from a row-count delta: under the left join 2 metadata rows fan out to 2
        output rows, so a height comparison would see no change at all. The
        duplicates are also not phantoms — both matched plate A.
        """
        import pandas as pd

        self._create_measurement_csvs(
            temp_output_dir,
            {
                "ds1": [
                    ("img_001", pd.DataFrame({"plate": ["A"], "area": [10]})),
                ],
            },
        )

        # Metadata has duplicate entries for plate A
        metadata_path = temp_output_dir / "metadata.csv"
        pd.DataFrame(
            {
                "plate": ["A", "A"],
                TREATMENT_LABEL: ["control", "drug_X"],
            }
        ).to_csv(metadata_path, index=False)

        with caplog.at_level(logging.WARNING):
            result = aggregate_measurements(
                output_dir=temp_output_dir,
                dataset_names=["ds1"],
                include_dataset_column=False,
                metadata_csv=metadata_path,
            )

        assert result is not None
        # Master archive stays at the original row count (no join applied).
        master = pd.read_csv(result)
        assert len(master) == 1
        assert TREATMENT_HEADER not in master.columns
        # Mirror reflects the duplicate-key inflation.
        mirror = pd.read_csv(measurements_csv_path(temp_output_dir))
        assert len(mirror) == 2
        assert "duplicate keys" in caplog.text
        # Duplicate-key fan-out is independent of the phantom signal.
        assert not mirror[METADATA_ONLY_HEADER].any()
        assert "matched no measured object" not in caplog.text

    def _write_unmatched_metadata_fixture(self, temp_output_dir):
        """Metadata for plates {A, B, C}; only A and B were ever measured."""
        import pandas as pd

        self._create_measurement_csvs(
            temp_output_dir,
            {
                "ds1": [
                    (
                        "img_001",
                        pd.DataFrame({"plate": ["A", "B"], "area": [10, 20]}),
                    ),
                ],
            },
        )
        metadata_path = temp_output_dir / "metadata.csv"
        pd.DataFrame(
            {
                "plate": ["A", "B", "C"],
                TREATMENT_LABEL: ["control", "drug_X", "drug_Y"],
            }
        ).to_csv(metadata_path, index=False)
        return metadata_path

    def test_aggregate_measurements_metadata_unmatched_becomes_phantom_row(
        self, temp_output_dir
    ):
        """A metadata key that matched no measured object survives as a phantom.

        Absence of a colony is data: plate C was never detected, so it must stay
        in the mirror carrying its metadata, null measurements, and the
        QC_MetadataOnly flag. The master archive stays clean.
        """
        import pandas as pd

        metadata_path = self._write_unmatched_metadata_fixture(temp_output_dir)

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=False,
            metadata_csv=metadata_path,
        )

        assert result is not None
        # Master archive stays clean: only measured rows, no metadata, no flag.
        master = pd.read_csv(result)
        assert set(master["plate"].tolist()) == {"A", "B"}
        assert TREATMENT_HEADER not in master.columns
        assert METADATA_ONLY_HEADER not in master.columns

        # Mirror keeps the undetected strain.
        mirror = pd.read_csv(measurements_csv_path(temp_output_dir))
        assert len(mirror) == 3
        assert set(mirror["plate"].tolist()) == {"A", "B", "C"}
        phantom = mirror.loc[mirror["plate"] == "C"]
        # Metadata values carried; measurements null.
        assert phantom[TREATMENT_HEADER].tolist() == ["drug_Y"]
        assert phantom["area"].isna().all()
        # Flagged, and only C is flagged.
        assert phantom[METADATA_ONLY_HEADER].tolist() == [True]
        assert not mirror.loc[
            mirror["plate"] != "C", METADATA_ONLY_HEADER
        ].any()

    def test_aggregate_measurements_metadata_unmatched_warns(
        self, temp_output_dir, caplog
    ):
        """Undetected metadata rows are reported at WARNING — the point of the join."""
        metadata_path = self._write_unmatched_metadata_fixture(temp_output_dir)

        with caplog.at_level(logging.WARNING):
            result = aggregate_measurements(
                output_dir=temp_output_dir,
                dataset_names=["ds1"],
                include_dataset_column=False,
                metadata_csv=metadata_path,
            )

        assert result is not None
        warnings_text = "\n".join(
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )
        assert "matched no measured object" in warnings_text
        assert "1/3" in warnings_text

    def test_phantom_row_does_not_float_promote_int_columns_under_post(
        self, temp_output_dir
    ):
        """An ``Int64`` column survives a phantom row + a post op as an integer.

        ``_apply_post_to_master`` round-trips the frame through pandas so post
        ops can run. ``to_pandas()`` has no nullable-int representation by
        default, so an ``Int64`` column carrying a phantom's null is promoted to
        ``float64`` and comes back as polars ``Float64`` — permanently. The
        mirror's dtype would then depend on whether a post op happened to be
        configured, which is unrelated to the data.

        This is the ONLY combination that triggers it: phantom row (supplies the
        null) AND a configured post op (supplies the round-trip). Neither alone
        reproduces it, which is why nothing else in the suite covers it.
        """
        import polars as pl

        from phenotypic import ImagePipeline
        from phenotypic.abc_._post_measurement import PostMeasurement

        class TouchNothing(PostMeasurement):
            """Forces the pandas round-trip without altering any value."""

            def _operate(self, df):
                return df

        metadata_path = self._write_unmatched_metadata_fixture(temp_output_dir)

        aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=False,
            metadata_csv=metadata_path,
            pipeline=ImagePipeline(post=[TouchNothing()]),
        )

        mirror = pl.read_parquet(measurements_parquet_path(temp_output_dir))
        # The phantom is present (so a null really did reach the round-trip)...
        assert mirror.filter(pl.col(METADATA_ONLY_HEADER)).height == 1
        # ...and the integer column is still an integer, not Float64.
        assert mirror.schema["area"] == pl.Int64, (
            f"'area' float-promoted to {mirror.schema['area']} by the pandas "
            "round-trip; the integer-dtype restore did not fire"
        )
        # The real rows keep their exact integer values.
        real = mirror.filter(~pl.col(METADATA_ONLY_HEADER))
        assert sorted(real["area"].to_list()) == [10, 20]

    def test_metadata_only_flag_round_trips_as_bool_in_both_mirrors(
        self, temp_output_dir
    ):
        """``QC_MetadataOnly`` re-reads as a real bool from parquet AND csv.

        The flag is the user-facing handle for "which strains went undetected" —
        the documented way to find them is a truth filter over the mirror
        (``mirror.filter(pl.col("QC_MetadataOnly"))`` /
        ``df[df["QC_MetadataOnly"]]``). If a writer ever emitted it as a string,
        every value would be truthy (``bool("False") is True``) and that filter
        would silently return *every* row instead of the missing strains.
        """
        import pandas as pd
        from pandas.api.types import is_bool_dtype

        metadata_path = self._write_unmatched_metadata_fixture(temp_output_dir)

        aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=False,
            metadata_csv=metadata_path,
        )

        from_parquet = pd.read_parquet(
            measurements_parquet_path(temp_output_dir)
        )
        from_csv = pd.read_csv(measurements_csv_path(temp_output_dir))

        assert is_bool_dtype(from_parquet[METADATA_ONLY_HEADER])
        assert is_bool_dtype(from_csv[METADATA_ONLY_HEADER])
        assert from_parquet[METADATA_ONLY_HEADER].sum() == 1
        assert from_csv[METADATA_ONLY_HEADER].sum() == 1

    def test_aggregate_measurements_metadata_dtype_mismatch(
        self, temp_output_dir
    ):
        """Join columns with mismatched dtypes (int vs str) still match."""
        import pandas as pd

        # Measurements have integer plate IDs
        self._create_measurement_csvs(
            temp_output_dir,
            {
                "ds1": [
                    (
                        "img_001",
                        pd.DataFrame({"plate": [1, 2], "area": [10, 20]}),
                    ),
                ],
            },
        )

        # Metadata has string plate IDs
        metadata_path = temp_output_dir / "metadata.csv"
        pd.DataFrame(
            {
                "plate": ["1", "2"],
                TREATMENT_LABEL: ["control", "drug_X"],
            }
        ).to_csv(metadata_path, index=False)

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=False,
            metadata_csv=metadata_path,
        )

        assert result is not None
        # Master archive stays clean — no metadata-join side effects.
        master = pd.read_csv(result)
        assert len(master) == 2
        assert TREATMENT_HEADER not in master.columns
        # Mirror carries the join even across int/str dtype mismatch on the key.
        mirror = pd.read_csv(measurements_csv_path(temp_output_dir))
        assert len(mirror) == 2
        assert TREATMENT_HEADER in mirror.columns
        assert mirror[TREATMENT_HEADER].notna().all()

    def test_aggregate_measurements_parquet_with_duckdb(self, temp_output_dir):
        """Standard .parquet files aggregate correctly via DuckDB."""
        import pandas as pd
        import polars as pl

        # Write PARQUET files (not arrow) to simulate pre-migration data
        meas_dir = temp_output_dir / "results" / "ds1" / "measurements"
        meas_dir.mkdir(parents=True, exist_ok=True)
        for stem, df in [
            ("img_001", pd.DataFrame({"area": [10, 20]})),
            ("img_002", pd.DataFrame({"area": [30]})),
        ]:
            pl.from_pandas(df).write_parquet(
                meas_dir / f"{stem}.parquet",
                compression="zstd",
                compression_level=3,
            )

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=True,
        )

        assert result is not None
        master = pd.read_csv(result)
        assert len(master) == 3
        assert DATASET_HEADER in master.columns

    def test_aggregate_measurements_post_seeds_post_applied_mirror(
        self, temp_output_dir
    ):
        """Pipeline post ops are applied to measurements.{csv,parquet} only.

        master_measurements.{csv,parquet} stay post-free (a clean archive of
        what per-image runs measured), while the seeded measurements.* mirror
        — what the GUI viewer reads — receives the post-applied frame.
        """
        import pandas as pd

        from phenotypic import ImagePipeline
        from phenotypic.abc_._post_measurement import PostMeasurement

        class AddPostColumn(PostMeasurement):
            def _operate(self, df):
                df = df.copy()
                df["post_marker"] = "applied"
                return df

        self._create_measurement_csvs(
            temp_output_dir,
            {
                "ds1": [
                    ("img_001", pd.DataFrame({"area": [10, 20]})),
                    ("img_002", pd.DataFrame({"area": [30]})),
                ],
            },
        )

        pipeline = ImagePipeline(post=[AddPostColumn()])

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=True,
            pipeline=pipeline,
        )

        assert result is not None

        master = pd.read_csv(master_measurements_csv_path(temp_output_dir))
        mirror = pd.read_csv(measurements_csv_path(temp_output_dir))

        # Master is clean — post column is absent.
        assert "post_marker" not in master.columns
        # Seeded mirror is post-applied — post column is present everywhere.
        assert "post_marker" in mirror.columns
        assert (mirror["post_marker"] == "applied").all()
        # Same number of rows in both.
        assert len(master) == len(mirror) == 3

    def test_finalize_post_master_outputs_seeds_post_applied_mirror(
        self, temp_output_dir
    ):
        """The unified post-master finalize helper seeds the post-applied frame.

        Both ``aggregate_measurements`` and the recompile worker now go
        through this helper, so this test pins the contract once: master_df
        is left untouched (clean), measurements.{csv,parquet} get the
        post-applied frame.
        """
        import polars as pl

        from phenotypic import ImagePipeline
        from phenotypic.abc_._post_measurement import PostMeasurement

        class TagPost(PostMeasurement):
            def _operate(self, df):
                df = df.copy()
                df["post_tag"] = "tagged"
                return df

        master_df = pl.DataFrame({"area": [10, 20, 30]})

        # Pre-write the master files so the helper's invariants match
        # what the real callers produce.
        master_measurements_csv_path(temp_output_dir).parent.mkdir(
            parents=True, exist_ok=True
        )
        master_measurements_csv_path(temp_output_dir).write_text(
            master_df.write_csv()
        )
        master_df.write_parquet(
            master_measurements_parquet_path(temp_output_dir),
            compression="zstd",
            compression_level=3,
        )

        finalize_post_master_outputs(
            temp_output_dir, master_df, ImagePipeline(post=[TagPost()])
        )

        mirror = pl.read_parquet(measurements_parquet_path(temp_output_dir))
        assert "post_tag" in mirror.columns
        assert mirror["post_tag"].to_list() == ["tagged", "tagged", "tagged"]

    def test_finalize_post_master_outputs_no_pipeline_skips_pipeline_steps(
        self, temp_output_dir
    ):
        """``pipeline=None`` still seeds the (clean) mirror; analysis is skipped."""
        import polars as pl

        master_df = pl.DataFrame({"area": [1, 2]})

        finalize_post_master_outputs(temp_output_dir, master_df, None)

        mirror = pl.read_parquet(measurements_parquet_path(temp_output_dir))
        # Mirror equals master when there's no pipeline to apply post.
        assert mirror.equals(master_df)
        # Pipeline-conditional artifacts are absent.
        assert not (
            deliverables_dir(temp_output_dir) / "analysis_manifest.json"
        ).exists()
        assert not pipeline_json_path(temp_output_dir).exists()

    def test_finalize_post_master_outputs_post_failure_falls_back_to_clean(
        self, temp_output_dir, caplog
    ):
        """If a post op raises, mirror is seeded with the clean master."""
        import logging

        import polars as pl

        from phenotypic import ImagePipeline
        from phenotypic.abc_._post_measurement import PostMeasurement

        class FailingPost(PostMeasurement):
            def _operate(self, df):
                raise RuntimeError("post op intentionally broken in test")

        master_df = pl.DataFrame({"area": [10, 20, 30]})

        with caplog.at_level(logging.WARNING):
            returned = finalize_post_master_outputs(
                temp_output_dir,
                master_df,
                ImagePipeline(post=[FailingPost()]),
            )

        # WARNING about the post-op exception is logged…
        assert any(
            "post-measurement transform raised" in rec.message.lower()
            for rec in caplog.records
        ), [rec.message for rec in caplog.records]

        # …and the helper returns the clean master so downstream callers
        # (e.g. finalization steps in the recompile worker) don't see
        # half-applied post columns.
        assert returned.equals(master_df)

        # Mirror on disk is seeded with the clean master, not a partial
        # post-applied frame.
        mirror = pl.read_parquet(measurements_parquet_path(temp_output_dir))
        assert mirror.equals(master_df)

    def test_aggregate_measurements_no_post_master_and_mirror_identical(
        self, temp_output_dir
    ):
        """With no post ops, measurements.* mirrors master data in canonical order."""
        import pandas as pd

        from phenotypic import ImagePipeline

        self._create_measurement_csvs(
            temp_output_dir,
            {
                "ds1": [
                    ("img_001", pd.DataFrame({"area": [10, 20]})),
                ],
            },
        )

        result = aggregate_measurements(
            output_dir=temp_output_dir,
            dataset_names=["ds1"],
            include_dataset_column=True,
            pipeline=ImagePipeline(),
        )

        assert result is not None
        master = pd.read_csv(master_measurements_csv_path(temp_output_dir))
        mirror = pd.read_csv(measurements_csv_path(temp_output_dir))
        expected = master[order_measurement_columns(master.columns)]
        pd.testing.assert_frame_equal(expected, mirror)
        assert mirror.columns.tolist() == [
            DATASET_HEADER,
            "area",
            IMAGE_NAME_HEADER,
        ]


# ---------------------------------------------------------------------------
# process-mode top-level CLI option: validation, ignored-flag warnings, dry-run
# ---------------------------------------------------------------------------


def test_layer_rejected_outside_process_mode(
    tmp_path, simple_pipeline_json, synth_one_level_input
):
    r = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(tmp_path / "o"),
            "--layer",
            "rgb",
            "--dry-run",
        ],
    )
    assert r.exit_code != 0
    assert "--layer" in r.output and "--mode process" in r.output


def test_process_only_warns_ignored_flags(
    tmp_path, simple_pipeline_json, synth_one_level_input
):
    r = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "process",
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(tmp_path / "o2"),
            "--layer",
            "rgb",
            "--no-qc",
            "--force-local",
            "--njobs",
            "1",
            "--dry-run",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "ignored" in r.output.lower()


def test_no_qc_is_preserved_on_execution_config(
    monkeypatch, tmp_path, simple_pipeline_json, synth_one_level_input
):
    captured = {}

    def _capture_dry_run(config, datasets, output_dir):
        captured["config"] = config

    monkeypatch.setattr(
        "phenotypic.phenotypicCLI.execute_dry_run", _capture_dry_run
    )
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(tmp_path / "no-qc-output"),
            "--no-qc",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["config"].no_qc is True


def test_process_only_dry_run_lists_plan(
    tmp_path, simple_pipeline_json, synth_one_level_input
):
    r = CliRunner().invoke(
        phenotypic_cli,
        [
            "--mode",
            "process",
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(tmp_path / "o3"),
            "--layer",
            "detect_mat",
            "--dry-run",
            "--force-local",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "process" in r.output.lower()
    assert ".phenotypic" in r.output
