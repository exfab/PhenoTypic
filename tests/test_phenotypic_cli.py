"""
Test suite for the PhenoTypic CLI (phenotypic_cli.py).

Tests the command-line interface for batch processing images with pipelines,
including argument parsing, file I/O, and output validation.
"""

import json
import tempfile
from pathlib import Path
from shutil import copy2

import pytest
from click.testing import CliRunner

from phenotypic import Image, GridImage, ImagePipeline
from phenotypic.phenotypic_cli import main, process_single_image
from phenotypic.prefab import RoundPeaksPipeline
from phenotypic.data import load_synthetic_detection_image


# Get path to synthetic plates in the phenotypic data directory
def get_synthetic_plates_dir() -> Path:
    """Get the path to synthetic plates directory."""
    import phenotypic

    phenotypic_dir = Path(phenotypic.__file__).parent
    return phenotypic_dir / "data" / "synthetic_plates"


class TestPhenotypicCLI:
    """Test suite for the PhenoTypic CLI command."""

    @pytest.fixture
    def runner(self):
        """Provide a Click CliRunner for CLI testing."""
        return CliRunner()

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_dir = tmpdir / "input"
            output_dir = tmpdir / "output"
            input_dir.mkdir()
            # Don't create output_dir - CLI should create it
            yield input_dir, output_dir

    @pytest.fixture
    def circular_pipeline_json(self):
        """Create a RoundPeaksPipeline and save it to a temporary JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Create RoundPeaksPipeline instance with sensible defaults
            pipeline = RoundPeaksPipeline(
                blur_sigma=3,
                detector_thresh_method="otsu",
                detector_subtract_background=True,
                detector_remove_noise=True,
            )
            # Serialize to JSON
            json_str = pipeline.to_json()
            f.write(json_str)
            pipeline_path = Path(f.name)

        yield pipeline_path

        # Cleanup
        if pipeline_path.exists():
            pipeline_path.unlink()

    @pytest.fixture
    def synthetic_grid_image(self, temp_dirs):
        """Create a synthetic test GridImage with detected objects."""
        input_dir, _ = temp_dirs

        # Load the synthetic detection image
        grid_image = load_synthetic_detection_image()

        # Save it as a PNG in the input directory
        img_path = input_dir / "test_grid.png"
        grid_image.rgb[:].astype("uint8")  # Ensure proper type
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
        pil_img.save(img_path)

        return img_path

    def test_cli_help(self, runner):
        """Test that the help command works."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "PIPELINE_JSON" in result.output
        assert "INPUT_DIR" in result.output
        assert "OUTPUT_DIR" in result.output

    def test_cli_missing_pipeline_arg(self, runner):
        """Test error when pipeline JSON argument is missing."""
        result = runner.invoke(main, ["input_dir", "output_dir"])
        assert result.exit_code != 0
        assert (
            "PIPELINE_JSON" in result.output
            or "Error" in result.output
            or "missing" in result.output.lower()
        )

    def test_cli_missing_input_dir_arg(self, runner, circular_pipeline_json):
        """Test error when input directory argument is missing."""
        result = runner.invoke(main, [str(circular_pipeline_json)])
        assert result.exit_code != 0

    def test_cli_missing_output_dir_arg(
        self, runner, circular_pipeline_json, temp_dirs
    ):
        """Test error when output directory argument is missing."""
        input_dir, _ = temp_dirs
        result = runner.invoke(main, [str(circular_pipeline_json), str(input_dir)])
        assert result.exit_code != 0

    def test_cli_invalid_pipeline_file(self, runner, temp_dirs):
        """Test error when pipeline JSON file does not exist."""
        input_dir, output_dir = temp_dirs
        result = runner.invoke(
            main, ["/nonexistent/pipeline.json", str(input_dir), str(output_dir)]
        )
        assert result.exit_code != 0
        assert (
            "exists" in result.output.lower()
            or "Error" in result.output
            or "not found" in result.output.lower()
        )

    def test_cli_invalid_input_dir(self, runner, circular_pipeline_json, temp_dirs):
        """Test error when input directory does not exist."""
        _, output_dir = temp_dirs
        result = runner.invoke(
            main, [str(circular_pipeline_json), "/nonexistent/input", str(output_dir)]
        )
        assert result.exit_code != 0
        assert (
            "exists" in result.output.lower()
            or "Error" in result.output
            or "not found" in result.output.lower()
        )

    def test_cli_no_images_in_input_dir(
        self, runner, circular_pipeline_json, temp_dirs
    ):
        """Test error when input directory contains no valid images."""
        input_dir, output_dir = temp_dirs
        # input_dir exists but is empty
        result = runner.invoke(
            main, [str(circular_pipeline_json), str(input_dir), str(output_dir)]
        )
        assert result.exit_code != 0
        assert "No valid images found" in result.output

    def test_cli_creates_output_directories(
        self, runner, circular_pipeline_json, synthetic_grid_image, temp_dirs
    ):
        """Test that CLI creates output directories if they don't exist."""
        input_dir, output_dir = temp_dirs

        # Verify output_dir doesn't exist yet
        assert not output_dir.exists()

        result = runner.invoke(
            main, [str(circular_pipeline_json), str(input_dir), str(output_dir)]
        )

        # Verify directories were created
        assert output_dir.exists()
        assert (output_dir / "measurements").exists()
        assert (output_dir / "overlays").exists()

    def test_cli_invalid_image_type_option(
        self, runner, circular_pipeline_json, temp_dirs
    ):
        """Test error with invalid image type option."""
        input_dir, output_dir = temp_dirs
        result = runner.invoke(
            main,
            [
                str(circular_pipeline_json),
                str(input_dir),
                str(output_dir),
                "--image-type",
                "InvalidType",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "choice" in result.output.lower()

    def test_cli_gridimage_options(
        self, runner, circular_pipeline_json, synthetic_grid_image, temp_dirs
    ):
        """Test GridImage with custom nrows and ncols."""
        input_dir, output_dir = temp_dirs

        result = runner.invoke(
            main,
            [
                str(circular_pipeline_json),
                str(input_dir),
                str(output_dir),
                "--image-type",
                "GridImage",
                "--nrows",
                "16",
                "--ncols",
                "24",
            ],
        )

        # Should succeed (even if image doesn't match grid - CLI doesn't validate that)
        assert result.exit_code == 0

    def test_cli_bit_depth_option(
        self, runner, circular_pipeline_json, synthetic_grid_image, temp_dirs
    ):
        """Test bit-depth option."""
        input_dir, output_dir = temp_dirs

        result = runner.invoke(
            main,
            [
                str(circular_pipeline_json),
                str(input_dir),
                str(output_dir),
                "--bit-depth",
                "8",
            ],
        )

        # Should succeed
        assert result.exit_code == 0

    def test_cli_n_jobs_option(
        self, runner, circular_pipeline_json, synthetic_grid_image, temp_dirs
    ):
        """Test n-jobs option."""
        input_dir, output_dir = temp_dirs

        result = runner.invoke(
            main,
            [
                str(circular_pipeline_json),
                str(input_dir),
                str(output_dir),
                "--n-jobs",
                "1",  # Use 1 job for testing to avoid parallelization issues
            ],
        )

        # Should succeed
        assert result.exit_code == 0

    def test_cli_master_csv_created(
        self, runner, circular_pipeline_json, synthetic_grid_image, temp_dirs
    ):
        """Test that master_measurements.csv is created with successful processing."""
        input_dir, output_dir = temp_dirs

        result = runner.invoke(
            main,
            [
                str(circular_pipeline_json),
                str(input_dir),
                str(output_dir),
                "--n-jobs",
                "1",
            ],
        )

        assert result.exit_code == 0

        master_csv = output_dir / "master_measurements.csv"
        assert master_csv.exists(), "master_measurements.csv should exist"
        assert master_csv.is_file(), "master_measurements.csv should be a file"
        assert master_csv.parent.is_dir(), "Parent directory should exist"
        assert master_csv.stat().st_size > 0, (
            "master_measurements.csv should not be empty"
        )
        assert master_csv.suffix == ".csv", "File should have .csv extension"

    def test_cli_output_structure(
        self, runner, circular_pipeline_json, synthetic_grid_image, temp_dirs
    ):
        """Test that the output directory structure is created correctly."""
        input_dir, output_dir = temp_dirs

        result = runner.invoke(
            main,
            [
                str(circular_pipeline_json),
                str(input_dir),
                str(output_dir),
                "--n-jobs",
                "1",
            ],
        )

        assert result.exit_code == 0

        # Verify output_dir structure using pathlib
        assert output_dir.is_dir(), "Output directory should exist"
        assert output_dir.exists(), "Output directory should exist"

        # Check structure
        meas_dir = output_dir / "measurements"
        overlay_dir = output_dir / "overlays"

        assert meas_dir.is_dir(), (
            "measurements directory should exist and be a directory"
        )
        assert overlay_dir.is_dir(), (
            "overlays directory should exist and be a directory"
        )

        # At least one measurement CSV should exist
        csv_files = list(meas_dir.glob("*.csv"))
        assert len(csv_files) > 0, "At least one measurement CSV should exist"
        for csv_file in csv_files:
            assert csv_file.is_file(), f"CSV file {csv_file.name} should be a file"
            assert csv_file.stat().st_size > 0, (
                f"CSV file {csv_file.name} should not be empty"
            )

        # At least one overlay PNG should exist
        png_files = list(overlay_dir.glob("*.png"))
        assert len(png_files) > 0, "At least one overlay PNG should exist"
        for png_file in png_files:
            assert png_file.is_file(), f"PNG file {png_file.name} should be a file"
            assert png_file.suffix == ".png", (
                f"File {png_file.name} should have .png extension"
            )
            assert png_file.stat().st_size > 0, (
                f"PNG file {png_file.name} should not be empty"
            )

    def test_cli_measurements_are_valid_csv(
        self, runner, circular_pipeline_json, synthetic_grid_image, temp_dirs
    ):
        """Test that generated measurements are valid CSV files."""
        input_dir, output_dir = temp_dirs

        result = runner.invoke(
            main,
            [
                str(circular_pipeline_json),
                str(input_dir),
                str(output_dir),
                "--n-jobs",
                "1",
            ],
        )

        assert result.exit_code == 0

        # Load and validate master CSV
        master_csv = output_dir / "master_measurements.csv"
        assert master_csv.is_file(), "master_measurements.csv should be a file"
        assert master_csv.suffix == ".csv", "File should have .csv extension"
        assert master_csv.stat().st_size > 0, (
            "master_measurements.csv should not be empty"
        )

        import pandas as pd

        df = pd.read_csv(master_csv)

        # Should have at least one row
        assert len(df) > 0, "Master CSV should contain at least one row"
        # Should have multiple columns
        assert len(df.columns) > 0, "Master CSV should have measurement columns"

        # Verify individual CSVs exist for each image
        meas_dir = output_dir / "measurements"
        csv_files = list(meas_dir.glob("*.csv"))

        for csv_file in csv_files:
            assert csv_file.is_file(), f"{csv_file.name} should be a regular file"
            df_individual = pd.read_csv(csv_file)
            assert len(df_individual) > 0, (
                f"{csv_file.name} should contain measurement data"
            )

    def test_cli_successful_full_workflow(
        self, runner, circular_pipeline_json, synthetic_grid_image, temp_dirs
    ):
        """Test a complete successful CLI workflow from start to finish."""
        input_dir, output_dir = temp_dirs

        result = runner.invoke(
            main,
            [
                str(circular_pipeline_json),
                str(input_dir),
                str(output_dir),
                "--image-type",
                "GridImage",
                "--nrows",
                "1",  # Synthetic image is 1x1 grid
                "--ncols",
                "1",
                "--n-jobs",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert (
            "Successfully processed" in result.output
            or "Found 1 images" in result.output
        )
        assert (
            "master_measurements.csv" in result.output
            or (output_dir / "master_measurements.csv").exists()
        )

    def test_process_single_image_with_real_pipeline(self, temp_dirs):
        """Test process_single_image function with a real RoundPeaksPipeline and image."""
        input_dir, output_dir = temp_dirs
        meas_dir = output_dir / "measurements"
        overlay_dir = output_dir / "overlays"
        meas_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        # Load real synthetic detection image
        grid_image = load_synthetic_detection_image()

        # Save it
        img_path = input_dir / "test.png"
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
        pil_img.save(img_path)

        # Create real pipeline
        pipeline = RoundPeaksPipeline(
            blur_sigma=3,
            detector_thresh_method="otsu",
        )

        # Process it
        result = process_single_image(
            img_path,
            meas_dir,
            overlay_dir,
            pipeline,
            GridImage,
            {"nrows": 1, "ncols": 1},
        )

        # Should return DataFrame on success
        assert result is not None
        import pandas as pd

        assert isinstance(result, pd.DataFrame)

        # Check that files were created
        csv_files = list(meas_dir.glob("*.csv"))
        assert len(csv_files) > 0
        png_files = list(overlay_dir.glob("*.png"))
        assert len(png_files) > 0

    def test_process_single_image_handles_exception(self, temp_dirs):
        """Test that process_single_image handles exceptions gracefully."""
        input_dir, output_dir = temp_dirs
        meas_dir = output_dir / "measurements"
        overlay_dir = output_dir / "overlays"
        meas_dir.mkdir(parents=True, exist_ok=True)
        overlay_dir.mkdir(parents=True, exist_ok=True)

        # Create a fake image file (not a real image)
        fake_img = input_dir / "fake.jpg"
        fake_img.write_text("not an image")

        # Create real pipeline
        pipeline = RoundPeaksPipeline()

        result = process_single_image(
            fake_img, meas_dir, overlay_dir, pipeline, Image, {}
        )

        # Should return None on failure
        assert result is None

    @pytest.mark.slow
    def test_cli_with_synthetic_plates(self, runner, circular_pipeline_json):
        """Test CLI with real synthetic plate images."""
        synthetic_dir = get_synthetic_plates_dir()

        # Skip if synthetic plates don't exist
        if not synthetic_dir.exists():
            pytest.skip(f"Synthetic plates directory not found at {synthetic_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "output"

            # Run CLI on synthetic plates directory
            result = runner.invoke(
                main,
                [
                    str(circular_pipeline_json),
                    str(synthetic_dir),
                    str(output_dir),
                    "--image-type",
                    "GridImage",
                    "--nrows",
                    "8",
                    "--ncols",
                    "12",
                    "--n-jobs",
                    "1",
                ],
            )

            # Should succeed
            assert result.exit_code == 0

            # Verify output structure
            assert output_dir.exists()
            assert (output_dir / "measurements").exists()
            assert (output_dir / "overlays").exists()
            assert (output_dir / "master_measurements.csv").exists()

    @pytest.mark.slow
    def test_cli_processes_multiple_synthetic_plates(
        self, runner, circular_pipeline_json
    ):
        """Test CLI processing multiple synthetic plate images."""
        synthetic_dir = get_synthetic_plates_dir()

        # Skip if synthetic plates don't exist
        if not synthetic_dir.exists():
            pytest.skip(f"Synthetic plates directory not found at {synthetic_dir}")

        # Verify we have multiple images
        image_files = list(synthetic_dir.glob("*.jpg")) + list(
            synthetic_dir.glob("*.png")
        )
        if len(image_files) < 2:
            pytest.skip("Not enough synthetic plate images for this test")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "output"

            result = runner.invoke(
                main,
                [
                    str(circular_pipeline_json),
                    str(synthetic_dir),
                    str(output_dir),
                    "--image-type",
                    "GridImage",
                    "--nrows",
                    "8",
                    "--ncols",
                    "12",
                    "--n-jobs",
                    "1",
                ],
            )

            assert result.exit_code == 0

            # Check that all images were processed
            meas_dir = output_dir / "measurements"
            csv_files = list(meas_dir.glob("*.csv"))

            # Should have processed multiple images
            assert len(csv_files) >= len(image_files), (
                f"Expected at least {len(image_files)} CSV files, got {len(csv_files)}"
            )

            # Master CSV should contain aggregated results
            master_csv = output_dir / "master_measurements.csv"
            import pandas as pd

            df = pd.read_csv(master_csv)

            # Should have many rows (from multiple images)
            assert len(df) > 10, (
                f"Master CSV should have many measurement rows, got {len(df)}"
            )

    @pytest.mark.slow
    def test_cli_synthetic_plates_output_validation(
        self, runner, circular_pipeline_json
    ):
        """Validate output from processing synthetic plates."""
        synthetic_dir = get_synthetic_plates_dir()

        if not synthetic_dir.exists():
            pytest.skip(f"Synthetic plates directory not found at {synthetic_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "output"

            result = runner.invoke(
                main,
                [
                    str(circular_pipeline_json),
                    str(synthetic_dir),
                    str(output_dir),
                    "--image-type",
                    "GridImage",
                    "--nrows",
                    "8",
                    "--ncols",
                    "12",
                    "--n-jobs",
                    "1",
                ],
            )

            assert result.exit_code == 0

            # Validate measurements CSV
            master_csv = output_dir / "master_measurements.csv"
            import pandas as pd

            df = pd.read_csv(master_csv)

            # Check expected columns from RoundPeaksPipeline measurements
            expected_measurement_keywords = [
                "Metadata",  # Metadata columns
                "Bbox",  # Bounding box measurements
                "Shape",  # Shape measurements
                "Intensity",  # Intensity measurements
            ]

            found_columns = False
            for keyword in expected_measurement_keywords:
                matching_cols = [col for col in df.columns if keyword in col]
                if matching_cols:
                    found_columns = True
                    break

            assert found_columns, (
                f"No expected measurement columns found in {df.columns.tolist()}"
            )

            # Verify overlay PNGs exist and are valid
            overlay_dir = output_dir / "overlays"
            png_files = list(overlay_dir.glob("*.png"))

            assert len(png_files) > 0, "No overlay PNG files created"

            for png_file in png_files:
                assert png_file.is_file()
                assert png_file.stat().st_size > 1000, (
                    f"PNG file {png_file.name} is too small (may be invalid)"
                )

            # Verify individual measurement CSVs
            meas_dir = output_dir / "measurements"
            csv_files = list(meas_dir.glob("*.csv"))

            assert len(csv_files) > 0, "No individual measurement CSVs created"

            for csv_file in csv_files:
                csv_df = pd.read_csv(csv_file)
                assert len(csv_df) > 0, f"{csv_file.name} has no measurement data"

    @pytest.mark.slow
    def test_cli_parallel_processing_synthetic_plates(
        self, runner, circular_pipeline_json
    ):
        """Test parallel processing with synthetic plates."""
        synthetic_dir = get_synthetic_plates_dir()

        if not synthetic_dir.exists():
            pytest.skip(f"Synthetic plates directory not found at {synthetic_dir}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "output"

            # Run with 2 parallel jobs
            result = runner.invoke(
                main,
                [
                    str(circular_pipeline_json),
                    str(synthetic_dir),
                    str(output_dir),
                    "--image-type",
                    "GridImage",
                    "--nrows",
                    "8",
                    "--ncols",
                    "12",
                    "--n-jobs",
                    "2",
                ],
            )

            # Should succeed with parallel execution
            assert result.exit_code == 0

            # Verify outputs are correct
            master_csv = output_dir / "master_measurements.csv"
            assert master_csv.exists()
            assert master_csv.stat().st_size > 0


class TestModuleCallable:
    """Test that the module can be called as python -m phenotypic."""

    def test_module_has_main(self):
        """Test that phenotypic_cli module has main function."""
        from phenotypic.phenotypic_cli import main

        assert callable(main)

    def test_main_module_exists(self):
        """Test that __main__.py exists in phenotypic package."""
        from phenotypic import __file__ as phenotypic_init

        main_file = Path(phenotypic_init).parent / "__main__.py"
        assert main_file.exists()

    def test_import_main_from_main_module(self):
        """Test that __main__.py can import main function."""
        import phenotypic.__main__

        # If this imports without error, the module is properly set up
        assert True

    def test_circular_pipeline_serialization(self):
        """Test that RoundPeaksPipeline can be serialized and deserialized."""
        # Use explicit parameters to avoid issues with default parameter combinations
        pipeline = RoundPeaksPipeline(
            blur_sigma=3,
            detector_thresh_method="otsu",
        )
        json_str = pipeline.to_json()
        assert json_str is not None
        assert len(json_str) > 0

        # Load it back
        loaded_pipeline = ImagePipeline.from_json(json_str)
        assert loaded_pipeline is not None
