"""
Unit tests for SLURM array job functionality in the CLI.

Tests cover array limit querying, chunking logic, script generation,
and sbatch submission parsing.
"""

import re
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from phenotypic._cli._cli_slurm_config import (
    get_slurm_array_limit,
    get_slurm_max_submit_jobs,
    calculate_optimal_array_chunks,
    validate_array_chunk,
)
from phenotypic._cli._cli_slurm_array_scripts import (
    generate_array_job_script,
    generate_all_array_job_scripts,
)
from phenotypic._cli._cli_types import Dataset, ExecutionConfig
from phenotypic._cli._cli_execution_strategies import AutonomousSLURMStrategy
from phenotypic._cli._cli_output_manager import OutputManager


class TestSLURMArrayLimitParsing:
    """Tests for SLURM configuration querying."""

    def test_calculate_optimal_array_chunks_single_chunk(self):
        """Test chunking with num_images <= array_limit."""
        chunks = calculate_optimal_array_chunks(500, 1000)
        assert chunks == [(0, 500)]

    def test_calculate_optimal_array_chunks_exact_multiple(self):
        """Test chunking when num_images is exact multiple of limit."""
        chunks = calculate_optimal_array_chunks(2000, 1000)
        assert chunks == [(0, 1000), (1000, 2000)]

    def test_calculate_optimal_array_chunks_with_remainder(self):
        """Test chunking when num_images is not exact multiple."""
        chunks = calculate_optimal_array_chunks(2500, 1000)
        assert chunks == [(0, 1000), (1000, 2000), (2000, 2500)]

    def test_calculate_optimal_array_chunks_boundary_case(self):
        """Test chunking at exact boundary (limit + 1)."""
        chunks = calculate_optimal_array_chunks(1001, 1000)
        assert chunks == [(0, 1000), (1000, 1001)]

    def test_calculate_optimal_array_chunks_equal_limit(self):
        """Test chunking when num_images equals limit."""
        chunks = calculate_optimal_array_chunks(1000, 1000)
        assert chunks == [(0, 1000)]

    def test_calculate_optimal_array_chunks_empty(self):
        """Test chunking with zero images."""
        chunks = calculate_optimal_array_chunks(0, 1000)
        assert chunks == []

    def test_calculate_optimal_array_chunks_small_limit(self):
        """Test chunking with small limit (many chunks)."""
        chunks = calculate_optimal_array_chunks(1000, 100)
        assert len(chunks) == 10
        assert chunks[0] == (0, 100)
        assert chunks[-1] == (900, 1000)

    @patch("subprocess.run")
    def test_get_slurm_array_limit_success(self, mock_run):
        """Test parsing MaxArraySize from scontrol output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="MaxArraySize           = 10000\n",
        )

        # Clear cache before test
        get_slurm_array_limit.cache_clear()

        limit = get_slurm_array_limit()
        assert limit == 10000

    @patch("subprocess.run")
    def test_get_slurm_array_limit_fallback(self, mock_run):
        """Test fallback when scontrol fails."""
        mock_run.side_effect = FileNotFoundError()

        # Clear cache before test
        get_slurm_array_limit.cache_clear()

        limit = get_slurm_array_limit()
        assert limit == 1000  # Default fallback

    @patch("subprocess.run")
    def test_get_slurm_array_limit_no_match(self, mock_run):
        """Test fallback when MaxArraySize not found in output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="SomeOtherConfig = 12345\n",
        )

        # Clear cache before test
        get_slurm_array_limit.cache_clear()

        limit = get_slurm_array_limit()
        assert limit == 1000  # Default fallback

    @patch("subprocess.run")
    def test_get_slurm_max_submit_jobs_success(self, mock_run):
        """Test parsing MaxSubmitJobsPerUser from sacctmgr."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="MaxSubmitJobsPerUser\n5000\n10000\n",
        )

        # Clear cache before test
        get_slurm_max_submit_jobs.cache_clear()

        limit = get_slurm_max_submit_jobs()
        assert limit == 10000  # Maximum of values

    @patch("subprocess.run")
    def test_get_slurm_max_submit_jobs_fallback(self, mock_run):
        """Test fallback when sacctmgr fails."""
        mock_run.side_effect = FileNotFoundError()

        # Clear cache before test
        get_slurm_max_submit_jobs.cache_clear()

        limit = get_slurm_max_submit_jobs()
        assert limit is None


class TestArrayChunkValidation:
    """Tests for array chunk validation."""

    def test_validate_array_chunk_valid(self):
        """Test validation of valid chunk."""
        assert validate_array_chunk((0, 500), 1000, 1000) is True

    def test_validate_array_chunk_exceeds_limit(self):
        """Test validation fails when chunk exceeds limit."""
        assert validate_array_chunk((0, 1500), 1000, 1000) is False

    def test_validate_array_chunk_negative_start(self):
        """Test validation fails with negative start."""
        assert validate_array_chunk((-1, 100), 1000, 1000) is False

    def test_validate_array_chunk_negative_end(self):
        """Test validation fails with negative end."""
        assert validate_array_chunk((0, -1), 1000, 1000) is False

    def test_validate_array_chunk_start_equals_end(self):
        """Test validation fails when start equals end."""
        assert validate_array_chunk((100, 100), 1000, 1000) is False

    def test_validate_array_chunk_end_before_start(self):
        """Test validation fails when end < start."""
        assert validate_array_chunk((500, 100), 1000, 1000) is False

    def test_validate_array_chunk_exceeds_num_images(self):
        """Test validation fails when end exceeds total images."""
        assert validate_array_chunk((0, 1500), 1000, 1000) is False


class TestArrayJobScriptGeneration:
    """Tests for array job script generation."""

    @pytest.fixture
    def dataset(self, tmp_path):
        """Create a test dataset."""
        # Create dummy image files
        images = []
        for i in range(10):
            img_path = tmp_path / f"image_{i:03d}.tif"
            img_path.touch()
            images.append(img_path)

        return Dataset(
            name="test_dataset",
            images=images,
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
        )

    @pytest.fixture
    def config(self, tmp_path):
        """Create a test execution config."""
        pipeline_json = tmp_path / "pipeline.json"
        pipeline_json.write_text('{"operations": []}')

        return ExecutionConfig(
            pipeline_json=pipeline_json,
            input_path=tmp_path,
            output_dir=tmp_path / "output",
            image_type="GridImage",
            nrows=8,
            ncols=12,
            bit_depth=None,
            n_jobs=-1,
            slurm_args={"slurm_partition": "short", "mem_gb": 16, "time": 60},
            force_local=False,
            wait=False,
            save_rgb=True,
            save_gray=False,
            save_enh_gray=False,
            save_objmask=True,
            save_objmap=False,
            save_objmap_overlay=False,
            save_enh_gray_overlay=False,
            save_objmask_overlay=False,
            rgb_ext="tiff",
            gray_ext="tiff",
            enh_gray_ext="tiff",
            objmask_ext="png",
            objmap_ext="png",
            objmap_overlay_ext="png",
            overlay_mode="image",
            overlay_alpha=0.3,
            include_dataset_column=False,
            dry_run=False,
            sample=None,
            resume=False,
            retry_failures=False,
            skip_validation=False,
        )

    def test_generate_array_job_script_basic(self, dataset, config, tmp_path):
        """Test basic array job script generation."""
        output_dir = tmp_path / "output"
        script_path = generate_array_job_script(
            dataset=dataset,
            array_indices=(0, 10),
            config=config,
            output_dir=output_dir,
            chunk_id=0,
        )

        assert script_path.exists()
        assert script_path.is_file()
        assert script_path.stat().st_mode & 0o111  # Executable

        # Read script content
        content = script_path.read_text()

        # Check SBATCH directives
        assert "#!/bin/bash" in content
        assert "#SBATCH --job-name=pheno-test_dataset" in content
        assert "#SBATCH --array=0-9" in content  # 10 images, 0-indexed
        assert "#SBATCH --partition=short" in content
        assert "#SBATCH --mem=16G" in content
        assert "#SBATCH --time=01:00:00" in content

        # Check image list
        for i in range(10):
            assert f"image_{i:03d}.tif" in content

        # Check processing command (split across lines with \ continuations)
        assert "python" in content
        assert "phenotypic._cli._cli_process_single" in content
        assert "--image-type" in content
        assert "GridImage" in content
        assert "--nrows" in content
        assert "8" in content
        assert "--ncols" in content
        assert "12" in content
        assert "--save-rgb" in content
        assert "--save-objmask" in content

        # Check array indexing logic
        assert "SLURM_ARRAY_TASK_ID" in content
        assert "IMAGE_LIST" in content
        assert "CURRENT_IMAGE=" in content

    def test_generate_array_job_script_chunked(self, dataset, config, tmp_path):
        """Test array job script generation for chunked dataset."""
        output_dir = tmp_path / "output"
        script_path = generate_array_job_script(
            dataset=dataset,
            array_indices=(0, 5),  # First half
            config=config,
            output_dir=output_dir,
            chunk_id=0,
        )

        content = script_path.read_text()

        # Should have chunk ID in job name
        assert "#SBATCH --job-name=pheno-test_dataset-chunk0" in content
        # Should only include first 5 images
        assert "#SBATCH --array=0-4" in content

    def test_generate_array_job_script_second_chunk(self, dataset, config, tmp_path):
        """Test array job script generation for second chunk."""
        output_dir = tmp_path / "output"
        script_path = generate_array_job_script(
            dataset=dataset,
            array_indices=(5, 10),  # Second half
            config=config,
            output_dir=output_dir,
            chunk_id=1,
        )

        content = script_path.read_text()

        # Should have chunk ID in job name and filename
        assert "#SBATCH --job-name=pheno-test_dataset-chunk1" in content
        assert "array_job_chunk1.sh" in str(script_path)
        # Should only include last 5 images
        assert "#SBATCH --array=0-4" in content  # Still 0-indexed for chunk

    def test_generate_all_array_job_scripts_single_dataset(
        self, dataset, config, tmp_path
    ):
        """Test generating scripts for single dataset."""
        output_dir = tmp_path / "output"
        all_scripts = generate_all_array_job_scripts(
            datasets=[dataset],
            config=config,
            output_dir=output_dir,
            array_limit=1000,
        )

        assert "test_dataset" in all_scripts
        assert len(all_scripts["test_dataset"]) == 1  # Single chunk
        assert all_scripts["test_dataset"][0].exists()

    def test_generate_all_array_job_scripts_chunked_dataset(
        self, tmp_path, config
    ):
        """Test generating scripts for dataset requiring chunking."""
        # Create large dataset (>1000 images)
        images = []
        for i in range(2500):
            img_path = tmp_path / f"image_{i:04d}.tif"
            # Don't create actual files for performance
            images.append(img_path)

        large_dataset = Dataset(
            name="large_dataset",
            images=images,
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
        )

        output_dir = tmp_path / "output"
        all_scripts = generate_all_array_job_scripts(
            datasets=[large_dataset],
            config=config,
            output_dir=output_dir,
            array_limit=1000,
        )

        # Should have 3 chunks (0-1000, 1000-2000, 2000-2500)
        assert len(all_scripts["large_dataset"]) == 3

    def test_generate_array_job_script_empty_chunk_raises(
        self, dataset, config, tmp_path
    ):
        """Test that empty chunk raises ValueError."""
        output_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="Empty chunk"):
            generate_array_job_script(
                dataset=dataset,
                array_indices=(10, 10),  # Empty range
                config=config,
                output_dir=output_dir,
                chunk_id=0,
            )


class TestSbatchJobIDParsing:
    """Tests for parsing sbatch output."""

    def test_sbatch_job_id_parsing_standard(self):
        """Test parsing standard sbatch output."""
        output = "Submitted batch job 12345\n"
        match = re.search(r"Submitted batch job (\d+)", output)
        assert match is not None
        assert match.group(1) == "12345"

    def test_sbatch_job_id_parsing_with_trailing_whitespace(self):
        """Test parsing sbatch output with trailing whitespace."""
        output = "Submitted batch job 67890  \n"
        match = re.search(r"Submitted batch job (\d+)", output)
        assert match is not None
        assert match.group(1) == "67890"

    def test_sbatch_job_id_parsing_multiline(self):
        """Test parsing sbatch output with multiple lines."""
        output = (
            "scontrol: some warning\n"
            "Submitted batch job 99999\n"
            "additional output\n"
        )
        match = re.search(r"Submitted batch job (\d+)", output)
        assert match is not None
        assert match.group(1) == "99999"

    def test_sbatch_job_id_parsing_no_match(self):
        """Test parsing when job ID pattern not found."""
        output = "Error: Invalid job submission\n"
        match = re.search(r"Submitted batch job (\d+)", output)
        assert match is None


class TestSLURMSubmissionErrors:
    """Tests for SLURM job submission error handling."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Create a mock ExecutionConfig for testing."""
        return ExecutionConfig(
            pipeline_json=tmp_path / "pipeline.json",
            input_path=tmp_path / "images",
            output_dir=tmp_path / "output",
            image_type="GridImage",
            nrows=8,
            ncols=12,
            bit_depth=None,
            n_jobs=-1,
            slurm_args={},
            force_local=False,
            wait=False,
            save_rgb=False,
            save_gray=False,
            save_enh_gray=False,
            save_objmask=False,
            save_objmap=False,
            save_objmap_overlay=False,
            save_enh_gray_overlay=False,
            save_objmask_overlay=False,
            rgb_ext="tiff",
            gray_ext="tiff",
            enh_gray_ext="tiff",
            objmask_ext="tiff",
            objmap_ext="tiff",
            objmap_overlay_ext="tiff",
            overlay_mode="image",
            overlay_alpha=0.3,
            include_dataset_column=False,
            dry_run=False,
            sample=None,
            resume=False,
            retry_failures=False,
            skip_validation=False,
        )

    @pytest.fixture
    def mock_output_manager(self, tmp_path):
        """Create a mock OutputManager for testing."""
        return OutputManager(
            base_dir=tmp_path / "output",
            save_layers={},
            extensions={},
            include_dataset_column=False,
        )

    @patch("subprocess.run")
    def test_submit_array_job_sbatch_not_found(self, mock_run, mock_config, mock_output_manager):
        """Test handling when sbatch command is not found."""
        mock_run.side_effect = FileNotFoundError("sbatch not found")

        strategy = AutonomousSLURMStrategy(mock_config, mock_output_manager)
        with pytest.raises(RuntimeError, match="sbatch"):
            strategy._submit_array_job_direct(Path("script.sh"), dependency_job_id=None)

    @patch("subprocess.run")
    def test_submit_array_job_sbatch_failure(self, mock_run, mock_config, mock_output_manager):
        """Test handling when sbatch returns non-zero exit code."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "sbatch", stderr="Invalid partition specified"
        )

        strategy = AutonomousSLURMStrategy(mock_config, mock_output_manager)
        with pytest.raises(RuntimeError, match="submission failed"):
            strategy._submit_array_job_direct(Path("script.sh"), dependency_job_id=None)

    @patch("subprocess.run")
    def test_submit_array_job_unparseable_output(self, mock_run, mock_config, mock_output_manager):
        """Test handling when sbatch output doesn't contain job ID."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Some other output\n"  # No "Submitted batch job" line
        )

        strategy = AutonomousSLURMStrategy(mock_config, mock_output_manager)
        with pytest.raises(RuntimeError, match="Could not parse job ID"):
            strategy._submit_array_job_direct(Path("script.sh"), dependency_job_id=None)

    @patch("subprocess.run")
    def test_submit_array_job_with_dependency_success(self, mock_run, mock_config, mock_output_manager):
        """Test successful submission with dependency."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Submitted batch job 12345\n"
        )

        strategy = AutonomousSLURMStrategy(mock_config, mock_output_manager)
        job_id = strategy._submit_array_job_direct(Path("script.sh"), dependency_job_id="11111")
        assert job_id == "12345"

    @patch("subprocess.run")
    def test_array_limit_validation_negative(self, mock_run, mock_config, mock_output_manager):
        """Test that negative array limit raises error."""
        with pytest.raises(ValueError, match="array_limit must be positive"):
            calculate_optimal_array_chunks(100, -1)

    @patch("subprocess.run")
    def test_array_limit_validation_zero(self, mock_run, mock_config, mock_output_manager):
        """Test that zero array limit raises error."""
        with pytest.raises(ValueError, match="array_limit must be positive"):
            calculate_optimal_array_chunks(100, 0)
