"""
Unit tests for SLURM array job functionality in the CLI.

Tests cover array limit querying, chunking logic, script generation,
and sbatch submission parsing.
"""

import subprocess
import sys
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

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SLURM not available on Windows")


class TestSLURMArrayLimitParsing:
    """Tests for SLURM configuration querying."""

    @pytest.mark.parametrize(
        "num_images, array_limit, expected",
        [
            (500, 1000, [(0, 500)]),
            (2000, 1000, [(0, 1000), (1000, 2000)]),
            (2500, 1000, [(0, 1000), (1000, 2000), (2000, 2500)]),
            (1001, 1000, [(0, 1000), (1000, 1001)]),
            (1000, 1000, [(0, 1000)]),
            (0, 1000, []),
            (1000, 100, [(i, i + 100) for i in range(0, 1000, 100)]),
        ],
        ids=[
            "single_chunk",
            "exact_multiple",
            "with_remainder",
            "boundary_case",
            "equal_limit",
            "empty",
            "small_limit",
        ],
    )
    def test_calculate_optimal_array_chunks(self, num_images, array_limit, expected):
        """Test chunking logic for various num_images and array_limit combinations."""
        chunks = calculate_optimal_array_chunks(num_images, array_limit)
        assert chunks == expected

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
            ext=".tiff",
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
        if sys.platform != "win32":
            assert script_path.stat().st_mode & 0o111  # Executable

        # Read script content
        content = script_path.read_text()

        # Check SBATCH directives
        assert "#!/bin/bash" in content
        assert "#SBATCH --job-name=pht-test_dataset" in content
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
        assert "#SBATCH --job-name=pht-test_dataset-chunk0" in content
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
        assert "#SBATCH --job-name=pht-test_dataset-chunk1" in content
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

        # Each chunk reserves slots for inserted checkpoint/manifest/finalizer
        # sentinels, so the per-chunk image count is slightly below array_limit;
        # 2500 images at limit=1000 still splits into 3 chunks.
        assert len(all_scripts["large_dataset"]) == 3

    def test_generate_all_array_job_scripts_array_directive_within_limit(
        self, tmp_path, config
    ):
        """Generated #SBATCH --array=0-N must satisfy N+1 <= array_limit
        even after sentinel insertion. Regression for the
        'Invalid job array specification' sbatch failure when chunk size
        equaled array_limit and sentinels pushed len(entries) past MaxArraySize.
        """
        import re

        images = [tmp_path / f"image_{i:05d}.tif" for i in range(3663)]
        for img in images:
            img.touch()
        ds = Dataset(
            name="big",
            images=images,
            input_dir=tmp_path,
            output_dir=tmp_path / "output",
        )

        array_limit = 2500
        all_scripts = generate_all_array_job_scripts(
            datasets=[ds],
            config=config,
            output_dir=tmp_path / "output",
            array_limit=array_limit,
        )

        for script_path in all_scripts["big"]:
            content = script_path.read_text()
            match = re.search(r"#SBATCH --array=0-(\d+)", content)
            assert match is not None, f"No array directive in {script_path}"
            top_index = int(match.group(1))
            assert top_index + 1 <= array_limit, (
                f"Array directive 0-{top_index} ({top_index + 1} entries) "
                f"exceeds array_limit={array_limit} in {script_path.name}"
            )

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


class TestSLURMSubmissionErrors:
    """Tests for shared submit_script error handling."""

    @patch("phenotypic.sdk_.slurm._sbatch.subprocess.run")
    def test_submit_script_sbatch_not_found(self, mock_run):
        """Test handling when sbatch command is not found."""
        from phenotypic.sdk_.slurm._sbatch import submit_script

        mock_run.side_effect = FileNotFoundError("sbatch not found")

        with pytest.raises(RuntimeError, match="sbatch"):
            submit_script(Path("script.sh"))

    @patch("phenotypic.sdk_.slurm._sbatch.subprocess.run")
    def test_submit_script_sbatch_failure(self, mock_run):
        """Test handling when sbatch returns non-zero exit code."""
        from phenotypic.sdk_.slurm._sbatch import submit_script

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "sbatch", stderr="Invalid partition specified"
        )

        with pytest.raises(RuntimeError, match="submission failed"):
            submit_script(Path("script.sh"))

    @patch("phenotypic.sdk_.slurm._sbatch.subprocess.run")
    def test_submit_script_parsable_output(self, mock_run):
        """Test successful submission with --parsable output."""
        from phenotypic.sdk_.slurm._sbatch import submit_script

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345\n"
        )

        job_id = submit_script(Path("script.sh"))
        assert job_id == "12345"

    @patch("phenotypic.sdk_.slurm._sbatch.subprocess.run")
    def test_submit_script_parsable_with_cluster(self, mock_run):
        """Test --parsable output with cluster name (id;cluster)."""
        from phenotypic.sdk_.slurm._sbatch import submit_script

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="67890;cluster_name\n"
        )

        job_id = submit_script(Path("script.sh"))
        assert job_id == "67890"

    @patch("phenotypic.sdk_.slurm._sbatch.subprocess.run")
    def test_submit_script_with_dependency(self, mock_run):
        """Test submission with dependency flag."""
        from phenotypic.sdk_.slurm._sbatch import submit_script

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="12345\n"
        )

        job_id = submit_script(Path("script.sh"), dependency_job_id="11111")
        assert job_id == "12345"

        # Verify --dependency was passed
        call_args = mock_run.call_args[0][0]
        assert "--dependency" in call_args
        assert "afterany:11111" in call_args

    def test_array_limit_validation_negative(self):
        """Test that negative array limit raises error."""
        with pytest.raises(ValueError, match="array_limit must be positive"):
            calculate_optimal_array_chunks(100, -1)

    def test_array_limit_validation_zero(self):
        """Test that zero array limit raises error."""
        with pytest.raises(ValueError, match="array_limit must be positive"):
            calculate_optimal_array_chunks(100, 0)


class TestSLURMScriptChainSubmission:
    """Tests for shared SLURM script chain submission plumbing."""

    def test_submit_slurm_script_chain_raises_for_empty_scripts(self, tmp_path):
        """Empty script lists fail before dispatcher generation."""
        from phenotypic._cli._cli_slurm_submission import submit_slurm_script_chain

        console = MagicMock()

        with patch(
            "phenotypic._cli._cli_slurm_submission.generate_dispatcher_chain"
        ) as mock_generate_dispatcher_chain:
            with pytest.raises(RuntimeError, match="No array job scripts"):
                submit_slurm_script_chain(
                    flat_chunk_scripts=[],
                    output_dir=tmp_path / "output",
                    slurm_args={},
                    console=console,
                )

        mock_generate_dispatcher_chain.assert_not_called()

    def test_submit_slurm_script_chain_preserves_script_order_and_submits(
        self, tmp_path
    ):
        """Dispatcher generation and submission receive scripts in input order."""
        from phenotypic._cli._cli_slurm_submission import submit_slurm_script_chain
        from phenotypic.sdk_ import logs_dir, slurm_scripts_dir

        output_dir = tmp_path / "output"
        slurm_args = {"slurm_partition": "short", "mem_gb": 16}
        chunk_scripts = [
            slurm_scripts_dir(output_dir) / "dataset_a_chunk0.sh",
            slurm_scripts_dir(output_dir) / "dataset_b_chunk0.sh",
        ]
        dispatcher_scripts = [slurm_scripts_dir(output_dir) / "dispatch_1.sh"]
        console = MagicMock()

        with patch(
            "phenotypic._cli._cli_slurm_submission.generate_dispatcher_chain",
            return_value=dispatcher_scripts,
        ) as mock_generate_dispatcher_chain, patch(
            "phenotypic._cli._cli_slurm_submission.submit_drip_feed_start",
            return_value=(["123", "124"], None),
        ) as mock_submit_drip_feed_start:
            result = submit_slurm_script_chain(
                flat_chunk_scripts=chunk_scripts,
                output_dir=output_dir,
                slurm_args=slurm_args,
                console=console,
            )

        mock_generate_dispatcher_chain.assert_called_once_with(
            chunk_scripts=chunk_scripts,
            output_dir=output_dir,
            slurm_args=slurm_args,
            log_dir=logs_dir(output_dir) / "slurm",
        )
        mock_submit_drip_feed_start.assert_called_once_with(
            chunk_scripts=chunk_scripts,
            dispatcher_scripts=dispatcher_scripts,
        )
        assert result.job_ids == ["123", "124"]
        assert result.warning is None
        assert result.flat_scripts == chunk_scripts
        assert result.dispatcher_scripts == dispatcher_scripts

        console.print.assert_any_call("[bold cyan]Submitting jobs to SLURM...[/bold cyan]")
        console.print.assert_any_call("  Chunk 0: [green]Job 123[/green]")
        console.print.assert_any_call(
            "  Dispatcher 1: [green]Job 124[/green] (depends on 123)"
        )
        console.print.assert_any_call(
            "  Remaining 1 chunk(s) will be auto-submitted as each completes"
        )
