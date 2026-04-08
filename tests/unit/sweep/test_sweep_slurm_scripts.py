"""Tests for sweep SLURM array job script generation and 2D indexing."""

import math
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SLURM not available on Windows")

from phenotypic.sweep._sweep_cli._sweep_slurm_scripts import (
    generate_sweep_array_script,
    generate_sweep_array_scripts_chunked,
)


@pytest.fixture
def image_paths(tmp_path):
    """Create dummy image paths."""
    paths = []
    for i in range(5):
        p = tmp_path / f"plate_{i}.tiff"
        p.touch()
        paths.append(p)
    return paths


@pytest.fixture
def pipeline_names():
    return ["Pipeline_0", "Pipeline_1", "Pipeline_2"]


@pytest.fixture
def manifest_path(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text("{}")
    return p


@pytest.fixture
def slurm_args():
    return {"slurm_partition": "short", "mem_gb": 16, "time": 60}


class TestGenerateSweepArrayScript:

    def test_script_created_and_executable(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        output_dir = tmp_path / "output"
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={"nrows": 8, "ncols": 12},
            slurm_args=slurm_args,
        )
        assert script.exists()
        if sys.platform != "win32":
            assert script.stat().st_mode & 0o111  # executable

    def test_2d_indexing_in_script(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Verify the generated script contains correct image-major 2D indexing."""
        output_dir = tmp_path / "output"
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={"nrows": 8, "ncols": 12},
            slurm_args=slurm_args,
        )
        content = script.read_text()

        assert "IMAGE_IDX=$((GLOBAL_TASK_ID % N_IMAGES))" in content
        assert "PIPE_IDX=$((GLOBAL_TASK_ID / N_IMAGES))" in content
        assert f"N_IMAGES={len(image_paths)}" in content

    def test_pipeline_names_array_in_script(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Verify pipeline names bash array is present."""
        output_dir = tmp_path / "output"
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
        )
        content = script.read_text()

        assert "PIPELINE_NAMES=(" in content
        for name in pipeline_names:
            assert name in content

    def test_pipeline_name_flag_in_command(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Verify --pipeline-name flag is passed to worker command."""
        output_dir = tmp_path / "output"
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
        )
        content = script.read_text()

        assert '--pipeline-name' in content
        assert '"${CURRENT_PIPELINE}"' in content

    def test_array_directive_size(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Array size should be N_images * N_pipelines."""
        output_dir = tmp_path / "output"
        total = len(image_paths) * len(pipeline_names)
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
        )
        content = script.read_text()

        assert f"#SBATCH --array=0-{total - 1}" in content

    def test_global_offset_zero_by_default(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """With no offset, BASE_TASK_ID = SLURM_ARRAY_TASK_ID."""
        output_dir = tmp_path / "output"
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
        )
        content = script.read_text()

        assert "BASE_TASK_ID=$SLURM_ARRAY_TASK_ID" in content
        assert "GLOBAL_OFFSET" not in content

    def test_global_offset_nonzero(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """With offset, BASE_TASK_ID should include the offset."""
        output_dir = tmp_path / "output"
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
            global_offset=100,
            num_local_tasks=5,
        )
        content = script.read_text()

        assert "GLOBAL_OFFSET=100" in content
        assert "BASE_TASK_ID=$((SLURM_ARRAY_TASK_ID + GLOBAL_OFFSET))" in content
        assert "#SBATCH --array=0-4" in content

    def test_log_header_shows_image_and_pipeline(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Log header should show image and pipeline info."""
        output_dir = tmp_path / "output"
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
        )
        content = script.read_text()

        assert 'echo "Image: $CURRENT_IMAGE"' in content
        assert 'echo "Pipeline: $CURRENT_PIPELINE"' in content

    def test_empty_image_paths_raises(
        self, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        with pytest.raises(ValueError, match="empty"):
            generate_sweep_array_script(
                image_paths=[],
                pipeline_names=pipeline_names,
                manifest_path=manifest_path,
                output_dir=tmp_path / "output",
                image_type="GridImage",
                read_kwargs={},
                slurm_args=slurm_args,
            )

    def test_empty_pipeline_names_raises(
        self, image_paths, manifest_path, tmp_path, slurm_args
    ):
        with pytest.raises(ValueError, match="empty"):
            generate_sweep_array_script(
                image_paths=image_paths,
                pipeline_names=[],
                manifest_path=manifest_path,
                output_dir=tmp_path / "output",
                image_type="GridImage",
                read_kwargs={},
                slurm_args=slurm_args,
            )


class TestGenerateSweepArrayScriptsChunked:

    def test_single_chunk_when_under_limit(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """No chunking needed when total tasks < array_limit."""
        output_dir = tmp_path / "output"
        scripts = generate_sweep_array_scripts_chunked(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
            array_limit=1000,
        )
        assert len(scripts) == 1
        assert scripts[0].name == "sweep_array_job.sh"

    def test_multiple_chunks_when_over_limit(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Should produce multiple chunk scripts when exceeding array limit."""
        output_dir = tmp_path / "output"
        # 5 images * 3 pipelines = 15 tasks, limit=10 -> 2 chunks
        scripts = generate_sweep_array_scripts_chunked(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
            array_limit=10,
        )
        assert len(scripts) == 2
        assert scripts[0].name == "sweep_array_job_chunk0.sh"
        assert scripts[1].name == "sweep_array_job_chunk1.sh"

    def test_chunk_offsets_are_correct(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Chunk scripts should have correct global offsets and local sizes."""
        output_dir = tmp_path / "output"
        # 5 images * 3 pipelines = 15 tasks, limit=10 -> chunks (0,10), (10,15)
        scripts = generate_sweep_array_scripts_chunked(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
            array_limit=10,
        )

        content0 = scripts[0].read_text()
        # First chunk: 10 local tasks, no offset
        assert "#SBATCH --array=0-9" in content0
        assert "BASE_TASK_ID=$SLURM_ARRAY_TASK_ID" in content0

        content1 = scripts[1].read_text()
        # Second chunk: 5 local tasks, offset=10
        assert "#SBATCH --array=0-4" in content1
        assert "GLOBAL_OFFSET=10" in content1

    def test_all_chunks_have_full_image_and_pipeline_lists(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Each chunk script should contain the full image and pipeline lists."""
        output_dir = tmp_path / "output"
        scripts = generate_sweep_array_scripts_chunked(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
            array_limit=10,
        )
        for script in scripts:
            content = script.read_text()
            for img in image_paths:
                assert img.name in content
            for pname in pipeline_names:
                assert pname in content


class TestBatchedProcessing:
    """Tests for the batch_size > 1 code path."""

    def test_batch_size_produces_batched_loop(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """batch_size > 1 should generate a batched for-loop."""
        output_dir = tmp_path / "output"
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
            batch_size=5,
        )
        content = script.read_text()

        assert "BATCH_SIZE=5" in content
        assert "BATCH_START=" in content
        assert "BATCH_END=" in content
        assert "for GLOBAL_TASK_ID in $(seq" in content
        assert "BATCH_OK=" in content
        assert "BATCH_FAIL=" in content

    def test_batch_size_one_produces_single_pair_section(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """batch_size=1 (default) should produce single-pair processing."""
        output_dir = tmp_path / "output"
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
            batch_size=1,
        )
        content = script.read_text()

        assert "# --- Single-pair processing ---" in content
        assert "BATCH_SIZE=" not in content
        assert "for GLOBAL_TASK_ID in $(seq" not in content

    def test_batch_size_reduces_array_directive(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """batch_size=5 with 15 total tasks -> array=0-2 (3 elements)."""
        output_dir = tmp_path / "output"
        total = len(image_paths) * len(pipeline_names)  # 15
        batch_size = 5
        effective = math.ceil(total / batch_size)  # 3
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
            batch_size=batch_size,
        )
        content = script.read_text()

        assert f"#SBATCH --array=0-{effective - 1}" in content

    def test_batch_size_clamping_in_script(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Batched script should clamp BATCH_END to TOTAL_TASKS."""
        output_dir = tmp_path / "output"
        total = len(image_paths) * len(pipeline_names)
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
            batch_size=4,
        )
        content = script.read_text()

        assert f"TOTAL_TASKS={total}" in content
        assert 'if [ "$BATCH_END" -gt "$TOTAL_TASKS" ]' in content

    def test_chunked_scripts_with_batch_size(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Chunking should use effective_tasks (after batching) not total_tasks."""
        output_dir = tmp_path / "output"
        # 5 * 3 = 15 tasks, batch_size=5 -> 3 effective, limit=2 -> 2 chunks
        scripts = generate_sweep_array_scripts_chunked(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
            array_limit=2,
            batch_size=5,
        )
        assert len(scripts) == 2
        # chunk0: array 0-1 (2 elements), chunk1: array 0-0 (1 element)
        content0 = scripts[0].read_text()
        assert "#SBATCH --array=0-1" in content0
        content1 = scripts[1].read_text()
        assert "#SBATCH --array=0-0" in content1

    def test_batch_exit_logic_in_script(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Batched script should only exit 1 if ALL pairs fail."""
        output_dir = tmp_path / "output"
        script = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type="GridImage",
            read_kwargs={},
            slurm_args=slurm_args,
            batch_size=3,
        )
        content = script.read_text()

        assert '[ "$BATCH_OK" -eq 0 ]' in content
        assert '[ "$BATCH_FAIL" -gt 0 ]' in content


class TestReservedSbatchKeys:
    """Tests for reserved SBATCH key filtering."""

    def test_reserved_keys_filtered(self):
        """Reserved keys (array, output, error, job-name) should be dropped."""
        from phenotypic._cli._cli_slurm_scripts import generate_slurm_directives
        from pathlib import Path

        directives = generate_slurm_directives(
            job_name="test-job",
            slurm_args={
                "slurm_partition": "short",
                "array": "0-99",
                "output": "/bad/path.log",
                "error": "/bad/err.log",
                "job-name": "override",
            },
            output_log=Path("/logs/out.log"),
            error_log=Path("/logs/err.log"),
        )

        # Should have the managed values, not user overrides
        assert "--job-name=test-job" in directives
        assert "--output=/logs/out.log" in directives
        assert "--error=/logs/err.log" in directives
        assert "--partition=short" in directives
        # User's reserved keys should NOT appear
        assert "--array=0-99" not in directives
        assert "/bad/path.log" not in directives
        assert "/bad/err.log" not in directives
        assert "--job-name=override" not in directives


class TestGetPythonCommandForSlurm:
    """Tests for get_python_command(for_slurm=True)."""

    def test_for_slurm_returns_sys_executable(self):
        from phenotypic._cli._cli_utils import get_python_command

        cmd, desc = get_python_command(for_slurm=True)
        assert cmd == [sys.executable]
        assert "direct venv" in desc

    def test_default_does_not_return_sys_executable(self):
        from phenotypic._cli._cli_utils import get_python_command

        cmd, _ = get_python_command()
        # Default should be either uv or plain python, not direct venv
        assert cmd != [sys.executable] or "uv" not in cmd[0]
