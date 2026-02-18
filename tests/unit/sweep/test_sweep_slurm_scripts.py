"""Tests for sweep SLURM array job script generation and 2D indexing."""

from pathlib import Path

import pytest

from phenotypic.sweep._sweep_slurm_scripts import (
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
        assert script.stat().st_mode & 0o111  # executable

    def test_2d_indexing_in_script(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """Verify the generated script contains correct 2D indexing math."""
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

        assert "IMAGE_IDX=$((GLOBAL_TASK_ID / N_PIPELINES))" in content
        assert "PIPE_IDX=$((GLOBAL_TASK_ID % N_PIPELINES))" in content
        assert f"N_PIPELINES={len(pipeline_names)}" in content

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
        """With no offset, GLOBAL_TASK_ID = SLURM_ARRAY_TASK_ID."""
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

        assert "GLOBAL_TASK_ID=$SLURM_ARRAY_TASK_ID" in content
        assert "GLOBAL_OFFSET" not in content

    def test_global_offset_nonzero(
        self, image_paths, pipeline_names, manifest_path, tmp_path, slurm_args
    ):
        """With offset, GLOBAL_TASK_ID should include the offset."""
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
        assert "GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + GLOBAL_OFFSET))" in content
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
        assert "GLOBAL_TASK_ID=$SLURM_ARRAY_TASK_ID" in content0

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
