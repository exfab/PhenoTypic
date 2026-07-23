"""Tests for SLURM drip-feed dispatcher script generation."""

import sys

import pytest

from phenotypic.sdk_.slurm._dispatcher import (
    generate_dispatcher_chain,
    generate_dispatcher_script,
)
from phenotypic.sdk_ import slurm_scripts_dir

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SLURM not available on Windows")


@pytest.fixture
def slurm_args():
    return {"slurm_partition": "short", "mem_gb": 16, "time": 60}


@pytest.fixture
def chunk_scripts(tmp_path):
    """Create dummy chunk script files."""
    scripts_dir = slurm_scripts_dir(tmp_path)
    scripts_dir.mkdir(parents=True)
    paths = []
    for i in range(4):
        p = scripts_dir / f"chunk{i}.sh"
        p.write_text(f"#!/bin/bash\necho chunk {i}")
        paths.append(p)
    return paths


class TestGenerateDispatcherScript:

    def test_script_submits_next_chunk(self, tmp_path, slurm_args):
        """Dispatcher delegates the next chunk to the lifecycle entry point."""
        chunk_script = tmp_path / "chunk1.sh"
        chunk_script.touch()
        dispatcher_script = tmp_path / "next_dispatch.sh"
        dispatcher_script.touch()
        log_dir = tmp_path / "logs"

        output = tmp_path / "dispatch_1.sh"
        generate_dispatcher_script(
            next_chunk_script=chunk_script,
            next_dispatcher_script=dispatcher_script,
            output_path=output,
            slurm_args=slurm_args,
            log_dir=log_dir,
        )

        content = output.read_text()
        assert "phenotypic._cli._cli_slurm_lifecycle" in content
        assert f"--chunk-script {chunk_script}" in content
        assert "sbatch --parsable" not in content

    def test_script_submits_next_dispatcher(self, tmp_path, slurm_args):
        """Dispatcher passes its successor to the lifecycle entry point."""
        chunk_script = tmp_path / "chunk1.sh"
        chunk_script.touch()
        next_dispatch = tmp_path / "dispatch_2.sh"
        next_dispatch.touch()
        log_dir = tmp_path / "logs"

        output = tmp_path / "dispatch_1.sh"
        generate_dispatcher_script(
            next_chunk_script=chunk_script,
            next_dispatcher_script=next_dispatch,
            output_path=output,
            slurm_args=slurm_args,
            log_dir=log_dir,
        )

        content = output.read_text()
        assert "--dispatcher-script" in content
        assert str(next_dispatch) in content

    def test_last_dispatcher_has_no_next(self, tmp_path, slurm_args):
        """Final dispatcher should only submit chunk, no next dispatcher."""
        chunk_script = tmp_path / "chunk_last.sh"
        chunk_script.touch()
        log_dir = tmp_path / "logs"

        output = tmp_path / "dispatch_last.sh"
        generate_dispatcher_script(
            next_chunk_script=chunk_script,
            next_dispatcher_script=None,
            output_path=output,
            slurm_args=slurm_args,
            log_dir=log_dir,
        )

        content = output.read_text()
        assert f"--chunk-script {chunk_script}" in content
        assert "no further dispatcher needed" in content
        assert "--dispatcher-script" not in content

    def test_dispatcher_minimal_resources(self, tmp_path, slurm_args):
        """Dispatcher should request minimal resources."""
        chunk_script = tmp_path / "chunk1.sh"
        chunk_script.touch()
        log_dir = tmp_path / "logs"

        output = tmp_path / "dispatch.sh"
        generate_dispatcher_script(
            next_chunk_script=chunk_script,
            next_dispatcher_script=None,
            output_path=output,
            slurm_args=slurm_args,
            log_dir=log_dir,
        )

        content = output.read_text()
        assert "--mem=100M" in content
        assert "--time=00:05:00" in content
        assert "--cpus-per-task=1" in content

    def test_dispatcher_uses_partition(self, tmp_path):
        """Dispatcher should use the partition from slurm_args."""
        chunk_script = tmp_path / "chunk.sh"
        chunk_script.touch()
        log_dir = tmp_path / "logs"

        output = tmp_path / "dispatch.sh"
        generate_dispatcher_script(
            next_chunk_script=chunk_script,
            next_dispatcher_script=None,
            output_path=output,
            slurm_args={"slurm_partition": "gpu"},
            log_dir=log_dir,
        )

        content = output.read_text()
        assert "--partition=gpu" in content

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not effective on Windows")
    def test_dispatcher_is_executable(self, tmp_path, slurm_args):
        """Generated dispatcher script should be executable."""
        chunk_script = tmp_path / "chunk.sh"
        chunk_script.touch()
        log_dir = tmp_path / "logs"

        output = tmp_path / "dispatch.sh"
        generate_dispatcher_script(
            next_chunk_script=chunk_script,
            next_dispatcher_script=None,
            output_path=output,
            slurm_args=slurm_args,
            log_dir=log_dir,
        )

        assert output.stat().st_mode & 0o111

    def test_dispatcher_has_shebang(self, tmp_path, slurm_args):
        """Generated dispatcher script should start with bash shebang."""
        chunk_script = tmp_path / "chunk.sh"
        chunk_script.touch()
        log_dir = tmp_path / "logs"

        output = tmp_path / "dispatch.sh"
        generate_dispatcher_script(
            next_chunk_script=chunk_script,
            next_dispatcher_script=None,
            output_path=output,
            slurm_args=slurm_args,
            log_dir=log_dir,
        )

        content = output.read_text()
        assert content.startswith("#!/bin/bash")


class TestGenerateDispatcherChain:

    def test_chain_length(self, chunk_scripts, tmp_path, slurm_args):
        """N chunks should produce N-1 dispatchers."""
        dispatchers = generate_dispatcher_chain(
            chunk_scripts=chunk_scripts,
            output_dir=tmp_path,
            slurm_args=slurm_args,
            log_dir=tmp_path / "logs" / "slurm",
        )
        assert len(dispatchers) == len(chunk_scripts) - 1

    def test_single_chunk_no_dispatcher(self, tmp_path, slurm_args):
        """1 chunk should produce 0 dispatchers."""
        chunk = slurm_scripts_dir(tmp_path) / "chunk0.sh"
        chunk.parent.mkdir(parents=True)
        chunk.write_text("#!/bin/bash\necho chunk")

        dispatchers = generate_dispatcher_chain(
            chunk_scripts=[chunk],
            output_dir=tmp_path,
            slurm_args=slurm_args,
            log_dir=tmp_path / "logs" / "slurm",
        )
        assert dispatchers == []

    def test_two_chunks_one_dispatcher(self, tmp_path, slurm_args):
        """2 chunks should produce 1 dispatcher."""
        scripts_dir = slurm_scripts_dir(tmp_path)
        scripts_dir.mkdir(parents=True)
        chunks = []
        for i in range(2):
            p = scripts_dir / f"chunk{i}.sh"
            p.write_text(f"#!/bin/bash\necho chunk {i}")
            chunks.append(p)

        dispatchers = generate_dispatcher_chain(
            chunk_scripts=chunks,
            output_dir=tmp_path,
            slurm_args=slurm_args,
            log_dir=tmp_path / "logs" / "slurm",
        )
        assert len(dispatchers) == 1
        # The dispatcher should submit chunk1 with no next dispatcher
        content = dispatchers[0].read_text()
        assert str(chunks[1]) in content
        assert "no further dispatcher needed" in content

    def test_dispatcher_naming(self, chunk_scripts, tmp_path, slurm_args):
        """Dispatchers should be named dispatch_1.sh, dispatch_2.sh, etc."""
        dispatchers = generate_dispatcher_chain(
            chunk_scripts=chunk_scripts,
            output_dir=tmp_path,
            slurm_args=slurm_args,
            log_dir=tmp_path / "logs" / "slurm",
        )
        for i, d in enumerate(dispatchers):
            assert d.name == f"dispatch_{i + 1}.sh"
            assert d.parent == slurm_scripts_dir(tmp_path)

    def test_dispatcher_chain_wiring(self, chunk_scripts, tmp_path, slurm_args):
        """Each dispatcher should reference the correct next chunk and dispatcher."""
        dispatchers = generate_dispatcher_chain(
            chunk_scripts=chunk_scripts,
            output_dir=tmp_path,
            slurm_args=slurm_args,
            log_dir=tmp_path / "logs" / "slurm",
        )

        # dispatch_1 submits chunk1, references dispatch_2
        content0 = dispatchers[0].read_text()
        assert str(chunk_scripts[1]) in content0
        assert "dispatch_2.sh" in content0

        # dispatch_2 submits chunk2, references dispatch_3
        content1 = dispatchers[1].read_text()
        assert str(chunk_scripts[2]) in content1
        assert "dispatch_3.sh" in content1

        # dispatch_3 submits chunk3, no next dispatcher
        content2 = dispatchers[2].read_text()
        assert str(chunk_scripts[3]) in content2
        assert "no further dispatcher needed" in content2

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod not effective on Windows")
    def test_all_dispatchers_are_executable(self, chunk_scripts, tmp_path, slurm_args):
        """All generated dispatchers should be executable."""
        dispatchers = generate_dispatcher_chain(
            chunk_scripts=chunk_scripts,
            output_dir=tmp_path,
            slurm_args=slurm_args,
            log_dir=tmp_path / "logs" / "slurm",
        )
        for d in dispatchers:
            assert d.stat().st_mode & 0o111

    def test_empty_chunk_list(self, tmp_path, slurm_args):
        """Empty chunk list should produce empty dispatcher list."""
        dispatchers = generate_dispatcher_chain(
            chunk_scripts=[],
            output_dir=tmp_path,
            slurm_args=slurm_args,
            log_dir=tmp_path / "logs" / "slurm",
        )
        assert dispatchers == []


class TestSbatchHelpers:
    """Tests for shared sbatch helper functions."""

    def test_parse_job_id_standard(self):
        from phenotypic.sdk_.slurm._sbatch import parse_job_id

        assert parse_job_id("Submitted batch job 12345\n") == "12345"

    def test_parse_job_id_multiline(self):
        from phenotypic.sdk_.slurm._sbatch import parse_job_id

        output = "scontrol: warning\nSubmitted batch job 99999\nextra\n"
        assert parse_job_id(output) == "99999"

    def test_parse_job_id_no_match_raises(self):
        from phenotypic.sdk_.slurm._sbatch import parse_job_id

        with pytest.raises(RuntimeError, match="Could not parse job ID"):
            parse_job_id("Error: something went wrong\n")

    def test_format_sbatch_directives_basic(self):
        from pathlib import Path
        from phenotypic.sdk_.slurm._sbatch import format_sbatch_directives

        directives = format_sbatch_directives(
            job_name="test-job",
            slurm_args={"slurm_partition": "short", "mem_gb": 16, "time": 90},
            output_log=Path("/logs/out.log"),
            error_log=Path("/logs/err.log"),
        )

        assert "--job-name=test-job" in directives
        assert "--partition=short" in directives
        assert "--mem=16G" in directives
        assert "--time=01:30:00" in directives
        assert "--output=/logs/out.log" in directives
        assert "--error=/logs/err.log" in directives

    def test_format_sbatch_directives_reserved_keys_filtered(self):
        from pathlib import Path
        from phenotypic.sdk_.slurm._sbatch import format_sbatch_directives

        directives = format_sbatch_directives(
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

        assert "--job-name=test-job" in directives
        assert "--output=/logs/out.log" in directives
        assert "--array=0-99" not in directives
        assert "/bad/path.log" not in directives
        assert "--job-name=override" not in directives
