"""Tests for SLURM drip-feed dispatcher script generation."""

import sys
from pathlib import Path
from typing import cast

import pytest

import phenotypic._cli._cli_slurm_lifecycle as lifecycle
from phenotypic._cli._cli_slurm_lifecycle import CancellationResult
from phenotypic.sdk_.slurm._dispatcher import (
    SlurmDependencyKind,
    _infer_output_dir,
    generate_dispatcher_chain,
    generate_dispatcher_script,
    submit_drip_feed_start,
)
from phenotypic.sdk_ import slurm_scripts_dir
from phenotypic.sdk_.slurm import (
    SLURM_PYTHONPATH_BOOTSTRAP_BASH,
    SLURM_PYTHONPATH_ENV_VAR,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="SLURM not available on Windows"
)


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


def test_infer_output_dir_from_nested_dataset_script(tmp_path: Path) -> None:
    """Dataset nesting below the canonical scripts root retains one ledger."""
    script = slurm_scripts_dir(tmp_path) / "dataset-a" / "array_job.sh"
    script.parent.mkdir(parents=True)
    script.touch()

    assert _infer_output_dir(script) == tmp_path.resolve()


def test_infer_output_dir_rejects_unscoped_slurm_scripts_name(
    tmp_path: Path,
) -> None:
    """A similarly named directory outside ``.phenotypic`` is not trusted."""
    script = tmp_path / "other" / "slurm_scripts" / "array_job.sh"
    script.parent.mkdir(parents=True)
    script.touch()

    assert _infer_output_dir(script) == script.parent.resolve()


def test_single_chunk_process_finalizer_depends_on_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process publisher cannot run concurrently with its image array."""
    scripts = slurm_scripts_dir(tmp_path)
    scripts.mkdir(parents=True)
    chunk = scripts / "chunk0.sh"
    finalizer = scripts / "process_finalizer.sh"
    chunk.touch()
    finalizer.touch()
    calls: list[dict[str, object]] = []

    def fake_submit(
        _output_dir: Path,
        **kwargs: object,
    ) -> str:
        calls.append(kwargs)
        return str(700 + len(calls))

    monkeypatch.setattr(lifecycle, "submit_with_lifecycle", fake_submit)

    job_ids, warning = submit_drip_feed_start(
        [chunk],
        [],
        finalizer_script=finalizer,
    )

    assert warning is None
    assert job_ids == ["701", "702"]
    assert calls[0]["role"] == "chunk"
    assert calls[1]["role"] == "finalizer"
    assert calls[1]["dependencies"] == ("701",)
    assert calls[1]["dependency_kind"] == "afterany"


def test_initial_continuation_accepts_afterok(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first dispatcher can be gated on successful chunk completion."""
    scripts = slurm_scripts_dir(tmp_path)
    scripts.mkdir(parents=True)
    chunk = scripts / "chunk0.sh"
    dispatcher = scripts / "dispatch_1.sh"
    chunk.touch()
    dispatcher.touch()
    calls: list[dict[str, object]] = []

    def fake_submit(_output_dir: Path, **kwargs: object) -> str:
        calls.append(kwargs)
        return str(700 + len(calls))

    monkeypatch.setattr(lifecycle, "submit_with_lifecycle", fake_submit)

    submit_drip_feed_start(
        [chunk],
        [dispatcher],
        continuation_dependency_kind="afterok",
    )

    assert calls[1]["dependency_kind"] == "afterok"


def test_initial_continuation_rejects_invalid_kind_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An invalid initial edge cannot submit chunk zero or create state."""
    scripts = slurm_scripts_dir(tmp_path)
    scripts.mkdir(parents=True)
    chunk = scripts / "chunk0.sh"
    dispatcher = scripts / "dispatch_1.sh"
    chunk.touch()
    dispatcher.touch()
    calls: list[dict[str, object]] = []

    def fake_submit(_output_dir: Path, **kwargs: object) -> str:
        calls.append(kwargs)
        return "701"

    monkeypatch.setattr(lifecycle, "submit_with_lifecycle", fake_submit)

    with pytest.raises(ValueError, match="dependency_kind"):
        submit_drip_feed_start(
            [chunk],
            [dispatcher],
            continuation_dependency_kind=cast(
                SlurmDependencyKind, "afterinvalid"
            ),
        )

    assert calls == []
    assert not lifecycle.lifecycle_state_path(tmp_path).exists()


def test_last_dispatcher_carries_process_finalizer(
    tmp_path: Path,
    slurm_args: dict[str, object],
) -> None:
    """Only the dispatcher for the final chunk can submit the publisher."""
    scripts = slurm_scripts_dir(tmp_path)
    scripts.mkdir(parents=True)
    chunks = [scripts / f"chunk{i}.sh" for i in range(3)]
    for chunk in chunks:
        chunk.touch()
    finalizer = scripts / "process_finalizer.sh"
    finalizer.touch()

    dispatchers = generate_dispatcher_chain(
        chunk_scripts=chunks,
        output_dir=tmp_path,
        slurm_args=slurm_args,
        log_dir=tmp_path / "logs",
        finalizer_script=finalizer,
    )

    assert "--finalizer-script" not in dispatchers[0].read_text()
    last = dispatchers[-1].read_text()
    assert f"--finalizer-script {finalizer}" in last
    assert "--dispatcher-script" not in last


class TestGenerateDispatcherScript:
    def test_invalid_dependency_kind_writes_no_dispatcher_or_state(
        self, tmp_path: Path, slurm_args: dict[str, object]
    ) -> None:
        """Invalid input fails before script and lifecycle state creation."""
        output_dir = tmp_path / "output"
        output = slurm_scripts_dir(output_dir) / "dispatch_1.sh"

        with pytest.raises(ValueError, match="dependency_kind"):
            generate_dispatcher_script(
                next_chunk_script=tmp_path / "chunk1.sh",
                next_dispatcher_script=None,
                output_path=output,
                slurm_args=slurm_args,
                log_dir=tmp_path / "logs",
                output_dir=output_dir,
                dependency_kind=cast(
                    SlurmDependencyKind, "afterinvalid"
                ),
            )

        assert not output.exists()
        assert not lifecycle.lifecycle_state_path(output_dir).exists()

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
        assert SLURM_PYTHONPATH_BOOTSTRAP_BASH in content
        assert SLURM_PYTHONPATH_ENV_VAR in content
        assert content.index(SLURM_PYTHONPATH_BOOTSTRAP_BASH) < content.index(
            "phenotypic._cli._cli_slurm_lifecycle"
        )

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

    @pytest.mark.skipif(
        sys.platform == "win32", reason="chmod not effective on Windows"
    )
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

    def test_dispatcher_chain_wiring(
        self, chunk_scripts, tmp_path, slurm_args
    ):
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

    @pytest.mark.skipif(
        sys.platform == "win32", reason="chmod not effective on Windows"
    )
    def test_all_dispatchers_are_executable(
        self, chunk_scripts, tmp_path, slurm_args
    ):
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

    def test_mixed_dependency_kinds_follow_continuation_edges(
        self, tmp_path, slurm_args
    ):
        """Generated dispatchers carry each later edge's dependency kind."""
        scripts_dir = slurm_scripts_dir(tmp_path)
        scripts_dir.mkdir(parents=True)
        chunks = [scripts_dir / f"chunk{i}.sh" for i in range(3)]
        for chunk in chunks:
            chunk.touch()
        finalizer = scripts_dir / "finalizer.sh"
        finalizer.touch()

        dispatchers = generate_dispatcher_chain(
            chunk_scripts=chunks,
            output_dir=tmp_path,
            slurm_args=slurm_args,
            log_dir=tmp_path / "logs" / "slurm",
            finalizer_script=finalizer,
            continuation_dependency_kinds=(
                "afterany",
                "afterok",
                "afterany",
            ),
        )

        assert "--dependency-kind afterok" in dispatchers[0].read_text()
        assert "--dependency-kind afterany" in dispatchers[1].read_text()

    def test_dependency_kind_count_must_match_edges(
        self, chunk_scripts, tmp_path, slurm_args
    ):
        """A misaligned edge sequence fails before scripts are generated."""
        with pytest.raises(ValueError, match="exactly 3 entries"):
            generate_dispatcher_chain(
                chunk_scripts=chunk_scripts,
                output_dir=tmp_path,
                slurm_args=slurm_args,
                log_dir=tmp_path / "logs" / "slurm",
                continuation_dependency_kinds=("afterok",),
            )

    def test_invalid_dependency_kind_writes_no_dispatchers(
        self, chunk_scripts, tmp_path, slurm_args
    ) -> None:
        """Invalid aligned edge values fail before chain side effects."""
        log_dir = tmp_path / "logs" / "slurm"

        with pytest.raises(ValueError, match="dependency_kind"):
            generate_dispatcher_chain(
                chunk_scripts=chunk_scripts,
                output_dir=tmp_path,
                slurm_args=slurm_args,
                log_dir=log_dir,
                continuation_dependency_kinds=(
                    "afterany",
                    cast(SlurmDependencyKind, "afterinvalid"),
                    "afterok",
                ),
            )

        assert not log_dir.exists()
        assert list(slurm_scripts_dir(tmp_path).glob("dispatch_*.sh")) == []
        assert not lifecycle.lifecycle_state_path(tmp_path).exists()


def test_initial_dispatcher_failure_fences_and_fails_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts_dir = slurm_scripts_dir(tmp_path)
    scripts_dir.mkdir(parents=True)
    chunk = scripts_dir / "chunk0.sh"
    dispatcher = scripts_dir / "dispatch_1.sh"
    chunk.touch()
    dispatcher.touch()
    submissions = 0
    cancelled: list[tuple[Path, str]] = []

    def fake_submit(*args, **kwargs):
        nonlocal submissions
        submissions += 1
        if submissions == 1:
            return "701"
        raise RuntimeError("dispatcher unavailable")

    def fake_cancel(output_dir, generation, **kwargs):
        cancelled.append((output_dir, generation))
        return CancellationResult(("701",), (), True)

    monkeypatch.setattr(lifecycle, "submit_with_lifecycle", fake_submit)
    monkeypatch.setattr(lifecycle, "cancel_generation", fake_cancel)

    with pytest.raises(
        RuntimeError, match="Initial dispatcher submission failed"
    ):
        submit_drip_feed_start([chunk], [dispatcher])

    assert cancelled == [
        (tmp_path, lifecycle.load_slurm_lifecycle(tmp_path)["generation"])
    ]


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
