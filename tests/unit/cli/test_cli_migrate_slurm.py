"""Behavior tests for dispatcher-fed ``--mode migrate`` SLURM work."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner


def _tasks(output_dir: Path, count: int):
    """Return explicit canonical migration tasks without touching image data."""
    from phenotypic._cli._cli_migrate_manifest import MigrationImageTask
    from phenotypic.sdk_ import (
        dataset_measurements_dir,
        dataset_overlays_dir,
        image_completion_marker_path,
        zarr_store_path,
    )

    return tuple(
        MigrationImageTask(
            index=index,
            dataset="ds",
            stem=f"image_{index}",
            hdf_path=(
                output_dir / "results" / "ds" / "hdf" / f"image_{index}.h5"
            ).resolve(),
            store_path=zarr_store_path(
                output_dir, "ds", f"image_{index}"
            ).resolve(),
            measurement_path=(
                dataset_measurements_dir(output_dir, "ds")
                / f"image_{index}.parquet"
            ).resolve(),
            overlay_path=(
                dataset_overlays_dir(output_dir, "ds") / f"image_{index}.png"
            ).resolve(),
            marker_path=image_completion_marker_path(
                output_dir, "ds", f"image_{index}"
            ).resolve(),
        )
        for index in range(count)
    )


def _script_entries(path: Path) -> list[int]:
    """Read the literal scheduler-index mapping from one generated script."""
    text = path.read_text(encoding="utf-8")
    block = text.split("TASK_INDICES=(\n", 1)[1].split("\n)", 1)[0]
    return [int(line.strip()) for line in block.splitlines()]


def test_plan_is_one_flat_afterany_chain_with_zero_based_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stage order and chunk windows cannot strand the terminal finalizer."""
    from phenotypic._cli import _cli_migrate_slurm as subject

    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        subject, "discover_migration_tasks", lambda _: _tasks(output_dir, 5)
    )
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 2)
    monkeypatch.setattr(subject, "get_slurm_max_submit_jobs", lambda: 20)

    plan = subject.generate_migration_slurm_plan(
        output_dir,
        slurm_args={"slurm_partition": "short"},
        delete_sources=True,
        generation="generation-1",
    )

    assert [path.stem for path in plan.flat_scripts] == [
        "metadata",
        "image_chunk0",
        "image_chunk1",
        "image_chunk2",
        "image_seal",
        "reclaim_chunk0",
        "reclaim_chunk1",
        "reclaim_chunk2",
        "reclaim_seal",
    ]
    assert [_script_entries(path) for path in plan.flat_scripts] == [
        [0],
        [0, 1],
        [2, 3],
        [4],
        [0],
        [0, 1],
        [2, 3],
        [4],
        [0],
    ]
    assert plan.finalizer_script.stem == "finalize"
    assert plan.task_count == 5
    assert all(
        "#SBATCH --array=0-" in path.read_text() for path in plan.flat_scripts
    )
    assert "#SBATCH --array=0-0" in plan.finalizer_script.read_text()


def test_plan_uses_tighter_limit_after_exact_two_slot_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A running dispatcher leaves room for its cohort and successor."""
    from phenotypic._cli import _cli_migrate_slurm as subject

    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        subject, "discover_migration_tasks", lambda _: _tasks(output_dir, 8)
    )
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 50)
    monkeypatch.setattr(subject, "get_slurm_max_submit_jobs", lambda: 5)

    plan = subject.generate_migration_slurm_plan(
        output_dir,
        slurm_args={"slurm_partition": "short"},
        generation="generation-1",
    )

    image_scripts = [
        path for path in plan.flat_scripts if "image_chunk" in path.stem
    ]
    assert [_script_entries(path) for path in image_scripts] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
    ]


@pytest.mark.parametrize("max_submit", [1, 2])
def test_plan_refuses_capacity_below_two_slot_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_submit: int
) -> None:
    """No array can launch when the dispatcher reservation consumes capacity."""
    from phenotypic._cli import _cli_migrate_slurm as subject

    monkeypatch.setattr(
        subject,
        "discover_migration_tasks",
        lambda _: _tasks(tmp_path / "run", 1),
    )
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(
        subject, "get_slurm_max_submit_jobs", lambda: max_submit
    )

    with pytest.raises(ValueError, match="at least 3"):
        subject.generate_migration_slurm_plan(
            tmp_path / "run",
            slurm_args={"slurm_partition": "short"},
            generation="generation-1",
        )


def test_submitter_delegates_the_whole_chain_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Migration cannot regress to an eager per-script sbatch loop."""
    from phenotypic._cli import _cli_migrate_slurm as subject

    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        subject, "discover_migration_tasks", lambda _: _tasks(output_dir, 2)
    )
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(subject, "get_slurm_max_submit_jobs", lambda: 100)
    plan = subject.generate_migration_slurm_plan(
        output_dir,
        slurm_args={"slurm_partition": "short"},
        generation="generation-1",
    )
    calls: list[dict[str, object]] = []

    def _submit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            job_ids=["11", "12"], flat_scripts=list(plan.flat_scripts)
        )

    monkeypatch.setattr(subject, "submit_slurm_script_chain", _submit)
    result = subject.submit_migration_slurm_plan(
        plan,
        slurm_args={"slurm_partition": "short"},
        console=SimpleNamespace(print=lambda *_: None),
    )

    assert result.job_ids == ["11", "12"]
    assert len(calls) == 1
    assert calls[0]["flat_chunk_scripts"] == plan.flat_scripts
    assert calls[0]["output_dir"] == output_dir.resolve()
    assert calls[0]["control_output_dir"] == plan.control_root
    assert calls[0]["finalizer_script"] == plan.finalizer_script
    assert calls[0]["continuation_dependency_kinds"] == (
        "afterany",
        "afterany",
        "afterany",
    )


def test_dry_run_places_every_control_path_outside_scientific_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scheduler bookkeeping cannot make a dry-run mutate the run tree."""
    from phenotypic._cli import _cli_migrate_slurm as subject

    output_dir = tmp_path / "run"
    cache_base = tmp_path / "cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache_base))
    monkeypatch.setattr(
        subject, "discover_migration_tasks", lambda _: _tasks(output_dir, 2)
    )
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(subject, "get_slurm_max_submit_jobs", lambda: 100)

    plan = subject.generate_migration_slurm_plan(
        output_dir,
        slurm_args={"slurm_partition": "short"},
        dry_run=True,
        generation="generation-1",
    )

    assert plan.control_root.is_relative_to(cache_base)
    assert not plan.control_root.is_relative_to(output_dir)
    assert plan.manifest_path.is_relative_to(plan.control_root)
    assert all(
        path.is_relative_to(plan.control_root) for path in plan.flat_scripts
    )
    assert plan.finalizer_script.is_relative_to(plan.control_root)
    assert not output_dir.exists()
    config = json.loads(
        (plan.control_root / "migration_config.json").read_text()
    )
    assert Path(config["output_dir"]) == output_dir.resolve()
    assert Path(config["control_root"]) == plan.control_root


def _worker_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    dry_run: bool = False,
):
    """Create one real manifest/config and activate its lifecycle fence."""
    from phenotypic._cli import _cli_migrate_slurm as slurm
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / "run"
    if dry_run:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(
        slurm, "discover_migration_tasks", lambda _: _tasks(output_dir, 2)
    )
    monkeypatch.setattr(slurm, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(slurm, "get_slurm_max_submit_jobs", lambda: 100)
    plan = slurm.generate_migration_slurm_plan(
        output_dir,
        slurm_args={"slurm_partition": "short"},
        dry_run=dry_run,
        generation="generation-1",
    )
    lifecycle_root = plan.control_root if dry_run else output_dir
    initialize_slurm_lifecycle(
        lifecycle_root, generation=plan.generation, mode="migrate"
    )
    return output_dir, plan


def test_failed_metadata_blocks_images_with_typed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upstream failure records blocked work without touching image science."""
    from phenotypic._cli import _cli_migrate_worker as worker

    output_dir, plan = _worker_plan(tmp_path, monkeypatch)
    config_path = plan.control_root / "migration_config.json"

    def _metadata_failure(*_args, **_kwargs):
        raise RuntimeError("metadata exploded")

    monkeypatch.setattr(worker, "run_metadata_pass", _metadata_failure)
    metadata = CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "metadata"],
    )
    assert metadata.exit_code == 1

    called = False

    def _must_not_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("blocked image entered scientific migration")

    monkeypatch.setattr(worker, "migrate_image_task", _must_not_run)
    image = CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "image", "--index", "1"],
    )

    assert image.exit_code == 0
    assert called is False
    status = json.loads(
        worker.migration_worker_status_path(
            plan.control_root, plan.generation, "image", 1
        ).read_text()
    )
    assert status["status"] == "blocked"
    assert status["failure_category"] == "metadata"
    assert "metadata exploded" in status["reason"]
    assert not (output_dir / "results").exists()


def test_image_failure_does_not_prevent_later_index_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One array item failure cannot suppress an independent later item."""
    from phenotypic._cli import _cli_migrate_worker as worker
    from phenotypic._cli._cli_migrate import MetadataPassResult
    from phenotypic.sdk_._metadata_migration import MetadataMigrationAuthority

    _output_dir, plan = _worker_plan(tmp_path, monkeypatch)
    config_path = plan.control_root / "migration_config.json"
    authority = MetadataMigrationAuthority(
        status_path=plan.control_root / "authority-status.json",
        terminal_receipt_path=plan.control_root / "receipt.json",
        terminal_receipt_digest="sha256:" + "a" * 64,
        plan_fingerprint="plan",
        source_fingerprint="source",
        resulting_fingerprint="result",
        compatible_noop=True,
    )
    monkeypatch.setattr(
        worker,
        "run_metadata_pass",
        lambda *_args, **_kwargs: MetadataPassResult(0, (), authority),
    )
    assert (
        CliRunner()
        .invoke(
            worker.migration_worker_cli,
            ["--config", str(config_path), "metadata"],
        )
        .exit_code
        == 0
    )

    from phenotypic._cli._cli_migrate_image import MigrationImageResult

    def _migrate(_output, task, **_kwargs):
        if task.index == 0:
            raise RuntimeError("bad image")
        return MigrationImageResult(
            index=task.index,
            dataset=task.dataset,
            stem=task.stem,
            work_id="work",
            converted=True,
            table_installed=False,
            overlay_rendered=True,
            marker_digest="b" * 64,
            skipped=False,
        )

    monkeypatch.setattr(worker, "migrate_image_task", _migrate)
    monkeypatch.setattr(
        worker, "publish_migration_task_status", lambda *_a, **_k: None
    )
    first = CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "image", "--index", "0"],
    )
    second = CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "image", "--index", "1"],
    )

    assert first.exit_code == 1
    assert second.exit_code == 0
    first_status = json.loads(
        worker.migration_worker_status_path(
            plan.control_root, plan.generation, "image", 0
        ).read_text()
    )
    second_status = json.loads(
        worker.migration_worker_status_path(
            plan.control_root, plan.generation, "image", 1
        ).read_text()
    )
    assert first_status["status"] == "failed"
    assert second_status["status"] == "complete"


def test_dry_finalizer_terminalizes_and_closes_failed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The afterany finalizer closes even an attempt with missing prerequisites."""
    from phenotypic._cli import _cli_migrate_worker as worker
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle

    _output_dir, plan = _worker_plan(tmp_path, monkeypatch, dry_run=True)
    result = CliRunner().invoke(
        worker.migration_worker_cli,
        [
            "--config",
            str(plan.control_root / "migration_config.json"),
            "finalize",
        ],
    )

    assert result.exit_code == 1
    terminal = json.loads(
        worker.migration_worker_status_path(
            plan.control_root, plan.generation, "terminal"
        ).read_text()
    )
    assert terminal["status"] == "failed"
    assert terminal["failure_category"] == "metadata"
    lifecycle = load_slurm_lifecycle(plan.control_root)
    assert lifecycle is not None
    assert lifecycle["active"] is False


def test_nondry_finalizer_validates_attempt_scoped_manifest_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Terminal validation uses the attempt control root, not a legacy default."""
    from phenotypic._cli import _cli_migrate_worker as worker
    from phenotypic._cli._cli_migrate import (
        MetadataPassResult,
        migration_terminal_status_path,
    )
    from phenotypic._cli._cli_migrate_image import MigrationImageResult
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle
    from phenotypic.sdk_._metadata_migration import MetadataMigrationAuthority

    output_dir, plan = _worker_plan(tmp_path, monkeypatch)
    config_path = plan.control_root / "migration_config.json"
    authority = MetadataMigrationAuthority(
        status_path=plan.control_root / "authority-status.json",
        terminal_receipt_path=plan.control_root / "receipt.json",
        terminal_receipt_digest="sha256:" + "c" * 64,
        plan_fingerprint="plan",
        source_fingerprint="source",
        resulting_fingerprint="result",
        compatible_noop=True,
    )
    monkeypatch.setattr(
        worker,
        "run_metadata_pass",
        lambda *_a, **_k: MetadataPassResult(0, (), authority),
    )
    monkeypatch.setattr(
        worker,
        "invalidate_migration_terminal_authority",
        lambda *_a, **_k: None,
    )
    assert (
        CliRunner()
        .invoke(
            worker.migration_worker_cli,
            ["--config", str(config_path), "metadata"],
        )
        .exit_code
        == 0
    )

    def _migrate(_output, task, **_kwargs):
        return MigrationImageResult(
            index=task.index,
            dataset=task.dataset,
            stem=task.stem,
            work_id="work",
            converted=False,
            table_installed=False,
            overlay_rendered=False,
            marker_digest="d" * 64,
            skipped=True,
        )

    monkeypatch.setattr(worker, "migrate_image_task", _migrate)
    monkeypatch.setattr(
        worker, "publish_migration_task_status", lambda *_a, **_k: None
    )
    for index in range(2):
        assert (
            CliRunner()
            .invoke(
                worker.migration_worker_cli,
                ["--config", str(config_path), "image", "--index", str(index)],
            )
            .exit_code
            == 0
        )

    monkeypatch.setattr(
        "phenotypic._cli._cli_migrate.metadata_migration_authority",
        lambda *_a, **_k: authority,
    )
    result = CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "finalize"],
    )

    assert result.exit_code == 1
    terminal_path = migration_terminal_status_path(
        plan.control_root, plan.generation
    )
    assert terminal_path.is_file(), repr(result.exception)
    terminal = json.loads(terminal_path.read_text())
    assert terminal["failure_category"] == "image_seal"
    assert "image seal is missing" in terminal["reason"]
    lifecycle = load_slurm_lifecycle(output_dir)
    assert lifecycle is not None
    assert lifecycle["active"] is False


@pytest.mark.parametrize("generation", ["../escape", "a/b", "a\\b", ".", ".."])
def test_generation_is_rejected_before_control_tree_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generation: str
) -> None:
    """A scheduler generation is one component, never a path supplied by input."""
    from phenotypic._cli import _cli_migrate_slurm as subject

    output_dir = tmp_path / "run"
    monkeypatch.setattr(subject, "discover_migration_tasks", lambda _: ())
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(subject, "get_slurm_max_submit_jobs", lambda: 100)

    with pytest.raises(ValueError, match="safe path component"):
        subject.generate_migration_slurm_plan(
            output_dir, slurm_args={}, generation=generation
        )
    assert not output_dir.exists()


def test_worker_status_path_rejects_generation_traversal(
    tmp_path: Path,
) -> None:
    """Worker status publication cannot escape its caller-bound control root."""
    from phenotypic._cli._cli_migrate_worker import (
        migration_worker_status_path,
    )

    with pytest.raises(ValueError, match="safe path component"):
        migration_worker_status_path(
            tmp_path / "control", "../escape", "metadata"
        )


def test_dry_run_rejects_cache_nested_in_scientific_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hostile cache environment cannot turn scheduler state into science writes."""
    from phenotypic._cli import _cli_migrate_slurm as subject

    output_dir = tmp_path / "run"
    monkeypatch.setenv("XDG_CACHE_HOME", str(output_dir / "cache"))
    monkeypatch.setattr(subject, "discover_migration_tasks", lambda _: ())
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(subject, "get_slurm_max_submit_jobs", lambda: 100)

    with pytest.raises(ValueError, match="outside scientific output"):
        subject.generate_migration_slurm_plan(
            output_dir, slurm_args={}, dry_run=True, generation="safe"
        )
    assert not output_dir.exists()


def test_image_rejects_corrupt_complete_metadata_before_task_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete label without typed authority cannot authorize image mutation."""
    from phenotypic._cli import _cli_migrate_worker as worker

    _output_dir, plan = _worker_plan(tmp_path, monkeypatch)
    config_path = plan.control_root / "migration_config.json"
    config = worker._load_worker_config(config_path)
    worker._publish_worker_status(
        config,
        "metadata",
        status="complete",
        extra={"headers_migrated": 0, "authority": {"compatible_noop": "yes"}},
    )
    monkeypatch.setattr(
        worker,
        "read_migration_task",
        lambda *_a, **_k: pytest.fail(
            "task loaded before authority validation"
        ),
    )
    monkeypatch.setattr(
        worker,
        "migrate_image_task",
        lambda *_a, **_k: pytest.fail("scientific mutation was reached"),
    )

    result = CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "image", "--index", "0"],
    )

    assert result.exit_code == 0
    status = json.loads(
        worker.migration_worker_status_path(
            plan.control_root, plan.generation, "image", 0
        ).read_text()
    )
    assert status["status"] == "blocked"
    assert "authority" in status["reason"]


@pytest.mark.parametrize(
    ("command", "stage"),
    [("seal", "seal"), ("reclaim-seal", "reclaim-seal")],
)
def test_barrier_failure_publishes_typed_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    stage: str,
) -> None:
    """Barrier primitives convert exceptions into generation-bound evidence."""
    from phenotypic._cli import _cli_migrate_worker as worker

    _output_dir, plan = _worker_plan(tmp_path, monkeypatch)
    primitive = (
        "seal_migration_image_stage"
        if command == "seal"
        else "seal_migration_reclaim_stage"
    )
    monkeypatch.setattr(
        worker,
        primitive,
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = CliRunner().invoke(
        worker.migration_worker_cli,
        [
            "--config",
            str(plan.control_root / "migration_config.json"),
            command,
        ],
    )

    assert result.exit_code == 1
    status = json.loads(
        worker.migration_worker_status_path(
            plan.control_root, plan.generation, stage
        ).read_text()
    )
    assert status["status"] == "failed"
    assert "RuntimeError: boom" in status["reason"]


def test_terminal_publication_failure_leaves_generation_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lifecycle closure never outruns durable terminal evidence."""
    from phenotypic._cli import _cli_migrate_worker as worker
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle

    output_dir, plan = _worker_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(
        worker,
        "publish_migration_terminal_status",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        worker,
        "_image_report",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("corrupt status")),
    )

    result = CliRunner().invoke(
        worker.migration_worker_cli,
        [
            "--config",
            str(plan.control_root / "migration_config.json"),
            "finalize",
        ],
    )

    assert result.exit_code != 0
    lifecycle = load_slurm_lifecycle(output_dir)
    assert lifecycle is not None
    assert lifecycle["active"] is True


def test_generic_dispatcher_separates_control_and_lifecycle_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dispatcher artifacts stay in control while lifecycle submissions stay scientific."""
    from phenotypic._cli._cli_slurm_submission import submit_slurm_script_chain
    from phenotypic._cli import _cli_slurm_lifecycle as lifecycle

    scientific = (tmp_path / "science").resolve()
    control = (tmp_path / "control").resolve()
    chunks = [tmp_path / "chunk0.sh", tmp_path / "chunk1.sh"]
    finalizer = tmp_path / "finalize.sh"
    calls: list[tuple[Path, str, Path]] = []

    def _submit(root, *, generation, token, script_path, **_kwargs):
        calls.append((Path(root), token, Path(script_path)))
        return str(len(calls) + 40)

    monkeypatch.setattr(lifecycle, "submit_with_lifecycle", _submit)
    result = submit_slurm_script_chain(
        flat_chunk_scripts=chunks,
        output_dir=scientific,
        control_output_dir=control,
        slurm_args={"slurm_partition": "short"},
        console=SimpleNamespace(print=lambda *_a: None),
        finalizer_script=finalizer,
        continuation_dependency_kinds=("afterany", "afterany"),
        generation="generation-1",
    )

    assert calls[0][:2] == (scientific, "chunk-0")
    assert calls[1][:2] == (scientific, "dispatcher-1")
    assert all(
        path.is_relative_to(control) for path in result.dispatcher_scripts
    )
    dispatcher = result.dispatcher_scripts[0].read_text()
    assert f"--output {scientific}" in dispatcher
    assert f"#SBATCH --output={control}" in dispatcher
    assert str(finalizer) in dispatcher


def test_ordinary_dispatcher_keeps_legacy_single_root_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optional control split does not change ordinary chain semantics."""
    from phenotypic._cli import _cli_slurm_lifecycle as lifecycle
    from phenotypic._cli._cli_slurm_submission import submit_slurm_script_chain

    output = (tmp_path / "ordinary").resolve()
    calls: list[Path] = []

    def _submit(root, **_kwargs):
        calls.append(Path(root))
        return str(len(calls) + 70)

    monkeypatch.setattr(lifecycle, "submit_with_lifecycle", _submit)
    result = submit_slurm_script_chain(
        flat_chunk_scripts=[tmp_path / "a.sh", tmp_path / "b.sh"],
        output_dir=output,
        slurm_args={},
        console=SimpleNamespace(print=lambda *_a: None),
        finalizer_script=tmp_path / "final.sh",
        generation="ordinary-1",
    )

    assert calls == [output, output]
    assert all(
        path.is_relative_to(output) for path in result.dispatcher_scripts
    )
    assert (
        str(tmp_path / "final.sh") in result.dispatcher_scripts[0].read_text()
    )


def test_reclaim_task_load_failure_is_typed_and_later_work_remains_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt manifest index fails only its reclaim array entry."""
    from phenotypic._cli import _cli_migrate_worker as worker

    _output_dir, plan = _worker_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(
        worker,
        "read_migration_task",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad task")),
    )

    result = CliRunner().invoke(
        worker.migration_worker_cli,
        [
            "--config",
            str(plan.control_root / "migration_config.json"),
            "reclaim",
            "--index",
            "0",
        ],
    )

    assert result.exit_code == 1
    status = json.loads(
        worker.migration_worker_status_path(
            plan.control_root, plan.generation, "reclaim", 0
        ).read_text()
    )
    assert status["status"] == "failed"
    assert "ValueError: bad task" in status["reason"]


def test_second_generation_reclaim_accepts_authoritative_already_absent_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial prior reclaim retries strong I/O only for the remaining source."""
    from phenotypic._cli import _cli_migrate_image as subject
    from phenotypic._cli._cli_migrate_manifest import MigrationImageTask
    from phenotypic.sdk_ import _hdf_to_zarr

    output = tmp_path / "run"
    store_only = MigrationImageTask(
        index=0,
        dataset="ds",
        stem="done",
        hdf_path=None,
        store_path=output / "results" / "ds" / "images" / "done.zarr",
        measurement_path=None,
        overlay_path=output / "deliverables" / "overlays" / "done.png",
        marker_path=output / ".phenotypic" / "done.json",
    )
    remaining_hdf = output / "results" / "ds" / "hdf" / "remaining.h5"
    remaining_hdf.parent.mkdir(parents=True)
    remaining_hdf.write_bytes(b"legacy")
    remaining = MigrationImageTask(
        index=1,
        dataset="ds",
        stem="remaining",
        hdf_path=remaining_hdf,
        store_path=output / "results" / "ds" / "images" / "remaining.zarr",
        measurement_path=None,
        overlay_path=output / "deliverables" / "overlays" / "remaining.png",
        marker_path=output / ".phenotypic" / "remaining.json",
    )
    strong_reads: list[Path] = []
    monkeypatch.setattr(subject, "_configured_work_id", lambda *_a: "work")
    monkeypatch.setattr(
        subject, "_current_marker_digest", lambda *_a: "d" * 64
    )
    monkeypatch.setattr(subject, "_marker_still_current", lambda *_a: True)
    monkeypatch.setattr(
        _hdf_to_zarr, "_marker_authority_permits_unlink", lambda *_a: True
    )

    def _faithful(source, _store):
        strong_reads.append(Path(source))
        return True

    monkeypatch.setattr(_hdf_to_zarr, "_conversion_is_faithful", _faithful)

    already_done = subject.reclaim_image_sources(
        output, store_only, metadata_csv=None
    )
    retried = subject.reclaim_image_sources(
        output, remaining, metadata_csv=None
    )

    assert already_done.reason is None
    assert already_done.deleted_paths == ()
    assert retried.reason is None
    assert retried.deleted_paths == (remaining_hdf,)
    assert strong_reads == [remaining_hdf]
    assert not remaining_hdf.exists()


def test_successful_nondry_finalizer_reads_canonical_seal_and_terminalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The successful afterany path consumes canonical authority and closes."""
    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic._cli import _cli_migrate_worker as worker
    from phenotypic._cli._cli_migrate_manifest import migration_image_seal_path
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle
    from phenotypic.sdk_._hdf_to_zarr import MigrationReport
    from phenotypic.sdk_._metadata_migration import MetadataMigrationAuthority

    output, plan = _worker_plan(tmp_path, monkeypatch)
    config_path = plan.control_root / "migration_config.json"
    authority = MetadataMigrationAuthority(
        status_path=plan.control_root / "authority.json",
        terminal_receipt_path=plan.control_root / "receipt.json",
        terminal_receipt_digest="sha256:" + "e" * 64,
        plan_fingerprint="plan",
        source_fingerprint="source",
        resulting_fingerprint="result",
        compatible_noop=True,
    )
    config = worker._load_worker_config(config_path)
    worker._publish_worker_status(
        config,
        "metadata",
        status="complete",
        extra={
            "headers_migrated": 0,
            "authority": worker._authority_payload(authority),
        },
    )
    seal_path = migration_image_seal_path(plan.control_root, plan.generation)
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": plan.generation,
                "manifest_digest": config.inventory_digest,
                "ordered_status_digest": "ordered",
                "metadata_terminal_digest": authority.terminal_receipt_digest,
                "clean": True,
                "failures": [],
            }
        )
    )
    monkeypatch.setattr(
        worker, "_image_report", lambda *_a: (MigrationReport(), ())
    )
    monkeypatch.setattr(
        migrate, "metadata_migration_authority", lambda *_a: authority
    )
    monkeypatch.setattr(
        migrate, "valid_migration_image_seal", lambda *_a, **_k: True
    )
    monkeypatch.setattr(
        migrate, "emit_canonical_metadata_view", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        migrate, "_publish_migration_aggregate", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        migrate, "publish_run_completion_evidence", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        migrate, "valid_run_completion", lambda *_a, **_k: object()
    )

    result = CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "finalize"],
    )

    assert result.exit_code == 0, repr(result.exception)
    terminal = json.loads(
        migrate.migration_terminal_status_path(
            plan.control_root, plan.generation
        ).read_text()
    )
    assert terminal["status"] == "succeeded"
    lifecycle = load_slurm_lifecycle(output)
    assert lifecycle is not None
    assert lifecycle["active"] is False


def test_corrupt_metadata_terminalizes_before_lifecycle_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed metadata evidence becomes a durable failed terminal attempt."""
    from phenotypic._cli import _cli_migrate as migrate
    from phenotypic._cli import _cli_migrate_worker as worker
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle

    output, plan = _worker_plan(tmp_path, monkeypatch)
    config_path = plan.control_root / "migration_config.json"
    config = worker._load_worker_config(config_path)
    worker._publish_worker_status(
        config,
        "metadata",
        status="complete",
        extra={"headers_migrated": "zero", "authority": None},
    )
    monkeypatch.setattr(
        worker, "_image_report", lambda *_a: (migrate.MigrationReport(), ())
    )

    result = CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "finalize"],
    )

    assert result.exit_code == 1
    terminal_path = migrate.migration_terminal_status_path(
        plan.control_root, plan.generation
    )
    assert terminal_path.is_file()
    terminal = json.loads(terminal_path.read_text())
    assert terminal["status"] == "failed"
    assert terminal["failure_category"] == "metadata"
    lifecycle = load_slurm_lifecycle(output)
    assert lifecycle is not None
    assert lifecycle["active"] is False


def test_finalizer_closes_when_upstream_status_payload_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed cached evidence cannot strand the lifecycle generation active."""
    from phenotypic._cli import _cli_migrate_worker as worker
    from phenotypic._cli._cli_migrate import MetadataPassResult
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle

    _output_dir, plan = _worker_plan(tmp_path, monkeypatch, dry_run=True)
    config_path = plan.control_root / "migration_config.json"
    monkeypatch.setattr(
        worker,
        "run_metadata_pass",
        lambda *_a, **_k: MetadataPassResult(0, (), None),
    )
    assert (
        CliRunner()
        .invoke(
            worker.migration_worker_cli,
            ["--config", str(config_path), "metadata"],
        )
        .exit_code
        == 0
    )
    corrupt_path = worker.migration_worker_status_path(
        plan.control_root, plan.generation, "image", 0
    )
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": plan.generation,
                "manifest_digest": json.loads(config_path.read_text())[
                    "inventory_digest"
                ],
                "stage": "image",
                "index": 0,
                "status": "complete",
                "failure_category": None,
                "reason": None,
                "result": {},
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "finalize"],
    )

    assert result.exit_code == 1
    terminal = json.loads(
        worker.migration_worker_status_path(
            plan.control_root, plan.generation, "terminal"
        ).read_text()
    )
    assert terminal["status"] == "failed"
    assert terminal["reason"] == "image seal status is missing"
    lifecycle = load_slurm_lifecycle(plan.control_root)
    assert lifecycle is not None
    assert lifecycle["active"] is False
