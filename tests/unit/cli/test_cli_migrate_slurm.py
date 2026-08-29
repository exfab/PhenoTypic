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
            store_path=zarr_store_path(output_dir, "ds", f"image_{index}").resolve(),
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
    monkeypatch.setattr(subject, "discover_migration_tasks", lambda _: _tasks(output_dir, 5))
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
    assert all("#SBATCH --array=0-" in path.read_text() for path in plan.flat_scripts)
    assert "#SBATCH --array=0-0" in plan.finalizer_script.read_text()


def test_plan_uses_tighter_limit_after_exact_two_slot_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A running dispatcher leaves room for its cohort and successor."""
    from phenotypic._cli import _cli_migrate_slurm as subject

    output_dir = tmp_path / "run"
    monkeypatch.setattr(subject, "discover_migration_tasks", lambda _: _tasks(output_dir, 8))
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 50)
    monkeypatch.setattr(subject, "get_slurm_max_submit_jobs", lambda: 5)

    plan = subject.generate_migration_slurm_plan(
        output_dir,
        slurm_args={"slurm_partition": "short"},
        generation="generation-1",
    )

    image_scripts = [path for path in plan.flat_scripts if "image_chunk" in path.stem]
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

    monkeypatch.setattr(subject, "discover_migration_tasks", lambda _: _tasks(tmp_path / "run", 1))
    monkeypatch.setattr(subject, "get_slurm_array_limit", lambda: 100)
    monkeypatch.setattr(subject, "get_slurm_max_submit_jobs", lambda: max_submit)

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
    monkeypatch.setattr(subject, "discover_migration_tasks", lambda _: _tasks(output_dir, 2))
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
        return SimpleNamespace(job_ids=["11", "12"], flat_scripts=list(plan.flat_scripts))

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
    monkeypatch.setattr(subject, "discover_migration_tasks", lambda _: _tasks(output_dir, 2))
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
    assert all(path.is_relative_to(plan.control_root) for path in plan.flat_scripts)
    assert plan.finalizer_script.is_relative_to(plan.control_root)
    assert not output_dir.exists()
    config = json.loads((plan.control_root / "migration_config.json").read_text())
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
    monkeypatch.setattr(slurm, "discover_migration_tasks", lambda _: _tasks(output_dir, 2))
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
    assert CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "metadata"],
    ).exit_code == 0

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
    monkeypatch.setattr(worker, "publish_migration_task_status", lambda *_a, **_k: None)
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
    monkeypatch.setattr(worker, "invalidate_migration_terminal_authority", lambda *_a, **_k: None)
    assert CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "metadata"],
    ).exit_code == 0

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
    monkeypatch.setattr(worker, "publish_migration_task_status", lambda *_a, **_k: None)
    for index in range(2):
        assert CliRunner().invoke(
            worker.migration_worker_cli,
            ["--config", str(config_path), "image", "--index", str(index)],
        ).exit_code == 0

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
        output_dir / ".phenotypic", plan.generation
    )
    assert terminal_path.is_file(), repr(result.exception)
    terminal = json.loads(
        terminal_path.read_text()
    )
    assert terminal["failure_category"] == "image_seal"
    assert "image seal is missing" in terminal["reason"]
    lifecycle = load_slurm_lifecycle(output_dir)
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
    assert CliRunner().invoke(
        worker.migration_worker_cli,
        ["--config", str(config_path), "metadata"],
    ).exit_code == 0
    corrupt_path = worker.migration_worker_status_path(
        plan.control_root, plan.generation, "image", 0
    )
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": plan.generation,
                "manifest_digest": json.loads(config_path.read_text())["inventory_digest"],
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
    assert "invalid upstream evidence" in terminal["reason"]
    lifecycle = load_slurm_lifecycle(plan.control_root)
    assert lifecycle is not None
    assert lifecycle["active"] is False
