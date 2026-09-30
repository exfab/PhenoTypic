"""The SLURM finalizer chain (``_cli_finalize_chain``), driven stage by stage.

Every stage runs its real code against a real tree -- real stores, real
records, a real lifecycle generation -- and only the scheduler is replaced:
``prepare``'s submissions are recorded rather than sent, and each later stage
is invoked the way its array task would invoke it. The fake-``sbatch``
integration test (``tests/integration/cli/test_finalize_chain_fake_slurm.py``)
covers the same chain through the CLI and a scheduler that honours
dependencies.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from phenotypic._cli import _cli_checkpoint_handler, _cli_finalize_chain
from phenotypic._cli._cli_checkpoint_handler import FinalizationIncomplete
from phenotypic._cli._cli_completion import (
    valid_aggregate_snapshot,
    valid_run_completion,
)
from phenotypic._cli._cli_finalize_chain import (
    MODE_ORDINARY,
    STAGE_MASTER,
    STAGE_OUTPUTS,
    STAGE_PREPARE,
    STAGE_PUBLISH,
    STAGE_QC,
    STAGE_SHARDS,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    load_chain_spec,
    read_stage_status,
    run_chain_task,
    stage_status_path,
    write_finalize_chain,
)
from phenotypic._cli._cli_finalize_fanout import (
    begin_aggregation_fanout,
    run_aggregation_shard,
)
from phenotypic._cli._cli_slurm_lifecycle import (
    initialize_slurm_lifecycle,
    load_slurm_lifecycle,
)
from phenotypic.sdk_ import (
    JobMetadataKey,
    atomic_write_json,
    deliverables_dir,
    job_metadata_path,
    master_measurements_parquet_path,
    measurements_parquet_path,
    metadata_csv_deliverable_path,
)

from .conftest import DATASET, _publish_successful_images

GENERATION = "gen-chain"
STEMS = ["a", "b", "c"]
SHARDS = 2
_SNAPSHOT = "Metadata_ImageName,Metadata_Strain\na.tiff,WT\nb.tiff,MUT\nc.tiff,WT\n"


# ---------------------------------------------------------------------------
# Fixture: a finished image phase, waiting for its finalizer chain
# ---------------------------------------------------------------------------


def _write_job_metadata(output_dir: Path) -> None:
    atomic_write_json(
        job_metadata_path(output_dir),
        {
            JobMetadataKey.START_TIME: "2026-09-30T00:00:00.000",
            JobMetadataKey.EXECUTION_MODE: "slurm",
            JobMetadataKey.DATASETS: {
                DATASET: {
                    "total": len(STEMS),
                    "images": [f"{stem}.tiff" for stem in STEMS],
                }
            },
            JobMetadataKey.CHUNK_SCRIPTS: [],
            JobMetadataKey.CHUNK_JOB_IDS: {},
            JobMetadataKey.INCLUDE_DATASET_COLUMN: True,
            JobMetadataKey.METADATA_CSV: str(
                metadata_csv_deliverable_path(output_dir)
            ),
            "slurm_metadata_version": 2,
            "slurm_generation": GENERATION,
        },
    )


@pytest.fixture
def chain_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An ordinary SLURM run whose image arrays are terminal and successful."""
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    _publish_successful_images(output_dir, stems=STEMS, snapshot=_SNAPSHOT)
    initialize_slurm_lifecycle(
        output_dir, generation=GENERATION, mode="ordinary"
    )
    _write_job_metadata(output_dir)
    begin_aggregation_fanout(
        output_dir,
        scheduler_epoch=GENERATION,
        shards=SHARDS,
        dataset_names=[DATASET],
    )
    write_finalize_chain(
        output_dir,
        mode=MODE_ORDINARY,
        generation=GENERATION,
        slurm_args={"slurm_partition": "batch", "slurm_time": "00:30:00"},
        shards=SHARDS,
    )
    # The event log is not what this suite is about, and with none written
    # the completion wait would sit out its full 600 s timeout.
    monkeypatch.setattr(
        _cli_checkpoint_handler, "_wait_for_completion", lambda *a, **k: None
    )
    return output_dir


def _run_shards(output_dir: Path) -> None:
    for task_index in range(SHARDS):
        run_aggregation_shard.callback(
            output_dir=output_dir, task_index=task_index, epoch=GENERATION
        )


def _run_after_shards(output_dir: Path, *, skip: set[tuple[str, int]] = frozenset()) -> None:
    """Run master, outputs, qc -- each task unless listed in *skip*."""
    for stage, tasks in ((STAGE_MASTER, 1), (STAGE_OUTPUTS, 2), (STAGE_QC, 1)):
        for task_index in range(tasks):
            if (stage, task_index) not in skip:
                run_chain_task(output_dir, GENERATION, stage, task_index)


# ---------------------------------------------------------------------------
# The shape the chain is written in
# ---------------------------------------------------------------------------


def test_every_stage_is_its_own_script_with_its_own_walltime(
    chain_run: Path,
) -> None:
    """One job per stage, each rendering the run's --slurm time on its own."""
    spec = load_chain_spec(chain_run, GENERATION)

    assert set(spec.scripts) == {
        STAGE_PREPARE,
        STAGE_SHARDS,
        STAGE_MASTER,
        STAGE_OUTPUTS,
        STAGE_QC,
        STAGE_PUBLISH,
    }
    assert spec.stages() == (
        STAGE_SHARDS,
        STAGE_MASTER,
        STAGE_OUTPUTS,
        STAGE_QC,
        STAGE_PUBLISH,
    )
    arrays = {
        STAGE_PREPARE: "0-0",
        STAGE_SHARDS: f"0-{SHARDS - 1}",
        STAGE_MASTER: "0-0",
        STAGE_OUTPUTS: "0-1",
        STAGE_QC: "0-0",
        STAGE_PUBLISH: "0-0",
    }
    for stage, script in spec.scripts.items():
        text = script.read_text(encoding="utf-8")
        assert f"#SBATCH --array={arrays[stage]}" in text, stage
        assert "#SBATCH --time=00:30:00" in text, stage
        assert "--partition=batch" in text, stage
        if stage == STAGE_SHARDS:
            assert "_cli_finalize_fanout" in text
            assert f"--epoch {GENERATION}" in text
        else:
            assert f"--stage {stage}" in text
            assert f"--generation {GENERATION}" in text


def test_no_stage_waits_on_its_own_shards_inside_its_walltime(
    chain_run: Path,
) -> None:
    """The defect the chain removes: index K shared an array with its shards."""
    spec = load_chain_spec(chain_run, GENERATION)
    shard_script = spec.scripts[STAGE_SHARDS].read_text(encoding="utf-8")

    assert "_cli_checkpoint_handler" not in shard_script
    assert "_cli_finalize_chain" not in shard_script


def test_prepare_submits_the_rest_as_an_afterany_chain(
    chain_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each stage waits on the one before; the first waits on nothing."""
    from phenotypic._cli import _cli_slurm_lifecycle

    calls: list[dict] = []

    def record(output_dir, **kwargs):
        calls.append(kwargs)
        return str(9000 + len(calls))

    monkeypatch.setattr(_cli_slurm_lifecycle, "submit_with_lifecycle", record)

    run_chain_task(chain_run, GENERATION, STAGE_PREPARE, 0)

    spec = load_chain_spec(chain_run, GENERATION)
    assert [call["token"] for call in calls] == [
        "finalize-shards",
        "finalize-master",
        "finalize-outputs",
        "finalize-qc",
        "finalize-publish",
    ]
    assert [call["script_path"] for call in calls] == [
        spec.scripts[stage]
        for stage in (
            STAGE_SHARDS,
            STAGE_MASTER,
            STAGE_OUTPUTS,
            STAGE_QC,
            STAGE_PUBLISH,
        )
    ]
    assert calls[0]["dependencies"] == ()
    for index, call in enumerate(calls[1:], start=1):
        assert call["dependencies"] == (str(9000 + index),)
        assert call["dependency_kind"] == "afterany"
    assert all(call["generation"] == GENERATION for call in calls)


# ---------------------------------------------------------------------------
# A chain that finishes
# ---------------------------------------------------------------------------


def test_a_finished_chain_certifies_the_run_and_closes_the_lifecycle(
    chain_run: Path,
) -> None:
    _run_shards(chain_run)
    _run_after_shards(chain_run)
    run_chain_task(chain_run, GENERATION, STAGE_PUBLISH, 0)

    spec = load_chain_spec(chain_run, GENERATION)
    for stage, tasks in ((STAGE_MASTER, 1), (STAGE_OUTPUTS, 2), (STAGE_QC, 1)):
        for task_index in range(tasks):
            status = read_stage_status(spec, stage, task_index)
            assert status is not None and status["status"] == STATUS_COMPLETED
    assert valid_aggregate_snapshot(chain_run) is not None
    assert valid_run_completion(chain_run) is not None
    assert load_slurm_lifecycle(chain_run)["active"] is False


def _table_outputs(output_dir: Path) -> dict[str, bytes]:
    """Every tabular deliverable the finalization writes, by relative path."""
    root = deliverables_dir(output_dir)
    wanted = [
        master_measurements_parquet_path(output_dir),
        measurements_parquet_path(output_dir),
        root / "measurements.csv",
        *sorted((root / "measurements_by_feature").glob("*")),
        *sorted((root / "measurements_by_category").glob("*")),
    ]
    return {
        str(path.relative_to(output_dir)): path.read_bytes() for path in wanted
    }


def test_the_chain_publishes_the_bytes_one_process_would(
    chain_run: Path, tmp_path: Path
) -> None:
    """Splitting the finalization across jobs must not change a byte of it.

    The in-process run reads the tables directly (no shards); the chain reads
    two shards and hands the mirror between jobs through the filesystem. Both
    differences are exactly where a divergence would come from.
    """
    from phenotypic._cli._cli_finalize_run import finalize_run

    single = tmp_path / "single"
    shutil.copytree(chain_run, single)
    finalize_run(
        single,
        dataset_names=[DATASET],
        metadata_csv=metadata_csv_deliverable_path(single),
    )

    _run_shards(chain_run)
    _run_after_shards(chain_run)
    run_chain_task(chain_run, GENERATION, STAGE_PUBLISH, 0)

    chained = _table_outputs(chain_run)
    assert len(chained) > 3, "the split directories are empty; vacuous"
    assert chained == _table_outputs(single)


# ---------------------------------------------------------------------------
# A chain that does not
# ---------------------------------------------------------------------------


def test_a_killed_output_task_leaves_the_run_incomplete(
    chain_run: Path,
) -> None:
    """A task killed at its walltime writes no status; that is a failure."""
    _run_shards(chain_run)
    _run_after_shards(chain_run, skip={(STAGE_OUTPUTS, 1)})

    with pytest.raises(FinalizationIncomplete, match=r"outputs\[1\]: no status"):
        run_chain_task(chain_run, GENERATION, STAGE_PUBLISH, 0)

    assert valid_run_completion(chain_run) is None
    assert valid_aggregate_snapshot(chain_run) is None
    lifecycle = load_slurm_lifecycle(chain_run)
    assert lifecycle["active"] is False
    # Closed as incomplete, not relabelled as a failed launch.
    assert lifecycle.get("terminal_status") != "failed"


def test_a_missing_shard_fails_the_master_and_skips_what_follows(
    chain_run: Path,
) -> None:
    run_aggregation_shard.callback(
        output_dir=chain_run, task_index=0, epoch=GENERATION
    )
    # Shard 1 was killed: no file, no status.
    _cli_finalize_chain.SHARD_STATUS_GRACE_SECONDS = 0.0
    try:
        with pytest.raises(RuntimeError, match="fan-out is incomplete"):
            run_chain_task(chain_run, GENERATION, STAGE_MASTER, 0)
    finally:
        _cli_finalize_chain.SHARD_STATUS_GRACE_SECONDS = 120.0

    spec = load_chain_spec(chain_run, GENERATION)
    assert read_stage_status(spec, STAGE_MASTER, 0)["status"] == STATUS_FAILED
    for stage, tasks in ((STAGE_OUTPUTS, 2), (STAGE_QC, 1)):
        for task_index in range(tasks):
            run_chain_task(chain_run, GENERATION, stage, task_index)
            assert (
                read_stage_status(spec, stage, task_index)["status"]
                == STATUS_SKIPPED
            )
    with pytest.raises(FinalizationIncomplete, match=r"master\[0\]"):
        run_chain_task(chain_run, GENERATION, STAGE_PUBLISH, 0)
    assert valid_run_completion(chain_run) is None


def test_a_mirror_rewritten_between_jobs_is_not_certified(
    chain_run: Path,
) -> None:
    """The handoff's digest is what stops a later job trusting other bytes."""
    _run_shards(chain_run)
    run_chain_task(chain_run, GENERATION, STAGE_MASTER, 0)
    mirror = measurements_parquet_path(chain_run)
    mirror.write_bytes(mirror.read_bytes() + b"\0")

    with pytest.raises(RuntimeError, match="changed after the master job"):
        run_chain_task(chain_run, GENERATION, STAGE_OUTPUTS, 1)

    spec = load_chain_spec(chain_run, GENERATION)
    status = read_stage_status(spec, STAGE_OUTPUTS, 1)
    assert status["status"] == STATUS_FAILED
    assert "changed after the master job" in status["error"]


def test_a_cancelled_generation_is_not_relabelled_failed(
    chain_run: Path,
) -> None:
    """A publish job that finds its generation closed leaves the verdict alone."""
    from phenotypic._cli._cli_slurm_lifecycle import deactivate_generation

    _run_shards(chain_run)
    _run_after_shards(chain_run)
    deactivate_generation(chain_run, GENERATION)

    with pytest.raises(Exception):
        run_chain_task(chain_run, GENERATION, STAGE_PUBLISH, 0)

    assert load_slurm_lifecycle(chain_run).get("terminal_status") != "failed"
    assert valid_run_completion(chain_run) is None


def test_status_files_live_in_the_chain_scratch_directory(
    chain_run: Path,
) -> None:
    _run_shards(chain_run)
    run_chain_task(chain_run, GENERATION, STAGE_MASTER, 0)

    path = stage_status_path(chain_run, GENERATION, STAGE_MASTER, 0)
    assert path.is_file()
    assert ".phenotypic" in path.parts
    assert json.loads(path.read_text(encoding="utf-8"))["stage"] == STAGE_MASTER


# ---------------------------------------------------------------------------
# Staged hand-off
# ---------------------------------------------------------------------------


def test_prepare_points_the_staged_controller_at_the_publish_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recovery controller must wait for ``publish``, not for ``prepare``."""
    from phenotypic._cli import _cli_staged_orchestration
    from phenotypic._cli._cli_finalize_chain import (
        MODE_STAGED,
        _hand_staged_controller_to,
    )
    from phenotypic._cli._cli_staged_orchestration import (
        initialize_orchestration,
        load_orchestration_state,
        save_orchestration_state,
    )

    output_dir = tmp_path / "staged"
    output_dir.mkdir()
    config = output_dir / "controller.json"
    config.write_text("{}", encoding="utf-8")
    state = initialize_orchestration(
        output_dir, epoch="epoch-1", mode="fresh", controller_config_path=config
    )
    state.update(
        {"phase": "finalizing", "active_job_id": "500", "expected_controller_id": "501"}
    )
    save_orchestration_state(output_dir, state)
    retargeted: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        _cli_staged_orchestration,
        "update_job_dependency",
        lambda job_id, deps: retargeted.append((job_id, list(deps))) or True,
    )
    write_finalize_chain(
        output_dir,
        mode=MODE_STAGED,
        generation="epoch-1",
        slurm_args={},
        shards=1,
    )

    _hand_staged_controller_to(load_chain_spec(output_dir, "epoch-1"), "599")

    assert load_orchestration_state(output_dir)["active_job_id"] == "599"
    assert retargeted == [("501", ["599"])]


def test_parallel_output_tasks_never_hold_the_lifecycle_lock(
    chain_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``outputs`` runs two tasks at once. A guard would hold the lifecycle
    lock for each whole write, starving the sibling (300 s) and cancellation
    (60 s) during a long analysis fit or QC rebuild."""
    from phenotypic._cli import _cli_output_manager

    seen: list[object] = []
    real = _cli_output_manager.publish_finalization_outputs

    def spy(*args, **kwargs):
        seen.append(kwargs.get("commit_guard"))
        return real(*args, **kwargs)

    monkeypatch.setattr(_cli_output_manager, "publish_finalization_outputs", spy)
    _run_shards(chain_run)
    _run_after_shards(chain_run)

    assert len(seen) == 3
    assert seen == [None, None, None]


def test_only_recompile_guards_its_serial_writes(tmp_path: Path) -> None:
    """Recompile's single-task finalizer held the guard throughout; the chain
    keeps it there (for less time), and adds none to ordinary or staged runs,
    which never had one."""
    from phenotypic._cli._cli_finalize_chain import (
        MODE_RECOMPILE,
        MODE_STAGED,
        ChainSpec,
        _commit_guard,
    )

    def spec(mode: str) -> ChainSpec:
        return ChainSpec(
            mode=mode,
            output_dir=tmp_path,
            generation="g",
            shards=0,
            scripts={},
        )

    assert _commit_guard(spec(MODE_ORDINARY)) is None
    assert _commit_guard(spec(MODE_STAGED)) is None
    assert callable(_commit_guard(spec(MODE_RECOMPILE)))
