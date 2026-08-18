"""Focused gates for SLURM metadata migration before recompile."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import polars as pl
import pytest

from phenotypic.schema import IMAGE
from phenotypic.sdk_ import (
    CHUNK_STATE_JSON,
    ChunkStateKey,
    dataset_overlays_dir,
    progress_dir,
)


def _write_legacy_hdf(path: Path, *, conflict: bool = False) -> None:
    """Write a minimal metadata-bearing HDF migration target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 1
        public = handle.create_group("public_metadata")
        public.attrs["MetadataGenetic_Strain"] = "S288C"
        if conflict:
            public.attrs["Metadata_Strain"] = "BY4741"


def _write_measurement(path: Path, area: int) -> None:
    """Write a tiny per-image measurement Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"Size_Area": [area]}).write_parquet(path)


def test_combined_chain_has_one_afterok_barrier_and_singleton_finalizer(
    tmp_path: Path,
) -> None:
    """Migration and recompile use one ordered dispatcher namespace."""
    from phenotypic.phenotypicCLI import _handle_recompile_slurm

    output_dir = tmp_path / "out"
    _write_legacy_hdf(
        output_dir / "results" / "plate_a" / "hdf" / "img1.h5"
    )
    _write_measurement(
        output_dir / "results" / "plate_a" / "measurements" / "img1.parquet",
        1,
    )
    metadata_csv = tmp_path / "external.csv"
    original_metadata = b"MetadataSample_Strain\nS288C\n"
    metadata_csv.write_bytes(original_metadata)

    def fake_submit(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            job_ids=["123"], flat_scripts=kwargs["flat_chunk_scripts"]
        )

    with (
        patch("phenotypic.phenotypicCLI.get_slurm_array_limit", return_value=8),
        patch(
            "phenotypic.phenotypicCLI.submit_slurm_script_chain",
            side_effect=fake_submit,
        ) as submit,
        patch("phenotypic._cli._dashboard.generate_dashboard"),
    ):
        _handle_recompile_slurm(
            output_dir=output_dir,
            metadata_csv=metadata_csv,
            include_dataset_column=True,
            overlay_alpha=0.3,
            checkpoint_interval=None,
            slurm_args={"slurm_partition": "compute"},
            wait=False,
        )

    kwargs = submit.call_args.kwargs
    scripts = list(kwargs["flat_chunk_scripts"])
    assert [path.name for path in scripts] == [
        "metadata_migration_chunk0.sh",
        "metadata_migration_finalizer.sh",
        "recompile_array_chunk0.sh",
    ]
    assert kwargs["continuation_dependency_kinds"] == [
        "afterany",
        "afterok",
    ]
    finalizer_text = scripts[1].read_text(encoding="utf-8")
    assert "#SBATCH --array" not in finalizer_text
    assert "--finalize" in finalizer_text
    assert metadata_csv.read_bytes() == original_metadata


def test_blocked_preflight_submits_and_writes_nothing(tmp_path: Path) -> None:
    """A legacy/canonical conflict aborts before machine-state creation."""
    from phenotypic.phenotypicCLI import _handle_recompile_slurm

    output_dir = tmp_path / "out"
    _write_legacy_hdf(
        output_dir / "results" / "plate_a" / "hdf" / "img1.h5",
        conflict=True,
    )

    with (
        patch("phenotypic.phenotypicCLI.submit_slurm_script_chain") as submit,
        pytest.raises(SystemExit),
    ):
        _handle_recompile_slurm(
            output_dir=output_dir,
            metadata_csv=None,
            include_dataset_column=True,
            overlay_alpha=0.3,
            checkpoint_interval=None,
            slurm_args={},
            wait=False,
        )

    submit.assert_not_called()
    assert not (output_dir / ".phenotypic").exists()
    assert not (output_dir / "deliverables").exists()


def test_hdf_only_migration_uses_slurm_without_recompile_publication(
    tmp_path: Path,
) -> None:
    """A migration-only bundle gets a SLURM barrier and no dashboard write."""
    from phenotypic.phenotypicCLI import _handle_recompile_slurm

    output_dir = tmp_path / "out"
    _write_legacy_hdf(
        output_dir / "results" / "plate_a" / "hdf" / "img1.h5"
    )
    overlay = dataset_overlays_dir(output_dir, "plate_a") / "img1.png"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"already rendered")

    def fake_submit(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            job_ids=["123"], flat_scripts=kwargs["flat_chunk_scripts"]
        )

    with (
        patch("phenotypic.phenotypicCLI.get_slurm_array_limit", return_value=8),
        patch(
            "phenotypic.phenotypicCLI.submit_slurm_script_chain",
            side_effect=fake_submit,
        ) as submit,
        patch("phenotypic._cli._dashboard.generate_dashboard") as dashboard,
        patch("phenotypic.phenotypicCLI._handle_recompile") as local,
    ):
        _handle_recompile_slurm(
            output_dir=output_dir,
            metadata_csv=None,
            include_dataset_column=True,
            overlay_alpha=0.3,
            checkpoint_interval=None,
            slurm_args={},
            wait=False,
        )

    scripts = list(submit.call_args.kwargs["flat_chunk_scripts"])
    assert [path.name for path in scripts] == [
        "metadata_migration_chunk0.sh",
        "metadata_migration_finalizer.sh",
    ]
    assert "continuation_dependency_kinds" not in submit.call_args.kwargs
    dashboard.assert_not_called()
    local.assert_not_called()


def test_worker_and_finalizer_validate_receipt_and_canonical_bundle(
    tmp_path: Path,
) -> None:
    """Successful target work releases the singleton validation barrier."""
    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        generate_metadata_migration_slurm_scripts,
        metadata_migration_finalizer_status_path,
        metadata_migration_task_status_path,
        plan_metadata_schema_for_slurm_recompile,
    )
    from phenotypic._cli._cli_recompile_metadata_migration_worker import (
        finalize_metadata_migration,
        run_metadata_migration_target,
    )
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / "out"
    generation = "migration-success"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    hdf_path = output_dir / "results" / "plate_a" / "hdf" / "img1.h5"
    _write_legacy_hdf(hdf_path)
    plan = plan_metadata_schema_for_slurm_recompile(
        output_dir, attempt_id=generation
    )
    generated = generate_metadata_migration_slurm_scripts(
        plan,
        slurm_args={},
        array_limit=5,
        attempt_id=generation,
        slurm_generation=generation,
        has_recompile_downstream=False,
    )
    assert generated is not None

    run_metadata_migration_target(
        plan.manifest_path,
        0,
        output_dir=output_dir,
        slurm_generation=generation,
        attempt_id=generation,
    )
    target_status = json.loads(
        metadata_migration_task_status_path(
            plan.manifest_path, 0
        ).read_text(encoding="utf-8")
    )
    assert target_status["status"] == "completed"
    assert Path(target_status["receipt_path"]).is_file()

    finalize_metadata_migration(
        plan.manifest_path,
        output_dir=output_dir,
        slurm_generation=generation,
        attempt_id=generation,
    )
    final_status = json.loads(
        metadata_migration_finalizer_status_path(
            plan.manifest_path
        ).read_text(encoding="utf-8")
    )
    assert final_status["status"] == "completed"
    assert final_status["target_count"] == 1
    with h5py.File(hdf_path, "r") as handle:
        assert handle["public_metadata"].attrs["Metadata_Strain"] == "S288C"
        assert "MetadataGenetic_Strain" not in handle["public_metadata"].attrs


def test_failed_worker_makes_migration_finalizer_fail(tmp_path: Path) -> None:
    """A target failure produces the status that keeps afterok unreleased."""
    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        generate_metadata_migration_slurm_scripts,
        metadata_migration_finalizer_status_path,
        plan_metadata_schema_for_slurm_recompile,
    )
    from phenotypic._cli._cli_recompile_metadata_migration_worker import (
        finalize_metadata_migration,
        run_metadata_migration_target,
    )
    from phenotypic._cli._cli_slurm_lifecycle import (
        generation_is_active,
        initialize_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    hdf_path = output_dir / "results" / "plate_a" / "hdf" / "img1.h5"
    _write_legacy_hdf(hdf_path)
    generation = "afterok-failure-attempt"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    plan = plan_metadata_schema_for_slurm_recompile(
        output_dir, attempt_id=generation
    )
    generated = generate_metadata_migration_slurm_scripts(
        plan,
        slurm_args={},
        array_limit=5,
        attempt_id=generation,
        slurm_generation=generation,
        has_recompile_downstream=True,
    )
    assert generated is not None
    shard_script = generated.shard_scripts[0].read_text(encoding="utf-8")
    finalizer_script = generated.finalizer_script.read_text(encoding="utf-8")
    assert "+    --output-dir" not in shard_script
    assert f"--slurm-generation {generation}" in shard_script
    assert f"--attempt-id {generation}" in shard_script
    assert f"--slurm-generation {generation}" in finalizer_script
    assert f"--attempt-id {generation}" in finalizer_script
    with h5py.File(hdf_path, "r+") as handle:
        handle["public_metadata"].attrs["unrelated"] = "changed"

    with pytest.raises(ValueError, match="planning fields"):
        run_metadata_migration_target(
            plan.manifest_path,
            0,
            output_dir=output_dir,
            slurm_generation=generation,
            attempt_id=generation,
        )
    with pytest.raises(RuntimeError, match="planning fields"):
        finalize_metadata_migration(
            plan.manifest_path,
            output_dir=output_dir,
            slurm_generation=generation,
            attempt_id=generation,
        )

    final_status = json.loads(
        metadata_migration_finalizer_status_path(
            plan.manifest_path
        ).read_text(encoding="utf-8")
    )
    assert final_status["status"] == "failed"
    assert not generation_is_active(output_dir, generation)


def test_worker_rejects_tampered_target_outside_bundle(tmp_path: Path) -> None:
    """Durable worker input cannot redirect mutation to an external file."""
    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        generate_metadata_migration_slurm_scripts,
        metadata_migration_task_status_path,
        plan_metadata_schema_for_slurm_recompile,
    )
    from phenotypic._cli._cli_recompile_metadata_migration_worker import (
        run_metadata_migration_target,
    )
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / "out"
    generation = "tampered-target"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    _write_legacy_hdf(
        output_dir / "results" / "plate_a" / "hdf" / "img1.h5"
    )
    external = tmp_path / "external.h5"
    _write_legacy_hdf(external)
    external_before = external.read_bytes()
    plan = plan_metadata_schema_for_slurm_recompile(
        output_dir, attempt_id=generation
    )
    assert (
        generate_metadata_migration_slurm_scripts(
            plan,
            slurm_args={},
            array_limit=5,
            attempt_id=generation,
            slurm_generation=generation,
        )
        is not None
    )
    manifest = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
    manifest["targets"][0]["path"] = str(external)
    plan.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the authoritative bundle"):
        run_metadata_migration_target(
            plan.manifest_path,
            0,
            output_dir=output_dir,
            slurm_generation=generation,
            attempt_id=generation,
        )

    status = json.loads(
        metadata_migration_task_status_path(
            plan.manifest_path, 0
        ).read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert external.read_bytes() == external_before


def test_worker_rejects_tampered_hdf_snapshot_plan_field(
    tmp_path: Path,
) -> None:
    """The SLURM manifest preserves and validates the HDF snapshot digest."""
    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        generate_metadata_migration_slurm_scripts,
        plan_metadata_schema_for_slurm_recompile,
    )
    from phenotypic._cli._cli_recompile_metadata_migration_worker import (
        run_metadata_migration_target,
    )
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / "out"
    generation = "tampered-hdf-snapshot"
    target = output_dir / "results" / "plate_a" / "hdf" / "img1.h5"
    _write_legacy_hdf(target)
    before = target.read_bytes()
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    plan = plan_metadata_schema_for_slurm_recompile(
        output_dir, attempt_id=generation
    )
    assert generate_metadata_migration_slurm_scripts(
        plan,
        slurm_args={},
        array_limit=5,
        attempt_id=generation,
        slurm_generation=generation,
    )
    manifest = json.loads(plan.manifest_path.read_text(encoding="utf-8"))
    assert manifest["targets"][0]["hdf_snapshot_fingerprint"].startswith(
        "sha256:"
    )
    manifest["targets"][0]["hdf_snapshot_fingerprint"] = "sha256:" + "0" * 64
    plan.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="planning fields"):
        run_metadata_migration_target(
            plan.manifest_path,
            0,
            output_dir=output_dir,
            slurm_generation=generation,
            attempt_id=generation,
        )

    assert target.read_bytes() == before


def test_waiter_observes_migration_failure_before_recompile_status(
    tmp_path: Path,
) -> None:
    """Wait mode cannot hang when afterok suppresses the recompile chain."""
    from phenotypic.phenotypicCLI import _wait_for_recompile_finalizer_status

    migration_status = tmp_path / "migration-finalizer.json"
    migration_status.write_text(
        json.dumps({"status": "failed", "error": "migration failed"}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="migration failed"):
        _wait_for_recompile_finalizer_status(
            tmp_path / "out",
            9,
            migration_finalizer_status_path=migration_status,
            poll_interval=0.001,
            timeout=0.01,
        )


def test_trailing_measurement_discovery_is_read_only_and_nonduplicating(
    tmp_path: Path,
) -> None:
    """Recompile combines the aggregate only with unrecorded individuals."""
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        TASK_MEASUREMENTS,
        build_recompile_tasks,
    )

    output_dir = tmp_path / "out"
    meas_dir = output_dir / "results" / "plate_a" / "measurements"
    aggregate = meas_dir / "_dataset_aggregated.parquet"
    image_1 = meas_dir / "img1.parquet"
    image_2 = meas_dir / "img2.parquet"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"Size_Area": [1], str(IMAGE.IMAGE_NAME): ["img1"]}
    ).write_parquet(aggregate)
    _write_measurement(image_1, 1)
    _write_measurement(image_2, 2)
    state_path = progress_dir(output_dir) / CHUNK_STATE_JSON
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                ChunkStateKey.CHUNKED_FILES: [],
                ChunkStateKey.NEXT_CHUNK_ID: 1,
            }
        ),
        encoding="utf-8",
    )
    before = state_path.read_bytes()
    existing_paths = {path for path in output_dir.rglob("*")}

    tasks = build_recompile_tasks(
        output_dir=output_dir,
        dataset_names=["plate_a"],
        include_dataset_column=True,
        overlay_alpha=0.3,
        shard_size=10,
    )

    measurement = next(
        task for task in tasks if task["task_type"] == TASK_MEASUREMENTS
    )
    assert measurement["files"] == [str(aggregate), str(image_2)]
    assert state_path.read_bytes() == before
    assert {path for path in output_dir.rglob("*")} == existing_paths


def test_canonical_plan_is_scriptless_no_op(tmp_path: Path) -> None:
    """Canonical bundles preserve the ordinary recompile chain shape."""
    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        generate_metadata_migration_slurm_scripts,
        plan_metadata_schema_for_slurm_recompile,
    )

    output_dir = tmp_path / "out"
    _write_measurement(
        output_dir / "results" / "plate_a" / "measurements" / "img1.parquet",
        1,
    )
    plan = plan_metadata_schema_for_slurm_recompile(output_dir)

    assert plan.report.status == "compatible"
    assert plan.targets == ()
    assert (
        generate_metadata_migration_slurm_scripts(
            plan, slurm_args={}, array_limit=5
        )
        is None
    )
    assert not plan.plan_dir.exists()


def test_finalizer_rejects_target_changed_after_worker(tmp_path: Path) -> None:
    """Durable receipt validation catches unrelated post-worker HDF edits."""
    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        generate_metadata_migration_slurm_scripts,
        metadata_migration_finalizer_status_path,
        plan_metadata_schema_for_slurm_recompile,
    )
    from phenotypic._cli._cli_recompile_metadata_migration_worker import (
        finalize_metadata_migration,
        run_metadata_migration_target,
    )
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    output_dir = tmp_path / "out"
    generation = "attempt-change"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    hdf_path = output_dir / "results" / "plate_a" / "hdf" / "img1.h5"
    _write_legacy_hdf(hdf_path)
    plan = plan_metadata_schema_for_slurm_recompile(
        output_dir, attempt_id=generation
    )
    assert (
        generate_metadata_migration_slurm_scripts(
            plan,
            slurm_args={},
            array_limit=5,
            attempt_id=generation,
            slurm_generation=generation,
        )
        is not None
    )
    run_metadata_migration_target(
        plan.manifest_path,
        0,
        output_dir=output_dir,
        slurm_generation=generation,
        attempt_id=generation,
    )
    with h5py.File(hdf_path, "r+") as handle:
        handle.attrs["unrelated_after_worker"] = "changed"

    with pytest.raises(RuntimeError, match="receipt validation failed"):
        finalize_metadata_migration(
            plan.manifest_path,
            output_dir=output_dir,
            slurm_generation=generation,
            attempt_id=generation,
        )

    status = json.loads(
        metadata_migration_finalizer_status_path(
            plan.manifest_path
        ).read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"


def test_attempt_scoped_shards_ignore_stale_prior_attempt(tmp_path: Path) -> None:
    """A later smaller recompile cannot ingest an old higher shard id."""
    from phenotypic._cli._cli_recompile_worker import (
        _write_master_outputs_from_shards,
    )

    output_dir = tmp_path / "out"
    old_attempt = progress_dir(output_dir) / "recompile" / "attempts" / "old"
    new_attempt = progress_dir(output_dir) / "recompile" / "attempts" / "new"
    _write_measurement(old_attempt / "measurement_shards" / "shard_0.parquet", 1)
    _write_measurement(old_attempt / "measurement_shards" / "shard_9.parquet", 99)
    _write_measurement(new_attempt / "measurement_shards" / "shard_0.parquet", 2)

    merged = _write_master_outputs_from_shards(output_dir, new_attempt)

    assert merged is not None
    assert merged["Size_Area"].to_list() == [2]


def test_same_plan_new_attempt_ignores_old_finalizer_failure(
    tmp_path: Path,
) -> None:
    """Waiters observe only their attempt-scoped migration finalizer."""
    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        plan_metadata_schema_for_slurm_recompile,
    )
    from phenotypic.phenotypicCLI import (
        _wait_for_metadata_migration_finalizer_status,
    )

    output_dir = tmp_path / "out"
    _write_legacy_hdf(
        output_dir / "results" / "plate_a" / "hdf" / "img1.h5"
    )
    old = plan_metadata_schema_for_slurm_recompile(
        output_dir, attempt_id="old-attempt"
    )
    new = plan_metadata_schema_for_slurm_recompile(
        output_dir, attempt_id="new-attempt"
    )
    old.finalizer_status_path.parent.mkdir(parents=True, exist_ok=True)
    new.finalizer_status_path.parent.mkdir(parents=True, exist_ok=True)
    old.finalizer_status_path.write_text(
        json.dumps({"status": "failed", "error": "old failure"}),
        encoding="utf-8",
    )
    new.finalizer_status_path.write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )

    _wait_for_metadata_migration_finalizer_status(
        new.finalizer_status_path, poll_interval=0.001, timeout=0.01
    )
    assert old.plan_dir != new.plan_dir


def test_consecutive_launches_use_new_generation_and_scripts(
    tmp_path: Path,
) -> None:
    """Terminally completed attempts cannot dedupe a later launch."""
    from phenotypic._cli._cli_slurm_lifecycle import deactivate_generation
    from phenotypic.phenotypicCLI import _handle_recompile_slurm

    output_dir = tmp_path / "out"
    _write_measurement(
        output_dir / "results" / "plate_a" / "measurements" / "img1.parquet",
        1,
    )
    generations: list[str] = []
    submitted_scripts: list[tuple[Path, ...]] = []

    def fake_submit(**kwargs: object) -> SimpleNamespace:
        generation = str(kwargs["generation"])
        generations.append(generation)
        scripts = tuple(kwargs["flat_chunk_scripts"])
        submitted_scripts.append(scripts)
        deactivate_generation(output_dir, generation)
        return SimpleNamespace(
            job_ids=[str(100 + len(generations))], flat_scripts=list(scripts)
        )

    with (
        patch("phenotypic.phenotypicCLI.get_slurm_array_limit", return_value=8),
        patch(
            "phenotypic.phenotypicCLI.submit_slurm_script_chain",
            side_effect=fake_submit,
        ) as submit,
        patch("phenotypic._cli._dashboard.generate_dashboard"),
    ):
        for _ in range(2):
            _handle_recompile_slurm(
                output_dir=output_dir,
                metadata_csv=None,
                include_dataset_column=True,
                overlay_alpha=0.3,
                checkpoint_interval=None,
                slurm_args={},
                wait=False,
            )

    assert submit.call_count == 2
    assert len(set(generations)) == 2
    assert submitted_scripts[0] != submitted_scripts[1]


def test_submission_failure_deactivates_attempt(tmp_path: Path) -> None:
    """A failed initial submission does not leave a blocking active fence."""
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle
    from phenotypic.phenotypicCLI import _handle_recompile_slurm

    output_dir = tmp_path / "out"
    _write_measurement(
        output_dir / "results" / "plate_a" / "measurements" / "img1.parquet",
        1,
    )
    with (
        patch("phenotypic.phenotypicCLI.get_slurm_array_limit", return_value=8),
        patch(
            "phenotypic.phenotypicCLI.submit_slurm_script_chain",
            side_effect=RuntimeError("submit failed"),
        ),
        pytest.raises(RuntimeError, match="submit failed"),
    ):
        _handle_recompile_slurm(
            output_dir=output_dir,
            metadata_csv=None,
            include_dataset_column=True,
            overlay_alpha=0.3,
            checkpoint_interval=None,
            slurm_args={},
            wait=False,
        )

    lifecycle = load_slurm_lifecycle(output_dir)
    assert lifecycle is not None
    assert lifecycle["active"] is False


def test_script_generation_failure_creates_no_active_attempt(
    tmp_path: Path,
) -> None:
    """Pure file-generation failure happens before lifecycle publication."""
    from phenotypic._cli._cli_slurm_lifecycle import load_slurm_lifecycle
    from phenotypic.phenotypicCLI import _handle_recompile_slurm

    output_dir = tmp_path / "out"
    _write_measurement(
        output_dir / "results" / "plate_a" / "measurements" / "img1.parquet",
        1,
    )
    with (
        patch("phenotypic.phenotypicCLI.get_slurm_array_limit", return_value=8),
        patch(
            "phenotypic.phenotypicCLI.generate_recompile_slurm_scripts",
            side_effect=RuntimeError("generation failed"),
        ),
        pytest.raises(RuntimeError, match="generation failed"),
    ):
        _handle_recompile_slurm(
            output_dir=output_dir,
            metadata_csv=None,
            include_dataset_column=True,
            overlay_alpha=0.3,
            checkpoint_interval=None,
            slurm_args={},
            wait=False,
        )

    assert load_slurm_lifecycle(output_dir) is None


@pytest.mark.parametrize("fails", [False, True])
def test_terminal_recompile_worker_deactivates_generation(
    tmp_path: Path, fails: bool
) -> None:
    """Ordinary terminal success and failure both release the attempt fence."""
    from phenotypic._cli._cli_recompile_worker import run_recompile_task
    from phenotypic._cli._cli_slurm_lifecycle import (
        generation_is_active,
        initialize_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    generation = f"terminal-{'failure' if fails else 'success'}"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    manifest = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "task_manifest.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "finalize",
                        "expected_non_finalizer_tasks": 0,
                        "slurm_generation": generation,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    side_effect = RuntimeError("terminal failed") if fails else None
    with patch(
        "phenotypic._cli._cli_recompile_worker._run_finalizer_task",
        side_effect=side_effect,
    ):
        if fails:
            with pytest.raises(RuntimeError, match="terminal failed"):
                run_recompile_task(
                    output_dir,
                    manifest,
                    0,
                    slurm_generation=generation,
                    attempt_id=generation,
                )
        else:
            run_recompile_task(
                output_dir,
                manifest,
                0,
                slurm_generation=generation,
                attempt_id=generation,
            )

    assert not generation_is_active(output_dir, generation)


@pytest.mark.parametrize("mode", ["ordinary", "recompile"])
def test_active_generation_is_refused_without_cancellation(
    tmp_path: Path, mode: str
) -> None:
    """Recompile never guesses that an active forward run is stale."""
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle
    from phenotypic.phenotypicCLI import _initialize_recompile_slurm_attempt

    output_dir = tmp_path / "out"
    initialize_slurm_lifecycle(
        output_dir, generation="live-forward", mode=mode
    )
    with (
        patch(
            "phenotypic._cli._cli_slurm_lifecycle.cancel_generation"
        ) as cancel,
        pytest.raises(RuntimeError, match="may still own live jobs"),
    ):
        _initialize_recompile_slurm_attempt(output_dir, "new-recompile")

    cancel.assert_not_called()


@pytest.mark.parametrize("identity", [None, " "])
def test_unusable_aggregate_identity_uses_individual_recovery(
    tmp_path: Path, identity: str | None
) -> None:
    """Null and blank aggregate identities cannot prove row membership."""
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        TASK_MEASUREMENTS,
        build_recompile_tasks,
    )

    output_dir = tmp_path / "out"
    meas_dir = output_dir / "results" / "plate_a" / "measurements"
    aggregate = meas_dir / "_dataset_aggregated.parquet"
    individual = meas_dir / "img1.parquet"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"Size_Area": [1], str(IMAGE.IMAGE_NAME): [identity]}
    ).write_parquet(aggregate)
    _write_measurement(individual, 1)

    tasks = build_recompile_tasks(
        output_dir=output_dir,
        dataset_names=["plate_a"],
        include_dataset_column=True,
        overlay_alpha=0.3,
        shard_size=10,
    )
    measurement = next(
        task for task in tasks if task["task_type"] == TASK_MEASUREMENTS
    )
    assert measurement["files"] == [str(individual)]


def test_uninspectable_aggregate_uses_individual_recovery(
    tmp_path: Path,
) -> None:
    """An unreadable aggregate cannot prove membership for safe combining."""
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        TASK_MEASUREMENTS,
        build_recompile_tasks,
    )

    output_dir = tmp_path / "out"
    meas_dir = output_dir / "results" / "plate_a" / "measurements"
    aggregate = meas_dir / "_dataset_aggregated.parquet"
    individual = meas_dir / "img1.parquet"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    aggregate.write_text("not parquet", encoding="utf-8")
    _write_measurement(individual, 1)

    tasks = build_recompile_tasks(
        output_dir=output_dir,
        dataset_names=["plate_a"],
        include_dataset_column=True,
        overlay_alpha=0.3,
        shard_size=10,
    )

    measurement = next(
        task for task in tasks if task["task_type"] == TASK_MEASUREMENTS
    )
    assert measurement["files"] == [str(individual)]


def test_sole_unusable_aggregate_is_migrated_then_recompiled(
    tmp_path: Path,
) -> None:
    """SLURM retains and migrates the sole aggregate before aggregation."""
    from phenotypic._cli._cli_recompile_metadata_migration_worker import (
        run_metadata_migration_target,
    )
    from phenotypic._cli._cli_recompile_slurm_scripts import TASK_MEASUREMENTS
    from phenotypic.sdk_ import JOB_METADATA_JSON
    from phenotypic.phenotypicCLI import _handle_recompile_slurm

    output_dir = tmp_path / "out"
    aggregate = (
        output_dir
        / "results"
        / "plate_a"
        / "measurements"
        / "_dataset_aggregated.parquet"
    )
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "Size_Area": [1],
            str(IMAGE.IMAGE_NAME): [""],
            "MetadataGenetic_Strain": ["S288C"],
        }
    ).write_parquet(aggregate)

    def fake_submit(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            job_ids=["123"], flat_scripts=kwargs["flat_chunk_scripts"]
        )

    with (
        patch("phenotypic.phenotypicCLI.get_slurm_array_limit", return_value=8),
        patch(
            "phenotypic.phenotypicCLI.submit_slurm_script_chain",
            side_effect=fake_submit,
        ),
        patch("phenotypic._cli._dashboard.generate_dashboard"),
    ):
        _handle_recompile_slurm(
            output_dir=output_dir,
            metadata_csv=None,
            include_dataset_column=True,
            overlay_alpha=0.3,
            checkpoint_interval=None,
            slurm_args={},
            wait=False,
        )

    metadata = json.loads(
        (progress_dir(output_dir) / JOB_METADATA_JSON).read_text(
            encoding="utf-8"
        )
    )
    attempt_id = str(metadata["recompile"]["attempt_id"])
    recompile_manifest = Path(metadata["recompile"]["task_manifest"])
    recompile_tasks = json.loads(
        recompile_manifest.read_text(encoding="utf-8")
    )["tasks"]
    measurement = next(
        task
        for task in recompile_tasks
        if task["task_type"] == TASK_MEASUREMENTS
    )
    assert measurement["files"] == [str(aggregate)]

    migration_manifest = Path(
        metadata["recompile"]["metadata_migration"]["task_manifest"]
    )
    run_metadata_migration_target(
        migration_manifest,
        0,
        output_dir=output_dir,
        slurm_generation=attempt_id,
        attempt_id=attempt_id,
    )
    canonical = pl.read_parquet(aggregate)
    assert "Metadata_Strain" in canonical.columns
    assert "MetadataGenetic_Strain" not in canonical.columns


@pytest.mark.parametrize("manifest_state", ["missing", "corrupt"])
def test_ordinary_manifest_bootstrap_failure_deactivates_attempt(
    tmp_path: Path, manifest_state: str
) -> None:
    """Ordinary workers fence their attempt without manifest metadata."""
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        recompile_task_status_path,
    )
    from phenotypic._cli._cli_recompile_worker import run_recompile_task
    from phenotypic._cli._cli_slurm_lifecycle import (
        generation_is_active,
        initialize_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    generation = f"ordinary-{manifest_state}"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    manifest = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "task_manifest.json"
    )
    if manifest_state == "corrupt":
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{broken", encoding="utf-8")
    terminal_status = recompile_task_status_path(manifest, 9)

    with pytest.raises((OSError, ValueError)):
        run_recompile_task(
            output_dir,
            manifest,
            0,
            slurm_generation=generation,
            attempt_id=generation,
            terminal_status_path=terminal_status,
        )

    status = json.loads(
        recompile_task_status_path(manifest, 0).read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    terminal = json.loads(terminal_status.read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["manifest_unreadable"] is True
    assert not generation_is_active(output_dir, generation)


def test_ordinary_semantic_manifest_failure_uses_script_fence(
    tmp_path: Path,
) -> None:
    """Readable but invalid task metadata cannot bypass terminal teardown."""
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        recompile_task_status_path,
    )
    from phenotypic._cli._cli_recompile_worker import run_recompile_task
    from phenotypic._cli._cli_slurm_lifecycle import (
        generation_is_active,
        initialize_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    generation = "ordinary-semantic-corruption"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    manifest = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "task_manifest.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"tasks": [{"slurm_generation": generation}]}),
        encoding="utf-8",
    )
    terminal_status = recompile_task_status_path(manifest, 9)

    with pytest.raises(ValueError, match="Unknown recompile task type"):
        run_recompile_task(
            output_dir,
            manifest,
            0,
            slurm_generation=generation,
            attempt_id=generation,
            terminal_status_path=terminal_status,
        )

    terminal = json.loads(terminal_status.read_text(encoding="utf-8"))
    assert terminal["status"] == "failed"
    assert terminal["manifest_unreadable"] is False
    assert not generation_is_active(output_dir, generation)


@pytest.mark.parametrize("manifest_state", ["missing", "corrupt"])
def test_migration_manifest_bootstrap_failure_deactivates_attempt(
    tmp_path: Path, manifest_state: str
) -> None:
    """Migration finalizer uses independent output/generation arguments."""
    from click.testing import CliRunner

    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        metadata_migration_finalizer_status_path,
    )
    from phenotypic._cli._cli_recompile_metadata_migration_worker import main
    from phenotypic._cli._cli_slurm_lifecycle import (
        generation_is_active,
        initialize_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    generation = f"migration-{manifest_state}"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    manifest = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "metadata_migration"
        / "plan"
        / "migration_plan.json"
    )
    if manifest_state == "corrupt":
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{broken", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "--task-manifest",
            str(manifest),
            "--finalize",
            "--output-dir",
            str(output_dir),
            "--slurm-generation",
            generation,
            "--attempt-id",
            generation,
        ],
    )

    assert result.exit_code != 0
    status = json.loads(
        metadata_migration_finalizer_status_path(manifest).read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "failed"
    assert status["manifest_unreadable"] is True
    assert not generation_is_active(output_dir, generation)


def test_migration_target_bootstrap_failure_is_visible_to_waiter(
    tmp_path: Path,
) -> None:
    """A target bootstrap failure publishes the singleton terminal status."""
    from click.testing import CliRunner

    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        metadata_migration_finalizer_status_path,
        metadata_migration_task_status_path,
    )
    from phenotypic._cli._cli_recompile_metadata_migration_worker import main
    from phenotypic._cli._cli_slurm_lifecycle import (
        generation_is_active,
        initialize_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    generation = "migration-corrupt-target"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    manifest = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "metadata_migration"
        / "plan"
        / "migration_plan.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{broken", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "--task-manifest",
            str(manifest),
            "--task-index",
            "0",
            "--output-dir",
            str(output_dir),
            "--slurm-generation",
            generation,
            "--attempt-id",
            generation,
        ],
    )

    assert result.exit_code != 0
    target = json.loads(
        metadata_migration_task_status_path(manifest, 0).read_text(
            encoding="utf-8"
        )
    )
    finalizer = json.loads(
        metadata_migration_finalizer_status_path(manifest).read_text(
            encoding="utf-8"
        )
    )
    assert target["status"] == finalizer["status"] == "failed"
    assert finalizer["manifest_unreadable"] is True
    assert not generation_is_active(output_dir, generation)


def test_migration_semantic_manifest_failure_uses_script_fence(
    tmp_path: Path,
) -> None:
    """An empty readable target plan still publishes terminal failure."""
    from click.testing import CliRunner

    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        metadata_migration_finalizer_status_path,
        metadata_migration_task_status_path,
    )
    from phenotypic._cli._cli_recompile_metadata_migration_worker import main
    from phenotypic._cli._cli_slurm_lifecycle import (
        generation_is_active,
        initialize_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    generation = "migration-semantic-corruption"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    manifest = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / generation
        / "metadata_migration"
        / "plan"
        / "migration_plan.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"targets": []}), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "--task-manifest",
            str(manifest),
            "--task-index",
            "0",
            "--output-dir",
            str(output_dir),
            "--slurm-generation",
            generation,
            "--attempt-id",
            generation,
        ],
    )

    assert result.exit_code != 0
    target = json.loads(
        metadata_migration_task_status_path(manifest, 0).read_text(
            encoding="utf-8"
        )
    )
    finalizer = json.loads(
        metadata_migration_finalizer_status_path(manifest).read_text(
            encoding="utf-8"
        )
    )
    assert target["status"] == finalizer["status"] == "failed"
    assert finalizer["manifest_unreadable"] is False
    assert not generation_is_active(output_dir, generation)


def test_stale_ordinary_worker_cannot_write_shard_or_status(
    tmp_path: Path,
) -> None:
    """A superseded measurement worker performs no attempt mutation."""
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        recompile_task_status_path,
    )
    from phenotypic._cli._cli_recompile_worker import run_recompile_task
    from phenotypic._cli._cli_slurm_lifecycle import (
        SlurmGenerationInactiveError,
        deactivate_generation,
        initialize_slurm_lifecycle,
        load_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    old_generation = "old-worker"
    new_generation = "new-owner"
    initialize_slurm_lifecycle(
        output_dir, generation=old_generation, mode="recompile"
    )
    source = output_dir / "results" / "plate_a" / "measurements" / "img.parquet"
    _write_measurement(source, 1)
    manifest = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / old_generation
        / "task_manifest.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "measurements",
                        "shard_id": 0,
                        "files": [str(source)],
                        "include_dataset_column": True,
                        "slurm_generation": old_generation,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    deactivate_generation(output_dir, old_generation)
    initialize_slurm_lifecycle(
        output_dir, generation=new_generation, mode="recompile"
    )

    with pytest.raises(SlurmGenerationInactiveError):
        run_recompile_task(
            output_dir,
            manifest,
            0,
            slurm_generation=old_generation,
            attempt_id=old_generation,
        )

    assert not recompile_task_status_path(manifest, 0).exists()
    assert not (manifest.parent / "measurement_shards").exists()
    state = load_slurm_lifecycle(output_dir)
    assert state["generation"] == new_generation
    assert state["active"] is True


def test_stale_migration_worker_cannot_mutate_target_or_status(
    tmp_path: Path,
) -> None:
    """A superseded migration worker cannot touch its planned target."""
    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        generate_metadata_migration_slurm_scripts,
        metadata_migration_finalizer_status_path,
        metadata_migration_task_status_path,
        plan_metadata_schema_for_slurm_recompile,
    )
    from phenotypic._cli._cli_recompile_metadata_migration_worker import (
        run_metadata_migration_target,
    )
    from phenotypic._cli._cli_slurm_lifecycle import (
        SlurmGenerationInactiveError,
        deactivate_generation,
        initialize_slurm_lifecycle,
        load_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    old_generation = "old-migration"
    new_generation = "new-migration-owner"
    target = output_dir / "results" / "plate_a" / "hdf" / "img.h5"
    _write_legacy_hdf(target)
    initialize_slurm_lifecycle(
        output_dir, generation=old_generation, mode="recompile"
    )
    plan = plan_metadata_schema_for_slurm_recompile(
        output_dir, attempt_id=old_generation
    )
    assert generate_metadata_migration_slurm_scripts(
        plan,
        slurm_args={},
        array_limit=5,
        attempt_id=old_generation,
        slurm_generation=old_generation,
    )
    before = target.read_bytes()
    deactivate_generation(output_dir, old_generation)
    initialize_slurm_lifecycle(
        output_dir, generation=new_generation, mode="recompile"
    )

    with pytest.raises(SlurmGenerationInactiveError):
        run_metadata_migration_target(
            plan.manifest_path,
            0,
            output_dir=output_dir,
            slurm_generation=old_generation,
            attempt_id=old_generation,
        )

    assert target.read_bytes() == before
    assert not metadata_migration_task_status_path(
        plan.manifest_path, 0
    ).exists()
    assert not metadata_migration_finalizer_status_path(
        plan.manifest_path
    ).exists()
    state = load_slurm_lifecycle(output_dir)
    assert state["generation"] == new_generation
    assert state["active"] is True


def test_superseded_finalizer_rechecks_before_master_publication(
    tmp_path: Path,
) -> None:
    """Ownership is checked again after status waiting and before publish."""
    import phenotypic._cli._cli_recompile_worker as worker

    from phenotypic._cli._cli_slurm_lifecycle import (
        SlurmGenerationInactiveError,
        deactivate_generation,
        initialize_slurm_lifecycle,
        load_slurm_lifecycle,
    )
    from phenotypic.sdk_ import master_measurements_csv_path

    output_dir = tmp_path / "out"
    old_generation = "old-finalizer"
    new_generation = "new-finalizer-owner"
    initialize_slurm_lifecycle(
        output_dir, generation=old_generation, mode="recompile"
    )
    manifest = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / old_generation
        / "task_manifest.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "finalize",
                        "expected_non_finalizer_tasks": 0,
                        "slurm_generation": old_generation,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_measurement(
        manifest.parent / "measurement_shards" / "shard_0.parquet", 1
    )

    def supersede_while_finalizer_waits(*_args: object) -> list[dict[str, object]]:
        deactivate_generation(output_dir, old_generation)
        initialize_slurm_lifecycle(
            output_dir, generation=new_generation, mode="recompile"
        )
        return []

    with (
        patch.object(
            worker,
            "_wait_for_non_finalizer_statuses",
            side_effect=supersede_while_finalizer_waits,
        ),
        pytest.raises(SlurmGenerationInactiveError),
    ):
        worker.run_recompile_task(
            output_dir,
            manifest,
            0,
            slurm_generation=old_generation,
            attempt_id=old_generation,
        )

    assert not master_measurements_csv_path(output_dir).exists()
    assert not (manifest.parent / "status" / "task_0.json").exists()
    state = load_slurm_lifecycle(output_dir)
    assert state["generation"] == new_generation
    assert state["active"] is True


def test_stale_migration_finalizer_cannot_publish_or_deactivate_new_owner(
    tmp_path: Path,
) -> None:
    """A superseded migration finalizer leaves its status absent."""
    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        generate_metadata_migration_slurm_scripts,
        metadata_migration_finalizer_status_path,
        plan_metadata_schema_for_slurm_recompile,
    )
    from phenotypic._cli._cli_recompile_metadata_migration_worker import (
        finalize_metadata_migration,
    )
    from phenotypic._cli._cli_slurm_lifecycle import (
        SlurmGenerationInactiveError,
        deactivate_generation,
        initialize_slurm_lifecycle,
        load_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    old_generation = "old-migration-finalizer"
    new_generation = "new-migration-finalizer-owner"
    _write_legacy_hdf(
        output_dir / "results" / "plate_a" / "hdf" / "img.h5"
    )
    initialize_slurm_lifecycle(
        output_dir, generation=old_generation, mode="recompile"
    )
    plan = plan_metadata_schema_for_slurm_recompile(
        output_dir, attempt_id=old_generation
    )
    assert generate_metadata_migration_slurm_scripts(
        plan,
        slurm_args={},
        array_limit=5,
        attempt_id=old_generation,
        slurm_generation=old_generation,
    )
    deactivate_generation(output_dir, old_generation)
    initialize_slurm_lifecycle(
        output_dir, generation=new_generation, mode="recompile"
    )

    with pytest.raises(SlurmGenerationInactiveError):
        finalize_metadata_migration(
            plan.manifest_path,
            output_dir=output_dir,
            slurm_generation=old_generation,
            attempt_id=old_generation,
        )

    assert not metadata_migration_finalizer_status_path(
        plan.manifest_path
    ).exists()
    state = load_slurm_lifecycle(output_dir)
    assert state["generation"] == new_generation
    assert state["active"] is True


def test_inflight_hdf_replacement_finishes_before_new_generation_starts(
    tmp_path: Path,
) -> None:
    """The HDF replace and lifecycle ownership change share one lock."""
    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        generate_metadata_migration_slurm_scripts,
        plan_metadata_schema_for_slurm_recompile,
    )
    from phenotypic._cli._cli_recompile_metadata_migration_worker import (
        run_metadata_migration_target,
    )
    from phenotypic._cli._cli_slurm_lifecycle import (
        SlurmGenerationInactiveError,
        deactivate_generation,
        initialize_slurm_lifecycle,
        load_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    old_generation = "inflight-hdf-old"
    new_generation = "inflight-hdf-new"
    target = output_dir / "results" / "plate_a" / "hdf" / "img.h5"
    _write_legacy_hdf(target)
    initialize_slurm_lifecycle(
        output_dir, generation=old_generation, mode="recompile"
    )
    plan = plan_metadata_schema_for_slurm_recompile(
        output_dir, attempt_id=old_generation
    )
    assert generate_metadata_migration_slurm_scripts(
        plan,
        slurm_args={},
        array_limit=5,
        attempt_id=old_generation,
        slurm_generation=old_generation,
        has_recompile_downstream=False,
    )

    replacement_entered = threading.Event()
    allow_replacement = threading.Event()
    supersession_started = threading.Event()
    new_generation_active = threading.Event()
    thread_errors: list[BaseException] = []
    real_replace = os.replace

    def guarded_replace(source: object, destination: object) -> None:
        destination_path = Path(os.fspath(destination)).resolve()
        if destination_path == target.resolve():
            replacement_entered.set()
            assert allow_replacement.wait(timeout=5)
            assert not new_generation_active.is_set()
        real_replace(source, destination)

    def run_old_worker() -> None:
        try:
            run_metadata_migration_target(
                plan.manifest_path,
                0,
                output_dir=output_dir,
                slurm_generation=old_generation,
                attempt_id=old_generation,
            )
        except SlurmGenerationInactiveError:
            # The target replacement may win while its later attempt-status
            # publication loses to the new generation. Both outcomes are safe.
            pass
        except BaseException as exc:  # pragma: no cover - assertion aid
            thread_errors.append(exc)

    def supersede_generation() -> None:
        try:
            supersession_started.set()
            assert deactivate_generation(output_dir, old_generation)
            initialize_slurm_lifecycle(
                output_dir, generation=new_generation, mode="recompile"
            )
            new_generation_active.set()
        except BaseException as exc:  # pragma: no cover - assertion aid
            thread_errors.append(exc)

    with patch(
        "phenotypic.sdk_._metadata_migration.os.replace",
        side_effect=guarded_replace,
    ):
        worker_thread = threading.Thread(target=run_old_worker)
        worker_thread.start()
        assert replacement_entered.wait(timeout=5)
        owner_thread = threading.Thread(target=supersede_generation)
        owner_thread.start()
        assert supersession_started.wait(timeout=5)
        assert not new_generation_active.wait(timeout=0.05)
        allow_replacement.set()
        worker_thread.join(timeout=10)
        owner_thread.join(timeout=10)

    assert not worker_thread.is_alive()
    assert not owner_thread.is_alive()
    assert not thread_errors
    assert new_generation_active.is_set()
    with h5py.File(target, "r") as handle:
        attrs = handle["public_metadata"].attrs
        assert attrs["Metadata_Strain"] == "S288C"
        assert "MetadataGenetic_Strain" not in attrs
    state = load_slurm_lifecycle(output_dir)
    assert state["generation"] == new_generation
    assert state["active"] is True


def test_inflight_finalizer_cannot_publish_after_new_generation_starts(
    tmp_path: Path,
) -> None:
    """Master publication completes first, then stale post output is fenced."""
    import phenotypic._cli._cli_recompile_worker as worker

    from phenotypic._cli._cli_slurm_lifecycle import (
        SlurmGenerationInactiveError,
        deactivate_generation,
        initialize_slurm_lifecycle,
        load_slurm_lifecycle,
    )
    from phenotypic.sdk_ import (
        master_measurements_csv_path,
        master_measurements_parquet_path,
    )

    output_dir = tmp_path / "out"
    old_generation = "inflight-finalizer-old"
    new_generation = "inflight-finalizer-new"
    initialize_slurm_lifecycle(
        output_dir, generation=old_generation, mode="recompile"
    )
    manifest = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / old_generation
        / "task_manifest.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "finalize",
                        "expected_non_finalizer_tasks": 0,
                        "dataset_names": ["plate_a"],
                        "slurm_generation": old_generation,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_measurement(
        manifest.parent / "measurement_shards" / "shard_0.parquet", 1
    )

    master_write_entered = threading.Event()
    allow_master_write = threading.Event()
    new_generation_active = threading.Event()
    pipeline_load_entered = threading.Event()
    thread_errors: list[BaseException] = []
    real_atomic_write = worker.atomic_write_with_writer

    def blocking_atomic_write(
        path: Path, writer: Callable[[Path], None]
    ) -> None:
        if Path(path) == master_measurements_csv_path(output_dir):
            master_write_entered.set()
            assert allow_master_write.wait(timeout=5)
            assert not new_generation_active.is_set()
        real_atomic_write(path, writer)

    def wait_for_new_owner(_output_dir: Path) -> object:
        pipeline_load_entered.set()
        assert new_generation_active.wait(timeout=5)
        return object()

    def run_old_finalizer() -> None:
        try:
            worker.run_recompile_task(
                output_dir,
                manifest,
                0,
                slurm_generation=old_generation,
                attempt_id=old_generation,
            )
        except SlurmGenerationInactiveError:
            pass
        except BaseException as exc:  # pragma: no cover - assertion aid
            thread_errors.append(exc)

    def supersede_generation() -> None:
        try:
            assert deactivate_generation(output_dir, old_generation)
            initialize_slurm_lifecycle(
                output_dir, generation=new_generation, mode="recompile"
            )
            new_generation_active.set()
        except BaseException as exc:  # pragma: no cover - assertion aid
            thread_errors.append(exc)

    with (
        patch.object(
            worker,
            "atomic_write_with_writer",
            side_effect=blocking_atomic_write,
        ),
        patch(
            "phenotypic._cli._cli_output_manager._load_pipeline_from_output_dir",
            side_effect=wait_for_new_owner,
        ),
        patch(
            "phenotypic._cli._cli_output_manager.finalize_post_master_outputs"
        ) as finalize_post,
        patch(
            "phenotypic._cli._dashboard.regenerate_dashboard_artifacts"
        ) as dashboard,
    ):
        worker_thread = threading.Thread(target=run_old_finalizer)
        worker_thread.start()
        assert master_write_entered.wait(timeout=5)
        owner_thread = threading.Thread(target=supersede_generation)
        owner_thread.start()
        assert not new_generation_active.wait(timeout=0.05)
        allow_master_write.set()
        assert pipeline_load_entered.wait(timeout=5)
        worker_thread.join(timeout=10)
        owner_thread.join(timeout=10)

    assert not worker_thread.is_alive()
    assert not owner_thread.is_alive()
    assert not thread_errors
    assert master_measurements_csv_path(output_dir).is_file()
    assert master_measurements_parquet_path(output_dir).is_file()
    finalize_post.assert_not_called()
    dashboard.assert_not_called()
    assert not (manifest.parent / "status" / "task_0.json").exists()
    state = load_slurm_lifecycle(output_dir)
    assert state["generation"] == new_generation
    assert state["active"] is True


def test_inflight_measurement_worker_cannot_stage_shard_for_new_owner(
    tmp_path: Path,
) -> None:
    """Attempt shards are fenced even when superseded after aggregation."""
    import phenotypic._cli._cli_recompile_worker as worker

    from phenotypic._cli._cli_slurm_lifecycle import (
        SlurmGenerationInactiveError,
        deactivate_generation,
        initialize_slurm_lifecycle,
    )

    output_dir = tmp_path / "out"
    old_generation = "inflight-shard-old"
    new_generation = "inflight-shard-new"
    source = output_dir / "results" / "plate_a" / "measurements" / "img.parquet"
    _write_measurement(source, 1)
    initialize_slurm_lifecycle(
        output_dir, generation=old_generation, mode="recompile"
    )
    manifest = (
        progress_dir(output_dir)
        / "recompile"
        / "attempts"
        / old_generation
        / "task_manifest.json"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_type": "measurements",
                        "shard_id": 0,
                        "files": [str(source)],
                        "include_dataset_column": True,
                        "slurm_generation": old_generation,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    aggregation_entered = threading.Event()
    allow_aggregation = threading.Event()
    thread_errors: list[BaseException] = []

    def delayed_aggregation(**_kwargs: object) -> pl.DataFrame:
        aggregation_entered.set()
        assert allow_aggregation.wait(timeout=5)
        return pl.DataFrame({"Size_Area": [1]})

    def run_old_worker() -> None:
        try:
            worker.run_recompile_task(
                output_dir,
                manifest,
                0,
                slurm_generation=old_generation,
                attempt_id=old_generation,
            )
        except SlurmGenerationInactiveError:
            pass
        except BaseException as exc:  # pragma: no cover - assertion aid
            thread_errors.append(exc)

    with patch(
        "phenotypic._cli._cli_parquet_agg.aggregate_parquet_files",
        side_effect=delayed_aggregation,
    ):
        worker_thread = threading.Thread(target=run_old_worker)
        worker_thread.start()
        assert aggregation_entered.wait(timeout=5)
        assert deactivate_generation(output_dir, old_generation)
        initialize_slurm_lifecycle(
            output_dir, generation=new_generation, mode="recompile"
        )
        allow_aggregation.set()
        worker_thread.join(timeout=10)

    assert not worker_thread.is_alive()
    assert not thread_errors
    assert not (manifest.parent / "measurement_shards").exists()
    assert not (manifest.parent / "status" / "task_0.json").exists()


def test_generated_ordinary_and_migration_scripts_execute_required_args(
    tmp_path: Path,
) -> None:
    """Rendered batch scripts parse and run with the independent fences."""
    from phenotypic._cli._cli_recompile_metadata_migration_slurm import (
        generate_metadata_migration_slurm_scripts,
        metadata_migration_finalizer_status_path,
        plan_metadata_schema_for_slurm_recompile,
    )
    from phenotypic._cli._cli_recompile_slurm_scripts import (
        build_recompile_tasks,
        generate_recompile_slurm_scripts,
        recompile_task_status_path,
    )
    from phenotypic._cli._cli_slurm_lifecycle import initialize_slurm_lifecycle

    ordinary_output = tmp_path / "ordinary"
    ordinary_generation = "execute-ordinary-script"
    _write_measurement(
        ordinary_output
        / "results"
        / "plate_a"
        / "measurements"
        / "img.parquet",
        1,
    )
    initialize_slurm_lifecycle(
        ordinary_output,
        generation=ordinary_generation,
        mode="recompile",
    )
    ordinary_tasks = build_recompile_tasks(
        ordinary_output,
        ["plate_a"],
        include_dataset_column=True,
        overlay_alpha=0.3,
        shard_size=10,
        attempt_id=ordinary_generation,
    )
    ordinary_scripts = generate_recompile_slurm_scripts(
        ordinary_tasks,
        ordinary_output,
        slurm_args={},
        array_limit=10,
        attempt_id=ordinary_generation,
    )
    ordinary_script = ordinary_scripts[0]
    ordinary_text = ordinary_script.read_text(encoding="utf-8")
    assert "+    --" not in ordinary_text
    assert f"--slurm-generation {ordinary_generation}" in ordinary_text
    assert f"--attempt-id {ordinary_generation}" in ordinary_text
    ordinary_env = dict(os.environ)
    ordinary_env.update(
        {"SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_ID": "local-test"}
    )
    ordinary_result = subprocess.run(
        ["bash", str(ordinary_script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=ordinary_env,
    )
    assert ordinary_result.returncode == 0, ordinary_result.stderr
    ordinary_manifest = (
        progress_dir(ordinary_output)
        / "recompile"
        / "attempts"
        / ordinary_generation
        / "task_manifest.json"
    )
    assert json.loads(
        recompile_task_status_path(ordinary_manifest, 0).read_text(
            encoding="utf-8"
        )
    )["status"] == "completed"

    migration_output = tmp_path / "migration"
    migration_generation = "execute-migration-script"
    migration_target = (
        migration_output / "results" / "plate_a" / "hdf" / "img.h5"
    )
    _write_legacy_hdf(migration_target)
    initialize_slurm_lifecycle(
        migration_output,
        generation=migration_generation,
        mode="recompile",
    )
    migration_plan = plan_metadata_schema_for_slurm_recompile(
        migration_output, attempt_id=migration_generation
    )
    generated = generate_metadata_migration_slurm_scripts(
        migration_plan,
        slurm_args={},
        array_limit=10,
        attempt_id=migration_generation,
        slurm_generation=migration_generation,
        has_recompile_downstream=False,
    )
    assert generated is not None
    migration_scripts = [
        *generated.shard_scripts,
        generated.finalizer_script,
    ]
    for script in migration_scripts:
        script_text = script.read_text(encoding="utf-8")
        assert "+    --" not in script_text
        assert f"--slurm-generation {migration_generation}" in script_text
        assert f"--attempt-id {migration_generation}" in script_text

    migration_env = dict(os.environ)
    migration_env.update(
        {"SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_ID": "local-test"}
    )
    migration_result = subprocess.run(
        ["bash", str(generated.shard_scripts[0])],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=migration_env,
    )
    assert migration_result.returncode == 0, migration_result.stderr
    finalizer_result = subprocess.run(
        ["bash", str(generated.finalizer_script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, SLURM_JOB_ID="local-test"),
    )
    assert finalizer_result.returncode == 0, finalizer_result.stderr
    assert json.loads(
        metadata_migration_finalizer_status_path(
            migration_plan.manifest_path
        ).read_text(encoding="utf-8")
    )["status"] == "completed"
    with h5py.File(migration_target, "r") as handle:
        attrs = handle["public_metadata"].attrs
        assert attrs["Metadata_Strain"] == "S288C"
        assert "MetadataGenetic_Strain" not in attrs


def test_waiter_fails_when_dynamic_dispatch_marks_attempt_terminal(
    tmp_path: Path,
) -> None:
    """A missing finalizer cannot leave wait mode polling forever."""
    from phenotypic._cli._cli_slurm_lifecycle import (
        initialize_slurm_lifecycle,
        mark_generation_failed,
    )
    from phenotypic.phenotypicCLI import (
        _wait_for_metadata_migration_finalizer_status,
        _wait_for_recompile_finalizer_status,
    )

    output_dir = tmp_path / "out"
    generation = "dynamic-dispatch-failure"
    initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="recompile"
    )
    mark_generation_failed(
        output_dir, generation, "dependent finalizer submission failed"
    )

    with pytest.raises(RuntimeError, match="finalizer submission failed"):
        _wait_for_recompile_finalizer_status(
            output_dir,
            9,
            recompile_finalizer_status_path=tmp_path / "missing-final.json",
            slurm_generation=generation,
            poll_interval=0,
        )
    with pytest.raises(RuntimeError, match="finalizer submission failed"):
        _wait_for_metadata_migration_finalizer_status(
            tmp_path / "missing-migration-final.json",
            output_dir=output_dir,
            slurm_generation=generation,
            poll_interval=0,
        )
