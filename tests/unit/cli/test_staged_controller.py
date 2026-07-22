"""Crash-boundary tests for staged SLURM controller orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from phenotypic._cli._cli_staged_controller import run_staged_controller
from phenotypic._cli._cli_staged_orchestration import (
    StagedManifestEntry,
    append_job_ledger,
    append_stage2_terminal_failure,
    cancel_staged_jobs,
    clear_stage2_sidecars,
    completed_inventory_images,
    current_slurm_job_id,
    initialize_orchestration,
    load_orchestration_state,
    quarantine_unchanged_restart_parquets,
    read_job_ledger,
    save_orchestration_state,
    snapshot_inventory_parquets,
    staged_completion_matches,
    staged_completion_path,
    submit_with_intent,
    terminal_stage2_identities,
    write_staged_manifest,
)
from phenotypic.sdk_ import atomic_write_json, dataset_hdf_dir, dataset_measurements_dir


def _controller_fixture(tmp_path: Path, epoch: str = "epoch-1") -> Path:
    manifest_path = tmp_path / "manifest.json"
    entries = [
        StagedManifestEntry("plate", "image.tif", "image", "/in/image.tif")
    ]
    write_staged_manifest(manifest_path, entries)
    hdf = dataset_hdf_dir(tmp_path, "plate") / "image.h5"
    hdf.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf, "w") as handle:
        handle.attrs["schema_version"] = 2
        layers = handle.create_group("layers")
        for name in ("gray", "detect_mat", "objmap"):
            layers.create_dataset(name, data=np.zeros((2, 2)))
    config_path = tmp_path / "controller.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "epoch": epoch,
                "output_dir": str(tmp_path),
                "resume": False,
                "manifest_path": str(manifest_path),
                "stage1_scripts": [str(tmp_path / "stage1.sh")],
                "stage2_script": str(tmp_path / "stage2.sh"),
                "stage3_scripts": [str(tmp_path / "stage3.sh")],
                "finalizer_script": str(tmp_path / "finalizer.sh"),
                "controller_script": str(tmp_path / "controller.sh"),
            }
        ),
        encoding="utf-8",
    )
    state = initialize_orchestration(
        tmp_path,
        epoch=epoch,
        mode="fresh",
        controller_config_path=config_path,
    )
    state.update({"phase": "stage2", "stage1_index": 1})
    from phenotypic._cli._cli_staged_orchestration import (
        save_orchestration_state,
    )

    save_orchestration_state(tmp_path, state)
    return config_path


def test_controller_allows_one_zero_progress_retry_then_advances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _controller_fixture(tmp_path)
    submitted: list[tuple[str, int]] = []

    def fake_submit(*args, **kwargs):
        role = kwargs["role"]
        round_index = kwargs["round_index"]
        submitted.append((role, round_index))
        return f"{role}-{len(submitted)}"

    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.submit_with_intent",
        fake_submit,
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.scheduler_job_is_active",
        lambda job_id: False,
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.update_job_dependency",
        lambda job_id, dependencies: True,
    )

    current = "100"
    for _ in range(3):
        monkeypatch.setenv("SLURM_JOB_ID", current)
        run_staged_controller(config_path)
        current_state = load_orchestration_state(tmp_path)
        assert current_state is not None
        current = str(current_state["expected_controller_id"])

    state = load_orchestration_state(tmp_path)
    assert state is not None
    assert state["round"] == 2
    assert state["zero_progress_rounds"] == 2
    assert state["phase"] == "stage3"
    assert [role for role, _ in submitted].count("stage2") == 2
    assert [role for role, _ in submitted].count("stage3") == 1
    assert "plate\0image.tif" in terminal_stage2_identities(
        tmp_path, "epoch-1"
    )


def test_controller_rearms_behind_discovered_active_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _controller_fixture(tmp_path)
    state = load_orchestration_state(tmp_path)
    assert state is not None
    state["active_job_id"] = "array-44"
    from phenotypic._cli._cli_staged_orchestration import (
        save_orchestration_state,
    )

    save_orchestration_state(tmp_path, state)
    monkeypatch.setenv("SLURM_JOB_ID", "200")
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.submit_with_intent",
        lambda *args, **kwargs: "controller-201",
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.scheduler_job_is_active",
        lambda job_id: job_id == "array-44",
    )
    dependencies: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.update_job_dependency",
        lambda job_id, deps: dependencies.append((job_id, list(deps))) or True,
    )

    run_staged_controller(config_path)

    assert dependencies == [("controller-201", ["200", "array-44"])]
    state = load_orchestration_state(tmp_path)
    assert state is not None and state["active_job_id"] == "array-44"


def test_unknown_scheduler_state_does_not_advance_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _controller_fixture(tmp_path)
    state = load_orchestration_state(tmp_path)
    assert state is not None
    state["active_job_id"] = "array-unknown"
    from phenotypic._cli._cli_staged_orchestration import save_orchestration_state

    save_orchestration_state(tmp_path, state)
    monkeypatch.setenv("SLURM_JOB_ID", "225")
    submitted_roles: list[str] = []

    def fake_submit(*args, **kwargs):
        submitted_roles.append(kwargs["role"])
        return "controller-226"

    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.submit_with_intent", fake_submit
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.scheduler_job_is_active",
        lambda job_id: None,
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.update_job_dependency",
        lambda job_id, dependencies: True,
    )

    run_staged_controller(config_path)

    assert submitted_roles == ["controller"]
    state = load_orchestration_state(tmp_path)
    assert state is not None
    assert state["active_job_id"] == "array-unknown"
    assert state["scheduler_query_failed"] is True


def test_dependency_update_failure_remains_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _controller_fixture(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "250")
    ids = iter(["controller-251", "array-252"])
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.submit_with_intent",
        lambda *args, **kwargs: next(ids),
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.scheduler_job_is_active",
        lambda job_id: False,
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.update_job_dependency",
        lambda job_id, dependencies: False,
    )

    run_staged_controller(config_path)

    state = load_orchestration_state(tmp_path)
    assert state is not None
    assert state["active_job_id"] == "array-252"
    assert state["expected_controller_id"] == "controller-251"
    assert state["dependency_update_failed"] is True


def test_stale_controller_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _controller_fixture(tmp_path, epoch="old")
    state = load_orchestration_state(tmp_path)
    assert state is not None
    state["epoch"] = "new"
    from phenotypic._cli._cli_staged_orchestration import (
        save_orchestration_state,
    )

    save_orchestration_state(tmp_path, state)
    monkeypatch.setenv("SLURM_JOB_ID", "300")
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.submit_with_intent",
        lambda *args, **kwargs: pytest.fail("stale controller submitted work"),
    )
    run_staged_controller(config_path)


def test_terminal_failures_are_namespaced_by_epoch(tmp_path: Path) -> None:
    entry = StagedManifestEntry("plate", "image.tif", "image", "/in/image.tif")
    append_stage2_terminal_failure(
        tmp_path,
        epoch="old",
        round_index=1,
        entry=entry,
        error_type="Failure",
        error_message="old failure",
    )
    assert terminal_stage2_identities(tmp_path, "new") == set()
    assert terminal_stage2_identities(tmp_path, "old") == {"plate\0image.tif"}


def test_submission_intent_discovers_job_after_recording_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_orchestration(
        tmp_path,
        epoch="epoch-1",
        mode="fresh",
        controller_config_path=tmp_path / "controller.json",
    )
    append_job_ledger(
        tmp_path,
        epoch="epoch-1",
        token="stage2-round-1",
        role="stage2",
        round_index=1,
        status="intent",
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_orchestration._job_from_scheduler_comment",
        lambda comment: "777",
    )
    job_id = submit_with_intent(
        tmp_path,
        epoch="epoch-1",
        token="stage2-round-1",
        role="stage2",
        round_index=1,
        script_path=tmp_path / "stage2.sh",
    )
    assert job_id == "777"
    assert read_job_ledger(tmp_path)[-1]["status"] == "submitted"


def test_duplicate_transition_uses_ledger_without_sbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_orchestration(
        tmp_path,
        epoch="epoch-1",
        mode="fresh",
        controller_config_path=tmp_path / "controller.json",
    )
    append_job_ledger(
        tmp_path,
        epoch="epoch-1",
        token="finalizer",
        role="finalizer",
        round_index=0,
        status="submitted",
        job_id="888",
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_orchestration._job_from_scheduler_comment",
        lambda comment: pytest.fail("scheduler discovery should not run"),
    )
    assert (
        submit_with_intent(
            tmp_path,
            epoch="epoch-1",
            token="finalizer",
            role="finalizer",
            round_index=0,
            script_path=tmp_path / "finalizer.sh",
        )
        == "888"
    )


def test_finalizer_without_completion_marker_marks_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _controller_fixture(tmp_path)
    state = load_orchestration_state(tmp_path)
    assert state is not None
    state.update({"phase": "finalizing", "active_job_id": "finalizer-1"})
    from phenotypic._cli._cli_staged_orchestration import save_orchestration_state

    save_orchestration_state(tmp_path, state)
    monkeypatch.setenv("SLURM_JOB_ID", "400")
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.submit_with_intent",
        lambda *args, **kwargs: "controller-401",
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.scheduler_job_is_active",
        lambda job_id: False,
    )

    run_staged_controller(config_path)

    state = load_orchestration_state(tmp_path)
    assert state is not None and state["phase"] == "failed"


def test_duplicate_controller_launches_finalizer_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _controller_fixture(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["stage3_scripts"] = []
    config_path.write_text(json.dumps(config), encoding="utf-8")
    from phenotypic._cli._cli_sidecar import sidecar_path

    sidecar = sidecar_path(tmp_path, "plate", "image")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.touch()
    submitted_roles: list[str] = []

    def fake_submit(*args, **kwargs):
        role = kwargs["role"]
        submitted_roles.append(role)
        return f"{role}-{len(submitted_roles)}"

    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.submit_with_intent", fake_submit
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.update_job_dependency",
        lambda job_id, dependencies: True,
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.scheduler_job_is_active",
        lambda job_id: job_id.startswith("finalizer"),
    )

    monkeypatch.setenv("SLURM_JOB_ID", "450")
    run_staged_controller(config_path)
    monkeypatch.setenv("SLURM_JOB_ID", "451")
    run_staged_controller(config_path)

    assert submitted_roles.count("finalizer") == 1
    assert submitted_roles.count("controller") == 1


def test_unexpected_duplicate_controller_does_not_spawn_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _controller_fixture(tmp_path)
    state = load_orchestration_state(tmp_path)
    assert state is not None
    state["expected_controller_id"] = "expected-700"
    from phenotypic._cli._cli_staged_orchestration import save_orchestration_state

    save_orchestration_state(tmp_path, state)
    monkeypatch.setenv("SLURM_JOB_ID", "duplicate-701")
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.submit_with_intent",
        lambda *args, **kwargs: pytest.fail("duplicate spawned a successor"),
    )

    run_staged_controller(config_path)


def test_ledgered_successor_adopts_ownership_after_state_write_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _controller_fixture(tmp_path)
    state = load_orchestration_state(tmp_path)
    assert state is not None
    state["expected_controller_id"] = "700"
    save_orchestration_state(tmp_path, state)
    append_job_ledger(
        tmp_path,
        epoch="epoch-1",
        token="controller-after-700",
        role="controller",
        round_index=0,
        status="submitted",
        job_id="701",
        dependencies=["700"],
    )
    submitted_roles: list[str] = []

    def fake_submit(*args, **kwargs):
        submitted_roles.append(kwargs["role"])
        return "702" if kwargs["role"] == "controller" else "703"

    monkeypatch.setenv("SLURM_JOB_ID", "701")
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.submit_with_intent", fake_submit
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.scheduler_job_is_active",
        lambda job_id: False,
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_controller.update_job_dependency",
        lambda job_id, dependencies: True,
    )

    run_staged_controller(config_path)

    assert submitted_roles == ["controller", "stage2"]
    state = load_orchestration_state(tmp_path)
    assert state is not None and state["expected_controller_id"] == "702"


def test_controller_uses_array_master_job_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "81234")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "81200")

    assert current_slurm_job_id() == "81200"


def test_restart_completion_requires_parquet_replacement(tmp_path: Path) -> None:
    parquet = dataset_measurements_dir(tmp_path, "plate") / "image.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"old")
    state = initialize_orchestration(
        tmp_path,
        epoch="restart-epoch",
        mode="restart",
        controller_config_path=tmp_path / "controller.json",
    )
    inventory = {"plate": ["image.tif"]}
    state["restart_parquet_fingerprints"] = snapshot_inventory_parquets(
        tmp_path, inventory
    )
    save_orchestration_state(tmp_path, state)

    assert completed_inventory_images(tmp_path, "plate", ["image.tif"]) == set()

    replacement = parquet.with_suffix(".replacement")
    replacement.write_bytes(b"current epoch")
    replacement.replace(parquet)
    assert completed_inventory_images(tmp_path, "plate", ["image.tif"]) == {
        "image.tif"
    }


def test_current_marker_contract_does_not_count_partial_parquet(
    tmp_path: Path,
) -> None:
    parquet = dataset_measurements_dir(tmp_path, "plate") / "image.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"partial")
    state = initialize_orchestration(
        tmp_path,
        epoch="current-epoch",
        mode="resume",
        controller_config_path=tmp_path / "controller.json",
    )
    state["stage3_markers_required"] = True
    save_orchestration_state(tmp_path, state)

    assert completed_inventory_images(tmp_path, "plate", ["image.tif"]) == set()

    from phenotypic._cli._cli_staged_resume import (
        write_stage3_completion_marker,
    )

    write_stage3_completion_marker(
        tmp_path, "plate", "image.tif", "image"
    )
    assert completed_inventory_images(tmp_path, "plate", ["image.tif"]) == {
        "image.tif"
    }


def test_restart_quarantines_unchanged_parquet_before_aggregation(
    tmp_path: Path,
) -> None:
    parquet = dataset_measurements_dir(tmp_path, "plate") / "image.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_bytes(b"old")
    state = initialize_orchestration(
        tmp_path,
        epoch="restart-epoch",
        mode="restart",
        controller_config_path=tmp_path / "controller.json",
    )
    state["restart_parquet_fingerprints"] = snapshot_inventory_parquets(
        tmp_path, {"plate": ["image.tif"]}
    )
    save_orchestration_state(tmp_path, state)

    assert quarantine_unchanged_restart_parquets(tmp_path, "restart-epoch") == 1
    assert not parquet.exists()
    assert (
        tmp_path
        / ".phenotypic"
        / "progress"
        / "restart_stale_parquets"
        / "plate"
        / "image.parquet"
    ).is_file()


def test_completion_marker_must_match_epoch(tmp_path: Path) -> None:
    atomic_write_json(staged_completion_path(tmp_path), {"epoch": "old"})

    assert not staged_completion_matches(tmp_path, "current")
    assert staged_completion_matches(tmp_path, "old")


def test_cancellation_fences_epoch_before_scancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _controller_fixture(tmp_path)
    append_job_ledger(
        tmp_path,
        epoch="epoch-1",
        token="stage2-round-1",
        role="stage2",
        round_index=1,
        status="submitted",
        job_id="501",
    )
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_orchestration.scheduler_job_is_active",
        lambda job_id: True,
    )
    observed_phase: list[str] = []

    def fake_run(command, **kwargs):
        state = load_orchestration_state(tmp_path)
        assert state is not None
        observed_phase.append(state["phase"])
        return None

    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_orchestration.subprocess.run", fake_run
    )
    assert config_path.is_file()
    assert cancel_staged_jobs(tmp_path) == ["501"]
    assert observed_phase == ["cancelled"]


def test_cancellation_after_failure_includes_reused_tokens_across_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _controller_fixture(tmp_path)
    from phenotypic._cli._cli_staged_orchestration import deactivate_orchestration

    for epoch, job_id in (("old-epoch", "601"), ("epoch-1", "602")):
        append_job_ledger(
            tmp_path,
            epoch=epoch,
            token="controller-initial",
            role="controller",
            round_index=0,
            status="submitted",
            job_id=job_id,
        )
    deactivate_orchestration(tmp_path, "failed")
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_orchestration.scheduler_job_is_active",
        lambda job_id: True,
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(
        "phenotypic._cli._cli_staged_orchestration.subprocess.run",
        lambda command, **kwargs: commands.append(command),
    )

    assert cancel_staged_jobs(tmp_path) == ["601", "602"]
    assert commands == [["scancel", "601", "602"]]


def test_restart_cleanup_removes_only_transient_sidecars(tmp_path: Path) -> None:
    sidecar = tmp_path / "results" / "plate" / "objmap" / "image.npy"
    partial = sidecar.parent / ".image.npy.deadbeef.tmp"
    parquet = tmp_path / "results" / "plate" / "measurements" / "image.parquet"
    sidecar.parent.mkdir(parents=True)
    parquet.parent.mkdir(parents=True)
    sidecar.touch()
    partial.touch()
    parquet.touch()

    assert clear_stage2_sidecars(tmp_path) == 2
    assert not sidecar.exists()
    assert not partial.exists()
    assert parquet.exists()
