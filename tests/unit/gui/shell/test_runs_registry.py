"""Unit tests for ``phenotypic.gui.shell._runs_registry``.

Coverage:

    * Basic CRUD: register / get / list / update_status / remove.
    * Concurrent updates serialise via the registry's :class:`threading.Lock`
      — many threads racing on ``update_status`` produce a deterministic
      final state.
    * ``rehydrate_from_sandbox`` walks a fake CLI-output layout and
      registers a record per discovered output dir.
    * Status comes from ``resolve_run_state``; mode and the SLURM job-id
      hint come from ``job_metadata.json``, with sane fallbacks.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

import pytest

import phenotypic.gui.shell._runs_registry as runs_registry_module
from phenotypic._cli._cli_update_state import append_event
from phenotypic.gui._config import DELIVERABLES_DIRNAME
from phenotypic.gui.shell._runs_registry import (
    RunRecord,
    RunRegistry,
)
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.sdk_ import (
    event_log_path,
    job_metadata_path,
    manifest_json_path,
    run_completion_marker_path,
    slurm_lifecycle_path,
    terminal_failures_jsonl_path,
)
from tests._output_layout import build_complete_run, build_incomplete_run


def _write_master_marker(out: Path) -> None:
    """Drop an empty ``deliverables/master_measurements.parquet`` marker.

    The shell classifier identifies a CLI output by this file (under
    ``deliverables/``) plus a root-level ``results/`` dir.
    """
    deliverables = out / DELIVERABLES_DIRNAME
    deliverables.mkdir(parents=True, exist_ok=True)
    (deliverables / "master_measurements.parquet").write_bytes(b"")


def _write_local_terminal_manifest(
    output: Path,
    *,
    start_time: float | str,
    gui_generation: object | None = None,
    is_complete: bool = True,
    completed: int = 1,
    failed: int = 0,
    total: int = 1,
    execution_mode: str = "local",
) -> None:
    """Write canonical local publication evidence for lifecycle tests."""
    path = manifest_json_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    start_text = (
        start_time
        if isinstance(start_time, str)
        else datetime.fromtimestamp(start_time).isoformat(
            timespec="milliseconds"
        )
    )
    payload = {
        "execution_mode": execution_mode,
        "start_time": start_text,
        "is_complete": is_complete,
        "completed": completed,
        "failed": failed,
        "total_images": total,
    }
    if gui_generation is not None:
        payload["gui_record_generation"] = str(gui_generation)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def test_register_get_list_remove(tmp_path: Path) -> None:
    reg = RunRegistry()
    rec = RunRecord(
        run_id="r1",
        mode="local",
        output_dir=tmp_path / "r1",
        rel_path="r1",
    )
    reg.register(rec)
    assert reg.get("r1") is rec
    assert reg.list() == [rec]
    assert reg.remove("r1") is True
    assert reg.get("r1") is None
    assert reg.remove("r1") is False  # idempotent


def test_register_replaces_on_same_id(tmp_path: Path) -> None:
    reg = RunRegistry()
    a = RunRecord(run_id="x", mode="local", output_dir=tmp_path, rel_path="x")
    b = RunRecord(run_id="x", mode="slurm", output_dir=tmp_path, rel_path="x")
    reg.register(a)
    reg.register(b)
    assert reg.get("x") is b


def test_update_status_returns_false_for_unknown() -> None:
    reg = RunRegistry()
    assert reg.update_status("missing", "complete") is False


def test_update_pid_and_slurm_job_id(tmp_path: Path) -> None:
    reg = RunRegistry()
    reg.register(
        RunRecord(
            run_id="r", mode="local",
            output_dir=tmp_path, rel_path="r",
        )
    )
    assert reg.update_pid("r", 4242) is True
    assert reg.get("r").pid == 4242  # type: ignore[union-attr]
    assert reg.update_slurm_job_id("r", "8675309") is True
    assert reg.get("r").slurm_job_id == "8675309"  # type: ignore[union-attr]


def test_allocate_persists_durable_generation_owner(tmp_path: Path) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="sha256:abc",
        status="running",
    )

    owner_path = (
        output / ".phenotypic" / "progress" / "gui_launch_owner.json"
    )
    payload = json.loads(owner_path.read_text(encoding="utf-8"))
    assert payload["generation"] == str(record.generation)
    assert payload["run_id"] == "run"
    assert payload["rel_path"] == "run"
    assert payload["command_digest"] == "sha256:abc"
    assert payload["lifecycle_epoch"] == str(record.generation)
    assert payload["status"] == "running"
    assert reg.revision == 1


def test_allocate_rejects_second_nonterminal_generation(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    first = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="one",
        status="running",
    )

    try:
        with pytest.raises(RuntimeError, match="nonterminal"):
            reg.allocate(
                mode="slurm",
                output_dir=output,
                rel_path="run",
                command_digest="two",
            )
    finally:
        reg.compare_and_set(
            "run",
            first.generation,  # type: ignore[arg-type]
            status="cancelled",
        )


def test_allocate_rejects_durable_owner_from_another_registry(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    first_registry = RunRegistry()
    first_registry.allocate(
        mode="slurm",
        output_dir=output,
        rel_path="run",
        command_digest="one",
        status="queued",
    )

    second_registry = RunRegistry()
    with pytest.raises(RuntimeError, match="durable nonterminal"):
        second_registry.allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="two",
        )
    assert second_registry.list() == []


def test_allocate_refuses_invalid_existing_owner(tmp_path: Path) -> None:
    output = tmp_path / "run"
    owner_path = (
        output / ".phenotypic" / "progress" / "gui_launch_owner.json"
    )
    owner_path.parent.mkdir(parents=True)
    owner_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid generation owner"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_two_registries_atomically_compete_for_one_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    registries = (RunRegistry(), RunRegistry())
    barrier = threading.Barrier(2)
    successes: list[RunRecord] = []
    failures: list[BaseException] = []
    result_lock = threading.Lock()

    def _allocate(registry: RunRegistry, digest: str) -> None:
        barrier.wait()
        try:
            record = registry.allocate(
                mode="local",
                output_dir=output,
                rel_path="run",
                command_digest=digest,
                status="running",
            )
            with result_lock:
                successes.append(record)
        except BaseException as exc:
            with result_lock:
                failures.append(exc)

    threads = [
        threading.Thread(target=_allocate, args=(registry, f"digest-{index}"))
        for index, registry in enumerate(registries)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    payload = json.loads(
        (
            output
            / ".phenotypic"
            / "progress"
            / "gui_launch_owner.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["generation"] == str(successes[0].generation)


# ---------------------------------------------------------------------------
# Claimability -- P6 Task 4
#
# One `resolve_run_state` call replaced three conflict predicates. The trees
# below are built by `tests._output_layout`'s real publishers rather than by
# hand-written JSON: the predicates these tests retired were pinned by
# hand-built `processing_state.json` fixtures carrying a `datasets.*.completed`
# shape P3 stopped writing, so those fixtures had already stopped describing
# any tree this build produces.
# ---------------------------------------------------------------------------

_RETIRED_CLAIMABILITY_MEMBERS = (
    "_processing_state_conflict",
    "_publication_evidence_conflict",
    "_orchestration_state_conflict",
    "_latest_event_states",
    "_read_status_from_manifest",
    "_string_set",
)


def test_the_retired_claimability_predicates_are_gone() -> None:
    """§11: three conflict predicates + two readers -> one resolve call.

    ``_string_set`` goes with them: it validated the
    ``datasets.*.{completed,failed}`` lists only ``_processing_state_conflict``
    read, and P3 stopped writing those keys.

    Fires if any of the six is reintroduced under its old name on the class or
    at module scope -- which is what a partial revert looks like.
    """
    for gone in _RETIRED_CLAIMABILITY_MEMBERS:
        assert not hasattr(RunRegistry, gone), gone
        assert not hasattr(runs_registry_module, gone), gone


def test_a_finished_run_can_be_claimed(tmp_path: Path) -> None:
    """A run whose proof covers its current inventory is not in anyone's way."""
    output = build_complete_run(tmp_path)

    record = RunRegistry().allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
    )
    assert record.generation is not None


def test_an_empty_directory_can_be_claimed(tmp_path: Path) -> None:
    """The existence gate's whole job.

    ``resolve_run_state`` reports ``incomplete`` for a directory holding no
    run, which is the same verdict it reports for an unfinished one -- so a
    claimability rule reading the verdict alone would refuse every fresh
    output directory the GUI ever creates.
    """
    output = tmp_path / "fresh"
    output.mkdir()

    record = RunRegistry().allocate(
        mode="local",
        output_dir=output,
        rel_path="fresh",
        command_digest="digest",
    )
    assert record.generation is not None


def test_an_unfinished_run_refuses_the_claim(tmp_path: Path) -> None:
    """One image published, one not: launching here would overwrite it."""
    output = build_incomplete_run(tmp_path)

    with pytest.raises(
        RuntimeError, match="incompatible non-GUI processing state"
    ):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_a_failed_image_refuses_the_claim(tmp_path: Path) -> None:
    """The terminal-failure journal is the failure authority (§4.1).

    A failure leaves no artifact, so it cannot be derived from the tree --
    which is why the retired predicate had to read a demoted
    ``datasets.*.failed`` list to see one at all.
    """
    output = build_incomplete_run(tmp_path)
    terminal_failures_jsonl_path(output).parent.mkdir(
        parents=True, exist_ok=True
    )
    terminal_failures_jsonl_path(output).write_text(
        json.dumps(
            {"work_id": "work-b", "exception_type": "SegmentationError"}
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match="failed non-GUI processing state"
    ):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_a_run_in_flight_refuses_the_claim(tmp_path: Path) -> None:
    """Rule 2 of the verdict ladder, reached through claimability.

    An active SLURM lifecycle fence at or above the run's restart epoch is a
    liveness authority; launching a second run into the output it is writing
    is the thing this gate exists to prevent.
    """
    output = build_incomplete_run(tmp_path)
    slurm_lifecycle_path(output).parent.mkdir(parents=True, exist_ok=True)
    slurm_lifecycle_path(output).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": "gen-1",
                "active": True,
                "restart_epoch": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="non-GUI run in flight"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_a_staged_orchestration_record_alone_refuses_the_claim(
    tmp_path: Path,
) -> None:
    """The second half of the existence gate.

    A staged controller writes ``staged_orchestration.json`` before its first
    stage writes any processing state, so a claim taken in that window would
    land on top of a controller that is mid-launch. Deriving "is there a run
    here" from ``RunState.identity`` would miss it: this tree yields no
    identity at all.
    """
    output = tmp_path / "run"
    state_path = (
        output / ".phenotypic" / "progress" / "staged_orchestration.json"
    )
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({"epoch": "other", "phase": "stage2"}),
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError, match="incompatible non-GUI processing state"
    ):
        RunRegistry().allocate(
            mode="slurm",
            output_dir=output,
            rel_path="run",
            command_digest="digest",
        )


def test_a_failed_event_in_the_log_does_not_gate_the_claim(
    tmp_path: Path,
) -> None:
    """Audit S5 / §4.2: one append-only log now has one parser.

    ``_latest_event_states`` was a **second** parser of
    ``processing_events.log``, and it differed from
    ``aggregate_state_from_events`` -- no inventory fence -- so the GUI and
    the CLI could read one file and disagree. The log is demoted here to a
    debugging artifact: nothing on this path reads it, so neither a ``failed``
    event nor a line no parser can read changes the answer.

    Fires if any replay is reintroduced: a reader would see ``b.tif`` failed
    and refuse the claim this test requires to succeed.
    """
    output = build_complete_run(tmp_path)
    append_event(event_log_path(output), "plate", "b.tif", "failed")
    with event_log_path(output).open("a", encoding="utf-8") as handle:
        handle.write("this line is not an event\n")

    record = RunRegistry().allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
    )
    assert record.generation is not None


def test_allocate_replaces_terminal_generation(tmp_path: Path) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    first = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="one",
        status="complete",
    )
    second = reg.allocate(
        mode="slurm",
        output_dir=output,
        rel_path="run",
        command_digest="two",
    )
    assert second.generation != first.generation
    assert reg.get("run") is second


def test_compare_and_set_rejects_stale_generation_without_revision_bump(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    revision = reg.revision
    assert (
        reg.compare_and_set(
            "run",
            uuid4(),
            status="failed",
            returncode=9,
        )
        is False
    )
    assert reg.revision == revision
    assert record.status == "running"
    assert record.returncode is None


def test_compare_and_set_updates_generalized_fields_and_aliases(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="slurm",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
    )
    submitted = datetime.now(timezone.utc)
    log_paths = (tmp_path / "submit.log", tmp_path / "slurm.log")
    assert record.generation is not None
    assert reg.compare_and_set(
        "run",
        record.generation,
        expected_statuses={"submitting"},
        expected_record_revision=0,
        status="queued",
        scheduler_ids=("22", "11", "22"),
        primary_scheduler_id="11",
        log_paths=log_paths,
        submitted_at=submitted,
        status_detail="waiting for resources",
    )
    updated = reg.get("run")
    assert updated is not None
    assert updated is not record
    assert record.status == "submitting"
    assert record.record_revision == 0
    assert updated.status == "queued"
    assert updated.scheduler_ids == ("22", "11")
    assert updated.primary_scheduler_id == "11"
    assert updated.slurm_job_id == "11"
    assert updated.log_paths == log_paths
    assert updated.log_path == log_paths[0]
    assert updated.submitted_at == submitted
    assert updated.record_revision == 1
    assert reg.revision == 2

    payload = json.loads(
        (
            updated.output_dir
            / ".phenotypic"
            / "progress"
            / "gui_launch_owner.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["scheduler_ids"] == ["22", "11"]
    assert payload["primary_scheduler_id"] == "11"
    assert payload["record_revision"] == 1


def test_compare_and_set_write_failure_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    owner_path = (
        record.output_dir
        / ".phenotypic"
        / "progress"
        / "gui_launch_owner.json"
    )
    owner_before = owner_path.read_bytes()
    registry_revision = reg.revision

    def _fail_write(*_args, **_kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(
        runs_registry_module,
        "atomic_write_json",
        _fail_write,
    )
    with pytest.raises(OSError, match="disk full"):
        reg.compare_and_set(
            "run",
            record.generation,
            status="failed",
            returncode=7,
        )

    assert reg.get("run") is record
    assert record.status == "running"
    assert record.returncode is None
    assert record.record_revision == 0
    assert reg.revision == registry_revision
    assert owner_path.read_bytes() == owner_before


def test_compare_and_set_rejects_durable_revision_changed_by_other_registry(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    first_registry = RunRegistry()
    first = first_registry.allocate(
        mode="slurm",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert first.generation is not None

    stale_registry = RunRegistry()
    stale_registry.rehydrate_from_sandbox(
        SandboxRoot.from_path(tmp_path)
    )
    stale = stale_registry.get("run")
    assert stale is not None
    assert stale.record_revision == 0

    assert first_registry.compare_and_set(
        "run",
        first.generation,
        status="complete",
    )
    assert (
        stale_registry.compare_and_set(
            "run",
            first.generation,
            status="failed",
        )
        is False
    )
    assert stale_registry.get("run") is stale
    assert stale.status == "running"
    payload = json.loads(
        (
            output
            / ".phenotypic"
            / "progress"
            / "gui_launch_owner.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "complete"
    assert payload["record_revision"] == 1


def test_compare_and_set_honors_expected_record_revision(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
    )
    assert record.generation is not None
    assert (
        reg.compare_and_set(
            "run",
            record.generation,
            expected_record_revision=7,
            status="running",
        )
        is False
    )
    assert record.status == "submitting"
    assert record.record_revision == 0


def test_observe_local_exit_maps_nonzero_and_rejects_stale_generation(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    assert reg.observe_local_exit("run", uuid4(), 0) is False
    assert record.status == "running"
    assert reg.observe_local_exit("run", record.generation, 3) is True
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert updated.returncode == 3
    assert updated.terminal_at is not None
    assert updated.status_detail == "local process exited with status 3"


def test_observe_validate_zero_exit_completes_without_publication(
    tmp_path: Path,
) -> None:
    """A dry-run validates configuration and intentionally publishes no output."""
    reg = RunRegistry()
    record = reg.allocate(
        mode="validate",
        output_dir=tmp_path / "validate",
        rel_path="validate",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None

    assert reg.observe_local_exit("validate", record.generation, 0)
    updated = reg.get("validate")
    assert updated is not None
    assert updated.status == "complete"
    assert updated.returncode == 0
    assert updated.status_detail is None


def test_observe_local_zero_exit_fails_without_canonical_manifest(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert updated.returncode == 0
    assert "no canonical terminal publication evidence" in (
        updated.status_detail or ""
    )


def _write_local_completion_marker(output: Path, generation: object) -> None:
    """Write exact successful local generation evidence."""
    path = run_completion_marker_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "generation": str(generation),
                "mode": "local",
                "status": "complete",
                "finalizer_succeeded": True,
            }
        ),
        encoding="utf-8",
    )


def test_observe_local_zero_exit_ignores_legacy_shadow_manifest(
    tmp_path: Path,
) -> None:
    """A legacy manifest cannot shadow missing current-generation evidence."""
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    legacy_path = output / "progress" / "manifest.json"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps(
            {
                "execution_mode": "local",
                "start_time": datetime.fromtimestamp(
                    record.started_at + 1.0
                ).isoformat(timespec="milliseconds"),
                "is_complete": True,
                "completed": 1,
                "failed": 0,
                "total_images": 1,
            }
        ),
        encoding="utf-8",
    )

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert "no canonical terminal publication evidence" in (
        updated.status_detail or ""
    )


def test_observe_local_zero_exit_rejects_incomplete_manifest(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=record.started_at + 1.0,
        gui_generation=record.generation,
        is_complete=False,
        completed=0,
        total=1,
    )

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert "publication is incomplete" in (updated.status_detail or "")


def test_observe_local_zero_exit_rejects_preexisting_complete_manifest(
    tmp_path: Path,
) -> None:
    """A prior run's complete manifest cannot terminalize a new generation."""
    reg = RunRegistry()
    output = tmp_path / "run"
    prior_generation = uuid4()
    _write_local_terminal_manifest(
        output,
        start_time=datetime.now().timestamp() - 60.0,
        gui_generation=prior_generation,
    )
    _write_local_completion_marker(output, prior_generation)
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert "different launch generation" in (
        updated.status_detail or ""
    )


@pytest.mark.parametrize(
    "fold_timestamp",
    [
        "2025-11-02T01:30:00-07:00",
        "2025-11-02T01:30:00-08:00",
    ],
)
def test_exact_generation_evidence_is_dst_fold_independent(
    tmp_path: Path,
    fold_timestamp: str,
) -> None:
    """Both repeated wall times accept the same exact-generation contract."""
    reg = RunRegistry()
    output = tmp_path / fold_timestamp[-6:].replace(":", "")
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path=output.name,
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=fold_timestamp,
        gui_generation=record.generation,
    )
    _write_local_completion_marker(output, record.generation)

    assert reg.observe_local_exit(record.run_id, record.generation, 0)
    updated = reg.get(record.run_id)
    assert updated is not None
    assert updated.status == "complete"


def test_cross_timezone_future_manifest_cannot_satisfy_new_generation(
    tmp_path: Path,
) -> None:
    """A copied artifact's wall time cannot override its stale generation."""
    reg = RunRegistry()
    output = tmp_path / "run"
    prior_generation = uuid4()
    _write_local_terminal_manifest(
        output,
        start_time="2099-01-01T00:00:00+14:00",
        gui_generation=prior_generation,
    )
    _write_local_completion_marker(output, prior_generation)
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert "different launch generation" in (updated.status_detail or "")


def test_observe_local_zero_exit_rejects_mismatched_completion_marker(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=record.started_at + 1.0,
        gui_generation=record.generation,
    )
    _write_local_completion_marker(output, uuid4())

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert "different launch generation" in (updated.status_detail or "")


def test_observe_local_zero_exit_rejects_manifest_without_exact_generation(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=record.started_at + 1.0,
    )

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "failed"
    assert updated.returncode == 0
    assert "manifest belongs to a different launch generation" in (
        updated.status_detail or ""
    )


def test_observe_local_zero_exit_accepts_matching_completion_marker(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    record = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="digest",
        status="running",
    )
    assert record.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=record.started_at + 1.0,
        gui_generation=record.generation,
    )
    _write_local_completion_marker(output, record.generation)

    assert reg.observe_local_exit("run", record.generation, 0)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "complete"
    assert updated.returncode == 0
    assert updated.status_detail is None


def test_stale_local_exit_cannot_terminalize_replacement_generation(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    output = tmp_path / "run"
    first = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="first",
        status="running",
    )
    assert first.generation is not None
    assert reg.compare_and_set(
        "run",
        first.generation,
        expected_statuses={"running"},
        status="failed",
    )
    replacement = reg.allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="replacement",
        status="running",
    )
    assert replacement.generation is not None
    _write_local_terminal_manifest(
        output,
        start_time=replacement.started_at + 1.0,
        gui_generation=replacement.generation,
    )
    _write_local_completion_marker(output, replacement.generation)

    assert reg.observe_local_exit("run", first.generation, 0) is False
    current = reg.get("run")
    assert current is not None
    assert current.generation == replacement.generation
    assert current.status == "running"

    assert reg.observe_local_exit("run", replacement.generation, 0)
    completed = reg.get("run")
    assert completed is not None
    assert completed.generation == replacement.generation
    assert completed.status == "complete"


def test_observe_local_exit_preserves_cancelled_status(tmp_path: Path) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="cancelled",
    )
    assert record.generation is not None
    assert reg.observe_local_exit("run", record.generation, -15)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "cancelled"
    assert updated.returncode == -15


def test_observe_local_exit_finishes_cancelling_as_cancelled(
    tmp_path: Path,
) -> None:
    reg = RunRegistry()
    record = reg.allocate(
        mode="local",
        output_dir=tmp_path / "run",
        rel_path="run",
        command_digest="digest",
        status="cancelling",
    )
    assert record.generation is not None
    assert reg.observe_local_exit("run", record.generation, -15)
    updated = reg.get("run")
    assert updated is not None
    assert updated.status == "cancelled"
    assert updated.status_detail is None


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_register_does_not_corrupt(tmp_path: Path) -> None:
    """Many threads racing on register produce a clean final state."""
    reg = RunRegistry()
    barrier = threading.Barrier(8)

    def _worker(i: int) -> None:
        barrier.wait()
        for j in range(20):
            run_id = f"w{i}-{j}"
            reg.register(
                RunRecord(
                    run_id=run_id,
                    mode="local",
                    output_dir=tmp_path / run_id,
                    rel_path=run_id,
                )
            )

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(reg.list()) == 8 * 20


def test_concurrent_update_status_is_serialised(tmp_path: Path) -> None:
    """Many threads racing on update_status leave the registry consistent.

    The final status is whichever update ran last, but the dict must not
    raise mid-iteration and ``list()`` must return a coherent snapshot.
    """
    reg = RunRegistry()
    reg.register(
        RunRecord(
            run_id="r", mode="local",
            output_dir=tmp_path, rel_path="r",
        )
    )
    statuses = ["running", "complete", "failed", "cancelled"]
    barrier = threading.Barrier(16)

    def _worker(s: str) -> None:
        barrier.wait()
        for _ in range(50):
            reg.update_status("r", s)

    threads = [
        threading.Thread(target=_worker, args=(statuses[i % 4],))
        for i in range(16)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    final = reg.get("r")
    assert final is not None
    assert final.status in statuses

    # Concurrent list() never raised.
    snapshots: List[int] = []
    for _ in range(20):
        snapshots.append(len(reg.list()))
    assert all(s == 1 for s in snapshots)


# ---------------------------------------------------------------------------
# rehydrate_from_sandbox
# ---------------------------------------------------------------------------

def _seed_discoverable(root: Path, name: str) -> Path:
    """Make ``<root>/<name>`` look like a CLI output to the sidebar classifier.

    The classifier keys on ``deliverables/master_measurements.parquet`` plus a
    root-level ``results/`` dir. Both are **created only if absent**: a tree
    built by ``build_complete_run`` already carries a real master whose digest
    the run proof binds, and stamping the empty marker over it would leave a
    fixture that is discoverable but no longer ``complete`` -- passing the
    discovery assertion while silently testing the wrong verdict.
    """
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    master = out / DELIVERABLES_DIRNAME / "master_measurements.parquet"
    if not master.exists():
        _write_master_marker(out)
    (out / "results").mkdir(exist_ok=True)
    return out


def _write_job_metadata(
    output: Path,
    *,
    execution_mode: str,
    chunk_job_ids: dict | None = None,
) -> None:
    """Write the CLI's submission record -- the mode + job-id source.

    ``job_metadata.json``, not ``manifest.json``: spec §4.2 demoted the
    dashboard manifest from evidence, and these two fields were only ever
    copied into it from here.
    """
    path = job_metadata_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"execution_mode": execution_mode}
    if chunk_job_ids is not None:
        payload["chunk_job_ids"] = chunk_job_ids
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_rehydrate_picks_up_cli_outputs(tmp_path: Path) -> None:
    """A finished run and an unfinished one, told apart by their proofs."""
    build_complete_run(tmp_path / "a")
    _seed_discoverable(tmp_path, "a/run")
    build_incomplete_run(tmp_path / "b")
    _seed_discoverable(tmp_path, "b/run")

    reg = RunRegistry()
    reg.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    statuses = {r.run_id: r.status for r in reg.list()}
    assert statuses == {"a/run": "complete", "b/run": "unknown"}


def test_rehydrate_marks_failed_from_the_terminal_journal(
    tmp_path: Path,
) -> None:
    """§4.1's failure authority, not a manifest ``failed`` count.

    The retired reader called a run failed when ``manifest.json`` said
    ``failed > 0``. That number is a report of what a run *said* happened; the
    journal is what it *recorded*, per work id, and survives a manifest that
    was never rewritten.
    """
    output = build_incomplete_run(tmp_path / "fr")
    terminal_failures_jsonl_path(output).parent.mkdir(
        parents=True, exist_ok=True
    )
    terminal_failures_jsonl_path(output).write_text(
        json.dumps({"work_id": "work-b", "exception_type": "OSError"}) + "\n",
        encoding="utf-8",
    )
    _seed_discoverable(tmp_path, "fr/run")

    reg = RunRegistry()
    reg.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    record = reg.get("fr/run")
    assert record is not None
    assert record.status == "failed"


def test_rehydrate_reports_a_live_run_as_running(tmp_path: Path) -> None:
    """The arm the manifest-count reader could not reach.

    Its own comment said progress counts cannot support a ``running`` claim
    after a GUI restart -- correct, and why it answered ``unknown`` for every
    run actually in flight. An active lifecycle fence can support that claim,
    so this row now says what is true.
    """
    output = build_incomplete_run(tmp_path / "live")
    slurm_lifecycle_path(output).parent.mkdir(parents=True, exist_ok=True)
    slurm_lifecycle_path(output).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation": "gen-1",
                "active": True,
                "restart_epoch": 0,
            }
        ),
        encoding="utf-8",
    )
    _seed_discoverable(tmp_path, "live/run")

    reg = RunRegistry()
    reg.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    record = reg.get("live/run")
    assert record is not None
    assert record.status == "running"
    assert record.generation is None
    assert "no GUI launch generation owns it" in (record.status_detail or "")


def test_rehydrate_extracts_slurm_job_id(tmp_path: Path) -> None:
    """The array-task suffix is dropped, exactly as before: ``12345_0`` -> ``12345``."""
    output = _seed_discoverable(tmp_path, "sr")
    _write_job_metadata(
        output,
        execution_mode="slurm",
        chunk_job_ids={"0": "12345_0", "1": "12345_1"},
    )

    reg = RunRegistry()
    reg.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    record = reg.get("sr")
    assert record is not None
    assert record.mode == "slurm"
    assert record.slurm_job_id == "12345"


def test_rehydrate_reads_local_mode_from_the_submission_record(
    tmp_path: Path,
) -> None:
    """A local run records its own mode; there is no job id to surface."""
    output = _seed_discoverable(tmp_path, "lr")
    _write_job_metadata(output, execution_mode="local")

    reg = RunRegistry()
    reg.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    record = reg.get("lr")
    assert record is not None
    assert record.mode == "local"
    assert record.slurm_job_id is None


def test_rehydrate_unknown_without_a_submission_record(
    tmp_path: Path,
) -> None:
    """``unknown``, deliberately not ``local``.

    ``resolve_execution_mode`` coerces a missing record to ``"local"``. Doing
    that here would label every foreign folder in a user's sandbox a local run
    of ours.
    """
    _seed_discoverable(tmp_path, "nm")

    reg = RunRegistry()
    reg.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    record = reg.get("nm")
    assert record is not None
    assert record.mode == "unknown"
    assert record.status == "unknown"
    assert "no observable nonterminal owner" in (record.status_detail or "")


def test_rehydrate_preserves_existing_records(tmp_path: Path) -> None:
    """A live run registered before scan must NOT be clobbered."""
    _seed_discoverable(tmp_path, "live")
    sandbox = SandboxRoot.from_path(tmp_path)
    reg = RunRegistry()
    pre = RunRecord(
        run_id="live",
        mode="local",
        output_dir=tmp_path / "live",
        rel_path="live",
        status="running",
        pid=999,
    )
    reg.register(pre)
    n = reg.rehydrate_from_sandbox(sandbox)
    assert n == 0  # didn't re-register
    assert reg.get("live") is pre  # same object


def test_rehydrate_ignores_a_corrupt_submission_record(
    tmp_path: Path,
) -> None:
    """A truncated write degrades to ``unknown`` rather than raising on boot."""
    output = _seed_discoverable(tmp_path, "broken")
    path = job_metadata_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    reg = RunRegistry()
    reg.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    record = reg.get("broken")
    assert record is not None
    assert record.mode == "unknown"
    assert record.status == "unknown"


def test_rehydrate_restores_persisted_generation_and_terminal_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "owned"
    source = RunRegistry()
    original = source.allocate(
        mode="local",
        output_dir=output,
        rel_path="owned",
        command_digest="digest",
        status="running",
    )
    assert original.generation is not None
    assert source.compare_and_set(
        "owned",
        original.generation,
        status="complete",
        returncode=0,
        log_paths=(output / ".gui_log" / "stdout.log",),
    )

    restored = RunRegistry()
    count = restored.rehydrate_from_sandbox(
        SandboxRoot.from_path(tmp_path)
    )
    assert count == 1
    record = restored.get("owned")
    assert record is not None
    assert record.generation == original.generation
    assert record.status == "complete"
    assert record.returncode == 0
    assert record.log_path == output / ".gui_log" / "stdout.log"
    assert record.record_revision == 1


def test_rehydrate_downgrades_unobserved_local_liveness(
    tmp_path: Path,
) -> None:
    output = tmp_path / "owned"
    source = RunRegistry()
    original = source.allocate(
        mode="local",
        output_dir=output,
        rel_path="owned",
        command_digest="digest",
        status="running",
    )
    assert original.generation is not None
    assert source.compare_and_set(
        "owned",
        original.generation,
        pid=12345,
    )

    restored = RunRegistry()
    restored.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))
    record = restored.get("owned")
    assert record is not None
    assert record.generation == original.generation
    assert record.status == "unknown"
    assert record.pid is None
    assert "restarted" in (record.status_detail or "")


def test_rehydrate_does_not_invent_a_generation(tmp_path: Path) -> None:
    """A finished run with no GUI owner record is still not GUI-owned."""
    build_complete_run(tmp_path / "hist")
    _seed_discoverable(tmp_path, "hist/run")

    restored = RunRegistry()
    restored.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    record = restored.get("hist/run")
    assert record is not None
    assert record.generation is None
    assert record.status == "complete"


def test_a_complete_manifest_no_longer_terminalizes_a_run(
    tmp_path: Path,
) -> None:
    """§4.2, at its bluntest.

    ``manifest.json`` claiming ``is_complete`` over a full inventory used to be
    enough to show ``complete`` in Recent Runs. It is a cache of what a run
    reported, it is never rewritten when the tree beneath it changes, and
    nothing now reads it here: an output whose only evidence is that file
    reads ``unknown``.
    """
    output = _seed_discoverable(tmp_path, "manifest-only")
    path = manifest_json_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "execution_mode": "local",
                "is_complete": True,
                "completed": 10,
                "failed": 0,
                "total_images": 10,
            }
        ),
        encoding="utf-8",
    )

    restored = RunRegistry()
    restored.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))

    record = restored.get("manifest-only")
    assert record is not None
    assert record.status == "unknown"
    assert record.mode == "unknown"
    assert "no observable nonterminal owner" in (record.status_detail or "")


def test_rehydrate_invalid_owner_does_not_invent_generation(
    tmp_path: Path,
) -> None:
    output = _seed_discoverable(tmp_path, "invalid-owner")
    owner_path = (
        output / ".phenotypic" / "progress" / "gui_launch_owner.json"
    )
    owner_path.parent.mkdir(parents=True, exist_ok=True)
    owner_path.write_text(
        json.dumps(
            {
                "version": 1,
                "run_id": "invalid-owner",
                "rel_path": "invalid-owner",
                "generation": "not-a-uuid",
            }
        ),
        encoding="utf-8",
    )

    restored = RunRegistry()
    restored.rehydrate_from_sandbox(SandboxRoot.from_path(tmp_path))
    record = restored.get("invalid-owner")
    assert record is not None
    assert record.generation is None


# ----------------------------------------------------------------------
# P6 Task 5 / DEFERRED D-2: a dead GUI must not own an output forever.
# ----------------------------------------------------------------------


def _a_dead_pid() -> int:
    """Return a pid no live process holds.

    Racy in principle -- the kernel could hand this number out between the
    probe and the assertion -- and safe in practice, because it starts above
    every pid currently allocated and pids are handed out ascending.
    """
    import psutil

    pid = max(psutil.pids()) + 1
    while psutil.pid_exists(pid):
        pid += 1
    return pid


def _abandoned_owner(
    output: Path,
    *,
    pid: int | None,
    mode: str = "local",
) -> RunRegistry:
    """Leave a nonterminal owner record on disk, written by the real writer.

    Deliberately not hand-written JSON. ``tests/unit/sdk_/test_run_state.py``
    hand-writes this record and says so, noting that P6 Task 5 "is where this
    record's reader and writer land in one place" -- so this is that place,
    and driving ``allocate`` / ``update_pid`` / ``update_status`` is what
    makes the fixture fail if the persisted schema ever moves.
    """
    registry = RunRegistry()
    record = registry.allocate(
        mode=mode,  # type: ignore[arg-type]
        output_dir=output,
        rel_path=output.name,
        command_digest="first",
    )
    if pid is not None:
        assert registry.update_pid(record.run_id, pid)
    assert registry.update_status(record.run_id, "running")
    return registry


def test_a_sigkilled_gui_does_not_lock_the_output_forever(
    tmp_path: Path,
) -> None:
    """DEFERRED D-2 / audit S7: the permanent dead end.

    Nothing in ``src/`` ever deletes or repairs ``gui_launch_owner.json``:
    ``remove`` and ``clear`` drop the in-memory record and leave the file,
    and ``rehydrate_from_sandbox`` downgrades with ``persist=False``. So the
    record a SIGKILLed GUI leaves behind said ``running`` forever and every
    later claim was refused, with no affordance anywhere in the UI to clear
    it. The record already stored the pid; nothing read it.
    """
    output = tmp_path / "run"
    _abandoned_owner(output, pid=_a_dead_pid())

    claimed = RunRegistry().allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="second",
    )
    assert claimed.generation is not None


def test_a_live_owner_still_refuses_the_claim(tmp_path: Path) -> None:
    """The liveness check must not become a rubber stamp.

    An owner whose process is alive still owns the output; releasing it
    would let two launches write one tree, which is the failure the refusal
    exists to prevent and is worse than the dead end it replaces.
    """
    import os

    output = tmp_path / "run"
    _abandoned_owner(output, pid=os.getpid())

    with pytest.raises(RuntimeError, match="durable nonterminal"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="second",
        )


def test_an_owner_carrying_no_pid_still_refuses_the_claim(
    tmp_path: Path,
) -> None:
    """Absence of evidence is not proof of death.

    A GUI killed between ``allocate`` and ``update_pid`` leaves a
    nonterminal record with ``pid: None``. Within one boot that is
    indistinguishable from a live owner, so it must keep refusing -- the
    repair is for records that are *provably* dead, and widening it to
    "cannot prove alive" would hand out an output someone is writing.
    """
    output = tmp_path / "run"
    _abandoned_owner(output, pid=None)

    with pytest.raises(RuntimeError, match="durable nonterminal"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="second",
        )


def test_a_slurm_owner_is_never_released_by_a_pid_probe(
    tmp_path: Path,
) -> None:
    """A scheduler job outlives the GUI that submitted it, by design.

    Its ``pid`` is legitimately ``None`` and its liveness belongs to the
    scheduler, so a pid probe can only ever be wrong about it. These are the
    same modes ``rehydrate_from_sandbox`` already excludes from its own
    downgrade.
    """
    output = tmp_path / "run"
    _abandoned_owner(output, pid=None, mode="slurm")

    with pytest.raises(RuntimeError, match="durable nonterminal"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="second",
        )


def test_a_record_predating_the_boot_is_released_without_any_pid(
    tmp_path: Path,
) -> None:
    """Nothing survives a reboot, so the pid stops mattering.

    This is the only arm that can retire a record carrying no pid at all,
    which is otherwise the one hole the repair cannot close.
    """
    from dataclasses import replace

    output = tmp_path / "run"
    registry = _abandoned_owner(output, pid=None)
    record = registry.list()[0]
    registry._persist_record_locked(
        replace(record, started_at=0.0, status="running", pid=None)
    )

    claimed = RunRegistry().allocate(
        mode="local",
        output_dir=output,
        rel_path="run",
        command_digest="second",
    )
    assert claimed.generation is not None


def test_releasing_a_dead_owner_reads_its_status_off_the_tree(
    tmp_path: Path,
) -> None:
    """A GUI killed *after* its child finished owns a complete run.

    Writing ``failed`` over that would put a second wrong answer where the
    first one was. The released record therefore carries the verdict the
    output's own evidence supports, and lands terminal -- ``unknown`` is
    nonterminal by ``run_status_is_nonterminal``, so a repair that wrote it
    would leave the dead end in place under a new name.
    """
    output = build_complete_run(tmp_path)
    _abandoned_owner(output, pid=_a_dead_pid())

    registry = RunRegistry()
    registry._assert_output_claimable_locked(
        output_dir=output,
        rel_path=output.name,
    )

    released = registry._read_owner_record(output, output.name)
    assert released is not None
    assert released.status == "complete"
    assert not runs_registry_module.run_status_is_nonterminal(
        released.status
    )
    assert released.pid is None


def test_the_registry_and_the_ladder_share_one_liveness_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two authorities for one fact is the defect this change removes.

    Spec §4.1 makes the owner record a liveness authority and P1 Task 5
    taught the Q2 ladder to believe it only while its pid is alive. A second
    probe here could disagree with the ladder about the same pid. Patching
    the ladder's probe must therefore move this verdict too -- if it does
    not, the registry grew its own.
    """
    output = tmp_path / "run"
    _abandoned_owner(output, pid=_a_dead_pid())

    monkeypatch.setattr(
        "phenotypic.sdk_._run_state._process_is_alive",
        lambda pid: True,
    )
    with pytest.raises(RuntimeError, match="durable nonterminal"):
        RunRegistry().allocate(
            mode="local",
            output_dir=output,
            rel_path="run",
            command_digest="second",
        )
