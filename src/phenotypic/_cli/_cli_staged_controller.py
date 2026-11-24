"""Recoverable array-level controller for staged SLURM GPU runs.

Config version 2 (#282 spec §4.3) adds a ``phases`` list. A run with a
``GpuDetector`` has ``["stage1", "stage2", "stage3", "stage4"]`` when it
embeds on the GPU; a run without one has ``["stage3", "stage4"]``. A version-1
config reads as ``["stage1", "stage2", "stage3"]``, so an in-flight run
submitted before the bump is driven exactly as before.

After the last Stage-3 chunk, a config with ``"stage4"`` sweeps orphan
signals, classifies Stage 4 (:func:`_classify_stage4`) and submits GPU rounds
over a per-round worklist (:func:`_submit_stage4_round`) until nothing is
retryable or two rounds in a row change nothing; then the finalizer. A stall
ends unfinished (U7): the leftover images keep no record and no journal row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from phenotypic.sdk_ import dataset_measurements_dir, zarr_store_path
from phenotypic.sdk_._file_locking import exclusive_path_lock

from ._cli_reference import reference_digest_for
from ._cli_slurm_lifecycle import generation_publication_guard
from ._cli_stage2_token import stage2_result_replayable
from ._cli_staged_resume import (
    stage3_completion_exists,
    stage4_status,
    valid_stage1_store,
)
from ._cli_staged_orchestration import (
    StagedManifestEntry,
    assert_active_epoch,
    current_slurm_job_id,
    load_orchestration_state,
    load_staged_manifest,
    mark_job_observed_terminal,
    orchestration_lock_path,
    read_job_ledger,
    retryable_digest,
    save_orchestration_state,
    scheduler_job_is_active,
    deactivated_epoch_phase,
    staged_completion_matches,
    staged_manifest_republishes,
    submit_with_intent,
    update_job_dependency,
    write_stage4_worklist,
)
from ._cli_failure_tracker import (
    read_terminal_failures,
    terminal_failure_work_ids_under_epoch,
)

logger = logging.getLogger(__name__)


def _is_ledgered_successor(
    output_dir: Path, epoch: str, job_id: str, predecessor_id: str
) -> bool:
    """Return whether the ledger proves *job_id* succeeds *predecessor_id*."""
    token = f"controller-after-{predecessor_id}"
    return any(
        row.get("status") == "submitted"
        and row.get("role") == "controller"
        and row.get("token") == token
        and str(row.get("job_id")) == job_id
        for row in read_job_ledger(output_dir, epoch=epoch)
    )


#: A version-1 config's phases: every staged run before Stage 4 existed.
_V1_PHASES = ("stage1", "stage2", "stage3")
#: The phases a version-2 config may list, in run order.
_KNOWN_PHASES = ("stage1", "stage2", "stage3", "stage4")
#: Keys a version-2 config must carry when it lists ``"stage4"``.
_STAGE4_CONFIG_KEYS = ("stage4_script", "stage4_worklist", "n_shards")


def _load_controller_config(path: Path) -> dict[str, Any]:
    """Read a version-1 or version-2 controller config; v1 gains its implied phases.

    Raises:
        ValueError: On another version, a ``phases`` list that is not an
            in-order subset of :data:`_KNOWN_PHASES` containing ``"stage3"``,
            or a ``"stage4"`` config without :data:`_STAGE4_CONFIG_KEYS`.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") not in (1, 2):
        raise ValueError(f"Unsupported staged controller config: {path}")
    if payload["version"] == 1:
        payload["phases"] = list(_V1_PHASES)
        return payload
    phases = payload.get("phases")
    if (
        not isinstance(phases, list)
        or "stage3" not in phases
        or phases != [phase for phase in _KNOWN_PHASES if phase in phases]
        or len(set(phases)) != len(phases)
    ):
        raise ValueError(
            f"Staged controller config {path} has unsupported phases {phases!r}"
        )
    if "stage4" in phases:
        missing = [key for key in _STAGE4_CONFIG_KEYS if key not in payload]
        if missing:
            raise ValueError(
                f"Staged controller config {path} lists stage4 without "
                f"{', '.join(missing)}"
            )
    return payload


def _phases(config: dict[str, Any]) -> list[str]:
    return list(config.get("phases", _V1_PHASES))


def _controller_commit_guard(output_dir: Path, epoch: str):
    """The epoch's publication guard, for the controller's own deletions."""

    @contextmanager
    def _guard() -> Iterator[None]:
        with generation_publication_guard(output_dir, epoch):
            assert_active_epoch(output_dir, epoch)
            yield

    return _guard


def _classify_stage2(
    config: dict[str, Any],
    entries: Sequence[StagedManifestEntry],
    round_index: int,
) -> tuple[list[StagedManifestEntry], list[StagedManifestEntry]]:
    """Return retryable and terminal entries from current artifact truth."""
    del round_index  # retained in the private signature for test compatibility
    output_dir = Path(config["output_dir"])
    epoch = str(config["epoch"])
    resume = bool(config.get("resume", False))
    markers_required = bool(config.get("stage3_markers_required", True))
    # The recovery controller has no ``StagePlan`` -- it runs from a JSON
    # config, not a pipeline. The submitter, which does have one, recorded the
    # slot for it in ``staged_controller.json``. Read with ``[]``, not
    # ``.get``: a config without it is a submitter/controller version skew, and
    # silently probing some other slot would make every image look un-done and
    # resubmit the whole GPU sweep.
    slot = str(config["detector_slot"])
    failed = {
        record.work_id
        for record in read_terminal_failures(output_dir)
        if record.lifecycle_epoch == epoch
    }
    retryable: list[StagedManifestEntry] = []
    terminal: list[StagedManifestEntry] = []
    for entry in entries:
        parquet = (
            dataset_measurements_dir(output_dir, entry.dataset)
            / f"{entry.stem}.parquet"
        )
        # BOTH halves: a token whose raw array is gone is not a finished
        # Stage 2, and skipping it here is what would strand the image --
        # nothing else in the SLURM path routes it back (ledger FLOW-17/M7).
        if stage2_result_replayable(
            output_dir, entry.dataset, entry.stem, slot
        ):
            continue
        if stage3_completion_exists(
            output_dir, entry.dataset, entry.stem
        ) or (resume and not markers_required and parquet.is_file()):
            continue
        if entry.work_id in failed:
            terminal.append(entry)
            continue
        store = zarr_store_path(output_dir, entry.dataset, entry.stem)
        if not valid_stage1_store(store):
            terminal.append(entry)
            continue
        # Re-planned since submission: every stage refuses this entry's pin,
        # and the pin cannot change within the run, so another GPU round would
        # refuse again (review M3). Done for this run; nothing is recorded, so
        # the next invocation re-plans it under a new work-id.
        if (
            reference_digest_for(output_dir, entry.dataset, entry.stem)
            != entry.reference_digest
        ):
            terminal.append(entry)
            continue
        retryable.append(entry)
    return retryable, terminal


def _prearm_successor(
    config: dict[str, Any], state: dict[str, Any], current_job_id: str
) -> str:
    output_dir = Path(config["output_dir"])
    epoch = str(config["epoch"])
    round_index = int(state.get("round", 0))
    token = f"controller-after-{current_job_id}"
    successor = submit_with_intent(
        output_dir,
        epoch=epoch,
        token=token,
        role="controller",
        round_index=round_index,
        script_path=Path(config["controller_script"]),
        dependencies=[current_job_id],
    )
    state["expected_controller_id"] = successor
    save_orchestration_state(output_dir, state)
    return successor


def _retarget_successor(
    state: dict[str, Any],
    successor_id: str,
    current_job_id: str,
    work_job_id: str,
) -> None:
    if not update_job_dependency(successor_id, [current_job_id, work_job_id]):
        state["dependency_update_failed"] = True


def _submit_stage2_round(
    config: dict[str, Any],
    state: dict[str, Any],
    successor_id: str,
    current_job_id: str,
    retryable: Sequence[StagedManifestEntry],
) -> None:
    output_dir = Path(config["output_dir"])
    epoch = str(config["epoch"])
    next_round = int(state.get("round", 0)) + 1
    digest, count = retryable_digest(retryable)
    previous_digest = state.get("last_retryable_digest")
    if previous_digest == digest:
        zero_progress = int(state.get("zero_progress_rounds", 0)) + 1
    else:
        zero_progress = 0

    if zero_progress >= 2:
        state["last_retryable_digest"] = digest
        state["last_retryable_count"] = count
        state["zero_progress_rounds"] = zero_progress
        state["phase"] = "stage3_starting"
        save_orchestration_state(output_dir, state)
        _submit_next_stage3_or_finalizer(
            config, state, successor_id, current_job_id
        )
        return

    token = f"stage2-round-{next_round}"
    job_id = submit_with_intent(
        output_dir,
        epoch=epoch,
        token=token,
        role="stage2",
        round_index=next_round,
        script_path=Path(config["stage2_script"]),
    )
    state.update(
        {
            "phase": "stage2",
            "round": next_round,
            "active_job_id": job_id,
            "last_retryable_digest": digest,
            "last_retryable_count": count,
            "zero_progress_rounds": zero_progress,
            "dependency_update_failed": False,
        }
    )
    _retarget_successor(state, successor_id, current_job_id, job_id)
    save_orchestration_state(output_dir, state)


def _submit_next_stage3_or_finalizer(
    config: dict[str, Any],
    state: dict[str, Any],
    successor_id: str,
    current_job_id: str,
) -> None:
    output_dir = Path(config["output_dir"])
    epoch = str(config["epoch"])
    scripts = [Path(path) for path in config["stage3_scripts"]]
    index = int(state.get("stage3_index", 0))
    if index < len(scripts):
        token = f"stage3-chunk-{index}"
        job_id = submit_with_intent(
            output_dir,
            epoch=epoch,
            token=token,
            role="stage3",
            round_index=index,
            script_path=scripts[index],
        )
        state.update(
            {
                "phase": "stage3",
                "stage3_index": index + 1,
                "active_job_id": job_id,
                "dependency_update_failed": False,
            }
        )
        _retarget_successor(state, successor_id, current_job_id, job_id)
        save_orchestration_state(output_dir, state)
        return

    if _resubmit_dead_tasks(config, state, successor_id, current_job_id, "stage3"):
        return
    if "stage4" in _phases(config):
        _advance_stage4(config, state, successor_id, current_job_id)
        return
    _submit_finalizer(config, state, successor_id, current_job_id, index)


#: The array name every image-stage chunk script maps its task ids through
#: (``SlurmArrayScriptSpec``'s default).
_TASK_INDICES_ARRAY = "TASK_INDICES"


def _chunk_task_indices(script: Path) -> list[int]:
    """The absolute manifest index of each array task of one Stage-1/3 chunk script.

    Read from the script's own ``TASK_INDICES=( ... )`` block, the mapping its
    tasks use (``_write_stage_script``), so a resubmission's ``--array``
    names exactly the tasks of the images it retries.

    Raises:
        ValueError: The script is unreadable or carries no such block.
    """
    try:
        lines = Path(script).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read chunk script {script}: {exc}") from exc
    try:
        start = lines.index(f"{_TASK_INDICES_ARRAY}=(") + 1
        end = lines.index(")", start)
        return [int(line.strip()) for line in lines[start:end] if line.strip()]
    except ValueError as exc:
        raise ValueError(f"chunk script {script} has no {_TASK_INDICES_ARRAY} block") from exc


def _stage_done_or_terminal(
    config: dict[str, Any], entry: StagedManifestEntry, failed: set[str]
) -> bool:
    """Common to both image stages: journalled under this epoch, published, or re-planned."""
    from ._cli_completion import valid_image_success

    output_dir = Path(config["output_dir"])
    if entry.work_id in failed:
        return True
    # Re-planned since submission: every stage refuses the pin (see
    # `_classify_stage2`), so a resubmission would refuse it again.
    if (
        reference_digest_for(output_dir, entry.dataset, entry.stem)
        != entry.reference_digest
    ):
        return True
    # Last, as the one that hashes artifacts: only images the cheaper checks
    # left (a few dead ones, not the run) reach it.
    return bool(
        entry.work_id
        and valid_image_success(
            output_dir, dataset=entry.dataset, image_stem=entry.stem, work_id=entry.work_id
        )
    )


def _dead_stage1_indices(
    config: dict[str, Any], entries: Sequence[StagedManifestEntry]
) -> list[int]:
    """Manifest indices whose Stage-1 task left nothing: no valid store and no journal row.

    A task killed by the host (OOM, node failure) writes neither, so it is
    node state, not a verdict on the image (U15/D5, smoke F2); a scientific
    failure is journalled, and stays terminal.
    """
    from ._cli_staged_resume import staged_store_matches_work_id

    output_dir = Path(config["output_dir"])
    failed = terminal_failure_work_ids_under_epoch(output_dir, str(config["epoch"]))
    dead: list[int] = []
    for index, entry in enumerate(entries):
        store = zarr_store_path(output_dir, entry.dataset, entry.stem)
        finished = (
            staged_store_matches_work_id(store, entry.work_id)
            if entry.work_id
            else valid_stage1_store(store)
        )
        if finished or _stage_done_or_terminal(config, entry, failed):
            continue
        dead.append(index)
    return dead


def _dead_stage3_indices(
    config: dict[str, Any], entries: Sequence[StagedManifestEntry]
) -> list[int]:
    """Manifest indices whose Stage-3 task left nothing: no marker, record or journal row.

    The ``"staged"`` role (a ``GpuDetector`` run) can retry only an image
    whose Stage-2 signal is still replayable: without it Stage 3 has no input,
    and Stage 2's own rounds already ended. The ``"single_pass"`` role runs
    the whole core, so it needs none. Either way an image Stage 4 owns
    (``stages_1_to_3_done``) is not Stage 3's (U15/D5, smoke F2).
    """
    from ._cli_staged_resume import stages_1_to_3_done

    output_dir = Path(config["output_dir"])
    failed = terminal_failure_work_ids_under_epoch(output_dir, str(config["epoch"]))
    if config.get("stage3_role") == "measure":
        return _dead_measure_indices(config, entries, failed)
    staged = config.get("stage3_role", "staged") != "single_pass"
    resume = bool(config.get("resume", False))
    markers_required = bool(config.get("stage3_markers_required", True))
    dead: list[int] = []
    for index, entry in enumerate(entries):
        if stage3_completion_exists(output_dir, entry.dataset, entry.stem):
            continue
        if entry.work_id and stages_1_to_3_done(
            output_dir, entry.dataset, entry.stem, entry.work_id
        ):
            continue
        parquet = dataset_measurements_dir(output_dir, entry.dataset) / f"{entry.stem}.parquet"
        if resume and not markers_required and parquet.is_file():
            continue
        if _stage_done_or_terminal(config, entry, failed):
            continue
        if staged and not stage2_result_replayable(
            output_dir, entry.dataset, entry.stem, str(config["detector_slot"])
        ):
            continue
        dead.append(index)
    return dead


def _dead_measure_indices(
    config: dict[str, Any], entries: Sequence[StagedManifestEntry], failed: set[str]
) -> list[int]:
    """The measure role's dead tasks (U15, P6 T2, U18).

    An image with a forward-published record, no journal row under this
    epoch, and no measure-done marker of this epoch
    (:func:`~phenotypic._cli._cli_staged_resume.write_measure_done_marker`)
    reads ``not_ready`` in the republish sense: its measure step was killed
    before it finished. The marker, not the run folder, is the evidence: with
    the forward pipeline the folder exists from the forward run, so a step
    killed before its table rewrite would otherwise look finished, and Stage
    4 would certify the un-re-measured table (P6 review I1). An image this
    pass already republished reads ``done`` and is not dead.
    """
    from ._cli_staged_resume import stage4_status

    output_dir = Path(config["output_dir"])
    dead: list[int] = []
    for index, entry in enumerate(entries):
        if entry.work_id in failed:
            continue
        status = stage4_status(
            output_dir, entry.dataset, entry.stem, entry.work_id, republish=True,
            pipeline_sha256=config.get("pipeline_sha256"), epoch=str(config["epoch"]),
        )
        if status == "not_ready" and _has_published_record(output_dir, entry):
            dead.append(index)
    return dead


def unfinished_measure_entries(
    output_dir: Path,
    entries: Sequence[StagedManifestEntry],
    *,
    pipeline_sha256: str | None,
    epoch: str,
) -> list[StagedManifestEntry]:
    """The images a SLURM ``--mode measure`` pass leaves unfinished (#282 U16).

    Each manifest image with a forward-published record that this pass
    neither republished (``stage4_status(republish=True, epoch=epoch) ==
    "done"``) nor journalled under *epoch*: a Stage-4 stall (U7), a measure
    step killed and given up on (U15, U18), or one Stage 4 never reached.
    Such an image still holds its forward record, which verifies, so the run
    state cannot see it; the measure finalizer asks this instead, and any
    entry here ends the pass unfinished (``phenotypic-wait`` exit 5). An
    image with no forward record is not this pass's to republish, and the
    forward run's own state accounts for it.
    """
    from ._cli_staged_resume import stage4_status

    failed = terminal_failure_work_ids_under_epoch(output_dir, epoch)
    unfinished: list[StagedManifestEntry] = []
    for entry in entries:
        if entry.work_id in failed or not _has_published_record(output_dir, entry):
            continue
        status = stage4_status(
            output_dir, entry.dataset, entry.stem, entry.work_id, republish=True,
            pipeline_sha256=pipeline_sha256, epoch=epoch,
        )
        if status != "done":
            unfinished.append(entry)
    return unfinished


def measure_pass_unfinished(output_dir: Path, epoch: str) -> list[StagedManifestEntry]:
    """:func:`unfinished_measure_entries` over the pass's own manifest, for the finalizer (U16).

    The manifest and the measure pipeline's sha256 are the controller
    config's, which the orchestration state names.

    Raises:
        RuntimeError: The state names no readable controller config, so the
            finalizer cannot tell what the pass covered; it fails closed.
    """
    state = load_orchestration_state(output_dir) or {}
    path = state.get("controller_config")
    try:
        config = _load_controller_config(Path(str(path)))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Staged controller config {path!r} is unreadable; cannot tell which "
            "images this measure pass covered"
        ) from exc
    return unfinished_measure_entries(
        output_dir,
        load_staged_manifest(Path(config["manifest_path"])),
        pipeline_sha256=config.get("pipeline_sha256"),
        epoch=epoch,
    )


def unfinished_measure_line(count: int) -> str:
    """The one line the measure finalizer logs, and records on the fence, for :func:`unfinished_measure_entries`."""
    return (
        f"{count} image(s) were neither re-measured and re-embedded by this "
        "measure pass nor recorded as failed (named below); the pass ends "
        "unfinished. Read the measure-step and stage4 logs for why, then "
        "re-run the same command."
    )


def _has_published_record(output_dir: Path, entry: StagedManifestEntry) -> bool:
    """Whether *entry*'s image has a record with ``artifacts`` (it was forward-published)."""
    from phenotypic.sdk_._image_record import read_image_record

    record = read_image_record(output_dir, entry.dataset, entry.stem)
    artifacts = record.get("artifacts") if record is not None else None
    return isinstance(artifacts, dict) and bool(artifacts)


def _resubmit_dead_tasks(
    config: dict[str, Any],
    state: dict[str, Any],
    successor_id: str,
    current_job_id: str,
    stage: str,
) -> bool:
    """Resubmit the Stage-1 or Stage-3 tasks that died leaving nothing; whether one was (U15/D5).

    Called once *stage*'s chunks are all submitted and the last has ended.
    The tasks whose images have no result, record or journal row
    (:func:`_dead_stage1_indices`, :func:`_dead_stage3_indices`) are
    resubmitted, a chunk at a time and only those tasks (``--array=<ids>``
    on the command line overrides the chunk's ``#SBATCH --array``), so one
    work array is active at a time and a finished image is never re-run.

    A **round** walks every chunk that has such a task. It is bounded as
    Stage 2's and Stage 4's are, on its stage's own counters
    (``<stage>_retry_round``, ``<stage>_retry_chunk``,
    ``<stage>_last_retryable_digest``, ``<stage>_zero_progress_rounds``,
    all read with defaults, so a state written before them, a #281 run's
    included, starts at zero): before each round the dead set is digested,
    and the second unchanged digest in a row stops retrying. The leftover
    images then stay unrecorded, as a Stage-4 stall's do (U7), and the run
    reads unfinished (exit 5) with a log line naming them. A chunk script
    whose task map cannot be read is not retried (one warning), which is
    the behaviour before D5.
    """
    scripts = [Path(path) for path in config[f"{stage}_scripts"]]
    entries = load_staged_manifest(Path(config["manifest_path"]))
    classify = _dead_stage1_indices if stage == "stage1" else _dead_stage3_indices
    dead = set(classify(config, entries))
    chunk_key = f"{stage}_retry_chunk"
    task_maps: list[list[int]] = []
    for script in scripts:
        try:
            task_maps.append(_chunk_task_indices(script))
        except ValueError as exc:
            if dead:
                logger.warning("%s: not retrying its dead tasks (%s)", stage, exc)
            task_maps.append([])

    def _next_chunk(start: int) -> tuple[int, list[int]] | None:
        for index in range(start, len(scripts)):
            task_ids = [
                task
                for task, manifest_index in enumerate(task_maps[index])
                if manifest_index in dead
            ]
            if task_ids:
                return index, task_ids
        return None

    chunk = int(state.get(chunk_key, 0))
    found = _next_chunk(chunk) if chunk > 0 else None
    if found is None:
        # A new round: the previous one (if any) has walked every chunk.
        state[chunk_key] = 0
        if not _start_retry_round(config, state, stage, entries, dead):
            return False
        found = _next_chunk(0)
        if found is None:
            return False
    index, task_ids = found
    _submit_chunk_retry(
        config, state, successor_id, current_job_id, stage,
        scripts[index], index, task_ids,
    )
    return True


def _start_retry_round(
    config: dict[str, Any],
    state: dict[str, Any],
    stage: str,
    entries: Sequence[StagedManifestEntry],
    dead: set[int],
) -> bool:
    """Apply the zero-progress rule before a retry round; whether the round runs."""
    digest, count = retryable_digest([entries[index] for index in sorted(dead)])
    digest_key = f"{stage}_last_retryable_digest"
    zero_key = f"{stage}_zero_progress_rounds"
    if state.get(digest_key) == digest:
        zero_progress = int(state.get(zero_key, 0)) + 1
    else:
        zero_progress = 0
    state[digest_key] = digest
    state[zero_key] = zero_progress
    if not dead:
        return False
    if zero_progress >= 2:
        logger.warning(dead_task_stall_line(stage, count))
        save_orchestration_state(Path(config["output_dir"]), state)
        return False
    state[f"{stage}_retry_round"] = int(state.get(f"{stage}_retry_round", 0)) + 1
    return True


def _submit_chunk_retry(
    config: dict[str, Any],
    state: dict[str, Any],
    successor_id: str,
    current_job_id: str,
    stage: str,
    script: Path,
    chunk_index: int,
    task_ids: Sequence[int],
) -> None:
    """Submit *script* narrowed to *task_ids*, as ``<stage>-retry-<round>-chunk-<i>``."""
    output_dir = Path(config["output_dir"])
    retry_round = int(state.get(f"{stage}_retry_round", 1))
    logger.info(
        "%s: resubmitting %d task(s) of chunk %d that died without a result "
        "(retry round %d)", stage, len(task_ids), chunk_index, retry_round,
    )
    job_id = submit_with_intent(
        output_dir,
        epoch=str(config["epoch"]),
        token=f"{stage}-retry-{retry_round}-chunk-{chunk_index}",
        role=stage,
        round_index=retry_round,
        script_path=script,
        extra_args=[f"--array={','.join(str(task) for task in task_ids)}"],
    )
    state.update(
        {
            "phase": stage,
            f"{stage}_retry_chunk": chunk_index + 1,
            "active_job_id": job_id,
            "dependency_update_failed": False,
        }
    )
    _retarget_successor(state, successor_id, current_job_id, job_id)
    save_orchestration_state(output_dir, state)


def _submit_finalizer(
    config: dict[str, Any],
    state: dict[str, Any],
    successor_id: str,
    current_job_id: str,
    round_index: int,
) -> None:
    output_dir = Path(config["output_dir"])
    job_id = submit_with_intent(
        output_dir,
        epoch=str(config["epoch"]),
        token="finalizer",
        role="finalizer",
        round_index=round_index,
        script_path=Path(config["finalizer_script"]),
    )
    state.update(
        {
            "phase": "finalizing",
            "active_job_id": job_id,
            "dependency_update_failed": False,
        }
    )
    _retarget_successor(state, successor_id, current_job_id, job_id)
    save_orchestration_state(output_dir, state)


def _classify_stage4(
    config: dict[str, Any],
    entries: Sequence[StagedManifestEntry],
) -> tuple[list[StagedManifestEntry], list[StagedManifestEntry]]:
    """``(retryable, terminal)`` Stage-4 entries from current artifact truth (spec §4.3).

    - done (in neither list): ``stage4_status`` is ``"done"``, its record
      is valid;
    - terminal: a terminal failure journalled under **this** epoch; or
      ``"not_ready"`` (no ``stages.stage3``, a store that does not match the
      work id, or no measurement table), which another GPU round cannot fix;
      or a reference digest changed since submission (:func:`_classify_stage2`'s
      rule): every stage refuses the entry's pin, so a round would too;
    - retryable: everything else, the ``"pending"`` images.

    A failure from an earlier epoch is not terminal here: the continuation
    that submitted this epoch chose to retry it.

    **Republish mode** (``--mode measure``, P6 T2) is read from the
    manifest (:func:`staged_manifest_republishes`, its one carrier), and
    ``stage4_status`` then asks of the measure pipeline's run folder
    (``config["pipeline_sha256"]``) and of this epoch, which a republished
    record carries. Measure mode plans no reference metadata, so the
    reference check is skipped.
    """
    output_dir = Path(config["output_dir"])
    epoch = str(config["epoch"])
    failed = terminal_failure_work_ids_under_epoch(output_dir, epoch)
    republish = staged_manifest_republishes(Path(config["manifest_path"]))
    retryable: list[StagedManifestEntry] = []
    terminal: list[StagedManifestEntry] = []
    for entry in entries:
        status = stage4_status(
            output_dir, entry.dataset, entry.stem, entry.work_id,
            republish=republish,
            pipeline_sha256=config.get("pipeline_sha256") if republish else None,
            epoch=epoch if republish else None,
        )
        if status == "done":
            continue
        if entry.work_id in failed or status == "not_ready":
            terminal.append(entry)
            continue
        if not republish and (
            reference_digest_for(output_dir, entry.dataset, entry.stem)
            != entry.reference_digest
        ):
            terminal.append(entry)
            continue
        retryable.append(entry)
    return retryable, terminal


def _log_stage4_terminal(
    config: dict[str, Any], terminal: Sequence[StagedManifestEntry]
) -> None:
    """Say how many images Stage 4 will not run, and why (review M5).

    A journalled failure under this epoch is a verdict on the image; one
    that never finished Stages 1-3 (``not_ready``) has no record and no
    row, so the run ends unfinished (exit 5) and a plain re-run retries it.
    """
    if not terminal:
        return
    output_dir = Path(config["output_dir"])
    failed = terminal_failure_work_ids_under_epoch(output_dir, str(config["epoch"]))
    journalled = sum(1 for entry in terminal if entry.work_id in failed)
    others = len(terminal) - journalled
    logger.info(
        "Stage 4 skips %d image(s): %d failed terminally this epoch, %d never "
        "finished Stages 1-3 or were re-planned (no record and no journal row, "
        "so the run ends unfinished and a re-run retries them)",
        len(terminal), journalled, others,
    )


def _stage4_progress_digest(
    config: dict[str, Any], retryable: Sequence[StagedManifestEntry]
) -> tuple[str, int]:
    """sha256 over the retryable images and, per key, their signal files' mtimes.

    The progress unit is the **signal key**, not the image (plan review C4):
    a round that embedded one backbone of several under a walltime committed
    nothing, but wrote signals the next round reuses, and that is progress.
    So the digest covers each retryable image's identity and, per key with a
    signal file, its ``st_mtime_ns``, from one ``stat`` each
    (:func:`~phenotypic._cli._cli_stage4_signal.stage4_signal_mtimes`). It
    never opens a signal: validity against the live root is the worker's
    check, which deletes a stale signal before re-embedding it, so a
    rewritten signal shows as a new mtime.

    Returns:
        ``(digest, number of retryable images)``.
    """
    from ._cli_stage4_signal import stage4_signal_mtimes

    output_dir = Path(config["output_dir"])
    by_dataset: dict[str, list[StagedManifestEntry]] = {}
    for entry in retryable:
        by_dataset.setdefault(entry.dataset, []).append(entry)
    lines: list[str] = []
    for dataset, dataset_entries in by_dataset.items():
        mtimes = stage4_signal_mtimes(
            output_dir, dataset, [entry.stem for entry in dataset_entries]
        )
        for entry in dataset_entries:
            keys = mtimes.get(entry.stem, {})
            lines.append(
                json.dumps(
                    [entry.identity, sorted(keys.items())], separators=(",", ":")
                )
            )
    digest = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()
    return digest, len(retryable)


def _submit_stage4_round(
    config: dict[str, Any],
    state: dict[str, Any],
    successor_id: str,
    current_job_id: str,
    retryable: Sequence[StagedManifestEntry],
) -> None:
    """Submit the next GPU round over *retryable*, or the finalizer after a stall.

    Stage 2's zero-progress rule (:func:`_submit_stage2_round`) on Stage 4's
    **own** counters (``stage4_round``, ``stage4_last_progress_digest``,
    ``stage4_zero_progress_rounds``) and on :func:`_stage4_progress_digest`.
    Below two unchanged rounds it writes the round's worklist, then submits
    the ``stage4`` array as ``stage4-round-<n>`` and retargets the successor.
    At two it submits the finalizer and records nothing for the leftover
    images (U7): no record and no journal row, so the run reads unfinished
    and a plain re-run retries them.
    """
    output_dir = Path(config["output_dir"])
    epoch = str(config["epoch"])
    next_round = int(state.get("stage4_round", 0)) + 1
    digest, count = _stage4_progress_digest(config, retryable)
    if state.get("stage4_last_progress_digest") == digest:
        zero_progress = int(state.get("stage4_zero_progress_rounds", 0)) + 1
    else:
        zero_progress = 0
    state["stage4_last_progress_digest"] = digest
    state["stage4_zero_progress_rounds"] = zero_progress

    if zero_progress >= 2:
        logger.warning(stage4_stall_line(count))
        save_orchestration_state(output_dir, state)
        _submit_finalizer(
            config, state, successor_id, current_job_id, next_round - 1
        )
        return

    token = f"stage4-round-{next_round}"
    worklist = Path(config["stage4_worklist"])
    if not _round_already_submitted(output_dir, epoch, token) or not worklist.is_file():
        write_stage4_worklist(
            worklist,
            round_index=next_round,
            manifest_path=Path(config["manifest_path"]),
            retryable=retryable,
            n_shards=int(config["n_shards"]),
        )
    # Review M13: only the shards the worklist fills get a GPU. The
    # partition is contiguous, so they are shards 0..k-1, and the command
    # line's --array overrides the script's #SBATCH --array=0-(n_shards-1).
    filled = _filled_stage4_shards(worklist)
    job_id = submit_with_intent(
        output_dir,
        epoch=epoch,
        token=token,
        role="stage4",
        round_index=next_round,
        script_path=Path(config["stage4_script"]),
        extra_args=[f"--array=0-{max(1, filled) - 1}"],
    )
    state.update(
        {
            "phase": "stage4",
            "stage4_round": next_round,
            "active_job_id": job_id,
            "dependency_update_failed": False,
        }
    )
    _retarget_successor(state, successor_id, current_job_id, job_id)
    save_orchestration_state(output_dir, state)


def stage4_stall_line(count: int | None) -> str:
    """The one line a Stage-4 stall logs, in the controller and again in the finalizer (M14)."""
    images = "the leftover image(s)" if count is None else f"{count} image(s)"
    return (
        f"Stage 4 made no progress in two rounds; {images} stay "
        "unfinished (U7). Read the stage4 shard logs for why: a shard that "
        "never commits its first image within its walltime needs a longer "
        "--gpu-slurm time=. Then re-run the same command to retry them."
    )


def dead_task_stall_line(stage: str, count: int | None) -> str:
    """The one line a Stage-1/3 retry stall logs, in the controller and the finalizer (U15/D5)."""
    images = "the leftover image(s)" if count is None else f"{count} image(s)"
    return (
        f"{stage}: {images} died without a result, record or journal row in "
        f"two unchanged retry rounds; they stay unfinished. Read the {stage} "
        "task logs for why (an OUT_OF_MEMORY kill needs a larger mem_gb in "
        "--slurm), then re-run the same command."
    )


def recorded_stall_lines(state: dict[str, Any]) -> list[str]:
    """Every stall the orchestration state records, as the controller logged it.

    The finalizer prints these, because ``phenotypic-wait``'s exit 5 points
    at its log, not the controller's (review M14).
    """
    lines = [
        dead_task_stall_line(stage, None)
        for stage in ("stage1", "stage3")
        if int(state.get(f"{stage}_zero_progress_rounds", 0) or 0) >= 2
    ]
    if int(state.get("stage4_zero_progress_rounds", 0) or 0) >= 2:
        lines.append(stage4_stall_line(None))
    return lines


def _round_already_submitted(output_dir: Path, epoch: str, token: str) -> bool:
    """Whether the ledger holds an intent or a submission for *token* under *epoch* (review M12).

    Then a predecessor wrote that round's worklist and may have a job
    running against it, which a rewritten partition would split under it.
    """
    return any(
        row.get("token") == token
        and row.get("status") in {"intent", "blocked", "submitted", "recovered"}
        for row in read_job_ledger(output_dir, epoch=epoch)
    )


def _filled_stage4_shards(worklist: Path) -> int:
    """The number of non-empty shards in the round's worklist."""
    payload = json.loads(worklist.read_text(encoding="utf-8"))
    return sum(1 for shard in payload.get("shards", []) if shard)


def _advance_stage4(
    config: dict[str, Any],
    state: dict[str, Any],
    successor_id: str,
    current_job_id: str,
) -> None:
    """Sweep orphan signals, classify Stage 4, then a GPU round or the finalizer."""
    from ._cli_stage4_worker import (
        sweep_republished_stage4_signals,
        sweep_run_orphan_stage4_signals,
    )

    output_dir = Path(config["output_dir"])
    epoch = str(config["epoch"])
    state["phase"] = "stage4"
    save_orchestration_state(output_dir, state)
    entries = load_staged_manifest(Path(config["manifest_path"]))
    # Before classifying (plan review I7): an image published in an earlier
    # round whose shard died before consuming its signals is done, and only
    # this sweep reaches its leftovers. In republish mode the forward sweep
    # would ask the forward record, which a same-pipeline measure leaves
    # verifying, and so delete the signals a killed round wrote for an image
    # this pass has not yet republished; it sweeps the images this pass
    # republished instead (review M1).
    guard = _controller_commit_guard(output_dir, epoch)
    if staged_manifest_republishes(Path(config["manifest_path"])):
        sweep_republished_stage4_signals(
            output_dir, entries, pipeline_sha256=config.get("pipeline_sha256"),
            epoch=epoch, commit_guard=guard,
        )
    else:
        sweep_run_orphan_stage4_signals(output_dir, commit_guard=guard)
    retryable, terminal = _classify_stage4(config, entries)
    _log_stage4_terminal(config, terminal)
    if retryable:
        _submit_stage4_round(
            config, state, successor_id, current_job_id, retryable
        )
        return
    _submit_finalizer(
        config,
        state,
        successor_id,
        current_job_id,
        int(state.get("stage4_round", 0)),
    )


def _submit_next_stage1(
    config: dict[str, Any],
    state: dict[str, Any],
    successor_id: str,
    current_job_id: str,
) -> bool:
    """Submit the next Stage-1 chunk, returning whether one was launched."""
    output_dir = Path(config["output_dir"])
    epoch = str(config["epoch"])
    scripts = [Path(path) for path in config["stage1_scripts"]]
    index = int(state.get("stage1_index", 0))
    if index >= len(scripts):
        if _resubmit_dead_tasks(
            config, state, successor_id, current_job_id, "stage1"
        ):
            return True
        state["phase"] = "stage2"
        save_orchestration_state(output_dir, state)
        return False
    job_id = submit_with_intent(
        output_dir,
        epoch=epoch,
        token=f"stage1-chunk-{index}",
        role="stage1",
        round_index=index,
        script_path=scripts[index],
    )
    state.update(
        {
            "phase": "stage1",
            "stage1_index": index + 1,
            "active_job_id": job_id,
            "dependency_update_failed": False,
        }
    )
    _retarget_successor(state, successor_id, current_job_id, job_id)
    save_orchestration_state(output_dir, state)
    return True


#: Phases after which a controller has nothing to do and exits 0. The fence
#: is closed in each, so ``assert_active_epoch`` would raise. The finalizer
#: closes a run with a terminal per-image failure, or a Stage-4 stall's
#: leftovers (U7), as ``"terminal_incomplete"``
#: (``_cli_checkpoint_handler``); without it here the trailing recovery
#: controller of every such run ended FAILED in ``sacct`` (review I3, smoke F4).
_QUIET_PHASES = frozenset({"complete", "cancelled", "failed", "terminal_incomplete"})


def run_staged_controller(config_path: Path) -> None:
    """Advance the staged orchestration by one scheduler-controlled transition."""
    config = _load_controller_config(config_path)
    output_dir = Path(config["output_dir"])
    epoch = str(config["epoch"])
    current_job_id = current_slurm_job_id()
    initial_state = load_orchestration_state(output_dir)
    if initial_state is None or initial_state.get("epoch") != epoch:
        return
    if initial_state.get("phase") in _QUIET_PHASES:
        return
    assert_active_epoch(output_dir, epoch)

    with exclusive_path_lock(
        orchestration_lock_path(output_dir), timeout=60.0
    ):
        state = load_orchestration_state(output_dir)
        if state is None or state.get("epoch") != epoch:
            return
        if state.get("phase") in _QUIET_PHASES:
            return

        expected_controller = state.get("expected_controller_id")
        if (
            expected_controller is not None
            and str(expected_controller) != current_job_id
        ):
            if not _is_ledgered_successor(
                output_dir,
                epoch,
                current_job_id,
                str(expected_controller),
            ):
                logger.warning(
                    "Ignoring duplicate staged controller %s; expected %s",
                    current_job_id,
                    expected_controller,
                )
                return
            state["expected_controller_id"] = current_job_id
            save_orchestration_state(output_dir, state)

        successor_id = _prearm_successor(config, state, current_job_id)

        active_job_id = state.get("active_job_id")
        active_status = (
            scheduler_job_is_active(str(active_job_id)) if active_job_id else False
        )
        if active_job_id and active_status is not False:
            _retarget_successor(
                state, successor_id, current_job_id, str(active_job_id)
            )
            state["scheduler_query_failed"] = active_status is None
            save_orchestration_state(output_dir, state)
            return
        if active_job_id:
            mark_job_observed_terminal(
                output_dir, epoch=epoch, job_id=str(active_job_id)
            )
        state["active_job_id"] = None

        if state.get("phase") == "finalizing":
            # The phase the finalizer closed this epoch with, read now, at
            # the decision (P6 review M5): a measure pass's finalizer writes
            # no staged-completion marker (U10), and one that closed
            # `complete` (or `terminal_incomplete`) just after this
            # controller loaded "finalizing" was otherwise relabelled
            # `failed`.
            closed = deactivated_epoch_phase(output_dir, epoch)
            state["phase"] = (
                "complete"
                if staged_completion_matches(output_dir, epoch)
                else closed
                if closed in ("complete", "terminal_incomplete")
                else "failed"
            )
            save_orchestration_state(output_dir, state)
            return

        phases = _phases(config)
        if "stage1" not in phases and state.get("phase") in {
            "submitted",
            "stage1",
            "stage2",
        }:
            # A run without a GpuDetector (phases ["stage3", "stage4"]) has
            # no Stage 1 or 2 to run: the single-pass core is its "stage3".
            state["phase"] = "stage3_starting"

        if state.get("phase") == "stage4" and "stage4" in phases:
            _advance_stage4(config, state, successor_id, current_job_id)
            return

        if state.get("phase") == "stage1" and _submit_next_stage1(
            config, state, successor_id, current_job_id
        ):
            return

        if state.get("phase") in {"stage3", "stage3_starting"}:
            _submit_next_stage3_or_finalizer(
                config, state, successor_id, current_job_id
            )
            return

        entries = load_staged_manifest(Path(config["manifest_path"]))
        retryable, _terminal = _classify_stage2(
            config, entries, int(state.get("round", 0))
        )
        if retryable:
            _submit_stage2_round(
                config,
                state,
                successor_id,
                current_job_id,
                retryable,
            )
            return

        state["phase"] = "stage3_starting"
        state["stage3_index"] = int(state.get("stage3_index", 0))
        save_orchestration_state(output_dir, state)
        _submit_next_stage3_or_finalizer(
            config, state, successor_id, current_job_id
        )


def staged_controller_cli(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for one staged-controller transition."""
    parser = argparse.ArgumentParser(prog="phenotypic-staged-controller")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    run_staged_controller(args.config)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(staged_controller_cli())
