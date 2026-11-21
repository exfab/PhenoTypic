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
    staged_completion_matches,
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

    if "stage4" in _phases(config):
        _advance_stage4(config, state, successor_id, current_job_id)
        return
    _submit_finalizer(config, state, successor_id, current_job_id, index)


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
    """
    output_dir = Path(config["output_dir"])
    failed = terminal_failure_work_ids_under_epoch(output_dir, str(config["epoch"]))
    retryable: list[StagedManifestEntry] = []
    terminal: list[StagedManifestEntry] = []
    for entry in entries:
        status = stage4_status(output_dir, entry.dataset, entry.stem, entry.work_id)
        if status == "done":
            continue
        if entry.work_id in failed or status == "not_ready":
            terminal.append(entry)
            continue
        if (
            reference_digest_for(output_dir, entry.dataset, entry.stem)
            != entry.reference_digest
        ):
            terminal.append(entry)
            continue
        retryable.append(entry)
    return retryable, terminal


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
        logger.warning(
            "Stage 4 made no progress in two rounds; %d image(s) stay "
            "unfinished. Re-run the same command to retry them.",
            count,
        )
        save_orchestration_state(output_dir, state)
        _submit_finalizer(
            config, state, successor_id, current_job_id, next_round - 1
        )
        return

    write_stage4_worklist(
        Path(config["stage4_worklist"]),
        round_index=next_round,
        manifest_path=Path(config["manifest_path"]),
        retryable=retryable,
        n_shards=int(config["n_shards"]),
    )
    job_id = submit_with_intent(
        output_dir,
        epoch=epoch,
        token=f"stage4-round-{next_round}",
        role="stage4",
        round_index=next_round,
        script_path=Path(config["stage4_script"]),
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


def _advance_stage4(
    config: dict[str, Any],
    state: dict[str, Any],
    successor_id: str,
    current_job_id: str,
) -> None:
    """Sweep orphan signals, classify Stage 4, then a GPU round or the finalizer."""
    from ._cli_stage4_worker import sweep_run_orphan_stage4_signals

    output_dir = Path(config["output_dir"])
    epoch = str(config["epoch"])
    state["phase"] = "stage4"
    save_orchestration_state(output_dir, state)
    # Before classifying (plan review I7): an image published in an earlier
    # round whose shard died before consuming its signals is done, and only
    # this sweep reaches its leftovers.
    sweep_run_orphan_stage4_signals(
        output_dir, commit_guard=_controller_commit_guard(output_dir, epoch)
    )
    entries = load_staged_manifest(Path(config["manifest_path"]))
    retryable, _terminal = _classify_stage4(config, entries)
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


def run_staged_controller(config_path: Path) -> None:
    """Advance the staged orchestration by one scheduler-controlled transition."""
    config = _load_controller_config(config_path)
    output_dir = Path(config["output_dir"])
    epoch = str(config["epoch"])
    current_job_id = current_slurm_job_id()
    initial_state = load_orchestration_state(output_dir)
    if initial_state is None or initial_state.get("epoch") != epoch:
        return
    if initial_state.get("phase") in {"complete", "cancelled", "failed"}:
        return
    assert_active_epoch(output_dir, epoch)

    with exclusive_path_lock(
        orchestration_lock_path(output_dir), timeout=60.0
    ):
        state = load_orchestration_state(output_dir)
        if state is None or state.get("epoch") != epoch:
            return
        if state.get("phase") in {"complete", "cancelled", "failed"}:
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
            state["phase"] = (
                "complete"
                if staged_completion_matches(output_dir, epoch)
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
