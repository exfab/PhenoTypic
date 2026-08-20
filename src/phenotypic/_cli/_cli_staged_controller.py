"""Recoverable array-level controller for staged SLURM GPU runs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

from phenotypic.sdk_ import dataset_measurements_dir, zarr_store_path
from phenotypic.sdk_._file_locking import exclusive_path_lock

from ._cli_stage2_token import stage2_result_replayable
from ._cli_staged_resume import stage3_completion_exists, valid_staged_store
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
)
from ._cli_failure_tracker import read_terminal_failures

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


def _load_controller_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError(f"Unsupported staged controller config: {path}")
    return payload


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
        if stage2_result_replayable(output_dir, entry.dataset, entry.stem):
            continue
        if stage3_completion_exists(
            output_dir, entry.dataset, entry.stem
        ) or (resume and not markers_required and parquet.is_file()):
            continue
        if entry.work_id in failed:
            terminal.append(entry)
            continue
        store = zarr_store_path(output_dir, entry.dataset, entry.stem)
        if not valid_staged_store(store):
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

    token = "finalizer"
    job_id = submit_with_intent(
        output_dir,
        epoch=epoch,
        token=token,
        role="finalizer",
        round_index=index,
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
