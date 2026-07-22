"""Durable orchestration state for staged SLURM GPU runs.

The staged controller is deliberately artifact-driven. Per-image HDF, objmap
sidecar, and terminal Stage 3 markers decide what work remains. This module
supplies the small amount of durable coordination needed to make scheduler
submissions and run cancellation crash-recoverable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from uuid import uuid4

from phenotypic.sdk_ import (
    JobMetadataKey,
    atomic_write_json,
    dataset_measurements_dir,
    job_metadata_path,
    progress_dir,
    results_dir,
)
from phenotypic.sdk_._file_locking import exclusive_path_lock

from ._cli_file_locking import atomic_append, atomic_read
from ._cli_staged_resume import stage3_completion_exists

_MANIFEST_VERSION = 2
_STATE_FILENAME = "staged_orchestration.json"
_LEDGER_FILENAME = "staged_jobs.jsonl"
_FAILURES_FILENAME = "stage2_terminal_failures.jsonl"
_DEACTIVATIONS_FILENAME = "staged_epoch_deactivations.jsonl"
_COMPLETION_FILENAME = "staged_finalization_complete.json"
_LOCK_FILENAME = ".staged_orchestration.lock"
_SUBMIT_ATTEMPTS = 3
_SUBMIT_BACKOFF_SECONDS = (1.0, 2.0, 4.0)


class SchedulerQueryUnavailable(RuntimeError):
    """Raised when SLURM cannot establish whether a token already exists."""


@dataclass(frozen=True)
class StagedManifestEntry:
    """One versioned staged-work manifest entry."""

    dataset: str
    image_name: str
    stem: str
    input_path: str

    @property
    def identity(self) -> str:
        """Return the stable event/state identity for the entry."""
        return f"{self.dataset}\0{self.image_name}"


def orchestration_state_path(output_dir: Path) -> Path:
    """Return the staged controller state path."""
    return progress_dir(output_dir) / _STATE_FILENAME


def orchestration_lock_path(output_dir: Path) -> Path:
    """Return the staged controller lock path."""
    return progress_dir(output_dir) / _LOCK_FILENAME


def staged_job_ledger_path(output_dir: Path) -> Path:
    """Return the append-only staged job ledger path."""
    return progress_dir(output_dir) / _LEDGER_FILENAME


def stage2_failure_journal_path(output_dir: Path) -> Path:
    """Return the current/past epoch Stage-2 terminal-failure journal path."""
    return progress_dir(output_dir) / _FAILURES_FILENAME


def epoch_deactivation_journal_path(output_dir: Path) -> Path:
    """Return the append-only epoch fencing journal path."""
    return progress_dir(output_dir) / _DEACTIVATIONS_FILENAME


def staged_completion_path(output_dir: Path) -> Path:
    """Return the atomic staged-finalization completion marker path."""
    return progress_dir(output_dir) / _COMPLETION_FILENAME


def staged_completion_matches(output_dir: Path, epoch: str) -> bool:
    """Return whether the atomic completion marker belongs to *epoch*."""
    try:
        marker = json.loads(
            staged_completion_path(output_dir).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(marker, dict) and marker.get("epoch") == epoch


def new_orchestration_epoch() -> str:
    """Return a collision-resistant orchestration epoch."""
    return uuid4().hex


def initialize_orchestration(
    output_dir: Path,
    *,
    epoch: str,
    mode: str,
    controller_config_path: Path,
) -> dict[str, Any]:
    """Create the active staged controller state for a new submission."""
    state: dict[str, Any] = {
        "schema_version": 1,
        "epoch": epoch,
        "mode": mode,
        "phase": "submitted",
        "stage1_index": 0,
        "round": 0,
        "stage3_index": 0,
        "last_retryable_digest": None,
        "last_retryable_count": None,
        "zero_progress_rounds": 0,
        "active_job_id": None,
        "expected_controller_id": None,
        "controller_config": str(controller_config_path),
        "created_at": datetime.now().isoformat(timespec="milliseconds"),
        "updated_at": datetime.now().isoformat(timespec="milliseconds"),
    }
    atomic_write_json(orchestration_state_path(output_dir), state)
    staged_completion_path(output_dir).unlink(missing_ok=True)
    return state


def load_orchestration_state(output_dir: Path) -> dict[str, Any] | None:
    """Read controller state, returning ``None`` for missing or invalid state."""
    path = orchestration_state_path(output_dir)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    deactivated_phase = _deactivated_epoch_phase(
        output_dir, str(payload.get("epoch", ""))
    )
    if deactivated_phase is not None:
        payload["phase"] = deactivated_phase
    return payload


def _deactivated_epoch_phase(output_dir: Path, epoch: str) -> str | None:
    """Return the terminal phase recorded by the independent epoch fence."""
    if not epoch:
        return None

    def _parse(content: str) -> str | None:
        for line in reversed(content.splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("epoch") == epoch:
                return str(row.get("phase", "cancelled"))
        return None

    return atomic_read(epoch_deactivation_journal_path(output_dir), _parse)


def save_orchestration_state(
    output_dir: Path, state: Mapping[str, Any]
) -> None:
    """Atomically replace controller state with an updated timestamp."""
    payload = dict(state)
    payload["updated_at"] = datetime.now().isoformat(timespec="milliseconds")
    atomic_write_json(orchestration_state_path(output_dir), payload)


def assert_active_epoch(output_dir: Path, epoch: str) -> None:
    """Raise when a worker/controller belongs to an inactive run epoch."""
    state = load_orchestration_state(output_dir)
    if state is None or state.get("epoch") != epoch:
        raise RuntimeError(
            f"Stale staged worker epoch {epoch!r}; active epoch is "
            f"{None if state is None else state.get('epoch')!r}"
        )
    if state.get("phase") in {"cancelled", "complete", "failed"}:
        raise RuntimeError(
            f"Staged orchestration {epoch} is already {state.get('phase')}"
        )


def epoch_is_active(output_dir: Path, epoch: str) -> bool:
    """Return whether *epoch* may still publish work."""
    try:
        assert_active_epoch(output_dir, epoch)
    except RuntimeError:
        return False
    return True


def write_staged_manifest(
    path: Path, entries: Sequence[StagedManifestEntry]
) -> Path:
    """Write the versioned staged manifest atomically."""
    atomic_write_json(
        path,
        {
            "version": _MANIFEST_VERSION,
            "images": [asdict(entry) for entry in entries],
        },
    )
    return path


def load_staged_manifest(path: Path) -> list[StagedManifestEntry]:
    """Load and validate a versioned staged manifest."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != _MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported staged manifest version in {path}; "
            f"expected {_MANIFEST_VERSION}"
        )
    images = raw.get("images")
    if not isinstance(images, list):
        raise ValueError(f"Staged manifest {path} has no image list")
    return [StagedManifestEntry(**entry) for entry in images]


def retryable_digest(
    entries: Iterable[StagedManifestEntry],
) -> tuple[str, int]:
    """Return a deterministic digest and count for a retryable identity set."""
    identities = sorted(entry.identity for entry in entries)
    digest = hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()
    return digest, len(identities)


def completed_inventory_images(
    output_dir: Path, dataset: str, image_names: Sequence[str]
) -> set[str]:
    """Return full-inventory image names with terminal staged artifacts."""
    measurements = dataset_measurements_dir(output_dir, dataset)
    state = load_orchestration_state(output_dir) or {}
    markers_required = bool(state.get("stage3_markers_required", False))
    old_fingerprints = state.get("restart_parquet_fingerprints", {})
    completed: set[str] = set()
    for image_name in image_names:
        stem = Path(image_name).stem
        if markers_required:
            if stage3_completion_exists(output_dir, dataset, stem):
                completed.add(image_name)
            continue
        parquet = measurements / f"{stem}.parquet"
        try:
            stat = parquet.stat()
        except FileNotFoundError:
            continue
        fingerprint = [stat.st_ino, stat.st_size, stat.st_mtime_ns]
        if old_fingerprints.get(f"{dataset}\0{image_name}") == fingerprint:
            continue
        completed.add(image_name)
    return completed


def snapshot_inventory_parquets(
    output_dir: Path, inventory: Mapping[str, Sequence[str]]
) -> dict[str, list[int]]:
    """Fingerprint parquets that predate an explicit restart epoch."""
    fingerprints: dict[str, list[int]] = {}
    for dataset, image_names in inventory.items():
        measurements = dataset_measurements_dir(output_dir, dataset)
        for image_name in image_names:
            parquet = measurements / f"{Path(image_name).stem}.parquet"
            try:
                stat = parquet.stat()
            except FileNotFoundError:
                continue
            fingerprints[f"{dataset}\0{image_name}"] = [
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
            ]
    return fingerprints


def quarantine_unchanged_restart_parquets(output_dir: Path, epoch: str) -> int:
    """Move unchanged pre-restart parquets out of final aggregation inputs."""
    state = load_orchestration_state(output_dir)
    if state is None or state.get("epoch") != epoch:
        return 0
    old_fingerprints = state.get("restart_parquet_fingerprints", {})
    if not isinstance(old_fingerprints, dict):
        return 0
    quarantine = progress_dir(output_dir) / "restart_stale_parquets"
    moved = 0
    for identity, old_fingerprint in old_fingerprints.items():
        try:
            dataset, image_name = str(identity).split("\0", 1)
        except ValueError:
            continue
        parquet = (
            dataset_measurements_dir(output_dir, dataset)
            / f"{Path(image_name).stem}.parquet"
        )
        try:
            stat = parquet.stat()
        except FileNotFoundError:
            continue
        if [stat.st_ino, stat.st_size, stat.st_mtime_ns] != old_fingerprint:
            continue
        destination = quarantine / dataset / parquet.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        parquet.replace(destination)
        moved += 1
    return moved


def append_stage2_terminal_failure(
    output_dir: Path,
    *,
    epoch: str,
    round_index: int,
    entry: StagedManifestEntry,
    error_type: str,
    error_message: str,
) -> None:
    """Record a non-retryable Stage-2 image outcome for the current epoch."""
    record = {
        "epoch": epoch,
        "round": round_index,
        "dataset": entry.dataset,
        "image_name": entry.image_name,
        "stem": entry.stem,
        "error_type": error_type,
        "error_message": error_message,
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
    }
    atomic_append(
        stage2_failure_journal_path(output_dir),
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n",
    )


def terminal_stage2_identities(output_dir: Path, epoch: str) -> set[str]:
    """Return ``dataset\0image_name`` identities terminal in *epoch*."""

    def _parse(content: str) -> set[str]:
        found: set[str] = set()
        for line in content.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("epoch") == epoch:
                found.add(
                    f"{row.get('dataset', '')}\0{row.get('image_name', '')}"
                )
        return found

    return atomic_read(stage2_failure_journal_path(output_dir), _parse)


def append_job_ledger(
    output_dir: Path,
    *,
    epoch: str,
    token: str,
    role: str,
    round_index: int,
    status: str,
    job_id: str | None = None,
    dependencies: Sequence[str] = (),
) -> None:
    """Append one scheduler-transition record to the staged job ledger."""
    record = {
        "epoch": epoch,
        "token": token,
        "role": role,
        "round": round_index,
        "status": status,
        "job_id": job_id,
        "dependencies": list(dependencies),
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
    }
    atomic_append(
        staged_job_ledger_path(output_dir),
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n",
    )


def read_job_ledger(
    output_dir: Path, *, epoch: str | None = None
) -> list[dict[str, Any]]:
    """Read valid ledger records, optionally restricted to one epoch."""

    def _parse(content: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in content.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and (
                epoch is None or row.get("epoch") == epoch
            ):
                rows.append(row)
        return rows

    return atomic_read(staged_job_ledger_path(output_dir), _parse)


def ledger_job_for_token(
    output_dir: Path, epoch: str, token: str
) -> str | None:
    """Return the most recently submitted job ID for a transition token."""
    for row in reversed(read_job_ledger(output_dir, epoch=epoch)):
        if row.get("token") == token and row.get("status") == "submitted":
            job_id = row.get("job_id")
            return str(job_id) if job_id else None
    return None


def mark_job_observed_terminal(
    output_dir: Path, *, epoch: str, job_id: str
) -> None:
    """Append a terminal observation for a previously submitted ledger job."""
    for row in reversed(read_job_ledger(output_dir, epoch=epoch)):
        if row.get("status") == "submitted" and str(row.get("job_id")) == job_id:
            append_job_ledger(
                output_dir,
                epoch=epoch,
                token=str(row.get("token")),
                role=str(row.get("role")),
                round_index=int(row.get("round", 0)),
                status="terminal",
                job_id=job_id,
                dependencies=tuple(row.get("dependencies", [])),
            )
            return


def _job_from_scheduler_comment(comment: str) -> str | None:
    """Discover a previously submitted job by its deterministic comment."""
    commands = [
        ["squeue", "--noheader", "--format=%i|%k"],
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--starttime=now-2days",
            "--format=JobIDRaw,Comment%200",
        ],
    ]
    successful_queries = 0
    for command in commands:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode != 0:
            continue
        successful_queries += 1
        for raw_line in result.stdout.splitlines():
            if "|" in raw_line:
                job_id, found_comment = (
                    part.strip() for part in raw_line.split("|", 1)
                )
                found_comment = found_comment.rstrip("|").strip()
            else:
                parts = raw_line.split(None, 1)
                if len(parts) != 2:
                    continue
                job_id, found_comment = parts
            if found_comment == comment and job_id.split("_")[0].isdigit():
                return job_id.split("_")[0]
    if successful_queries == 0:
        raise SchedulerQueryUnavailable(
            "Could not query squeue or sacct for an incomplete submission intent"
        )
    return None


def _mirror_job_to_metadata(
    output_dir: Path, *, token: str, role: str, job_id: str
) -> None:
    """Expose dynamic jobs without polluting the numeric chunk registry."""
    path = job_metadata_path(output_dir)
    if not path.is_file():
        return
    lock = path.with_name(f".{path.name}.lock")
    with exclusive_path_lock(lock, timeout=60.0):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        all_jobs = payload.setdefault(JobMetadataKey.SLURM_JOB_IDS, {})
        if isinstance(all_jobs, dict):
            all_jobs[token] = job_id
        if role in {"stage1", "stage2", "stage3"} and token.isdecimal():
            chunks = payload.setdefault(JobMetadataKey.CHUNK_JOB_IDS, {})
            if isinstance(chunks, dict):
                chunks[token] = job_id
        atomic_write_json(path, payload)


def submit_with_intent(
    output_dir: Path,
    *,
    epoch: str,
    token: str,
    role: str,
    round_index: int,
    script_path: Path,
    dependencies: Sequence[str] = (),
) -> str:
    """Idempotently submit an SBATCH script with bounded retry and ledgering."""
    assert_active_epoch(output_dir, epoch)
    existing = ledger_job_for_token(output_dir, epoch, token)
    if existing:
        return existing
    comment = f"phenotypic:{epoch}:{token}"
    prior_intent = any(
        row.get("token") == token and row.get("status") in {"intent", "blocked"}
        for row in read_job_ledger(output_dir, epoch=epoch)
    )
    if prior_intent:
        discovered = _job_from_scheduler_comment(comment)
        if discovered:
            append_job_ledger(
                output_dir,
                epoch=epoch,
                token=token,
                role=role,
                round_index=round_index,
                status="submitted",
                job_id=discovered,
                dependencies=dependencies,
            )
            _mirror_job_to_metadata(
                output_dir, token=token, role=role, job_id=discovered
            )
            return discovered

    append_job_ledger(
        output_dir,
        epoch=epoch,
        token=token,
        role=role,
        round_index=round_index,
        status="intent",
        dependencies=dependencies,
    )
    command = ["sbatch", "--parsable", "--comment", comment]
    if dependencies:
        command.extend(["--dependency", f"afterany:{':'.join(dependencies)}"])
    command.append(str(script_path))
    last_error = "unknown submission failure"
    for attempt in range(_SUBMIT_ATTEMPTS):
        assert_active_epoch(output_dir, epoch)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            job_id = result.stdout.strip().split(";", 1)[0]
            if not job_id.isdigit():
                raise RuntimeError(f"Invalid sbatch job id: {result.stdout!r}")
            append_job_ledger(
                output_dir,
                epoch=epoch,
                token=token,
                role=role,
                round_index=round_index,
                status="submitted",
                job_id=job_id,
                dependencies=dependencies,
            )
            _mirror_job_to_metadata(
                output_dir, token=token, role=role, job_id=job_id
            )
            return job_id
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            RuntimeError,
        ) as exc:
            last_error = str(exc)
            if isinstance(exc, (subprocess.TimeoutExpired, RuntimeError)):
                discovered = _job_from_scheduler_comment(comment)
                if discovered:
                    append_job_ledger(
                        output_dir,
                        epoch=epoch,
                        token=token,
                        role=role,
                        round_index=round_index,
                        status="submitted",
                        job_id=discovered,
                        dependencies=dependencies,
                    )
                    _mirror_job_to_metadata(
                        output_dir,
                        token=token,
                        role=role,
                        job_id=discovered,
                    )
                    return discovered
            if attempt + 1 < _SUBMIT_ATTEMPTS:
                time.sleep(_SUBMIT_BACKOFF_SECONDS[attempt])
    append_job_ledger(
        output_dir,
        epoch=epoch,
        token=token,
        role=role,
        round_index=round_index,
        status="blocked",
        dependencies=dependencies,
    )
    raise RuntimeError(
        f"Could not submit {role} after {_SUBMIT_ATTEMPTS} attempts: {last_error}"
    )


def update_job_dependency(job_id: str, dependencies: Sequence[str]) -> bool:
    """Retarget a pending controller to wait for all supplied job IDs."""
    if not dependencies:
        return True
    try:
        result = subprocess.run(
            [
                "scontrol",
                "update",
                f"JobId={job_id}",
                f"Dependency=afterany:{':'.join(dependencies)}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def scheduler_job_is_active(job_id: str) -> bool | None:
    """Return active/inactive, or ``None`` when SLURM cannot answer."""
    try:
        result = subprocess.run(
            ["squeue", "--noheader", "--jobs", str(job_id), "--format=%T"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return any(line.strip() for line in result.stdout.splitlines())


def active_ledger_job_ids(output_dir: Path) -> list[str]:
    """Return all ledgered job IDs still visible in SLURM's active queue."""
    latest: dict[tuple[str, str], str] = {}
    for row in read_job_ledger(output_dir):
        key = (str(row.get("epoch")), str(row.get("token")))
        if row.get("status") == "submitted" and row.get("job_id"):
            latest[key] = str(row.get("job_id"))
        elif row.get("status") == "terminal":
            latest.pop(key, None)
    return sorted(
        {
            job_id
            for job_id in latest.values()
            if scheduler_job_is_active(job_id) is not False
        }
    )


def deactivate_orchestration(
    output_dir: Path, phase: str = "cancelled"
) -> bool:
    """Fence an epoch immediately, without waiting for its controller lock.

    The append-only fence is authoritative over the mutable controller state.
    This lets cancellation stop publication even when a controller is blocked
    inside a scheduler query or submission retry.
    """
    state = load_orchestration_state(output_dir)
    if state is None or state.get("phase") in {
        "complete",
        "failed",
        "cancelled",
    }:
        return False
    epoch = str(state.get("epoch", ""))
    atomic_append(
        epoch_deactivation_journal_path(output_dir),
        json.dumps(
            {
                "epoch": epoch,
                "phase": phase,
                "timestamp": datetime.now().isoformat(
                    timespec="milliseconds"
                ),
            },
            sort_keys=True,
        )
        + "\n",
    )
    state["phase"] = phase
    save_orchestration_state(output_dir, state)
    return True


def cancel_staged_jobs(output_dir: Path) -> list[str]:
    """Fence and cancel every active job recorded for a staged run."""
    deactivate_orchestration(output_dir, "cancelled")
    job_ids = active_ledger_job_ids(output_dir)
    if job_ids:
        try:
            subprocess.run(["scancel", *job_ids], check=False)
        except FileNotFoundError:
            pass
    return job_ids


def clear_stage2_sidecars(output_dir: Path) -> int:
    """Delete transient Stage-2 objmap sidecars during an explicit restart."""
    root = results_dir(output_dir)
    removed = 0
    if not root.is_dir():
        return removed
    for objmap_dir in root.glob("*/objmap"):
        if not objmap_dir.is_dir():
            continue
        for path in objmap_dir.iterdir():
            if path.is_file() and path.suffix in {".npy", ".tmp"}:
                path.unlink()
                removed += 1
    return removed


def mark_staged_complete(output_dir: Path, epoch: str) -> None:
    """Atomically mark successful remote finalization for *epoch*."""
    assert_active_epoch(output_dir, epoch)
    marker = {
        "epoch": epoch,
        "completed_at": datetime.now().isoformat(timespec="milliseconds"),
    }
    atomic_write_json(staged_completion_path(output_dir), marker)
    state = load_orchestration_state(output_dir)
    if state is not None and state.get("epoch") == epoch:
        state["phase"] = "complete"
        save_orchestration_state(output_dir, state)


def mark_local_staged_complete(output_dir: Path, pipeline_sha256: str) -> None:
    """Atomically mark successful local staged publication."""
    atomic_write_json(
        staged_completion_path(output_dir),
        {
            "mode": "local",
            "pipeline_sha256": pipeline_sha256,
            "completed_at": datetime.now().isoformat(timespec="milliseconds"),
        },
    )


def iter_manifest_entries(path: Path) -> Iterator[StagedManifestEntry]:
    """Yield entries from a staged manifest."""
    yield from load_staged_manifest(path)


def current_slurm_job_id() -> str:
    """Return the current SLURM array-master/job ID."""
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get(
        "SLURM_JOB_ID", ""
    )
    if not job_id:
        raise RuntimeError("Staged controller requires SLURM_JOB_ID")
    return job_id


__all__ = [
    "StagedManifestEntry",
    "active_ledger_job_ids",
    "append_job_ledger",
    "append_stage2_terminal_failure",
    "assert_active_epoch",
    "cancel_staged_jobs",
    "clear_stage2_sidecars",
    "completed_inventory_images",
    "current_slurm_job_id",
    "deactivate_orchestration",
    "epoch_deactivation_journal_path",
    "epoch_is_active",
    "initialize_orchestration",
    "load_orchestration_state",
    "load_staged_manifest",
    "mark_local_staged_complete",
    "mark_staged_complete",
    "mark_job_observed_terminal",
    "new_orchestration_epoch",
    "orchestration_lock_path",
    "orchestration_state_path",
    "quarantine_unchanged_restart_parquets",
    "read_job_ledger",
    "retryable_digest",
    "save_orchestration_state",
    "scheduler_job_is_active",
    "snapshot_inventory_parquets",
    "stage2_failure_journal_path",
    "staged_completion_path",
    "staged_completion_matches",
    "staged_job_ledger_path",
    "submit_with_intent",
    "terminal_stage2_identities",
    "update_job_dependency",
    "write_staged_manifest",
]
