"""Crash-recoverable lifecycle primitives for every SLURM submission role."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal, cast
from uuid import uuid4

import psutil

from phenotypic.sdk_ import (
    JobMetadataKey,
    atomic_write_json,
    job_metadata_path,
    progress_dir,
)
from phenotypic.sdk_._file_locking import exclusive_path_lock
from phenotypic.sdk_.slurm import sbatch_submission_environment

from ._cli_file_locking import atomic_append, atomic_read

SCHEMA_VERSION = 2
_STATE_FILENAME = "slurm_lifecycle.json"
_LEDGER_FILENAME = "slurm_jobs.jsonl"
_LEGACY_STAGED_LEDGER_FILENAME = "staged_jobs.jsonl"
_LOCK_FILENAME = ".slurm_submit_cancel.lock"
_SUBMIT_BACKOFF_SECONDS = (1.0, 2.0)

SlurmDependencyKind = Literal["afterany", "afterok"]


class SchedulerQueryUnavailable(RuntimeError):
    """Raised when neither scheduler accounting source can be queried."""


class SlurmGenerationInactiveError(RuntimeError):
    """Raised when a worker no longer owns the output lifecycle fence."""


def slurm_generation_inactive_cause(
    exception: BaseException,
) -> SlurmGenerationInactiveError | None:
    """Find a lifecycle rejection hidden by scientific exception translation."""
    current: BaseException | None = exception
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, SlurmGenerationInactiveError):
            return current
        if current.__cause__ is not None:
            current = current.__cause__
        else:
            current = current.__context__
    return None


@dataclass(frozen=True)
class CancellationResult:
    """Result of fencing and cancelling one launch generation."""

    job_ids: tuple[str, ...]
    unresolved_tokens: tuple[str, ...]
    quiescent: bool


def new_slurm_generation() -> str:
    """Return a collision-resistant launch generation."""
    return uuid4().hex


def lifecycle_state_path(output_dir: Path) -> Path:
    """Return the mutable active-generation fence path."""
    return progress_dir(output_dir) / _STATE_FILENAME


def lifecycle_ledger_path(output_dir: Path) -> Path:
    """Return the append-only scheduler ledger path."""
    return progress_dir(output_dir) / _LEDGER_FILENAME


def lifecycle_lock_path(output_dir: Path) -> Path:
    """Return the lock coordinating submission and cancellation."""
    return progress_dir(output_dir) / _LOCK_FILENAME


def initialize_slurm_lifecycle(
    output_dir: Path,
    *,
    generation: str,
    mode: str,
    owner_kind: str | None = None,
    control_root: Path | None = None,
) -> dict[str, Any]:
    """Publish an active launch fence before any scheduler call."""
    with exclusive_path_lock(lifecycle_lock_path(output_dir), timeout=60.0):
        existing = load_slurm_lifecycle(output_dir)
        if existing is not None and existing.get("active") is True:
            existing_generation = str(existing["generation"])
            if existing_generation != generation:
                raise RuntimeError(
                    "Output already has an active SLURM generation "
                    f"{existing_generation!r}; refusing conflicting generation "
                    f"{generation!r}"
                )
            return existing
        state = {
            "schema_version": SCHEMA_VERSION,
            "generation": generation,
            "mode": mode,
            "active": True,
            "created_at": _timestamp(),
            "updated_at": _timestamp(),
        }
        if owner_kind is not None:
            if owner_kind not in {"local", "slurm"}:
                raise ValueError("lifecycle owner_kind must be local or slurm")
            if control_root is None:
                raise ValueError("owned lifecycle requires a control root")
            state.update(
                {
                    "owner_kind": owner_kind,
                    "owner_pid": os.getpid(),
                    "owner_started_at": psutil.Process(os.getpid()).create_time(),
                    "owner_host": socket.gethostname(),
                    "control_root": str(Path(control_root).resolve()),
                }
            )
        atomic_write_json(lifecycle_state_path(output_dir), state)
    return state


def load_slurm_lifecycle(output_dir: Path) -> dict[str, Any] | None:
    """Read the active-generation fence, accepting version 1 fields."""
    try:
        raw = json.loads(
            lifecycle_state_path(output_dir).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    generation = raw.get("generation", raw.get("epoch"))
    if not isinstance(generation, str) or not generation:
        return None
    raw["generation"] = generation
    raw.setdefault("schema_version", 1)
    raw.setdefault(
        "active",
        raw.get("phase")
        not in {
            "cancelled",
            "failed",
            "complete",
        },
    )
    return raw


def generation_is_active(output_dir: Path, generation: str) -> bool:
    """Return whether *generation* is still the active launch fence."""
    state = load_slurm_lifecycle(output_dir)
    return bool(
        state
        and state.get("generation") == generation
        and state.get("active") is True
    )


def assert_generation_active(output_dir: Path, generation: str) -> None:
    """Reject stale or cancelled continuations before they can submit."""
    if not generation_is_active(output_dir, generation):
        raise SlurmGenerationInactiveError(
            f"SLURM generation {generation!r} is inactive or superseded"
        )


@contextmanager
def generation_publication_guard(
    output_dir: Path, generation: str
) -> Iterator[None]:
    """Serialize generation validation with canonical publication.

    Cancellation and initialization use the same lifecycle lock, so neither
    can deactivate or supersede the generation between this validation and
    the guarded mutation.
    """
    # Every recompile/measure worker task (up to the account's concurrency
    # cap, observed 60-90+) serializes through this single lock via
    # _write_status on completion. A 60s timeout is too tight when that many
    # tasks finish in a similar window; one spurious ArtifactLockTimeout here
    # cascades (via the worker's failure handler) into deactivating the whole
    # shared generation and failing every sibling task. 300s gives real
    # headroom under legitimate contention without masking a true deadlock.
    with exclusive_path_lock(lifecycle_lock_path(output_dir), timeout=300.0):
        assert_generation_active(output_dir, generation)
        yield


def append_lifecycle_entry(
    output_dir: Path,
    *,
    generation: str,
    token: str,
    role: str,
    status: str,
    job_id: str | None = None,
    dependencies: Sequence[str] = (),
    dependency_kind: SlurmDependencyKind = "afterany",
    round_index: int = 0,
    comment: str | None = None,
) -> None:
    """Append one versioned scheduler transition to the durable ledger."""
    validated_dependency_kind = _validate_dependency_kind(dependency_kind)
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generation": generation,
        "epoch": generation,
        "token": token,
        "role": role,
        "round": round_index,
        "status": status,
        "dependencies": list(dependencies),
        "dependency_kind": validated_dependency_kind,
        "comment": comment or scheduler_comment(generation, token),
        "timestamp": _timestamp(),
    }
    if job_id is not None:
        row["job_id"] = str(job_id)
    atomic_append(
        lifecycle_ledger_path(output_dir),
        json.dumps(row, sort_keys=True) + "\n",
    )


def read_lifecycle_ledger(
    output_dir: Path,
    *,
    generation: str | None = None,
) -> list[dict[str, Any]]:
    """Read versioned and pre-versioned ledgers into one normalized stream."""
    rows: list[dict[str, Any]] = []
    paths = (
        progress_dir(output_dir) / _LEGACY_STAGED_LEDGER_FILENAME,
        lifecycle_ledger_path(output_dir),
    )
    for path in paths:
        parsed = atomic_read(path, _parse_json_lines)
        for source in parsed:
            normalized = dict(source)
            found_generation = normalized.get(
                "generation", normalized.get("epoch")
            )
            if found_generation is None:
                continue
            normalized["generation"] = str(found_generation)
            normalized.setdefault("epoch", str(found_generation))
            normalized.setdefault("schema_version", 1)
            normalized.setdefault("role", "unknown")
            normalized.setdefault("dependency_kind", "afterany")
            if generation is None or str(found_generation) == generation:
                rows.append(normalized)
    return rows


def ledger_job_for_token(
    output_dir: Path, generation: str, token: str
) -> str | None:
    """Return the latest durable job id for a generation/token pair."""
    for row in reversed(
        read_lifecycle_ledger(output_dir, generation=generation)
    ):
        if (
            row.get("token") == token
            and row.get("status") in {"submitted", "recovered"}
            and row.get("job_id")
        ):
            return str(row["job_id"])
    return None


def scheduler_comment(generation: str, token: str) -> str:
    """Return the deterministic scheduler identity for one submission intent."""
    return f"phenotypic:{generation}:{token}"


def query_scheduler_comments(
    *,
    exact: str | None = None,
    prefix: str | None = None,
    include_accounting: bool = True,
    run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, set[str]]:
    """Return scheduler jobs whose comments match *exact* or *prefix*."""
    runner = run_command or subprocess.run
    commands = [["squeue", "--noheader", "--format=%i|%k"]]
    if include_accounting:
        commands.append(
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--starttime=now-2days",
                "--format=JobIDRaw,Comment%200",
            ]
        )
    matches: dict[str, set[str]] = {}
    successful_queries = 0
    for command in commands:
        try:
            result = runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result is None:
            continue
        if result.returncode != 0:
            continue
        successful_queries += 1
        for raw_line in result.stdout.splitlines():
            parts = raw_line.split("|", 1)
            if len(parts) != 2:
                continue
            job_id = parts[0].strip().split("_", 1)[0]
            comment = parts[1].rstrip("|").strip()
            if not job_id.isdigit():
                continue
            if exact is not None and comment != exact:
                continue
            if prefix is not None and not comment.startswith(prefix):
                continue
            matches.setdefault(comment, set()).add(job_id)
    if successful_queries == 0:
        raise SchedulerQueryUnavailable(
            "Could not query squeue or sacct for scheduler comments"
        )
    return matches


def query_scheduler_job_states(
    job_ids: Sequence[str],
    *,
    run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, str]:
    """Return one scheduler state for every requested durable job ID.

    Both the live queue and accounting are queried because a quiescent chain
    normally disappears from ``squeue`` before recovery begins. Missing state
    for even one ID is treated as unavailable authority rather than terminal.
    """
    requested = {str(job_id) for job_id in job_ids}
    if not requested or any(not job_id.isdecimal() for job_id in requested):
        raise ValueError("scheduler state query requires numeric job IDs")
    runner = run_command or subprocess.run
    joined = ",".join(sorted(requested))
    commands = (
        ["squeue", "--noheader", "--jobs", joined, "--format=%i|%T"],
        [
            "sacct", "--noheader", "--parsable2", "--jobs", joined,
            "--format=JobIDRaw,State",
        ],
    )
    states: dict[str, str] = {}
    successful_queries = 0
    for command in commands:
        try:
            result = runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result is None or result.returncode != 0:
            continue
        successful_queries += 1
        for raw_line in result.stdout.splitlines():
            parts = raw_line.rstrip("|").split("|", 1)
            if len(parts) != 2:
                continue
            raw_id, raw_state = (part.strip() for part in parts)
            base_id = raw_id.split(".", 1)[0].split("_", 1)[0]
            state = raw_state.split(maxsplit=1)[0].rstrip("+").upper()
            if base_id in requested and state:
                states.setdefault(base_id, state)
    if successful_queries == 0 or set(states) != requested:
        missing = ", ".join(sorted(requested - set(states)))
        raise SchedulerQueryUnavailable(
            "Could not determine scheduler state for job IDs"
            + (f": {missing}" if missing else "")
        )
    return states


def mirror_job_to_metadata(
    output_dir: Path,
    *,
    generation: str,
    token: str,
    role: str,
    job_id: str,
) -> None:
    """Mirror one scheduler role while retaining the legacy chunk index map."""
    path = job_metadata_path(output_dir)
    if not path.is_file():
        return
    metadata_lock = path.with_name(f".{path.name}.lock")
    with exclusive_path_lock(metadata_lock, timeout=60.0):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        payload["slurm_metadata_version"] = SCHEMA_VERSION
        payload["slurm_generation"] = generation
        jobs = payload.setdefault(JobMetadataKey.SLURM_JOB_IDS, {})
        if not isinstance(jobs, dict):
            jobs = {}
            payload[JobMetadataKey.SLURM_JOB_IDS] = jobs
        jobs[token] = {
            "job_id": str(job_id),
            "role": role,
            "generation": generation,
        }
        if role == "chunk" and token.startswith("chunk-"):
            index = token.removeprefix("chunk-")
            if index.isdecimal():
                chunks = payload.setdefault(JobMetadataKey.CHUNK_JOB_IDS, {})
                if isinstance(chunks, dict):
                    chunks[index] = str(job_id)
        atomic_write_json(path, payload)


def submit_with_lifecycle(
    output_dir: Path,
    *,
    generation: str,
    token: str,
    role: str,
    script_path: Path,
    dependencies: Sequence[str] = (),
    dependency_kind: SlurmDependencyKind = "afterany",
    round_index: int = 0,
    active_check: Callable[[], bool] | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    discover: Callable[[str], str | None] | None = None,
) -> str:
    """Submit exactly once with intent, fence, and job record under one lock."""
    validated_dependency_kind = _validate_dependency_kind(dependency_kind)
    comment = scheduler_comment(generation, token)
    runner = run_command or subprocess.run
    is_active = active_check or (
        lambda: generation_is_active(output_dir, generation)
    )
    with exclusive_path_lock(lifecycle_lock_path(output_dir), timeout=60.0):
        if not is_active():
            raise RuntimeError(
                f"SLURM generation {generation!r} is inactive or superseded"
            )
        existing = ledger_job_for_token(output_dir, generation, token)
        if existing:
            return existing

        rows = read_lifecycle_ledger(output_dir, generation=generation)
        intent_row = next(
            (
                row
                for row in reversed(rows)
                if row.get("token") == token
                and row.get("status") in {"intent", "blocked"}
            ),
            None,
        )
        has_intent = intent_row is not None
        effective_dependency_kind = (
            _dependency_kind_from_row(intent_row)
            if intent_row is not None
            else validated_dependency_kind
        )
        if has_intent:
            recovered = _metadata_job_for_token(
                output_dir, generation=generation, token=token
            )
            if recovered is None:
                recovered = (
                    discover(comment)
                    if discover is not None
                    else _single_job_for_comment(comment, run_command=runner)
                )
            if recovered:
                return _record_submission(
                    output_dir,
                    generation=generation,
                    token=token,
                    role=role,
                    job_id=recovered,
                    dependencies=dependencies,
                    dependency_kind=effective_dependency_kind,
                    round_index=round_index,
                    status="submitted",
                )
        else:
            append_lifecycle_entry(
                output_dir,
                generation=generation,
                token=token,
                role=role,
                status="intent",
                dependencies=dependencies,
                dependency_kind=effective_dependency_kind,
                round_index=round_index,
                comment=comment,
            )

        command = [
            "sbatch",
            "--parsable",
            "--export=ALL",
            "--comment",
            comment,
        ]
        if dependencies:
            command.extend(
                [
                    "--dependency",
                    f"{effective_dependency_kind}:{':'.join(dependencies)}",
                ]
            )
        command.append(str(script_path))
        last_error = "unknown submission failure"
        for attempt in range(3):
            if not is_active():
                raise RuntimeError(
                    f"SLURM generation {generation!r} became inactive"
                )
            try:
                result = runner(
                    command,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                    env=sbatch_submission_environment(),
                )
                job_id = result.stdout.strip().split(";", 1)[0]
                if not job_id.isdigit():
                    raise RuntimeError(
                        f"Invalid sbatch job id: {result.stdout!r}"
                    )
                return _record_submission(
                    output_dir,
                    generation=generation,
                    token=token,
                    role=role,
                    job_id=job_id,
                    dependencies=dependencies,
                    dependency_kind=effective_dependency_kind,
                    round_index=round_index,
                    status="submitted",
                )
            except (
                FileNotFoundError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                RuntimeError,
            ) as exc:
                last_error = str(exc)
                recovered = (
                    discover(comment)
                    if discover is not None
                    else _single_job_for_comment(comment, run_command=runner)
                )
                if recovered:
                    return _record_submission(
                        output_dir,
                        generation=generation,
                        token=token,
                        role=role,
                        job_id=recovered,
                        dependencies=dependencies,
                        dependency_kind=effective_dependency_kind,
                        round_index=round_index,
                        status="submitted",
                    )
                if isinstance(
                    exc,
                    (
                        FileNotFoundError,
                        subprocess.TimeoutExpired,
                        RuntimeError,
                    ),
                ):
                    append_lifecycle_entry(
                        output_dir,
                        generation=generation,
                        token=token,
                        role=role,
                        status="blocked",
                        dependencies=dependencies,
                        dependency_kind=effective_dependency_kind,
                        round_index=round_index,
                        comment=comment,
                    )
                    raise RuntimeError(
                        f"Ambiguous or unavailable {role} submission: "
                        f"{last_error}"
                    ) from exc
                if attempt < 2:
                    time.sleep(_SUBMIT_BACKOFF_SECONDS[attempt])
        append_lifecycle_entry(
            output_dir,
            generation=generation,
            token=token,
            role=role,
            status="blocked",
            dependencies=dependencies,
            dependency_kind=effective_dependency_kind,
            round_index=round_index,
            comment=comment,
        )
        raise RuntimeError(
            f"Could not submit {role} after 3 attempts: {last_error}"
        )


def deactivate_generation(output_dir: Path, generation: str) -> bool:
    """Write the inactive fence under the lifecycle lock."""
    with exclusive_path_lock(lifecycle_lock_path(output_dir), timeout=60.0):
        return _deactivate_generation_locked(output_dir, generation)


def _deactivate_generation_locked(output_dir: Path, generation: str) -> bool:
    """Write the inactive fence while the caller holds the lifecycle lock."""
    state = load_slurm_lifecycle(output_dir)
    if state is None or state.get("generation") != generation:
        return False
    if state.get("active") is not True:
        return False
    state["active"] = False
    state["updated_at"] = _timestamp()
    atomic_write_json(lifecycle_state_path(output_dir), state)
    return True


def mark_generation_failed(
    output_dir: Path, generation: str, error: str
) -> bool:
    """Durably mark one owned generation failed and inactive.

    The generation comparison prevents a late dispatcher from changing the
    lifecycle state of a newer launch that now owns the output directory.
    """
    with exclusive_path_lock(lifecycle_lock_path(output_dir), timeout=60.0):
        state = load_slurm_lifecycle(output_dir)
        if state is None or state.get("generation") != generation:
            return False
        state["active"] = False
        state["terminal_status"] = "failed"
        state["terminal_error"] = str(error)
        state["updated_at"] = _timestamp()
        atomic_write_json(lifecycle_state_path(output_dir), state)
    return True


def cancel_generation(
    output_dir: Path,
    generation: str,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    max_rescans: int = 3,
) -> CancellationResult:
    """Fence, reconcile, cancel, and rescan until the generation is quiescent."""
    runner = run_command or subprocess.run
    known_ids: set[str] = set()
    unresolved: set[str] = set()
    with exclusive_path_lock(lifecycle_lock_path(output_dir), timeout=60.0):
        _deactivate_generation_locked(output_dir, generation)
        rows = read_lifecycle_ledger(output_dir, generation=generation)
        submitted_tokens: set[str] = set()
        latest_active: dict[str, str] = {}
        intent_rows: dict[str, dict[str, Any]] = {}
        for row in rows:
            token = str(row.get("token", ""))
            if row.get("status") in {"submitted", "recovered"} and row.get(
                "job_id"
            ):
                latest_active[token] = str(row["job_id"])
                submitted_tokens.add(token)
            elif row.get("status") == "terminal":
                latest_active.pop(token, None)
            elif row.get("status") in {"intent", "blocked"}:
                intent_rows[token] = row
        known_ids.update(latest_active.values())
        unresolved = set(intent_rows) - submitted_tokens
        for token in tuple(unresolved):
            metadata_id = _metadata_job_for_token(
                output_dir, generation=generation, token=token
            )
            if metadata_id is None:
                continue
            row = intent_rows[token]
            known_ids.add(metadata_id)
            _record_submission(
                output_dir,
                generation=generation,
                token=token,
                role=str(row.get("role", "unknown")),
                job_id=metadata_id,
                dependencies=tuple(row.get("dependencies", [])),
                dependency_kind=_dependency_kind_from_row(row),
                round_index=int(row.get("round", 0)),
                status="submitted",
            )
            unresolved.discard(token)
        try:
            found = query_scheduler_comments(
                prefix=f"phenotypic:{generation}:",
                run_command=runner,
            )
        except SchedulerQueryUnavailable:
            found = {}
            scheduler_reconciled = False
        else:
            scheduler_reconciled = True
        found_tokens: set[str] = set()
        for comment, ids in found.items():
            token = comment.rsplit(":", 1)[-1]
            found_tokens.add(token)
            known_ids.update(ids)
            for job_id in sorted(ids):
                row = intent_rows.get(token, {})
                _record_submission(
                    output_dir,
                    generation=generation,
                    token=token,
                    role=str(row.get("role", "unknown")),
                    job_id=job_id,
                    dependencies=tuple(row.get("dependencies", [])),
                    dependency_kind=_dependency_kind_from_row(row),
                    round_index=int(row.get("round", 0)),
                    status="submitted",
                )
            unresolved.discard(token)
        if scheduler_reconciled:
            for token in sorted(unresolved - found_tokens):
                row = intent_rows[token]
                append_lifecycle_entry(
                    output_dir,
                    generation=generation,
                    token=token,
                    role=str(row.get("role", "unknown")),
                    status="reconciled-no-job",
                    dependencies=tuple(row.get("dependencies", [])),
                    dependency_kind=_dependency_kind_from_row(row),
                    round_index=int(row.get("round", 0)),
                )
                unresolved.discard(token)

    for _ in range(max(1, max_rescans)):
        if known_ids:
            try:
                runner(
                    ["scancel", *sorted(known_ids)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        try:
            found = query_scheduler_comments(
                prefix=f"phenotypic:{generation}:",
                include_accounting=False,
                run_command=runner,
            )
        except SchedulerQueryUnavailable:
            break
        rescanned = {job_id for ids in found.values() for job_id in ids}
        if not rescanned:
            return CancellationResult(
                tuple(sorted(known_ids)),
                tuple(sorted(unresolved)),
                not unresolved,
            )
        known_ids.update(rescanned)
    return CancellationResult(
        tuple(sorted(known_ids)), tuple(sorted(unresolved)), False
    )


def dispatch_continuation(
    output_dir: Path,
    *,
    generation: str,
    chunk_index: int,
    chunk_script: Path,
    dispatcher_script: Path | None = None,
    finalizer_script: Path | None = None,
    dependency_kind: SlurmDependencyKind = "afterany",
) -> tuple[str, str | None]:
    """Submit a later chunk and its optional dependent continuation safely."""
    validated_dependency_kind = _validate_dependency_kind(dependency_kind)
    chunk_id: str | None = None
    continuation_id = None
    try:
        chunk_id = submit_with_lifecycle(
            output_dir,
            generation=generation,
            token=f"chunk-{chunk_index}",
            role="chunk",
            script_path=chunk_script,
        )
        if dispatcher_script is not None:
            continuation_id = submit_with_lifecycle(
                output_dir,
                generation=generation,
                token=f"dispatcher-{chunk_index + 1}",
                role="dispatcher",
                script_path=dispatcher_script,
                dependencies=(chunk_id,),
                dependency_kind=validated_dependency_kind,
            )
        elif finalizer_script is not None:
            continuation_id = submit_with_lifecycle(
                output_dir,
                generation=generation,
                token="finalizer",
                role="finalizer",
                script_path=finalizer_script,
                dependencies=(chunk_id,),
                dependency_kind=validated_dependency_kind,
            )
    except Exception as exc:
        try:
            cancellation = cancel_generation(output_dir, generation)
        except Exception:
            # Cancellation performs scheduler reconciliation, but the local
            # fence is the load-bearing safety boundary. Preserve it even if
            # the scheduler query/cancel path itself fails unexpectedly.
            deactivate_generation(output_dir, generation)
            detail = (
                "the generation was fenced but scheduler reconciliation "
                "failed"
            )
        else:
            detail = (
                "the generation was fenced and reconciled"
                if cancellation.quiescent
                else (
                    "the generation was fenced but reconciliation is "
                    "incomplete"
                )
            )
        failure_stage = (
            "Next chunk submission failed"
            if chunk_id is None
            else "Continuation submission failed after the next chunk was submitted"
        )
        terminal_error = f"{failure_stage}; {detail}: {exc}"
        mark_generation_failed(output_dir, generation, terminal_error)
        raise RuntimeError(terminal_error) from exc
    if chunk_id is None:  # pragma: no cover - guarded by the exception path
        raise RuntimeError("Dynamic chunk submission returned no job id")
    return chunk_id, continuation_id


def _record_submission(
    output_dir: Path,
    *,
    generation: str,
    token: str,
    role: str,
    job_id: str,
    dependencies: Sequence[str],
    dependency_kind: SlurmDependencyKind,
    round_index: int,
    status: str,
) -> str:
    append_lifecycle_entry(
        output_dir,
        generation=generation,
        token=token,
        role=role,
        status=status,
        job_id=job_id,
        dependencies=dependencies,
        dependency_kind=dependency_kind,
        round_index=round_index,
    )
    mirror_job_to_metadata(
        output_dir,
        generation=generation,
        token=token,
        role=role,
        job_id=job_id,
    )
    return job_id


def _single_job_for_comment(
    comment: str,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[str]],
) -> str | None:
    matches = query_scheduler_comments(exact=comment, run_command=run_command)
    ids = sorted(matches.get(comment, ()))
    if len(ids) > 1:
        raise RuntimeError(
            f"Ambiguous scheduler comment {comment!r} matched jobs "
            f"{', '.join(ids)}"
        )
    return ids[0] if ids else None


def _metadata_job_for_token(
    output_dir: Path, *, generation: str, token: str
) -> str | None:
    """Read a token from either versioned or pre-versioned job metadata."""
    try:
        payload = json.loads(
            job_metadata_path(output_dir).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    jobs = payload.get(JobMetadataKey.SLURM_JOB_IDS)
    if not isinstance(jobs, dict):
        return None
    value = jobs.get(token)
    if isinstance(value, str):
        return value if value.isdigit() else None
    if not isinstance(value, dict):
        return None
    found_generation = value.get("generation")
    if found_generation not in {None, generation}:
        return None
    job_id = value.get("job_id")
    return str(job_id) if str(job_id).isdigit() else None


def _parse_json_lines(content: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in content.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _validate_dependency_kind(value: str) -> SlurmDependencyKind:
    """Validate one SLURM dependency condition at a public API boundary."""
    if value not in {"afterany", "afterok"}:
        raise ValueError("dependency_kind must be 'afterany' or 'afterok'")
    return cast(SlurmDependencyKind, value)


def _dependency_kind_from_row(row: Mapping[str, Any]) -> SlurmDependencyKind:
    """Read a journaled dependency kind with legacy ``afterany`` fallback."""
    return _validate_dependency_kind(
        str(row.get("dependency_kind", "afterany"))
    )


def _timestamp() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _dispatch_from_argv(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--chunk-script", type=Path, required=True)
    parser.add_argument("--dispatcher-script", type=Path)
    parser.add_argument("--finalizer-script", type=Path)
    parser.add_argument(
        "--dependency-kind",
        choices=("afterany", "afterok"),
        default="afterany",
    )
    args = parser.parse_args(argv)
    dispatch_continuation(
        args.output,
        generation=args.generation,
        chunk_index=args.chunk_index,
        chunk_script=args.chunk_script,
        dispatcher_script=args.dispatcher_script,
        finalizer_script=args.finalizer_script,
        dependency_kind=args.dependency_kind,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_dispatch_from_argv())


__all__ = [
    "CancellationResult",
    "SchedulerQueryUnavailable",
    "append_lifecycle_entry",
    "assert_generation_active",
    "cancel_generation",
    "deactivate_generation",
    "dispatch_continuation",
    "generation_is_active",
    "initialize_slurm_lifecycle",
    "ledger_job_for_token",
    "lifecycle_ledger_path",
    "lifecycle_lock_path",
    "lifecycle_state_path",
    "load_slurm_lifecycle",
    "mirror_job_to_metadata",
    "new_slurm_generation",
    "query_scheduler_comments",
    "query_scheduler_job_states",
    "read_lifecycle_ledger",
    "scheduler_comment",
    "submit_with_lifecycle",
]
