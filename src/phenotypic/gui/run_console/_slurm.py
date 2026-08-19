"""Dash-free SLURM submission and scheduler-metadata reconciliation.

The CLI owns scheduler submission. This module launches that CLI and reads its
durable metadata, append-only lifecycle ledger, and scheduler comments. It
never parses Rich-formatted CLI output for scheduler identity.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from phenotypic._cli._cli_slurm_lifecycle import (
    SchedulerQueryUnavailable,
    append_lifecycle_entry,
    cancel_generation,
    load_slurm_lifecycle,
    mirror_job_to_metadata,
    query_scheduler_comments,
    read_lifecycle_ledger,
)
from phenotypic._services.argv import (
    slurm_argv_extension,
    to_subprocess_argv,
)
from phenotypic.gui.run_console._state import RunConsoleState
from phenotypic.sdk_ import JobMetadataKey, job_metadata_path

__all__ = [
    "SlurmSubmitError",
    "SlurmSubmitPending",
    "SlurmSubmitResult",
    "SubmittedJobSet",
    "read_submitted_job_set",
    "submit_slurm",
    "wait_for_job_id",
]

logger = logging.getLogger(__name__)

_PRIMARY_ROLE_ORDER = (
    "controller-initial",
    "chunk",
    "controller",
    "finalizer",
    "dispatcher",
    "recovery-controller",
    "recovery",
    "unknown",
)
_SUBMITTER_TAIL_CHARS = 128 * 1024
_SUBMITTER_READ_CHARS = 8 * 1024
_SUBMITTER_TAIL_CHUNKS = (
    _SUBMITTER_TAIL_CHARS // _SUBMITTER_READ_CHARS
)
_SUBMITTER_TERM_GRACE_SECONDS = 2.0


class SlurmSubmitError(RuntimeError):
    """Raised when submission cannot be attached to durable scheduler work."""


class SlurmSubmitPending(SlurmSubmitError):
    """Raised when a durable submission intent is still recoverable.

    This is not terminal failure evidence. S3 must retain the generation's
    registry record as ``submitting`` or ``unknown``, bind its scheduler
    generation to the lifecycle observer, and let comment reconciliation
    continue server-side.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        generation: UUID,
        unresolved_tokens: tuple[str, ...],
        submitted_jobs: SubmittedJobSet | None,
        scheduler_available: bool,
        returncode: int,
    ) -> None:
        availability = (
            "scheduler query returned no matching job"
            if scheduler_available
            else "scheduler query is unavailable"
        )
        super().__init__(
            "SLURM submission remains recoverable because durable intent(s) "
            f"{', '.join(unresolved_tokens)} are unresolved and {availability}."
        )
        self.output_dir = output_dir
        self.generation = generation
        self.unresolved_tokens = unresolved_tokens
        self.submitted_jobs = submitted_jobs
        self.scheduler_available = scheduler_available
        self.returncode = returncode


@dataclass(frozen=True)
class SubmittedJobSet:
    """Every scheduler handle belonging to one launch generation.

    Attributes:
        primary_id: Deterministic compact scheduler handle.
        all_ids: De-duplicated scheduler ids in deterministic order.
        roles: Mapping from scheduler role to ids with that role.
        generation: Durable scheduler launch generation.
    """

    primary_id: str
    all_ids: tuple[str, ...]
    roles: Mapping[str, tuple[str, ...]]
    generation: UUID


@dataclass(frozen=True)
class SlurmSubmitResult:
    """Outcome of a successful or successfully reconciled submission."""

    job_id: str
    output_dir: Path
    stdout: str
    stderr: str
    returncode: int
    submitted_jobs: SubmittedJobSet | None = None
    reconciled: bool = False


@dataclass(frozen=True)
class _StreamedProcessResult:
    """Bounded result of one live-tee submitter subprocess."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    stream_error: str | None = None


@dataclass(frozen=True)
class _JobEvidence:
    """One scheduler id together with the strongest known role and token."""

    job_id: str
    role: str
    token: str
    source_rank: int


@dataclass(frozen=True)
class _SubmissionReconciliation:
    """Durable state recovered after an ambiguous submitter exit."""

    jobs: SubmittedJobSet | None
    generation: UUID | None
    unresolved_tokens: tuple[str, ...]
    scheduler_available: bool
    cancelled: bool = False


# Both emitters were promoted to ``phenotypic._services.argv`` so the MCP
# server renders the identical command line (spec 05 §5.4 digests it). These
# names are bindings, not wrappers: a wrapper that re-added the ``--slurm``
# pairs on top of the promoted composition would double-emit every one of
# them, and both halves would still read as correct in isolation.
_slurm_argv_extension = slurm_argv_extension
_build_subprocess_argv = to_subprocess_argv


def _read_metadata(output_dir: Path) -> dict[str, object]:
    """Return valid scheduler metadata or an empty mapping."""
    try:
        payload = json.loads(
            job_metadata_path(output_dir).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _uuid_from_generation(value: object) -> UUID | None:
    """Parse a UUID generation emitted with or without hyphens."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return UUID(value)
    except ValueError:
        return None


def _generation_for(
    output_dir: Path,
    metadata: Mapping[str, object],
    expected_generation: UUID | None,
) -> UUID:
    """Resolve generation without allowing stale metadata to win."""
    if expected_generation is not None:
        return expected_generation
    candidates: tuple[object, ...] = (
        (load_slurm_lifecycle(output_dir) or {}).get("generation"),
        metadata.get("slurm_generation"),
        metadata.get(JobMetadataKey.ORCHESTRATION_EPOCH),
    )
    for candidate in candidates:
        parsed = _uuid_from_generation(candidate)
        if parsed is not None:
            return parsed
    for row in reversed(read_lifecycle_ledger(output_dir)):
        parsed = _uuid_from_generation(row.get("generation"))
        if parsed is not None:
            return parsed
    # Pre-versioned metadata did not carry a generation. A stable namespace
    # UUID lets the typed compatibility reader represent it without confusing
    # it with a newly allocated GUI launch generation.
    return uuid5(
        NAMESPACE_URL,
        f"phenotypic:legacy-slurm:{output_dir.resolve(strict=False)}",
    )


def _canonical_job_id(value: object) -> str | None:
    """Normalize a scheduler task id to its array/job master id."""
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None
    text = str(value).strip()
    base = text.split("_", 1)[0].split(".", 1)[0]
    return base if base.isdigit() else None


def _token_sort_key(token: str) -> tuple[str, int, str]:
    """Sort numbered lifecycle tokens naturally and deterministically."""
    prefix, separator, suffix = token.rpartition("-")
    if separator and suffix.isdecimal():
        return (prefix, int(suffix), token)
    return (token, -1, token)


def _collect_job_evidence(
    output_dir: Path,
    metadata: Mapping[str, object],
    generation: UUID,
) -> list[_JobEvidence]:
    """Merge versioned metadata, legacy mappings, and the lifecycle ledger."""
    found: dict[str, _JobEvidence] = {}
    generation_texts = {str(generation), generation.hex}
    metadata_generation = _uuid_from_generation(
        metadata.get("slurm_generation")
        or metadata.get(JobMetadataKey.ORCHESTRATION_EPOCH)
    )
    metadata_matches = (
        metadata_generation is None or metadata_generation == generation
    )

    def add(job_id: object, role: str, token: str, rank: int) -> None:
        canonical = _canonical_job_id(job_id)
        if canonical is None:
            return
        prior = found.get(canonical)
        evidence = _JobEvidence(canonical, role or "unknown", token, rank)
        if prior is None or evidence.source_rank > prior.source_rank:
            found[canonical] = evidence

    raw_jobs = metadata.get(JobMetadataKey.SLURM_JOB_IDS)
    if metadata_matches and isinstance(raw_jobs, Mapping):
        for raw_token, raw_value in raw_jobs.items():
            token = str(raw_token)
            if isinstance(raw_value, Mapping):
                row_generation = raw_value.get("generation")
                if (
                    row_generation is not None
                    and str(row_generation) not in generation_texts
                ):
                    continue
                add(
                    raw_value.get("job_id"),
                    str(raw_value.get("role", "unknown")),
                    token,
                    2,
                )
            else:
                add(raw_value, "unknown", token, 1)

    staged = JobMetadataKey.ORCHESTRATION_EPOCH in metadata
    raw_chunks = metadata.get(JobMetadataKey.CHUNK_JOB_IDS)
    if metadata_matches and isinstance(raw_chunks, Mapping):
        for raw_index, raw_id in raw_chunks.items():
            index = str(raw_index)
            add(
                raw_id,
                "unknown" if staged else "chunk",
                f"chunk-{index}",
                1,
            )

    for row in read_lifecycle_ledger(output_dir):
        row_generation = row.get("generation", row.get("epoch"))
        if str(row_generation) not in generation_texts:
            continue
        if row.get("status") not in {"submitted", "recovered", "terminal"}:
            continue
        add(
            row.get("job_id"),
            str(row.get("role", "unknown")),
            str(row.get("token", "unknown")),
            3,
        )
    return list(found.values())


def _primary_evidence(items: list[_JobEvidence]) -> _JobEvidence:
    """Choose the primary by explicit role and natural token order."""
    if not items:
        raise SlurmSubmitError(
            "SLURM metadata and lifecycle ledger contain no submitted jobs."
        )
    role_rank = {
        role: index for index, role in enumerate(_PRIMARY_ROLE_ORDER)
    }
    return min(
        items,
        key=lambda item: (
            role_rank.get(item.role, len(role_rank)),
            _token_sort_key(item.token),
            int(item.job_id),
        ),
    )


def read_submitted_job_set(
    output_dir: Path,
    *,
    expected_generation: UUID | None = None,
) -> SubmittedJobSet | None:
    """Read and merge all durable scheduler identities for an output.

    Args:
        output_dir: CLI output root.
        expected_generation: Optional launch generation fence. Versioned
            evidence from another generation is ignored.

    Returns:
        A typed job set, or ``None`` before any job id is durably visible.
    """
    metadata = _read_metadata(output_dir)
    generation = _generation_for(
        output_dir, metadata, expected_generation
    )
    evidence = _collect_job_evidence(output_dir, metadata, generation)
    if not evidence:
        return None
    primary = _primary_evidence(evidence)
    role_groups: dict[str, list[_JobEvidence]] = defaultdict(list)
    for item in evidence:
        role_groups[item.role].append(item)
    roles = {
        role: tuple(
            item.job_id
            for item in sorted(
                role_items,
                key=lambda item: (_token_sort_key(item.token), int(item.job_id)),
            )
        )
        for role, role_items in sorted(role_groups.items())
    }
    all_ids = tuple(
        dict.fromkeys(
            (
                primary.job_id,
                *(
                    item.job_id
                    for item in sorted(
                        evidence,
                        key=lambda item: (
                            _token_sort_key(item.token),
                            int(item.job_id),
                        ),
                    )
                ),
            )
        )
    )
    return SubmittedJobSet(
        primary_id=primary.job_id,
        all_ids=all_ids,
        roles=MappingProxyType(roles),
        generation=generation,
    )


def _submitter_log_paths(
    output_dir: Path,
    generation: UUID,
) -> tuple[Path, Path]:
    """Return generation-specific GUI submitter log paths."""
    log_dir = output_dir / ".phenotypic" / "logs" / "gui"
    log_dir.mkdir(parents=True, exist_ok=True)
    token = generation.hex
    return (
        log_dir / f"submitter.{token}.stdout.log",
        log_dir / f"submitter.{token}.stderr.log",
    )


def _tee_submitter_stream(
    stream: IO[str],
    path: Path,
    tail: deque[str],
    tail_lock: threading.Lock,
    errors: list[str],
    errors_lock: threading.Lock,
) -> None:
    """Drain one child pipe to disk and a bounded in-memory line tail."""
    handle: IO[str] | None = None
    try:
        try:
            handle = path.open("w", encoding="utf-8")
        except OSError as error:
            with errors_lock:
                errors.append(f"{path.name}: {error}")
        for chunk in iter(
            lambda: stream.readline(_SUBMITTER_READ_CHARS),
            "",
        ):
            if handle is not None:
                try:
                    handle.write(chunk)
                    handle.flush()
                except OSError as error:
                    with errors_lock:
                        errors.append(f"{path.name}: {error}")
                    try:
                        handle.close()
                    except OSError:
                        pass
                    handle = None
            with tail_lock:
                tail.append(chunk)
    except OSError as error:
        with errors_lock:
            errors.append(f"{path.name} stream: {error}")
    finally:
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        try:
            stream.close()
        except OSError:
            pass


def _bounded_tail_text(
    lines: deque[str],
    *,
    lock: threading.Lock,
) -> str:
    """Join a line tail while enforcing a final character bound."""
    with lock:
        return "".join(lines)[-_SUBMITTER_TAIL_CHARS:]


def _run_submitter_streamed(
    argv: list[str],
    *,
    output_dir: Path,
    log_generation: UUID,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> _StreamedProcessResult:
    """Run the CLI submitter while teeing stdout and stderr as they arrive."""
    stdout_path, stderr_path = _submitter_log_paths(
        output_dir,
        log_generation,
    )
    stdout_tail: deque[str] = deque(maxlen=_SUBMITTER_TAIL_CHUNKS)
    stderr_tail: deque[str] = deque(maxlen=_SUBMITTER_TAIL_CHUNKS)
    stdout_tail_lock = threading.Lock()
    stderr_tail_lock = threading.Lock()
    stream_errors: list[str] = []
    stream_errors_lock = threading.Lock()
    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess,
            "CREATE_NEW_PROCESS_GROUP",
            0,
        )
    else:
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=str(cwd),
        env=env,
        **popen_kwargs,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    readers = (
        threading.Thread(
            target=_tee_submitter_stream,
            args=(
                process.stdout,
                stdout_path,
                stdout_tail,
                stdout_tail_lock,
                stream_errors,
                stream_errors_lock,
            ),
            daemon=True,
            name=f"phenotypic-submit-stdout-{log_generation.hex[:8]}",
        ),
        threading.Thread(
            target=_tee_submitter_stream,
            args=(
                process.stderr,
                stderr_path,
                stderr_tail,
                stderr_tail_lock,
                stream_errors,
                stream_errors_lock,
            ),
            daemon=True,
            name=f"phenotypic-submit-stderr-{log_generation.hex[:8]}",
        ),
    )
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _signal_submitter_tree(process, force=False)
        try:
            returncode = process.wait(timeout=_SUBMITTER_TERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_submitter_tree(process, force=True)
            returncode = process.wait()
        if os.name != "nt":
            # The direct CLI may exit on SIGTERM while a pipe-inheriting
            # descendant ignores it. Kill the isolated group once more so
            # resistant descendants cannot survive or pin the reader pipes.
            _signal_submitter_tree(process, force=True)
    streams = (process.stdout, process.stderr)
    for reader, stream in zip(readers, streams, strict=True):
        reader.join(timeout=_SUBMITTER_TERM_GRACE_SECONDS)
        if reader.is_alive():
            try:
                os.close(stream.fileno())
            except OSError:
                pass
            reader.join(timeout=_SUBMITTER_TERM_GRACE_SECONDS)
        if reader.is_alive():
            with stream_errors_lock:
                stream_errors.append(
                    f"{reader.name} did not stop after pipe closure"
                )
    with stream_errors_lock:
        stream_error = "; ".join(stream_errors) or None
    return _StreamedProcessResult(
        stdout=_bounded_tail_text(stdout_tail, lock=stdout_tail_lock),
        stderr=_bounded_tail_text(stderr_tail, lock=stderr_tail_lock),
        returncode=returncode,
        timed_out=timed_out,
        stream_error=stream_error,
    )


def _signal_submitter_tree(
    process: subprocess.Popen[str],
    *,
    force: bool,
) -> None:
    """Signal the isolated submitter process tree after a timeout."""
    if os.name != "nt":
        try:
            os.killpg(
                process.pid,
                signal.SIGKILL if force else signal.SIGTERM,
            )
            return
        except ProcessLookupError:
            return
    if force:
        process.kill()
    else:
        process.terminate()


def _reconcile_submission(
    output_dir: Path,
) -> _SubmissionReconciliation:
    """Resolve metadata, ledger, and generation comments after ambiguity."""
    jobs = read_submitted_job_set(output_dir)
    state = load_slurm_lifecycle(output_dir)
    generation_raw = state.get("generation") if state else None
    generation = _uuid_from_generation(generation_raw)
    if generation is None:
        return _SubmissionReconciliation(jobs, None, (), False)
    if state is not None and state.get("active") is False:
        cancel_generation(output_dir, str(generation_raw))
        return _SubmissionReconciliation(
            None, generation, (), True, cancelled=True
        )
    rows = read_lifecycle_ledger(
        output_dir, generation=str(generation_raw)
    )
    latest: dict[str, Mapping[str, object]] = {}
    for row in rows:
        token = str(row.get("token", ""))
        if token:
            latest[token] = row
    unresolved = {
        token: row
        for token, row in latest.items()
        if row.get("status") in {"intent", "blocked"}
    }
    if not unresolved:
        return _SubmissionReconciliation(jobs, generation, (), True)
    prefix = f"phenotypic:{generation_raw}:"
    try:
        matches = query_scheduler_comments(prefix=prefix)
    except SchedulerQueryUnavailable:
        return _SubmissionReconciliation(
            jobs,
            generation,
            tuple(sorted(unresolved)),
            False,
        )
    for comment, ids in matches.items():
        if not comment.startswith(prefix):
            continue
        token = comment.removeprefix(prefix)
        intent = unresolved.get(token)
        if intent is None:
            continue
        role = str(intent.get("role", "unknown"))
        dependencies_raw = intent.get("dependencies", ())
        dependencies = (
            tuple(str(item) for item in dependencies_raw)
            if isinstance(dependencies_raw, (list, tuple))
            else ()
        )
        round_raw = intent.get("round", 0)
        round_index = (
            round_raw
            if isinstance(round_raw, int) and not isinstance(round_raw, bool)
            else 0
        )
        for job_id in sorted(ids, key=int):
            append_lifecycle_entry(
                output_dir,
                generation=str(generation_raw),
                token=token,
                role=role,
                status="recovered",
                job_id=job_id,
                dependencies=dependencies,
                round_index=round_index,
                comment=comment,
            )
            mirror_job_to_metadata(
                output_dir,
                generation=str(generation_raw),
                token=token,
                role=role,
                job_id=job_id,
            )
    rows = read_lifecycle_ledger(
        output_dir, generation=str(generation_raw)
    )
    latest = {
        str(row.get("token", "")): row
        for row in rows
        if str(row.get("token", ""))
    }
    remaining = tuple(
        sorted(
            token
            for token, row in latest.items()
            if row.get("status") in {"intent", "blocked"}
        )
    )
    return _SubmissionReconciliation(
        read_submitted_job_set(
            output_dir, expected_generation=generation
        ),
        generation,
        remaining,
        True,
    )


def submit_slurm(
    state: RunConsoleState,
    *,
    sandbox_root: Path,
    record_generation: UUID | None = None,
    timeout: float = 60.0,
) -> SlurmSubmitResult:
    """Run the CLI submitter and attach to all durably submitted jobs.

    Timeout and abnormal-exit paths reconcile the lifecycle records before
    reporting failure. If durable evidence proves that work was submitted,
    the method returns a reconciled result instead of orphaning that work.

    Args:
        state: Validated Run Console request.
        sandbox_root: Working directory for the CLI subprocess.
        record_generation: Durable GUI owner generation propagated into CLI
            scheduler metadata.
        timeout: Maximum submitter subprocess duration in seconds.
    """
    if state.mode != "slurm":
        raise SlurmSubmitError(
            "SLURM submission requires RunConsoleState.mode='slurm'."
        )
    if not _slurm_argv_extension(state.slurm_args or {}):
        raise SlurmSubmitError(
            "SLURM submission requires a non-empty SLURM configuration."
        )
    if state.output_dir is None:
        raise SlurmSubmitError(
            "RunConsoleState.output_dir is required for SLURM submission."
        )
    try:
        argv = _build_subprocess_argv(state)
    except ValueError as err:
        raise SlurmSubmitError(str(err)) from err
    output_dir = Path(state.output_dir)
    stdout = ""
    stderr = ""
    returncode = 0
    ambiguous_error: BaseException | None = None
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if record_generation is not None:
        env["PHENOTYPIC_GUI_RECORD_GENERATION"] = str(record_generation)
    log_generation = record_generation or uuid4()
    try:
        completed = _run_submitter_streamed(
            argv,
            output_dir=output_dir,
            log_generation=log_generation,
            cwd=sandbox_root,
            env=env,
            timeout=timeout,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.stream_error is not None:
            stream_diagnostic = (
                "Submitter log streaming failed: "
                f"{completed.stream_error}"
            )
            stderr = (
                f"{stderr.rstrip()}\n{stream_diagnostic}"
                if stderr
                else stream_diagnostic
            )
        returncode = -1 if completed.timed_out else completed.returncode
        if completed.timed_out:
            ambiguous_error = SlurmSubmitError(
                f"SLURM submission timed out after {timeout:.1f}s. "
                f"stderr:\n{stderr or '<empty>'}\n"
                f"stdout:\n{stdout or '<empty>'}"
            )
        elif completed.stream_error is not None:
            ambiguous_error = SlurmSubmitError(
                "SLURM submitter logging failed, so submission status is "
                f"ambiguous. stderr:\n{stderr or '<empty>'}\n"
                f"stdout:\n{stdout or '<empty>'}"
            )
        elif returncode != 0:
            ambiguous_error = SlurmSubmitError(
                "SLURM submission subprocess exited with code "
                f"{returncode}. stderr:\n{stderr or '<empty>'}\n"
                f"stdout:\n{stdout or '<empty>'}"
            )
    except FileNotFoundError as err:
        raise SlurmSubmitError(
            f"Failed to launch SLURM submitter subprocess: {err}"
        ) from err

    jobs = read_submitted_job_set(output_dir)
    reconciliation = (
        _reconcile_submission(output_dir)
        if ambiguous_error is not None or jobs is None
        else _SubmissionReconciliation(jobs, jobs.generation, (), True)
    )
    jobs = reconciliation.jobs
    if (
        reconciliation.generation is not None
        and reconciliation.unresolved_tokens
    ):
        raise SlurmSubmitPending(
            output_dir=output_dir,
            generation=reconciliation.generation,
            unresolved_tokens=reconciliation.unresolved_tokens,
            submitted_jobs=jobs,
            scheduler_available=reconciliation.scheduler_available,
            returncode=returncode,
        )
    if jobs is None:
        if ambiguous_error is not None:
            raise ambiguous_error
        raise SlurmSubmitError(
            "SLURM submission subprocess exited cleanly but "
            f"{job_metadata_path(output_dir)} and the lifecycle ledger "
            f"contain no submitted jobs. stderr:\n{stderr or '<empty>'}"
        )
    return SlurmSubmitResult(
        job_id=jobs.primary_id,
        output_dir=output_dir,
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        submitted_jobs=jobs,
        reconciled=ambiguous_error is not None,
    )


def wait_for_job_id(
    output_dir: Path,
    *,
    timeout: float = 5.0,
    poll_interval: float = 0.25,
) -> str | None:
    """Poll durable scheduler evidence until a primary id appears."""
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        jobs = read_submitted_job_set(output_dir)
        if jobs is not None:
            return jobs.primary_id
        if time.monotonic() >= deadline:
            return None
        time.sleep(max(0.0, poll_interval))
