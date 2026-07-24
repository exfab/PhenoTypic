"""Dash-free SLURM submission and scheduler-metadata reconciliation.

The CLI owns scheduler submission. This module launches that CLI and reads its
durable metadata, append-only lifecycle ledger, and scheduler comments. It
never parses Rich-formatted CLI output for scheduler identity.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from uuid import NAMESPACE_URL, UUID, uuid5

from phenotypic._cli._cli_slurm_lifecycle import (
    SchedulerQueryUnavailable,
    append_lifecycle_entry,
    cancel_generation,
    load_slurm_lifecycle,
    mirror_job_to_metadata,
    query_scheduler_comments,
    read_lifecycle_ledger,
)
from phenotypic.gui.run_console._state import RunConsoleState, to_argv
from phenotypic.sdk_ import JobMetadataKey, job_metadata_path

__all__ = [
    "SlurmSubmitError",
    "SlurmSubmitResult",
    "SubmittedJobSet",
    "read_submitted_job_set",
    "submit_slurm",
    "wait_for_job_id",
]

logger = logging.getLogger(__name__)

_SLURM_DIRECT_KEYS: tuple[str, ...] = (
    "partition",
    "time",
    "mem",
    "cpus_per_task",
    "gpus",
)
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


class SlurmSubmitError(RuntimeError):
    """Raised when submission cannot be attached to durable scheduler work."""


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
class _JobEvidence:
    """One scheduler id together with the strongest known role and token."""

    job_id: str
    role: str
    token: str
    source_rank: int


def _slurm_argv_extension(slurm_args: dict[str, object]) -> list[str]:
    """Build repeated ``--slurm key=value`` arguments."""
    if not slurm_args:
        return []
    pairs: list[tuple[str, str]] = []
    for key in _SLURM_DIRECT_KEYS:
        value = slurm_args.get(key)
        if value is not None and value != "":
            pairs.append((key, str(value)))
    extra = slurm_args.get("extra") or {}
    if isinstance(extra, dict):
        for key, value in extra.items():
            if key is not None and value is not None and str(key) and str(value):
                pairs.append((str(key), str(value)))
    argv: list[str] = []
    for key, value in pairs:
        argv.extend(["--slurm", f"{key}={value}"])
    return argv


def _build_subprocess_argv(state: RunConsoleState) -> list[str]:
    """Assemble the CLI subprocess argument vector."""
    return [
        sys.executable,
        "-m",
        "phenotypic",
        *to_argv(state),
        *_slurm_argv_extension(state.slurm_args or {}),
    ]


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


def _write_submitter_logs(
    output_dir: Path, stdout: str, stderr: str
) -> tuple[Path, Path]:
    """Persist bounded-source submitter output beside scheduler logs."""
    log_dir = output_dir / ".phenotypic" / "logs" / "gui"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "submitter.stdout.log"
    stderr_path = log_dir / "submitter.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return stdout_path, stderr_path


def _reconcile_submission(
    output_dir: Path,
) -> SubmittedJobSet | None:
    """Resolve metadata, ledger, and generation comments after ambiguity."""
    jobs = read_submitted_job_set(output_dir)
    state = load_slurm_lifecycle(output_dir)
    generation_raw = state.get("generation") if state else None
    generation = _uuid_from_generation(generation_raw)
    if generation is None:
        return jobs
    if state is not None and state.get("active") is False:
        cancel_generation(output_dir, str(generation_raw))
        return None
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
        return jobs
    prefix = f"phenotypic:{generation_raw}:"
    try:
        matches = query_scheduler_comments(prefix=prefix)
    except SchedulerQueryUnavailable:
        return jobs
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
    return read_submitted_job_set(output_dir)


def submit_slurm(
    state: RunConsoleState,
    *,
    sandbox_root: Path,
    timeout: float = 60.0,
) -> SlurmSubmitResult:
    """Run the CLI submitter and attach to all durably submitted jobs.

    Timeout and abnormal-exit paths reconcile the lifecycle records before
    reporting failure. If durable evidence proves that work was submitted,
    the method returns a reconciled result instead of orphaning that work.
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
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(sandbox_root),
            check=False,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        returncode = completed.returncode
        if returncode != 0:
            ambiguous_error = SlurmSubmitError(
                "SLURM submission subprocess exited with code "
                f"{returncode}. stderr:\n{stderr or '<empty>'}\n"
                f"stdout:\n{stdout or '<empty>'}"
            )
    except subprocess.TimeoutExpired as err:
        stdout = err.stdout if isinstance(err.stdout, str) else ""
        stderr = err.stderr if isinstance(err.stderr, str) else ""
        returncode = -1
        ambiguous_error = SlurmSubmitError(
            f"SLURM submission timed out after {timeout:.1f}s. "
            f"stderr:\n{stderr or '<empty>'}\n"
            f"stdout:\n{stdout or '<empty>'}"
        )
    except FileNotFoundError as err:
        raise SlurmSubmitError(
            f"Failed to launch SLURM submitter subprocess: {err}"
        ) from err

    _write_submitter_logs(output_dir, stdout, stderr)
    jobs = (
        _reconcile_submission(output_dir)
        if ambiguous_error is not None
        else read_submitted_job_set(output_dir)
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
