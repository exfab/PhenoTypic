"""Bounded, Dash-free observation of SLURM lifecycle state and logs."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import UUID

from phenotypic._cli._cli_slurm_lifecycle import (
    SchedulerQueryUnavailable,
    append_lifecycle_entry,
    load_slurm_lifecycle,
    mirror_job_to_metadata,
    query_scheduler_comments,
    read_lifecycle_ledger,
)
from phenotypic._cli._cli_staged_orchestration import (
    load_orchestration_state,
    staged_completion_matches,
)
from phenotypic._cli._cli_staged_resume import stage3_completion_exists
from phenotypic.gui.run_console._slurm import (
    SubmittedJobSet,
    read_submitted_job_set,
)
from phenotypic.gui.shell._runs_registry import (
    RunRecord,
    RunRegistry,
    RunStatus,
)
from phenotypic.sdk_ import (
    DashboardManifestKey,
    JobMetadataKey,
    job_metadata_path,
    resolve_manifest_json_path,
    run_completion_marker_path,
)

logger = logging.getLogger(__name__)

__all__ = [
    "IncrementalLogBatch",
    "IncrementalLogReader",
    "LogCursor",
    "SchedulerClient",
    "SchedulerCommentQueryResult",
    "SchedulerGenerationBinding",
    "SchedulerQueryResult",
    "SlurmCommandScheduler",
    "SlurmLifecycleObserver",
    "discover_log_files",
]

_TERMINAL_RUN_STATUSES = frozenset({"complete", "failed", "cancelled"})
_QUEUED_STATES = frozenset(
    {
        "PENDING",
        "CONFIGURING",
        "RESV_DEL_HOLD",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "REQUEUED",
    }
)
_RUNNING_STATES = frozenset(
    {"RUNNING", "COMPLETING", "SIGNALING", "STAGE_OUT"}
)
_SUCCESS_STATES = frozenset({"COMPLETED"})
_FAILURE_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "TIMEOUT",
    }
)
_TERMINAL_SCHEDULER_STATES = _SUCCESS_STATES | _FAILURE_STATES
_STAGED_FAILURE_PHASES = frozenset(
    {"failed", "failure", "exhausted", "stage2_failed"}
)


@dataclass(frozen=True)
class SchedulerQueryResult:
    """One bounded scheduler query result."""

    states: Mapping[str, str]
    available: bool = True
    detail: str | None = None


@dataclass(frozen=True)
class SchedulerCommentQueryResult:
    """One bounded deterministic-comment reconciliation query."""

    matches: Mapping[str, tuple[str, ...]]
    available: bool = True
    detail: str | None = None


@dataclass(frozen=True)
class SchedulerGenerationBinding:
    """Explicit binding between one GUI CAS generation and scheduler epoch."""

    run_id: str
    record_generation: UUID
    scheduler_generation: UUID
    scheduler_epoch: str


class SchedulerClient(Protocol):
    """Scheduler interface accepted by :class:`SlurmLifecycleObserver`."""

    def query(self, job_ids: Sequence[str]) -> SchedulerQueryResult:
        """Return normalized states for the supplied master job ids."""

    def find_by_comments(
        self,
        scheduler_epoch: str,
        tokens: Sequence[str],
    ) -> SchedulerCommentQueryResult:
        """Find jobs by exact generation-scoped lifecycle comments."""


class SlurmCommandScheduler:
    """Query ``squeue`` and ``sacct`` without raising on controller outages."""

    def __init__(
        self,
        *,
        run_command: Callable[..., subprocess.CompletedProcess[str]]
        | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self._run_command = run_command or subprocess.run
        self._timeout_seconds = timeout_seconds

    def query(self, job_ids: Sequence[str]) -> SchedulerQueryResult:
        """Query active and accounting state once each."""
        normalized_ids = tuple(dict.fromkeys(str(item) for item in job_ids))
        if not normalized_ids:
            return SchedulerQueryResult(states={})
        states: dict[str, str] = {}
        successes = 0
        commands = (
            [
                "squeue",
                "--noheader",
                "--jobs",
                ",".join(normalized_ids),
                "--format=%i|%T",
            ],
            [
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                ",".join(normalized_ids),
                "--format=JobIDRaw,State",
            ],
        )
        errors: list[str] = []
        for command in commands:
            try:
                result = self._run_command(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self._timeout_seconds,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                errors.append(str(exc))
                continue
            if result.returncode != 0:
                errors.append(result.stderr.strip() or "scheduler query failed")
                continue
            successes += 1
            for line in result.stdout.splitlines():
                parts = line.rstrip("|").split("|", 1)
                if len(parts) != 2:
                    continue
                job_id = _master_job_id(parts[0])
                state = _normalize_scheduler_state(parts[1])
                if job_id is None or not state:
                    continue
                prior = states.get(job_id)
                if prior is None or _state_rank(state) > _state_rank(prior):
                    states[job_id] = state
        if successes == 0:
            return SchedulerQueryResult(
                states={},
                available=False,
                detail="; ".join(item for item in errors if item)
                or "scheduler unavailable",
            )
        return SchedulerQueryResult(states=states)

    def find_by_comments(
        self,
        scheduler_epoch: str,
        tokens: Sequence[str],
    ) -> SchedulerCommentQueryResult:
        """Query exact generation-scoped comments for unresolved intents."""
        prefix = f"phenotypic:{scheduler_epoch}:"
        try:
            raw_matches = query_scheduler_comments(
                prefix=prefix,
                run_command=self._run_command,
            )
        except SchedulerQueryUnavailable as exc:
            return SchedulerCommentQueryResult(
                matches={},
                available=False,
                detail=str(exc),
            )
        wanted = set(tokens)
        matches: dict[str, tuple[str, ...]] = {}
        for comment, raw_ids in raw_matches.items():
            if not comment.startswith(prefix):
                continue
            token = comment.removeprefix(prefix)
            if token not in wanted:
                continue
            ids = tuple(
                sorted(
                    {
                        job_id
                        for raw_id in raw_ids
                        if (job_id := _master_job_id(str(raw_id))) is not None
                    },
                    key=int,
                )
            )
            if ids:
                matches[token] = ids
        return SchedulerCommentQueryResult(matches=matches)


@dataclass(frozen=True)
class LogCursor:
    """Incremental position in one file identity."""

    device: int
    inode: int
    offset: int


@dataclass(frozen=True)
class IncrementalLogBatch:
    """Bounded text returned by one incremental multi-file read."""

    text: str
    cursors: Mapping[Path, LogCursor]
    bytes_read: int
    lines_read: int
    reset_paths: tuple[Path, ...] = ()


class IncrementalLogReader:
    """Maintain one cursor per file with byte and line budgets."""

    def __init__(
        self,
        *,
        byte_budget: int = 64 * 1024,
        line_budget: int = 500,
    ) -> None:
        if byte_budget <= 0 or line_budget <= 0:
            raise ValueError("log budgets must be positive")
        self.byte_budget = byte_budget
        self.line_budget = line_budget
        self._cursors: dict[Path, LogCursor] = {}
        self._lock = threading.Lock()

    @property
    def cursors(self) -> Mapping[Path, LogCursor]:
        """Return a stable cursor snapshot."""
        with self._lock:
            return dict(self._cursors)

    def read(
        self, files: Mapping[str, Path] | Sequence[Path]
    ) -> IncrementalLogBatch:
        """Read new content across files without exceeding either budget."""
        if isinstance(files, Mapping):
            labelled = [(str(label), Path(path)) for label, path in files.items()]
        else:
            labelled = [(path.name, Path(path)) for path in files]
        with self._lock:
            batch = _read_incremental_logs(
                labelled,
                self._cursors,
                byte_budget=self.byte_budget,
                line_budget=self.line_budget,
            )
            self._cursors = dict(batch.cursors)
            return batch


@dataclass(frozen=True)
class _LifecycleInventory:
    """Latest intent and job evidence for one scheduler generation."""

    job_ids: tuple[str, ...]
    unresolved_tokens: tuple[str, ...]
    terminal_job_ids: frozenset[str]
    roles: Mapping[str, tuple[str, ...]]
    unresolved_rows: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class _Observation:
    """Effective registry mutation computed by one observation."""

    status: RunStatus
    scheduler_ids: tuple[str, ...]
    primary_scheduler_id: str | None
    log_paths: tuple[Path, ...]
    detail: str | None
    terminal: bool = False


@dataclass
class SlurmLifecycleObserver:
    """Poll nonterminal registry records outside Dash callback threads."""

    registry: RunRegistry
    scheduler: SchedulerClient = field(default_factory=SlurmCommandScheduler)
    poll_interval_seconds: float = 2.0
    reconciliation_grace_seconds: float = 30.0
    monotonic: Callable[[], float] = time.monotonic
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False
    )
    _thread: threading.Thread | None = field(default=None, init=False)
    _reconciling_since: dict[tuple[str, UUID], float] = field(
        default_factory=dict, init=False
    )
    _bindings: dict[tuple[str, UUID], SchedulerGenerationBinding] = field(
        default_factory=dict, init=False
    )

    def bind_generation(
        self,
        *,
        run_id: str,
        record_generation: UUID,
        scheduler_generation: UUID,
    ) -> SchedulerGenerationBinding:
        """Bind one registry generation to the exact durable scheduler epoch.

        S3 must call this after submission returns either a successful job set
        or a recoverable pending submission. The observer intentionally does
        not infer this association from mutable metadata.

        Raises:
            ValueError: If the registry generation or lifecycle epoch differs.
        """
        record = self.registry.get(run_id)
        if record is None or record.generation != record_generation:
            raise ValueError("registry generation is absent or has been superseded")
        lifecycle = load_slurm_lifecycle(record.output_dir)
        lifecycle_epoch = lifecycle.get("generation") if lifecycle else None
        lifecycle_generation = _parse_generation(lifecycle_epoch)
        if lifecycle_generation != scheduler_generation:
            raise ValueError(
                "scheduler generation does not match the durable lifecycle epoch"
            )
        binding = SchedulerGenerationBinding(
            run_id=run_id,
            record_generation=record_generation,
            scheduler_generation=scheduler_generation,
            scheduler_epoch=str(lifecycle_epoch),
        )
        self._bindings[(run_id, record_generation)] = binding
        return binding

    def proven_binding(
        self,
        record: RunRecord,
    ) -> SchedulerGenerationBinding | None:
        """Return an already-proven durable binding without reading output state."""
        if record.generation is None or record.lifecycle_epoch is None:
            return None
        binding = self._bindings.get((record.run_id, record.generation))
        if (
            binding is None
            or binding.scheduler_epoch != record.lifecycle_epoch
        ):
            return None
        return binding

    @property
    def tracked_generation_counts(self) -> tuple[int, int]:
        """Return retained binding and reconciliation-timer counts."""
        return len(self._bindings), len(self._reconciling_since)

    def reconcile_durable_binding(self, record: RunRecord) -> bool:
        """Bind only when metadata durably proves both sides of the identity."""
        if record.generation is None or record.mode != "slurm":
            return False
        try:
            metadata = json.loads(
                job_metadata_path(record.output_dir).read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(metadata, dict):
            return False
        if (
            _parse_generation(
                metadata.get(JobMetadataKey.GUI_RECORD_GENERATION)
            )
            != record.generation
        ):
            return False
        scheduler_generation = _parse_generation(
            metadata.get("slurm_generation")
            or metadata.get(JobMetadataKey.ORCHESTRATION_EPOCH)
        )
        lifecycle = load_slurm_lifecycle(record.output_dir)
        raw_epoch = lifecycle.get("generation") if lifecycle else None
        if (
            scheduler_generation is None
            or _parse_generation(raw_epoch) != scheduler_generation
        ):
            return False
        if not self.registry.compare_and_set(
            record.run_id,
            record.generation,
            lifecycle_epoch=str(raw_epoch),
        ):
            return False
        self.bind_generation(
            run_id=record.run_id,
            record_generation=record.generation,
            scheduler_generation=scheduler_generation,
        )
        return True

    def reconcile_durable_bindings(self) -> int:
        """Retry restart-window bindings for every nonterminal SLURM record."""
        reconciled = 0
        for record in self.registry.list():
            if (
                record.status not in _TERMINAL_RUN_STATUSES
                and (record.run_id, record.generation) not in self._bindings
                and self.reconcile_durable_binding(record)
            ):
                reconciled += 1
        return reconciled

    def start(self) -> None:
        """Start at most one daemon observer thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._observe_loop,
            name="phenotypic-slurm-observer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Request shutdown and bound the join."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(0.0, timeout))

    def observe_once(self, run_id: str | None = None) -> int:
        """Observe one record or every nonterminal SLURM record once.

        Returns:
            Number of records whose effective registry state changed.
        """
        records = self.registry.list()
        self._prune_registry_generations(records)
        changed = 0
        for record in records:
            if run_id is not None and record.run_id != run_id:
                continue
            if record.status in _TERMINAL_RUN_STATUSES:
                self._evict_generation(record)
                continue
            if (
                record.mode != "slurm"
                or record.generation is None
            ):
                continue
            try:
                observation = self._observe_record(record)
            except (OSError, ValueError, TypeError):
                logger.exception("Could not observe SLURM run %s", record.run_id)
                observation = _Observation(
                    status="unknown",
                    scheduler_ids=record.scheduler_ids,
                    primary_scheduler_id=record.primary_scheduler_id,
                    log_paths=record.log_paths,
                    detail="durable SLURM state is unreadable",
                )
            if self._apply(record, observation):
                changed += 1
            if observation.terminal:
                self._evict_generation(record)
        return changed

    def _observe_loop(self) -> None:
        """Poll until stopped, never allowing one malformed run to kill it."""
        while not self._stop_event.is_set():
            try:
                self.observe_once()
            except Exception:  # noqa: BLE001
                logger.exception("SLURM observer iteration failed")
            self._stop_event.wait(max(0.05, self.poll_interval_seconds))

    def _observe_record(self, record: RunRecord) -> _Observation:
        assert record.generation is not None
        binding = self._bindings.get((record.run_id, record.generation))
        if binding is None and self.reconcile_durable_binding(record):
            binding = self._bindings.get((record.run_id, record.generation))
        logs = discover_log_files(
            record.output_dir,
            record.log_paths,
            record_generation=record.generation,
        )
        if binding is None:
            if record.status in {"submitting", "cancelling"}:
                return _Observation(
                    record.status,
                    record.scheduler_ids,
                    record.primary_scheduler_id,
                    logs,
                    "awaiting explicit scheduler lifecycle generation binding",
                )
            return _Observation(
                "unknown",
                record.scheduler_ids,
                record.primary_scheduler_id,
                logs,
                "scheduler lifecycle generation is not explicitly bound",
            )
        lifecycle = load_slurm_lifecycle(record.output_dir)
        if _parse_generation(
            lifecycle.get("generation") if lifecycle else None
        ) != binding.scheduler_generation:
            return _Observation(
                "unknown",
                record.scheduler_ids,
                record.primary_scheduler_id,
                logs,
                "bound scheduler lifecycle generation is no longer durable",
            )
        jobs = read_submitted_job_set(
            record.output_dir,
            expected_generation=binding.scheduler_generation,
        )
        inventory = _lifecycle_inventory(record, binding, jobs)
        comment_result: SchedulerCommentQueryResult | None = None
        if inventory.unresolved_tokens:
            comment_result = _reconcile_unresolved_intents(
                record,
                binding,
                inventory,
                self.scheduler,
            )
            if comment_result.matches:
                jobs = read_submitted_job_set(
                    record.output_dir,
                    expected_generation=binding.scheduler_generation,
                )
                inventory = _lifecycle_inventory(record, binding, jobs)
        job_ids = inventory.job_ids
        primary = jobs.primary_id if jobs is not None else None
        scheduler_result = _query_scheduler(self.scheduler, job_ids)
        normalized_states = {
            job_id: _normalize_scheduler_state(state)
            for job_id, state in scheduler_result.states.items()
        }
        fence_matches = bool(
            lifecycle
            and _parse_generation(lifecycle.get("generation"))
            == binding.scheduler_generation
        )
        inactive = bool(
            fence_matches and lifecycle and lifecycle.get("active") is False
        )
        all_confirmed_inactive = (
            not inventory.unresolved_tokens
            and all(
                job_id in inventory.terminal_job_ids
                or normalized_states.get(job_id)
                in _TERMINAL_SCHEDULER_STATES
                for job_id in job_ids
            )
        )
        staged = _staged_terminal_observation(
            record, binding.scheduler_generation, inventory, normalized_states
        )
        marker = _run_marker_observation(
            record,
            binding.scheduler_generation,
            inventory,
            normalized_states,
        )
        if record.status == "cancelling":
            if inactive and all_confirmed_inactive:
                return _Observation(
                    "cancelled",
                    job_ids,
                    primary,
                    logs,
                    "cancellation fence is inactive and all jobs are inactive",
                    terminal=True,
                )
            return _Observation(
                "cancelling",
                job_ids,
                primary,
                logs,
                "cancellation requested; awaiting inactive fence and "
                "scheduler quiescence",
            )
        for terminal_evidence in (staged, marker):
            if (
                terminal_evidence is not None
                and terminal_evidence.status == "failed"
            ):
                return _with_identity(
                    terminal_evidence, job_ids, primary, logs
                )
        if staged is not None and staged.status in {"complete", "reconciling"}:
            if staged.status == "reconciling" and not all_confirmed_inactive:
                self._clear_grace(record)
            else:
                staged = self._apply_publication_grace(record, staged)
            return _with_identity(staged, job_ids, primary, logs)
        if marker is not None and marker.status in {"complete", "reconciling"}:
            if marker.status == "reconciling" and not all_confirmed_inactive:
                self._clear_grace(record)
            else:
                marker = self._apply_publication_grace(record, marker)
            return _with_identity(marker, job_ids, primary, logs)
        if inactive:
            if all_confirmed_inactive:
                return _Observation(
                    "cancelled",
                    job_ids,
                    primary,
                    logs,
                    "cancellation fence is inactive and all jobs are inactive",
                    terminal=True,
                )
            return _Observation(
                "cancelling",
                job_ids,
                primary,
                logs,
                "cancellation fence is inactive; awaiting scheduler quiescence",
            )

        if staged is not None:
            staged = self._apply_publication_grace(record, staged)
            return _with_identity(staged, job_ids, primary, logs)

        if marker is not None:
            marker = self._apply_publication_grace(record, marker)
            return _with_identity(marker, job_ids, primary, logs)

        states = tuple(
            normalized_states[job_id]
            for job_id in job_ids
            if job_id in normalized_states
        )
        if any(state in _RUNNING_STATES for state in states):
            self._clear_grace(record)
            return _Observation(
                "running", job_ids, primary, logs, "scheduler reports active work"
            )
        if any(state in _QUEUED_STATES for state in states):
            self._clear_grace(record)
            return _Observation(
                "queued", job_ids, primary, logs, "scheduler reports queued work"
            )

        all_completed = (
            bool(job_ids)
            and not inventory.unresolved_tokens
            and all(
                normalized_states.get(job_id) in _SUCCESS_STATES
                or job_id in inventory.terminal_job_ids
                for job_id in job_ids
            )
        )
        if all_completed:
            elapsed = self._grace_elapsed(record)
            if elapsed < self.reconciliation_grace_seconds:
                return _Observation(
                    "reconciling",
                    job_ids,
                    primary,
                    logs,
                    "scheduler completed; awaiting shared-filesystem publication",
                )
            return _Observation(
                "failed",
                job_ids,
                primary,
                logs,
                "scheduler completed but required publication evidence "
                "did not appear before the reconciliation grace expired",
                terminal=True,
            )

        failures = [
            (job_id, state)
            for job_id, state in normalized_states.items()
            if state in _FAILURE_STATES
        ]
        all_jobs_terminal = bool(job_ids) and all(
            normalized_states.get(job_id) in _TERMINAL_SCHEDULER_STATES
            or job_id in inventory.terminal_job_ids
            for job_id in job_ids
        )
        if (
            failures
            and all_jobs_terminal
            and not inventory.unresolved_tokens
        ):
            detail = ", ".join(f"{job_id}={state}" for job_id, state in failures)
            return _Observation(
                "failed",
                job_ids,
                primary,
                logs,
                f"scheduler reported terminal failure: {detail}",
                terminal=True,
            )
        if not scheduler_result.available:
            detail = scheduler_result.detail or "scheduler unavailable"
            if inventory.unresolved_tokens and comment_result is not None:
                detail = (
                    comment_result.detail
                    or "scheduler comment reconciliation is unavailable"
                )
            return _Observation(
                "unknown",
                job_ids,
                primary,
                logs,
                detail,
            )
        if inventory.unresolved_tokens:
            if comment_result is not None and not comment_result.available:
                return _Observation(
                    "unknown",
                    job_ids,
                    primary,
                    logs,
                    comment_result.detail
                    or "scheduler comment reconciliation is unavailable",
                )
            return _Observation(
                "submitting",
                job_ids,
                primary,
                logs,
                "submission intents remain unresolved: "
                + ", ".join(inventory.unresolved_tokens),
            )
        return _Observation(
            "unknown",
            job_ids,
            primary,
            logs,
            "scheduler returned no active or terminal state",
        )

    def _apply_publication_grace(
        self,
        record: RunRecord,
        observation: _Observation,
    ) -> _Observation:
        """Bound durable-marker reconciliation after terminal publication."""
        if observation.status != "reconciling":
            self._clear_grace(record)
            return observation
        if self._grace_elapsed(record) < self.reconciliation_grace_seconds:
            return observation
        return _Observation(
            "failed",
            observation.scheduler_ids,
            observation.primary_scheduler_id,
            observation.log_paths,
            f"{observation.detail or 'required publication evidence is absent'}; "
            "reconciliation grace expired",
            terminal=True,
        )

    def _grace_elapsed(self, record: RunRecord) -> float:
        """Start or read the shared-filesystem reconciliation grace."""
        assert record.generation is not None
        key = (record.run_id, record.generation)
        now = self.monotonic()
        started = self._reconciling_since.setdefault(key, now)
        return max(0.0, now - started)

    def _clear_grace(self, record: RunRecord) -> None:
        """Forget grace state when work is visibly active again."""
        if record.generation is not None:
            self._reconciling_since.pop(
                (record.run_id, record.generation), None
            )

    def _evict_generation(self, record: RunRecord) -> None:
        """Drop all observer state for one terminal GUI generation."""
        if record.generation is None:
            return
        key = (record.run_id, record.generation)
        self._bindings.pop(key, None)
        self._reconciling_since.pop(key, None)

    def _prune_registry_generations(
        self,
        records: Sequence[RunRecord],
    ) -> None:
        """Drop state for generations no longer current in the registry."""
        current = {
            (record.run_id, record.generation)
            for record in records
            if record.generation is not None
        }
        for tracked in (self._bindings, self._reconciling_since):
            for key in tuple(tracked):
                if key in current:
                    continue
                latest = self.registry.get(key[0])
                if latest is not None and latest.generation == key[1]:
                    continue
                tracked.pop(key, None)

    def _apply(self, record: RunRecord, observation: _Observation) -> bool:
        """CAS only when the effective state differs."""
        terminal_at = (
            datetime.now(timezone.utc)
            if observation.terminal and record.terminal_at is None
            else record.terminal_at
        )
        effective = (
            record.status,
            record.scheduler_ids,
            record.primary_scheduler_id,
            record.log_paths,
            record.status_detail,
            record.terminal_at,
        )
        candidate = (
            observation.status,
            observation.scheduler_ids,
            observation.primary_scheduler_id,
            observation.log_paths,
            observation.detail,
            terminal_at,
        )
        if effective == candidate or record.generation is None:
            return False
        return self.registry.compare_and_set(
            record.run_id,
            record.generation,
            expected_record_revision=record.record_revision,
            status=observation.status,
            scheduler_ids=observation.scheduler_ids,
            primary_scheduler_id=observation.primary_scheduler_id,
            log_paths=observation.log_paths,
            status_detail=observation.detail,
            terminal_at=terminal_at,
        )


def discover_log_files(
    output_dir: Path,
    existing: Sequence[Path] = (),
    *,
    record_generation: UUID | None = None,
) -> tuple[Path, ...]:
    """Return GUI submitter and scheduler logs in deterministic role order."""
    paths: list[Path] = [Path(path) for path in existing]
    for role_dir in (
        output_dir / ".phenotypic" / "logs" / "gui",
        output_dir / ".phenotypic" / "logs" / "slurm",
    ):
        try:
            paths.extend(
                sorted(path for path in role_dir.iterdir() if path.is_file())
            )
        except OSError:
            continue
    if record_generation is None:
        return tuple(dict.fromkeys(paths))
    generation_token = record_generation.hex
    return tuple(
        dict.fromkeys(
            path
            for path in paths
            if not (
                _is_generation_scoped_submitter_log(path)
                and generation_token not in path.name
            )
        )
    )


def _is_generation_scoped_submitter_log(path: Path) -> bool:
    """Return whether a GUI log name embeds a launch-generation token."""
    parts = path.name.split(".")
    if (
        len(parts) != 4
        or parts[0] != "submitter"
        or parts[2] not in {"stdout", "stderr"}
        or parts[3] != "log"
    ):
        return False
    token = parts[1]
    return len(token) == 32 and all(
        character in "0123456789abcdef" for character in token.lower()
    )


def _read_incremental_logs(
    files: Sequence[tuple[str, Path]],
    cursors: Mapping[Path, LogCursor],
    *,
    byte_budget: int,
    line_budget: int,
) -> IncrementalLogBatch:
    """Read new bytes from several files using stable inode-aware cursors."""
    next_cursors = dict(cursors)
    parts: list[str] = []
    resets: list[Path] = []
    used_bytes = 0
    used_lines = 0
    for label, path in files:
        if used_bytes >= byte_budget or used_lines >= line_budget:
            break
        try:
            stat_result = path.stat()
        except OSError:
            continue
        prior = next_cursors.get(path)
        reset = bool(
            prior is not None
            and (
                prior.device != stat_result.st_dev
                or prior.inode != stat_result.st_ino
                or stat_result.st_size < prior.offset
            )
        )
        offset = 0 if prior is None or reset else prior.offset
        if reset:
            resets.append(path)
        if stat_result.st_size <= offset:
            next_cursors[path] = LogCursor(
                stat_result.st_dev, stat_result.st_ino, offset
            )
            continue
        chunks: list[bytes] = []
        with path.open("rb") as handle:
            handle.seek(offset)
            while used_bytes < byte_budget and used_lines < line_budget:
                remaining = byte_budget - used_bytes
                line = handle.readline(remaining)
                if not line:
                    break
                chunks.append(line)
                used_bytes += len(line)
                used_lines += 1
            new_offset = handle.tell()
        heading = f"== {label} ==\n"
        if reset:
            heading += "[log rotated or truncated; cursor reset]\n"
        parts.append(heading + b"".join(chunks).decode("utf-8", errors="replace"))
        next_cursors[path] = LogCursor(
            stat_result.st_dev, stat_result.st_ino, new_offset
        )
    return IncrementalLogBatch(
        text="\n".join(parts),
        cursors=next_cursors,
        bytes_read=used_bytes,
        lines_read=used_lines,
        reset_paths=tuple(resets),
    )


def _lifecycle_inventory(
    record: RunRecord,
    binding: SchedulerGenerationBinding,
    jobs: SubmittedJobSet | None,
) -> _LifecycleInventory:
    """Reduce the append-only ledger to latest token state."""
    latest: dict[str, dict[str, object]] = {}
    for row in read_lifecycle_ledger(record.output_dir):
        if (
            _parse_generation(row.get("generation"))
            != binding.scheduler_generation
        ):
            continue
        token = str(row.get("token", ""))
        if token:
            latest[token] = row
    job_ids = list(jobs.all_ids if jobs is not None else ())
    terminal: set[str] = set()
    unresolved: list[str] = []
    unresolved_rows: dict[str, Mapping[str, object]] = {}
    for token, row in latest.items():
        status = str(row.get("status", ""))
        job_id = _master_job_id(str(row.get("job_id", "")))
        if job_id is not None:
            job_ids.append(job_id)
            if status == "terminal":
                terminal.add(job_id)
        if status in {"intent", "blocked"}:
            unresolved.append(token)
            unresolved_rows[token] = row
    return _LifecycleInventory(
        job_ids=tuple(dict.fromkeys(job_ids)),
        unresolved_tokens=tuple(sorted(unresolved)),
        terminal_job_ids=frozenset(terminal),
        roles=jobs.roles if jobs is not None else {},
        unresolved_rows=unresolved_rows,
    )


def _reconcile_unresolved_intents(
    record: RunRecord,
    binding: SchedulerGenerationBinding,
    inventory: _LifecycleInventory,
        scheduler: SchedulerClient,
) -> SchedulerCommentQueryResult:
    """Recover exact-comment matches without terminalizing ambiguity."""
    result = _query_scheduler_comments(
        scheduler,
        binding.scheduler_epoch,
        inventory.unresolved_tokens,
    )
    if not result.available:
        return result
    generation = binding.scheduler_epoch
    for token in inventory.unresolved_tokens:
        row = inventory.unresolved_rows[token]
        role = str(row.get("role", "unknown"))
        dependencies_raw = row.get("dependencies", ())
        dependencies = (
            tuple(str(item) for item in dependencies_raw)
            if isinstance(dependencies_raw, (list, tuple))
            else ()
        )
        round_raw = row.get("round", 0)
        round_index = (
            round_raw
            if isinstance(round_raw, int) and not isinstance(round_raw, bool)
            else 0
        )
        comment = f"phenotypic:{generation}:{token}"
        for job_id in result.matches.get(token, ()):
            append_lifecycle_entry(
                record.output_dir,
                generation=generation,
                token=token,
                role=role,
                status="recovered",
                job_id=job_id,
                dependencies=dependencies,
                round_index=round_index,
                comment=comment,
            )
            mirror_job_to_metadata(
                record.output_dir,
                generation=generation,
                token=token,
                role=role,
                job_id=job_id,
            )
    return result


def _query_scheduler_comments(
    scheduler: SchedulerClient,
    scheduler_epoch: str,
    tokens: Sequence[str],
) -> SchedulerCommentQueryResult:
    """Use an optional comment-query capability on scheduler test doubles."""
    finder = getattr(scheduler, "find_by_comments", None)
    if finder is None:
        return SchedulerCommentQueryResult(
            matches={},
            available=False,
            detail="scheduler comment reconciliation is unavailable",
        )
    result = finder(scheduler_epoch, tokens)
    if isinstance(result, SchedulerCommentQueryResult):
        return result
    if isinstance(result, Mapping):
        return SchedulerCommentQueryResult(
            matches={
                str(token): tuple(str(job_id) for job_id in job_ids)
                for token, job_ids in result.items()
            }
        )
    raise TypeError("scheduler comment query returned an unsupported result")


def _query_scheduler(
    scheduler: SchedulerClient, job_ids: Sequence[str]
) -> SchedulerQueryResult:
    """Accept the protocol result and a plain mapping for simple fakes."""
    result = scheduler.query(job_ids)
    if isinstance(result, SchedulerQueryResult):
        return result
    if isinstance(result, Mapping):
        return SchedulerQueryResult(
            states={str(key): str(value) for key, value in result.items()}
        )
    raise TypeError("scheduler query returned an unsupported result")


def _staged_terminal_observation(
    record: RunRecord,
    generation: UUID,
    inventory: _LifecycleInventory,
    states: Mapping[str, str],
) -> _Observation | None:
    """Read exact staged terminal evidence before scheduler heuristics."""
    orchestration = load_orchestration_state(record.output_dir)
    if orchestration is None:
        return None
    epoch = str(orchestration.get("epoch", ""))
    if _parse_generation(epoch) != generation:
        return None
    phase = str(orchestration.get("phase", "")).lower()
    if phase == "cancelled":
        return _Observation("cancelled", (), None, (), "staged orchestration cancelled", True)
    if phase in _STAGED_FAILURE_PHASES:
        return _Observation(
            "failed",
            (),
            None,
            (),
            f"staged orchestration ended in phase {phase!r}",
            True,
        )
    if phase != "complete":
        return None
    if inventory.unresolved_tokens:
        return None
    if not _all_stage3_markers_exist(record.output_dir):
        return _Observation(
            "reconciling",
            (),
            None,
            (),
            "staged orchestration completed; awaiting Stage-3 publication markers",
        )
    if not staged_completion_matches(record.output_dir, epoch):
        return _Observation(
            "reconciling",
            (),
            None,
            (),
            "staged orchestration completed; awaiting matching completion marker",
        )
    finalizer_ids = inventory.roles.get("finalizer", ())
    failed_finalizers = [
        (job_id, states.get(job_id))
        for job_id in finalizer_ids
        if states.get(job_id) in _FAILURE_STATES
    ]
    if failed_finalizers:
        detail = ", ".join(
            f"{job_id}={state}" for job_id, state in failed_finalizers
        )
        return _Observation(
            "failed",
            (),
            None,
            (),
            f"staged finalizer failed after publication: {detail}",
            True,
        )
    all_jobs_terminal = (
        not inventory.unresolved_tokens
        and all(
            job_id in inventory.terminal_job_ids
            or states.get(job_id) in _TERMINAL_SCHEDULER_STATES
            for job_id in inventory.job_ids
        )
    )
    finalizer_succeeded = not finalizer_ids or all(
        states.get(job_id) == "COMPLETED" for job_id in finalizer_ids
    )
    if not all_jobs_terminal or not finalizer_succeeded:
        return _Observation(
            "reconciling",
            (),
            None,
            (),
            "staged publication is visible; awaiting terminal jobs and finalizer",
        )
    return _Observation(
        "complete",
        (),
        None,
        (),
        "staged orchestration and publication completed",
        True,
    )


def _run_marker_observation(
    record: RunRecord,
    generation: UUID,
    inventory: _LifecycleInventory,
    states: Mapping[str, str],
) -> _Observation | None:
    """Interpret a generation-bearing ordinary run terminal marker."""
    try:
        marker = json.loads(
            run_completion_marker_path(record.output_dir).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(marker, dict):
        return None
    marker_generation = marker.get("generation", marker.get("epoch"))
    if _parse_generation(marker_generation) != generation:
        return None
    status = str(marker.get("status", marker.get("result", ""))).lower()
    if status in {"failed", "failure", "error"}:
        return _Observation(
            "failed", (), None, (), "CLI terminal marker reports failure", True
        )
    if status not in {"complete", "completed", "success", "succeeded", "ok"}:
        return None
    all_jobs_terminal = (
        not inventory.unresolved_tokens
        and all(
            job_id in inventory.terminal_job_ids
            or states.get(job_id) in _TERMINAL_SCHEDULER_STATES
            for job_id in inventory.job_ids
        )
    )
    finalizer_ids = inventory.roles.get("finalizer", ())
    failed_finalizers = [
        (job_id, states.get(job_id))
        for job_id in finalizer_ids
        if states.get(job_id) in _FAILURE_STATES
    ]
    if failed_finalizers:
        detail = ", ".join(
            f"{job_id}={state}" for job_id, state in failed_finalizers
        )
        return _Observation(
            "failed",
            (),
            None,
            (),
            f"finalizer scheduler role failed after terminal publication: {detail}",
            True,
        )
    finalizer_succeeded = (
        bool(marker.get("finalizer_succeeded"))
        and bool(finalizer_ids)
        and all(states.get(job_id) == "COMPLETED" for job_id in finalizer_ids)
    )
    if not all_jobs_terminal:
        return _Observation(
            "reconciling",
            (),
            None,
            (),
            "terminal marker is visible; awaiting terminal jobs and finalizer",
        )
    if not finalizer_succeeded:
        return _Observation(
            "reconciling",
            (),
            None,
            (),
            "terminal marker is visible; awaiting terminal jobs and finalizer",
        )
    if not _manifest_is_complete(record.output_dir):
        return _Observation(
            "reconciling",
            (),
            None,
            (),
            "terminal marker is visible; awaiting complete manifest",
        )
    return _Observation(
        "complete", (), None, (), "SLURM publication completed", True
    )


def _all_stage3_markers_exist(output_dir: Path) -> bool:
    """Require one Stage-3 marker for every metadata inventory image."""
    metadata = _read_json(job_metadata_path(output_dir))
    datasets = metadata.get(JobMetadataKey.DATASETS)
    if not isinstance(datasets, Mapping) or not datasets:
        return False
    for dataset, raw_inventory in datasets.items():
        if not isinstance(raw_inventory, Mapping):
            return False
        images = raw_inventory.get("images")
        if not isinstance(images, list):
            return False
        for image_name in images:
            if not isinstance(image_name, str) or not stage3_completion_exists(
                output_dir, str(dataset), Path(image_name).stem
            ):
                return False
    return True


def _manifest_is_complete(output_dir: Path) -> bool:
    """Require an atomic complete, failure-free inventory manifest."""
    manifest = _read_json(resolve_manifest_json_path(output_dir))
    if manifest.get(DashboardManifestKey.IS_COMPLETE) is not True:
        return False
    failed = manifest.get(DashboardManifestKey.FAILED)
    completed = manifest.get(DashboardManifestKey.COMPLETED)
    total = manifest.get(DashboardManifestKey.TOTAL_IMAGES)
    return (
        isinstance(failed, int)
        and not isinstance(failed, bool)
        and failed == 0
        and isinstance(completed, int)
        and not isinstance(completed, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and completed == total
    )


def _read_json(path: Path) -> dict[str, object]:
    """Read a JSON object, returning an empty object on partial visibility."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _with_identity(
    observation: _Observation,
    job_ids: tuple[str, ...],
    primary: str | None,
    logs: tuple[Path, ...],
) -> _Observation:
    """Attach scheduler and log identity to a durable-state observation."""
    return _Observation(
        observation.status,
        job_ids,
        primary,
        logs,
        observation.detail,
        observation.terminal,
    )


def _master_job_id(raw: str) -> str | None:
    """Normalize array task and step ids to a numeric master id."""
    base = raw.strip().split("_", 1)[0].split(".", 1)[0]
    return base if base.isdigit() else None


def _parse_generation(raw: object) -> UUID | None:
    """Parse a durable UUID epoch written with or without hyphens."""
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return UUID(raw)
    except ValueError:
        return None


def _normalize_scheduler_state(raw: str) -> str:
    """Strip SLURM decorations such as ``CANCELLED by 123`` and ``+``."""
    return raw.strip().upper().split()[0].rstrip("+") if raw.strip() else ""


def _state_rank(state: str) -> int:
    """Prefer active squeue state over accounting terminal rows."""
    if state in _RUNNING_STATES:
        return 4
    if state in _QUEUED_STATES:
        return 3
    if state in _FAILURE_STATES:
        return 2
    if state in _SUCCESS_STATES:
        return 1
    return 0
