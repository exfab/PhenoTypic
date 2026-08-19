"""Process-wide ``RunRegistry`` for live + historical pipeline runs.

A single :class:`RunRegistry` instance per shell process tracks every local
subprocess that the Run console has spawned and every SLURM job it has
submitted. The registry survives Run-console UI release/rebuild — when the
user navigates away from ``/run/`` the UI scratch state is dropped via
:class:`ToolSession.release()` but the registry's ``RunRecord`` entries stay
alive so:

    * The ``Recent Runs`` panel can re-hydrate the same rows on next visit.
    * Local subprocesses keep streaming stdout to disk + their in-memory
      log buffer (owned by the runner, not the UI session).
    * SLURM polling keeps tracking ``progress/manifest.json`` updates.

Threading model
    Every public method takes ``self._lock`` (a :class:`threading.Lock`)
    so concurrent Dash callback threads + the runner's daemon thread do
    not interleave. The lock is fine-grained around the dict mutations,
    NOT around long-running side effects (rehydrate scans, manifest reads
    happen lock-free).

Boot rehydration
    :meth:`rehydrate_from_sandbox` walks the sandbox to a configurable
    depth, picks up any directory looking like a CLI output (master
    parquet + ``results/`` dir), and registers a :class:`RunRecord` for
    each. Status is read from ``progress/manifest.json`` if present;
    otherwise a sentinel "unknown" status is set.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    IO,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    cast,
)
from uuid import UUID, uuid4

from phenotypic._cli._cli_gui_lifecycle import (
    local_manifest_completion_problem,
)
from phenotypic.sdk_ import (
    RUN_LOG_DIRNAME,
    STDOUT_LOG,
    BundleLayout,
    DashboardManifestKey,
    DashboardManifestSlurmInfoKey,
    atomic_write_json,
    gui_launch_owner_path,
    manifest_json_path,
    resolve_event_log_path,
    resolve_manifest_json_path,
    resolve_processing_state_path,
    run_completion_marker_path,
)
from phenotypic.sdk_._file_locking import exclusive_path_lock
from phenotypic.sdk_._io_constants import GUI_RECORD_GENERATION_ENV_VAR

from phenotypic._services.sandbox import SandboxRoot
from phenotypic.gui.shell._classifier import classify

logger = logging.getLogger(__name__)

__all__ = [
    "RunMode",
    "RunStatus",
    "RunRecord",
    "RunRegistry",
    "run_status_is_nonterminal",
    "LocalRunHandle",
    "LocalRunner",
]

# Mode and status tags typed as Literal aliases. We keep them as ``str``
# supersets (via Literal) so the records survive ``json.dumps`` for any
# future persistence step while gaining static narrowability.
RunMode = Literal["local", "slurm", "validate", "unknown"]
RunStatus = Literal[
    "queued",
    "submitting",
    "running",
    "reconciling",
    "cancelling",
    "complete",
    "failed",
    "cancelled",
    "unknown",
]

_RUN_MODES: frozenset[str] = frozenset(
    {"local", "slurm", "validate", "unknown"}
)
_RUN_STATUSES: frozenset[str] = frozenset(
    {
        "queued",
        "submitting",
        "running",
        "reconciling",
        "cancelling",
        "complete",
        "failed",
        "cancelled",
        "unknown",
    }
)
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"complete", "failed", "cancelled"}
)
_OWNER_RECORD_VERSION = 1
_UNSET = object()
_BACKUP_NAME_SUFFIXES = ("-backup", "_backup", ".backup")


def run_status_is_nonterminal(status: object) -> bool:
    """Return whether ``status`` is a recognized nonterminal run status."""
    return (
        isinstance(status, str)
        and status in _RUN_STATUSES
        and status not in _TERMINAL_STATUSES
    )


def _owner_record_path(output_dir: Path) -> Path:
    """Return the canonical SDK-owned launch owner-record path."""
    return gui_launch_owner_path(output_dir)


def _owner_lock_path(output_dir: Path) -> Path:
    """Return the interprocess acquisition lock beside the owner record."""
    return _owner_record_path(output_dir).with_suffix(".lock")


def _is_recognized_backup_artifact_name(name: str) -> bool:
    """Return whether ``name`` follows a reserved backup convention.

    Backup-shaped trees are excluded from Recent Runs unless their directory
    carries a valid current GUI owner record. The suffix rule includes the
    private ``_legacy_*_backup`` migration convention without making backup
    words in the middle of an ordinary run name special.
    """
    return name.endswith(_BACKUP_NAME_SUFFIXES)


@dataclass
class RunRecord:
    """One row in the registry.

    Attributes:
        run_id: Sandbox-relative output identity used as the registry key.
        generation: Durable launch generation. ``None`` is reserved for
            historical outputs that predate the GUI owner record; rehydration
            never invents a generation for such outputs.
        mode: One of :data:`RunMode` — ``"local"``, ``"slurm"``,
            ``"validate"``, or ``"unknown"``. ``"validate"`` is injected
            by the run console's pre-flight pipeline-validation flow
            (`_callbacks._validate_pipeline`).
        output_dir: Absolute path to the run's output directory (where
            ``progress/manifest.json`` lives). Stored as :class:`Path`.
        rel_path: ``output_dir.relative_to(sandbox.root)`` as a string —
            cached so the UI does not re-compute it on every render.
        status: Current status — one of :data:`RunStatus` — ``"running"``,
            ``"submitting"``, ``"complete"``, ``"failed"``, ``"cancelled"``,
            or ``"unknown"``. ``"submitting"`` is the transient state for
            SLURM runs between sbatch dispatch and the first chunk's
            sentinel update.
        pid: Subprocess PID for local runs (``None`` for SLURM and
            rehydrated historical runs).
        scheduler_ids: Every known scheduler id for this launch generation.
        primary_scheduler_id: Deterministic scheduler id shown in compact UI.
        started_at: Monotonic-style ``time.time()`` when the run was
            registered. Persisted on disk only as the manifest's
            ``start_time`` for SLURM rehydration.
        log_paths: Submitter, local-process, and scheduler log paths.
        submitted_at: Scheduler submission timestamp, if applicable.
        terminal_at: Timestamp when terminal evidence was observed.
        returncode: Local process or submitter return code, if known.
        status_detail: Reader-facing lifecycle diagnostic.
        lifecycle_epoch: Durable epoch/fence identity. Defaults to generation.
        record_revision: Persisted per-record mutation revision.
        slurm_job_id: Backward-compatible alias storage for
            ``primary_scheduler_id``.
        log_path: Backward-compatible alias storage for the first log path.
    """

    run_id: str
    mode: RunMode
    output_dir: Path
    rel_path: str
    generation: UUID | None = None
    status: RunStatus = "unknown"
    pid: int | None = None
    scheduler_ids: tuple[str, ...] = ()
    primary_scheduler_id: str | None = None
    log_paths: tuple[Path, ...] = ()
    submitted_at: datetime | None = None
    terminal_at: datetime | None = None
    returncode: int | None = None
    status_detail: str | None = None
    command_digest: str | None = None
    lifecycle_epoch: str | None = None
    record_revision: int = 0
    started_at: float = field(default_factory=time.time)
    # Compatibility fields retained until all callback consumers migrate to
    # the plural/generalized scheduler and log contracts.
    slurm_job_id: str | None = None
    log_path: Path | None = None

    def __post_init__(self) -> None:
        """Normalize compatibility aliases without inventing identity."""
        self.output_dir = Path(self.output_dir)
        self.log_paths = tuple(Path(path) for path in self.log_paths)
        self.scheduler_ids = tuple(dict.fromkeys(self.scheduler_ids))
        if self.primary_scheduler_id is None:
            self.primary_scheduler_id = self.slurm_job_id
        if self.slurm_job_id is None:
            self.slurm_job_id = self.primary_scheduler_id
        if (
            self.primary_scheduler_id is not None
            and self.primary_scheduler_id not in self.scheduler_ids
        ):
            self.scheduler_ids = (
                self.primary_scheduler_id,
                *self.scheduler_ids,
            )
        if not self.log_paths and self.log_path is not None:
            self.log_paths = (Path(self.log_path),)
        if self.log_path is None and self.log_paths:
            self.log_path = self.log_paths[0]
        if self.lifecycle_epoch is None and self.generation is not None:
            self.lifecycle_epoch = str(self.generation)


class RunRegistry:
    """Process-wide thread-safe registry of pipeline runs.

    Methods that mutate the underlying dict acquire ``self._lock``. Read
    methods that return snapshots also acquire the lock briefly to avoid
    iterating over a mid-mutation dict.

    Example:
        >>> from phenotypic.gui.shell._runs_registry import (
        ...     RunRecord, RunRegistry,
        ... )
        >>> reg = RunRegistry()
        >>> rec = RunRecord(
        ...     run_id="my-run", mode="local",
        ...     output_dir=Path("/tmp/out"), rel_path="out",
        ... )
        >>> reg.register(rec)
        >>> reg.get("my-run").status
        'unknown'
        >>> reg.update_status("my-run", "complete")
        >>> reg.get("my-run").status
        'complete'
    """

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._lock = threading.Lock()
        self._revision = 0

    # ------------------------------------------------------------------
    # CRUD-ish API
    # ------------------------------------------------------------------

    @property
    def revision(self) -> int:
        """Return the process-wide mutation revision."""
        with self._lock:
            return self._revision

    def allocate(
        self,
        *,
        mode: RunMode,
        output_dir: Path,
        rel_path: str,
        command_digest: str,
        status: RunStatus = "submitting",
        run_id: str | None = None,
        lifecycle_epoch: str | None = None,
    ) -> RunRecord:
        """Allocate, persist, and register a new launch generation.

        The owner record is atomically durable before this method returns.
        Existing nonterminal generations for the same resolved output are
        rejected.
        """
        generation = uuid4()
        record = RunRecord(
            run_id=run_id or rel_path,
            generation=generation,
            mode=mode,
            output_dir=output_dir,
            rel_path=rel_path,
            status=status,
            command_digest=command_digest,
            lifecycle_epoch=lifecycle_epoch or str(generation),
        )
        with self._lock:
            target = output_dir.resolve(strict=False)
            for existing in self._records.values():
                if (
                    existing.output_dir.resolve(strict=False) == target
                    and existing.status not in _TERMINAL_STATUSES
                ):
                    raise RuntimeError(
                        "output already has a nonterminal launch generation: "
                        f"{existing.run_id}"
                    )
            # The file lock serializes independent GUI processes. All durable
            # state checks are deliberately repeated while it is held so two
            # registries cannot both pass preflight and overwrite ownership.
            with exclusive_path_lock(_owner_lock_path(output_dir)):
                self._assert_output_claimable_locked(
                    output_dir=output_dir,
                    rel_path=rel_path,
                )
                self._persist_record_locked(record)
            self._records[record.run_id] = record
            self._bump_revision_locked(record)
        return record

    def register(
        self,
        record: RunRecord,
        *,
        persist: bool | None = None,
    ) -> None:
        """Insert or replace ``record`` and optionally persist its owner.

        ``persist=None`` persists records with a generation and leaves legacy
        historical records (``generation is None``) read-only.
        """
        with self._lock:
            should_persist = (
                record.generation is not None if persist is None else persist
            )
            if should_persist:
                if record.generation is None:
                    raise ValueError(
                        "cannot persist a run record without a generation"
                    )
                self._persist_record_locked(record)
            self._records[record.run_id] = record
            self._bump_revision_locked(record)

    def get(self, run_id: str) -> RunRecord | None:
        """Return the record for ``run_id``, or ``None`` if missing."""
        with self._lock:
            return self._records.get(run_id)

    def list(self) -> list[RunRecord]:
        """Snapshot the records. Caller may iterate without holding the lock."""
        with self._lock:
            return list(self._records.values())

    def update_status(self, run_id: str, status: RunStatus) -> bool:
        """Backward-compatible unguarded status update.

        New asynchronous callers must use :meth:`compare_and_set` with the
        launch generation. This accessor remains for synchronous legacy
        callbacks while they migrate.
        """
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return False
            if record.status == status:
                return True
            record.status = status
            if status in _TERMINAL_STATUSES and record.terminal_at is None:
                record.terminal_at = datetime.now(timezone.utc)
            self._commit_mutation_locked(record)
            return True

    def update_pid(self, run_id: str, pid: int | None) -> bool:
        """Set ``pid`` (e.g. once Popen returns the subprocess handle)."""
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return False
            if record.pid == pid:
                return True
            record.pid = pid
            self._commit_mutation_locked(record)
            return True

    def update_slurm_job_id(self, run_id: str, job_id: str | None) -> bool:
        """Set ``slurm_job_id`` (post-submit, after job_metadata.json exists)."""
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return False
            if record.primary_scheduler_id == job_id:
                return True
            record.slurm_job_id = job_id
            record.primary_scheduler_id = job_id
            if job_id is not None and job_id not in record.scheduler_ids:
                record.scheduler_ids = (*record.scheduler_ids, job_id)
            self._commit_mutation_locked(record)
            return True

    def compare_and_set(
        self,
        run_id: str,
        generation: UUID,
        *,
        expected_statuses: Iterable[RunStatus] | None = None,
        expected_record_revision: int | None = None,
        status: RunStatus | object = _UNSET,
        pid: int | None | object = _UNSET,
        scheduler_ids: Sequence[str] | object = _UNSET,
        primary_scheduler_id: str | None | object = _UNSET,
        log_paths: Sequence[Path] | object = _UNSET,
        submitted_at: datetime | None | object = _UNSET,
        terminal_at: datetime | None | object = _UNSET,
        returncode: int | None | object = _UNSET,
        status_detail: str | None | object = _UNSET,
        lifecycle_epoch: str | None | object = _UNSET,
    ) -> bool:
        """Atomically mutate a record only when generation and guards match.

        Returns ``False`` for a missing record, a stale generation, or a
        failed expected-status/revision check. No rejected update is persisted
        and the registry revision is unchanged.
        """
        with self._lock:
            record = self._records.get(run_id)
            if record is None or record.generation != generation:
                return False
            if (
                expected_record_revision is not None
                and record.record_revision != expected_record_revision
            ):
                return False
            if expected_statuses is not None:
                allowed = frozenset(expected_statuses)
                if record.status not in allowed:
                    return False

            candidate = replace(record)
            changed = False
            changed |= self._set_if_changed(candidate, "status", status)
            changed |= self._set_if_changed(candidate, "pid", pid)
            if scheduler_ids is not _UNSET:
                supplied_scheduler_ids = cast(Sequence[str], scheduler_ids)
                normalized_ids = tuple(
                    dict.fromkeys(
                        str(item) for item in supplied_scheduler_ids
                    )
                )
                changed |= self._set_if_changed(
                    candidate, "scheduler_ids", normalized_ids
                )
            changed |= self._set_if_changed(
                candidate, "primary_scheduler_id", primary_scheduler_id
            )
            if log_paths is not _UNSET:
                supplied_log_paths = cast(Sequence[Path], log_paths)
                normalized_logs = tuple(
                    Path(path) for path in supplied_log_paths
                )
                changed |= self._set_if_changed(
                    candidate, "log_paths", normalized_logs
                )
            changed |= self._set_if_changed(
                candidate, "submitted_at", submitted_at
            )
            changed |= self._set_if_changed(
                candidate, "terminal_at", terminal_at
            )
            changed |= self._set_if_changed(
                candidate, "returncode", returncode
            )
            changed |= self._set_if_changed(
                candidate, "status_detail", status_detail
            )
            changed |= self._set_if_changed(
                candidate, "lifecycle_epoch", lifecycle_epoch
            )

            if not changed:
                return True
            self._synchronize_compatibility_fields(candidate)
            if (
                candidate.status in _TERMINAL_STATUSES
                and candidate.terminal_at is None
            ):
                candidate.terminal_at = datetime.now(timezone.utc)
            candidate.record_revision += 1
            # Persist first. A failed write leaves the published record,
            # per-record revision, and registry revision untouched.
            if not self._persist_candidate_if_current_locked(
                current=record,
                candidate=candidate,
            ):
                return False
            self._records[run_id] = candidate
            self._revision += 1
            return True

    def publish_if_current_generation(
        self,
        run_id: str,
        generation: UUID,
        publisher: Callable[[], object],
    ) -> bool:
        """Publish an artifact only while memory and durable owner agree.

        The registry lock and output owner lock remain held through
        ``publisher``. Callers should prepare the complete payload first and
        perform only the final atomic write inside the callback.

        Args:
            run_id: Stable registry identity.
            generation: Exact launch generation allowed to publish.
            publisher: Final atomic artifact writer.

        Returns:
            ``True`` when ``publisher`` ran, otherwise ``False`` for a stale or
            missing in-memory/durable generation.
        """
        with self._lock:
            current = self._records.get(run_id)
            if current is None or current.generation != generation:
                return False
            with exclusive_path_lock(_owner_lock_path(current.output_dir)):
                persisted = self._read_owner_record(
                    current.output_dir,
                    current.rel_path,
                )
                if persisted is None or persisted.generation != generation:
                    return False
                publisher()
            return True

    def observe_local_exit(
        self,
        run_id: str,
        generation: UUID,
        returncode: int,
    ) -> bool:
        """Record generation-matched local terminal evidence.

        A validation dry-run has no output publication contract, so its
        process return code is authoritative. A real local run may report
        return code zero even when final dashboard publication failed; it is
        complete only when the canonical manifest proves that this launch
        published a successful terminal inventory and the CLI's atomic
        completion marker carries the exact durable GUI generation.
        """
        with self._lock:
            record = self._records.get(run_id)
            if record is None or record.generation != generation:
                return False
            candidate = replace(record)
            status: RunStatus
            detail: str | None
            if record.status in {"cancelling", "cancelled"}:
                status = "cancelled"
                detail = None
            elif returncode != 0:
                status = "failed"
                detail = f"local process exited with status {returncode}"
            elif record.mode == "validate":
                status = "complete"
                detail = None
            elif record.mode == "local":
                detail = self._local_completion_evidence_conflict(record)
                status = "failed" if detail is not None else "complete"
            else:
                status = "failed"
                detail = (
                    "local exit observer received unsupported run mode "
                    f"{record.mode!r}"
                )
            candidate.status = status
            candidate.returncode = returncode
            candidate.terminal_at = datetime.now(timezone.utc)
            candidate.status_detail = detail
            candidate.record_revision += 1
            if not self._persist_candidate_if_current_locked(
                current=record,
                candidate=candidate,
            ):
                return False
            self._records[run_id] = candidate
            self._revision += 1
            return True

    @staticmethod
    def _local_completion_evidence_conflict(
        record: RunRecord,
    ) -> str | None:
        """Return why a zero-exit local generation cannot publish complete."""
        from phenotypic._cli._cli_completion import current_run_is_complete

        marker_complete = current_run_is_complete(record.output_dir)
        if marker_complete is False:
            return (
                "local process exited successfully but current marker evidence "
                "is incomplete"
            )
        if marker_complete is True:
            marker_path = run_completion_marker_path(record.output_dir)
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                return (
                    "local process exited successfully but its completion "
                    f"marker is unreadable at {marker_path}: {exc}"
                )
            if not isinstance(marker, dict) or (
                marker.get("status") != "complete"
                or marker.get("finalizer_succeeded") is not True
            ):
                return "local completion marker is missing successful publication status"
            # The exact observe_local_exit(run_id, generation) CAS already
            # fences stale child processes. A scientific no-op intentionally
            # retains its prior marker instead of rewriting it for GUI chrome.
            return None

        # Schema-2 compatibility: old runs have no general image markers and
        # continue through the manifest/generation contract below.
        path = manifest_json_path(record.output_dir)
        if not path.is_file():
            return (
                "local process exited successfully but the current generation "
                f"has no canonical terminal publication evidence at {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("manifest is not an object")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            return (
                "local process exited successfully but its canonical "
                f"publication manifest is unreadable at {path}: {exc}"
            )

        manifest_problem = local_manifest_completion_problem(
            payload,
            str(record.generation),
        )
        if manifest_problem == "non_local":
            return (
                "local process exited successfully but terminal publication "
                "mode does not match the current local generation"
            )
        if manifest_problem == "wrong_generation":
            return (
                "local process exited successfully but its canonical "
                "manifest belongs to a different launch generation"
            )
        if manifest_problem == "incomplete":
            return (
                "local process exited successfully but terminal publication "
                "is incomplete, failed, or has invalid inventory counts"
            )

        marker_path = run_completion_marker_path(record.output_dir)
        if not marker_path.is_file():
            return (
                "local process exited successfully but has no exact "
                f"generation completion evidence at {marker_path}"
            )
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if not isinstance(marker, dict):
                raise TypeError("completion marker is not an object")
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
        ) as exc:
            return (
                "local process exited successfully but its completion "
                f"marker is unreadable at {marker_path}: {exc}"
            )
        if marker.get("generation") != str(record.generation):
            return (
                "local process exited successfully but completion "
                "evidence belongs to a different launch generation"
            )
        if (
            marker.get("mode") != "local"
            or marker.get("status") != "complete"
            or marker.get("finalizer_succeeded") is not True
        ):
            return (
                "local process exited successfully but its generation "
                "completion marker is not a successful local publication"
            )
        return None

    def remove(self, run_id: str) -> bool:
        """Drop ``run_id`` from the registry."""
        with self._lock:
            removed = self._records.pop(run_id, None)
            if removed is None:
                return False
            self._revision += 1
            return True

    def clear(self) -> None:
        """Drop every record. Used by tests."""
        with self._lock:
            if self._records:
                self._revision += 1
            self._records.clear()

    # ------------------------------------------------------------------
    # Boot rehydration
    # ------------------------------------------------------------------

    def rehydrate_from_sandbox(
        self,
        sandbox: SandboxRoot,
        *,
        max_depth: int = 3,
    ) -> int:
        """Walk the sandbox + register a record for each output dir found.

        Args:
            sandbox: Sandbox root.
            max_depth: How many levels below the root to scan (default 3,
                matches the spec's ``--scan-depth``). Set to 1 for "only
                immediate children", larger for deep trees.

        Returns:
            Number of new records registered. Existing run_ids are
            preserved (e.g. live local runs registered before boot scan
            don't get clobbered).
        """
        registered = 0
        for output_dir in self._discover_output_dirs(sandbox, max_depth):
            try:
                rel = output_dir.relative_to(sandbox.root).as_posix()
            except ValueError:
                continue
            owner_record = self._read_owner_record(output_dir, rel)
            run_id = (
                owner_record.run_id if owner_record is not None else rel
            )
            if self.get(run_id) is not None:
                continue
            if owner_record is None:
                mode, status, slurm_job_id = self._read_status_from_manifest(
                    output_dir
                )
                record = RunRecord(
                    run_id=run_id,
                    generation=None,
                    mode=mode,
                    output_dir=output_dir,
                    rel_path=rel,
                    status=status,
                    slurm_job_id=slurm_job_id,
                    status_detail=(
                        "historical output has no GUI launch generation"
                        if status in _TERMINAL_STATUSES
                        else (
                            "historical output has no observable "
                            "nonterminal owner"
                        )
                    ),
                )
            else:
                record = owner_record
                # A persisted local nonterminal state is not proof that its
                # process survived the GUI. Preserve identity but downgrade
                # the unsupported liveness claim.
                if (
                    record.mode in {"local", "validate"}
                    and record.status not in _TERMINAL_STATUSES
                ):
                    record.status = "unknown"
                    record.status_detail = (
                        "GUI restarted before local process exit was observed"
                    )
                    record.pid = None
            self.register(record, persist=False)
            registered += 1
        logger.debug(
            "rehydrate_from_sandbox: registered %d output dir(s)",
            registered,
        )
        return registered

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _discover_output_dirs(
        self,
        sandbox: SandboxRoot,
        max_depth: int,
    ) -> Iterator[Path]:
        """Walk the sandbox up to ``max_depth`` and yield CLI output dirs.

        TODO(perf): on a sandbox with thousands of plate folders this walk
        runs synchronously on shell boot before the HTTP listener accepts
        requests. Consider deferring to a background thread + surfacing a
        "Scanning…" badge in the Recent Runs panel.

        Hidden directories (``.gui_log``, ``.phenotypic-gui``, etc.) are
        intentionally skipped: ``SandboxRoot.list_children`` defaults to
        ``include_hidden=False``. This means a user-named ``.run_2026/``
        output directory would also be skipped — that's an accepted
        limitation of the conventional dotfile-as-hidden semantic.
        """
        root = sandbox.root

        def _has_valid_owner(path: Path) -> bool:
            try:
                rel_path = path.relative_to(root).as_posix()
            except ValueError:
                return False
            return bool(
                _owner_record_path(path).is_file()
                and self._read_owner_record(path, rel_path) is not None
            )

        stack: list[tuple[Path, int]] = [(root, 0)]
        seen_output_dirs: set[Path] = set()
        while stack:
            current, depth = stack.pop()
            if depth > max_depth:
                continue
            try:
                children = list(
                    sandbox.list_children(
                        current if current != sandbox.root else None
                    )
                )
            except (PermissionError, FileNotFoundError, OSError):
                continue
            for child in children:
                try:
                    if not child.is_dir():
                        continue
                    owner_is_valid = _has_valid_owner(child)
                    if (
                        _is_recognized_backup_artifact_name(child.name)
                        and not owner_is_valid
                    ):
                        # Backups are private artifacts at every sandbox depth,
                        # including root-level siblings of current outputs. A
                        # valid durable generation owner takes precedence so a
                        # legitimate run is not hidden merely because its chosen
                        # name ends in a backup-shaped suffix.
                        continue
                    caps = classify(child)
                    output_dir: Path | None = None
                    if owner_is_valid:
                        output_dir = child
                    elif caps.is_cli_output:
                        output_dir = self._canonical_cli_output_dir(child)
                    elif caps.is_process_only_output:
                        output_dir = child

                    if output_dir is not None:
                        if (
                            _is_recognized_backup_artifact_name(output_dir.name)
                            and not _has_valid_owner(output_dir)
                        ):
                            # A promoted ``deliverables/`` child can canonicalize
                            # to the sandbox root. Reapply the same owner-aware
                            # exclusion after canonicalization so a depth-zero
                            # ``*-backup`` root cannot leak back in as ``"."``.
                            output_dir = None
                    if output_dir is not None:
                        key = output_dir.resolve()
                        if key not in seen_output_dirs:
                            seen_output_dirs.add(key)
                            yield output_dir
                    # Recurse regardless. Nested outputs are uncommon but remain
                    # a supported compatibility layout, and an invalid owner file
                    # must not hide valid descendants.
                    stack.append((child, depth + 1))
                except (PermissionError, FileNotFoundError, OSError):
                    # A single unreadable or concurrently removed entry must not
                    # prevent valid sibling runs from being discovered.
                    continue

    @staticmethod
    def _canonical_cli_output_dir(path: Path) -> Path:
        """Return the run root for openable CLI-output paths.

        The sidebar classifier intentionally marks ``run/deliverables`` as
        openable so users can launch the Results Viewer from that folder. The
        recent-runs registry still needs one row per run, so collapse that
        promoted deliverables path back to its resolved ``output_root``.
        """
        try:
            layout = BundleLayout.detect(path)
        except FileNotFoundError:
            return path
        return layout.output_root if layout.output_root is not None else path

    @staticmethod
    def _read_status_from_manifest(
        output_dir: Path,
    ) -> tuple[RunMode, RunStatus, str | None]:
        """Best-effort read of mode + status + SLURM job id from manifest.

        Returns ``("unknown", "unknown", None)`` on any failure.
        """
        manifest_path = resolve_manifest_json_path(output_dir)
        if not manifest_path.is_file():
            return ("unknown", "unknown", None)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ("unknown", "unknown", None)

        execution_mode = manifest.get(
            DashboardManifestKey.EXECUTION_MODE, "unknown"
        )
        has_completion_flag = DashboardManifestKey.IS_COMPLETE in manifest
        is_complete = (
            manifest.get(DashboardManifestKey.IS_COMPLETE) is True
        )
        failed = int(manifest.get(DashboardManifestKey.FAILED, 0) or 0)
        completed = int(manifest.get(DashboardManifestKey.COMPLETED, 0) or 0)
        total = int(manifest.get(DashboardManifestKey.TOTAL_IMAGES, 0) or 0)

        if is_complete:
            status: RunStatus = "complete" if failed == 0 else "failed"
        elif (
            not has_completion_flag
            and total > 0
            and (completed + failed) >= total
        ):
            status = "complete" if failed == 0 else "failed"
        else:
            # A legacy manifest records progress, not current liveness.
            # Without a durable GUI owner, local process handle, or scheduler
            # observation it cannot support a ``running`` claim after restart.
            status = "unknown"

        mode: RunMode
        if execution_mode == "slurm":
            mode = "slurm"
        elif execution_mode == "local":
            mode = "local"
        else:
            mode = "unknown"

        slurm_info = manifest.get(DashboardManifestKey.SLURM_INFO) or {}
        chunk_job_ids = (
            slurm_info.get(DashboardManifestSlurmInfoKey.CHUNK_JOB_IDS) or {}
        )
        # ``chunk_job_ids`` is a dict[str, str] of chunk index -> job id.
        # The "primary" array id is the common prefix of all values when
        # SLURM submits as an array; we just surface the first as a hint.
        slurm_job_id: str | None = None
        if isinstance(chunk_job_ids, dict) and chunk_job_ids:
            first_value = next(iter(chunk_job_ids.values()))
            if isinstance(first_value, str):
                # ``45678901_0`` → ``45678901`` (drop array suffix).
                slurm_job_id = first_value.split("_")[0]

        return (mode, status, slurm_job_id)

    @staticmethod
    def _read_owner_record(
        output_dir: Path,
        rel_path: str,
    ) -> RunRecord | None:
        """Read a durable owner record without manufacturing identity."""
        path = _owner_record_path(output_dir)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            if payload.get("version") != _OWNER_RECORD_VERSION:
                return None
            generation = UUID(str(payload["generation"]))
            run_id = str(payload["run_id"])
            stored_rel_path = str(payload["rel_path"])
            mode_value = str(payload["mode"])
            status_value = str(payload["status"])
            if run_id != stored_rel_path or stored_rel_path != rel_path:
                logger.warning(
                    "Ignoring GUI owner with mismatched output identity: %s",
                    path,
                )
                return None
            if mode_value not in _RUN_MODES or status_value not in _RUN_STATUSES:
                return None
            scheduler_ids_raw = payload.get("scheduler_ids", [])
            log_paths_raw = payload.get("log_paths", [])
            if not isinstance(scheduler_ids_raw, list) or not isinstance(
                log_paths_raw, list
            ):
                return None
            submitted_at = RunRegistry._parse_datetime(
                payload.get("submitted_at")
            )
            terminal_at = RunRegistry._parse_datetime(
                payload.get("terminal_at")
            )
            started_at_raw = payload.get("started_at")
            started_at = (
                float(started_at_raw)
                if isinstance(started_at_raw, (int, float))
                else time.time()
            )
            return RunRecord(
                run_id=run_id,
                generation=generation,
                mode=mode_value,  # type: ignore[arg-type]
                output_dir=output_dir,
                rel_path=rel_path,
                status=status_value,  # type: ignore[arg-type]
                pid=RunRegistry._optional_int(payload.get("pid")),
                scheduler_ids=tuple(str(item) for item in scheduler_ids_raw),
                primary_scheduler_id=RunRegistry._optional_str(
                    payload.get("primary_scheduler_id")
                ),
                log_paths=tuple(Path(str(item)) for item in log_paths_raw),
                submitted_at=submitted_at,
                terminal_at=terminal_at,
                returncode=RunRegistry._optional_int(
                    payload.get("returncode")
                ),
                status_detail=RunRegistry._optional_str(
                    payload.get("status_detail")
                ),
                command_digest=RunRegistry._optional_str(
                    payload.get("command_digest")
                ),
                lifecycle_epoch=RunRegistry._optional_str(
                    payload.get("lifecycle_epoch")
                ),
                record_revision=max(
                    0,
                    RunRegistry._optional_int(
                        payload.get("record_revision")
                    )
                    or 0,
                ),
                started_at=started_at,
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ):
            logger.warning("Ignoring invalid GUI owner record: %s", path)
            return None

    def _assert_output_claimable_locked(
        self,
        *,
        output_dir: Path,
        rel_path: str,
    ) -> None:
        """Reject conflicting durable state while ownership lock is held."""
        owner_path = _owner_record_path(output_dir)
        if owner_path.exists():
            persisted = self._read_owner_record(output_dir, rel_path)
            if persisted is None:
                raise RuntimeError(
                    f"output has an invalid generation owner: {owner_path}"
                )
            if persisted.status not in _TERMINAL_STATUSES:
                raise RuntimeError(
                    "output already has a durable nonterminal launch "
                    f"generation: {persisted.generation}"
                )

        processing_conflict = self._processing_state_conflict(output_dir)
        if processing_conflict is not None:
            raise RuntimeError(processing_conflict)

        orchestration_conflict = self._orchestration_state_conflict(output_dir)
        if orchestration_conflict is not None:
            raise RuntimeError(orchestration_conflict)

    @staticmethod
    def _processing_state_conflict(output_dir: Path) -> str | None:
        """Return a blocker unless reconciled CLI state published successfully."""
        path = resolve_processing_state_path(output_dir)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("processing state is not an object")
            datasets = payload.get("datasets")
            if not isinstance(datasets, dict):
                raise TypeError("processing state has no dataset mapping")
            event_log = resolve_event_log_path(output_dir)
            event_states = (
                RunRegistry._latest_event_states(event_log)
                if event_log.exists()
                else {}
            )
            unexpected_datasets = set(event_states) - set(datasets)
            if unexpected_datasets:
                raise ValueError(
                    "event log contains datasets absent from processing "
                    f"inventory: {sorted(unexpected_datasets)!r}"
                )
            inventory_total = 0
            reconciled_completed = 0
            for dataset_name, raw_state in datasets.items():
                if not isinstance(raw_state, dict):
                    raise TypeError(
                        f"dataset {dataset_name!r} state is not an object"
                    )
                initial = RunRegistry._string_set(
                    raw_state.get("initial_images")
                )
                inventory_total += len(initial)
                event_state = event_states.get(str(dataset_name))
                if event_state is None:
                    completed = RunRegistry._string_set(
                        raw_state.get("completed")
                    )
                    failed = RunRegistry._string_set(
                        raw_state.get("failed")
                    )
                else:
                    unexpected_images = set(event_state) - initial
                    if unexpected_images:
                        raise ValueError(
                            f"event log dataset {dataset_name!r} contains "
                            "images absent from processing inventory: "
                            f"{sorted(unexpected_images)!r}"
                        )
                    completed = {
                        image
                        for image, status in event_state.items()
                        if status == "completed"
                    }
                    failed = {
                        image
                        for image, status in event_state.items()
                        if status == "failed"
                    }
                failed_images = initial & failed
                if failed_images:
                    return (
                        "output has failed non-GUI processing state with "
                        f"{len(failed_images)} failed image(s) in dataset "
                        f"{dataset_name!r}"
                    )
                remaining = initial - completed - failed
                if remaining:
                    return (
                        "output has incompatible non-GUI processing state "
                        f"with {len(remaining)} unfinished image(s) in "
                        f"dataset {dataset_name!r}"
                    )
                reconciled_completed += len(initial & completed)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return f"output has unreadable processing state at {path}: {exc}"
        return RunRegistry._publication_evidence_conflict(
            output_dir,
            expected_total=inventory_total,
            expected_completed=reconciled_completed,
        )

    @staticmethod
    def _latest_event_states(
        event_log: Path,
    ) -> dict[str, dict[str, str]]:
        """Replay the log so each image's last event is authoritative."""
        from phenotypic._cli._cli_file_locking import atomic_read
        from phenotypic._cli._cli_update_state import parse_event_line
        from phenotypic._cli._stages import STAGED_TERMINAL_STAGE

        def _replay(content: str) -> dict[str, dict[str, str]]:
            states: dict[str, dict[str, str]] = {}
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    event = parse_event_line(line)
                except ValueError:
                    continue
                status = event.status
                if (
                    status == "completed"
                    and event.stage is not None
                    and event.stage != STAGED_TERMINAL_STAGE
                ):
                    status = "started"
                states.setdefault(event.dataset, {})[event.image] = status
            return states

        return atomic_read(event_log, _replay, timeout=60.0)

    @staticmethod
    def _publication_evidence_conflict(
        output_dir: Path,
        *,
        expected_total: int,
        expected_completed: int,
    ) -> str | None:
        """Require a successful atomic manifest before reclaiming CLI state."""
        path = resolve_manifest_json_path(output_dir)
        if not path.is_file():
            return (
                "output processing state has no terminal publication "
                f"evidence at {path}"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("manifest is not an object")
            is_complete = payload.get(DashboardManifestKey.IS_COMPLETE)
            failed = payload.get(DashboardManifestKey.FAILED)
            completed = payload.get(DashboardManifestKey.COMPLETED)
            total = payload.get(DashboardManifestKey.TOTAL_IMAGES)
            if (
                is_complete is not True
                or not isinstance(failed, int)
                or isinstance(failed, bool)
                or not isinstance(completed, int)
                or isinstance(completed, bool)
                or not isinstance(total, int)
                or isinstance(total, bool)
            ):
                return (
                    "output processing state lacks successful terminal "
                    "publication evidence"
                )
            if failed != 0:
                return (
                    "output terminal publication reports "
                    f"{failed} failed image(s)"
                )
            if total != expected_total:
                return (
                    "output terminal publication inventory does not match "
                    f"processing state: manifest={total}, "
                    f"reconciled={expected_total}"
                )
            if completed != expected_completed:
                return (
                    "output terminal publication completion count does not "
                    "match processing state: "
                    f"manifest={completed}, "
                    f"reconciled={expected_completed}"
                )
            if completed != total:
                return (
                    "output terminal publication inventory is incomplete: "
                    f"{completed}/{total} completed"
                )
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            return f"output has unreadable publication manifest at {path}: {exc}"
        return None

    @staticmethod
    def _orchestration_state_conflict(output_dir: Path) -> str | None:
        """Require matching successful staged completion before reclaim."""
        from phenotypic._cli._cli_staged_orchestration import (
            load_orchestration_state,
            orchestration_state_path,
            staged_completion_matches,
        )

        path = orchestration_state_path(output_dir)
        if not path.exists():
            return None
        payload = load_orchestration_state(output_dir)
        if payload is None:
            return f"output has unreadable orchestration state at {path}"
        phase = payload.get("phase")
        if not isinstance(phase, str) or not phase:
            return f"output has unreadable orchestration state at {path}"
        if phase != "complete":
            return (
                "output has unsuccessful or active non-GUI staged "
                "orchestration "
                f"in phase {phase!r}"
            )
        epoch = payload.get("epoch")
        if not isinstance(epoch, str) or not epoch:
            return f"output has unreadable orchestration state at {path}"
        if not staged_completion_matches(output_dir, epoch):
            return (
                "output staged orchestration has no matching successful "
                f"completion evidence for epoch {epoch!r}"
            )
        return None

    @staticmethod
    def _string_set(value: object) -> set[str]:
        """Validate one processing-state image-name collection."""
        if not isinstance(value, list):
            raise TypeError("image state must be a list")
        if not all(isinstance(item, str) for item in value):
            raise TypeError("image state entries must be strings")
        return set(value)

    def _persist_record_locked(self, record: RunRecord) -> None:
        """Atomically persist one generation owner while holding the lock."""
        if record.generation is None:
            return
        atomic_write_json(
            _owner_record_path(record.output_dir),
            {
                "version": _OWNER_RECORD_VERSION,
                "run_id": record.run_id,
                "generation": str(record.generation),
                "mode": record.mode,
                "output_dir": str(record.output_dir),
                "rel_path": record.rel_path,
                "status": record.status,
                "pid": record.pid,
                "scheduler_ids": list(record.scheduler_ids),
                "primary_scheduler_id": record.primary_scheduler_id,
                "log_paths": [str(path) for path in record.log_paths],
                "submitted_at": self._format_datetime(record.submitted_at),
                "terminal_at": self._format_datetime(record.terminal_at),
                "returncode": record.returncode,
                "status_detail": record.status_detail,
                "command_digest": record.command_digest,
                "lifecycle_epoch": record.lifecycle_epoch,
                "record_revision": record.record_revision,
                "started_at": record.started_at,
                "created_at": datetime.fromtimestamp(
                    record.started_at, timezone.utc
                ).isoformat(),
            },
        )

    def _persist_candidate_if_current_locked(
        self,
        *,
        current: RunRecord,
        candidate: RunRecord,
    ) -> bool:
        """Persist a CAS candidate only if durable generation is unchanged."""
        with exclusive_path_lock(_owner_lock_path(current.output_dir)):
            persisted = self._read_owner_record(
                current.output_dir,
                current.rel_path,
            )
            if (
                persisted is None
                or persisted.generation != current.generation
                or persisted.record_revision != current.record_revision
            ):
                return False
            self._persist_record_locked(candidate)
        return True

    def _commit_mutation_locked(self, record: RunRecord) -> None:
        """Persist and publish one effective record mutation."""
        record.record_revision += 1
        self._persist_record_locked(record)
        self._revision += 1

    def _bump_revision_locked(self, record: RunRecord) -> None:
        """Publish a registration revision without rewriting its owner."""
        self._revision += 1
        if record.record_revision > self._revision:
            self._revision = record.record_revision

    @staticmethod
    def _set_if_changed(
        record: RunRecord,
        field_name: str,
        value: object,
    ) -> bool:
        """Set one field when ``value`` is supplied and differs."""
        if value is _UNSET or getattr(record, field_name) == value:
            return False
        setattr(record, field_name, value)
        return True

    @staticmethod
    def _synchronize_compatibility_fields(record: RunRecord) -> None:
        """Keep legacy single-value accessors aligned with canonical fields."""
        if (
            record.primary_scheduler_id is not None
            and record.primary_scheduler_id not in record.scheduler_ids
        ):
            record.scheduler_ids = (
                record.primary_scheduler_id,
                *record.scheduler_ids,
            )
        record.slurm_job_id = record.primary_scheduler_id
        record.log_path = record.log_paths[0] if record.log_paths else None

    @staticmethod
    def _format_datetime(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("datetime must be an ISO-8601 string")
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def _optional_str(value: object) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None


# Re-export Iterable so type-checker sees it; import lifted to module
# scope to avoid the ``del`` shenanigans elsewhere.


#: Maximum lines retained in the in-memory ring buffer per run. Beyond this
#: the deque silently drops the oldest entry. 5000 is enough for the Run
#: console's UI log tail, which only renders the last ~100 lines anyway.
_LOG_BUFFER_MAXLEN = 5000

#: Seconds between SIGTERM and SIGKILL. The CLI's atexit handlers and
#: progress-flush logic typically finish in well under a second; 10 s is
#: a generous grace window.
_TERM_GRACE_SECONDS = 10.0

#: How often the tee thread polls when ``Popen.stdout.readline`` returns
#: empty bytes. We use ``readline``'s blocking semantics instead, but a
#: timeout-style fall-back exists for race conditions on Windows.
_TEE_THREAD_NAME_PREFIX = "phenotypic-runner-tee-"
_EXIT_THREAD_NAME_PREFIX = "phenotypic-runner-exit-"
_DEFAULT_RETENTION_LIMIT = 64
#: A descendant may inherit the child's stdout descriptor and keep the pipe
#: open after the observed process exits. Terminal state must not wait for
#: that unrelated descendant, so log-drain joining is deliberately bounded.
_TEE_DRAIN_GRACE_SECONDS = 0.25


@dataclass
class LocalRunHandle:
    """Immutable view of a live local subprocess.

    Attributes:
        run_id: Stable identifier within the registry.
        output_dir: Directory the run is writing into.
        process: The :class:`subprocess.Popen` object. ``poll()`` reveals
            exit status; ``returncode`` is set after termination.
        stdout_log_path: ``<output_dir>/.gui_log/stdout.log`` — the on-disk
            tee target. Stays valid even after the deque has been GC'd.
        buffer: In-memory ring buffer (``collections.deque(maxlen=...)``).
            Caller MUST acquire ``buffer_lock`` before reading.
        buffer_lock: Guards the deque against concurrent
            ``append`` (tee thread) and ``snapshot`` (Dash callback)
            calls.
        generation: Durable launch generation used by every lifecycle access
            and the exit callback.
        exit_thread: Observer that waits for process termination.
        finished_at: Monotonic completion timestamp used for bounded eviction.
    """

    run_id: str
    output_dir: Path
    process: subprocess.Popen[bytes]
    stdout_log_path: Path
    generation: UUID
    buffer: "deque[str]" = field(default_factory=lambda: deque(maxlen=_LOG_BUFFER_MAXLEN))
    buffer_lock: threading.Lock = field(default_factory=threading.Lock)
    #: Daemon thread teeing ``process.stdout`` into ``buffer`` and disk.
    #: Exposed so callers (and tests) can ``join()`` and know the pipe is
    #: fully drained — ``process.wait()`` only signals subprocess exit; the
    #: OS pipe may still hold buffered bytes the tee thread has yet to read.
    tee_thread: threading.Thread | None = None
    exit_thread: threading.Thread | None = None
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None


ExitCallback = Callable[[LocalRunHandle, int], None]
ThreadFactory = Callable[..., threading.Thread]


class LocalRunner:
    """Spawns + tracks local pipeline subprocesses.

    A single :class:`LocalRunner` owns the ``atexit`` hook for the process
    so multiple registry entries don't double-register the cleanup. The
    Phase 5 composer instantiates one per Flask app.
    """

    _atexit_registered: bool = False
    # TODO(perf): every ``LocalRunner()`` appends itself to this list and
    # never removes itself. In production exactly one runner exists per
    # process so the list stays bounded, but pytest fixtures that build
    # many run-console apps will accumulate instances and retain their
    # ``_handles`` dicts. If/when a long-running test suite exhibits the
    # leak, switch to ``weakref.WeakSet``. Bounded enough today.
    _instances: list["LocalRunner"] = []
    _atexit_lock = threading.Lock()

    def __init__(
        self,
        *,
        retention_limit: int = _DEFAULT_RETENTION_LIMIT,
        thread_factory: ThreadFactory = threading.Thread,
    ) -> None:
        if retention_limit < 1:
            raise ValueError("retention_limit must be at least 1")
        self._handles: dict[str, LocalRunHandle | None] = {}
        self._lock = threading.Lock()
        self._retention_limit = retention_limit
        self._thread_factory = thread_factory
        with LocalRunner._atexit_lock:
            LocalRunner._instances.append(self)
            if not LocalRunner._atexit_registered:
                atexit.register(LocalRunner._atexit_cleanup_all)
                LocalRunner._atexit_registered = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        run_id: str,
        argv: list[str],
        *,
        output_dir: Path,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        generation: UUID,
        on_exit: ExitCallback | None = None,
    ) -> LocalRunHandle:
        """Spawn ``argv`` and tee its stdout.

        Args:
            run_id: Stable id (typically the registry key).
            argv: Command + args. Caller is responsible for building the
                CLI invocation (``["python", "-m", "phenotypic", ...]``).
            output_dir: Where the CLI is writing its results. The tee
                thread creates ``<output_dir>/.gui_log/`` and writes
                ``stdout.log`` there.
            cwd: Subprocess working directory. Defaults to ``output_dir``.
            env: Subprocess environment. Defaults to inheriting the
                parent's full environment.
            generation: Required durable launch generation. Every later
                lifecycle access must present the same value.
            on_exit: Callback invoked exactly once by the lifecycle observer
                after the process exits. It receives the retained handle and
                integer return code.

        Returns:
            :class:`LocalRunHandle` with the live process + ring buffer.

        Raises:
            FileNotFoundError: If ``argv[0]`` cannot be found.
            RuntimeError: If a handle already exists for ``run_id``.
        """
        # Reserve the run_id under the lock BEFORE the (slow) Popen so a
        # concurrent ``start()`` with the same id can't slip past the
        # duplicate check. We use a sentinel ``None`` placeholder; if the
        # spawn fails we drop the reservation in the ``except`` block so
        # a retry can proceed.
        with self._lock:
            existing = self._handles.get(run_id)
            if run_id in self._handles:
                can_replace_generation = (
                    existing is not None
                    and existing.process.poll() is not None
                    and generation != existing.generation
                )
                if not can_replace_generation:
                    raise RuntimeError(
                        f"run_id already running or retained: {run_id!r}"
                    )
            # Starting a new generation for this output evicts its finished
            # predecessor only now, preserving logs/handles until replacement.
            self._handles[run_id] = None  # type: ignore[assignment]

        process: subprocess.Popen[bytes] | None = None
        handle: LocalRunHandle | None = None
        try:
            log_dir = output_dir / RUN_LOG_DIRNAME
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / STDOUT_LOG

            child_env = dict(os.environ if env is None else env)
            # The token is the only GUI-owned value added to the inherited
            # child environment. It is never logged, and an explicitly
            # supplied test/worker environment remains otherwise unchanged.
            child_env[GUI_RECORD_GENERATION_ENV_VAR] = str(generation)
            process = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge so dashboard sees both
                cwd=str(cwd or output_dir),
                env=child_env,
                # ``bufsize=0`` (unbuffered binary) so ``readline``
                # returns full lines as the subprocess flushes them.
                bufsize=0,
            )

            handle = LocalRunHandle(
                run_id=run_id,
                output_dir=output_dir,
                process=process,
                stdout_log_path=log_path,
                generation=generation,
            )
        except BaseException:
            with self._lock:
                self._handles.pop(run_id, None)
            raise

        with self._lock:
            self._handles[run_id] = handle

        try:
            tee_thread = self._thread_factory(
                target=self._tee_loop,
                args=(handle,),
                name=f"{_TEE_THREAD_NAME_PREFIX}{run_id}",
                daemon=True,
            )
            exit_thread = self._thread_factory(
                target=self._observe_exit,
                args=(handle, on_exit),
                name=f"{_EXIT_THREAD_NAME_PREFIX}{run_id}",
                daemon=True,
            )
            handle.tee_thread = tee_thread
            handle.exit_thread = exit_thread
            tee_thread.start()
            exit_thread.start()
        except BaseException:
            # A child without an exit observer is unsafe. Terminate and reap
            # it synchronously, release the reservation/handle, and preserve
            # any log bytes already drained by the tee thread.
            self._terminate_and_reap(handle, grace_seconds=1.0)
            if handle.tee_thread is not None and handle.tee_thread.is_alive():
                handle.tee_thread.join(timeout=1.0)
            with self._lock:
                if self._handles.get(run_id) is handle:
                    self._handles.pop(run_id, None)
            raise

        with self._lock:
            self._evict_finished_handles_locked(protected_run_id=run_id)
        logger.debug("LocalRunner.start: run_id=%s pid=%d", run_id, process.pid)
        return handle

    def get(self, run_id: str, *, generation: UUID) -> LocalRunHandle | None:
        """Return the exact generation handle, or ``None`` if absent/stale.

        Returns ``None`` for both "never registered" and "reservation
        held but ``Popen`` not yet completed" (the reservation window
        in ``start()``), as well as a mismatched retained generation.
        """
        with self._lock:
            handle = self._handles.get(run_id)
            if handle is None or handle.generation != generation:
                return None
            return handle

    def is_running(
        self,
        run_id: str,
        *,
        generation: UUID,
    ) -> bool:
        """Return whether the exact retained generation is still running."""
        handle = self.get(run_id, generation=generation)
        if handle is None:
            return False
        return handle.process.poll() is None

    def stop(
        self,
        run_id: str,
        *,
        generation: UUID,
        grace_seconds: float = _TERM_GRACE_SECONDS,
    ) -> bool:
        """Send SIGTERM, wait ``grace_seconds``, escalate to SIGKILL.

        Args:
            run_id: Stable registry identity.
            generation: Required exact generation fence. A mismatched retained
                handle is left untouched.
            grace_seconds: Seconds to wait before escalating to SIGKILL.

        Note:
            This method **does not** update the :class:`RunRegistry`. The
            runner and registry are deliberately decoupled (the registry
            persists across UI release while the runner does not). Callers
            that need the registry to reflect a cancellation MUST call
            ``registry.update_status(run_id, "cancelled")`` after this
            method returns ``True``. The Phase 6 ``Cancel`` callback owns
            that wiring.

        Returns:
            ``True`` if a handle was found and termination was attempted;
            ``False`` if no live handle exists for ``run_id``.
        """
        handle = self.get(run_id, generation=generation)
        if handle is None:
            return False
        proc = handle.process
        if proc.poll() is not None:
            return False  # already exited

        logger.debug("LocalRunner.stop: SIGTERM %s pid=%d", run_id, proc.pid)
        try:
            proc.terminate()
        except ProcessLookupError:
            return False
        try:
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            logger.warning(
                "LocalRunner.stop: SIGTERM grace expired (%ss); SIGKILL %s",
                grace_seconds,
                run_id,
            )
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:  # pragma: no cover
                logger.error(
                    "LocalRunner.stop: process %s did not exit after SIGKILL",
                    run_id,
                )
        return True

    def snapshot_log(
        self,
        run_id: str,
        *,
        generation: UUID,
        tail: int | None = None,
    ) -> list[str]:
        """Return a snapshot of the in-memory log buffer for ``run_id``.

        Args:
            run_id: Registry key.
            generation: Required exact generation fence.
            tail: If non-None, return at most this many trailing lines.

        Returns:
            List of decoded log lines (newline-terminated). Empty list if
            ``run_id`` is unknown.
        """
        handle = self.get(run_id, generation=generation)
        if handle is None:
            return []
        with handle.buffer_lock:
            lines = list(handle.buffer)
        if tail is None:
            return lines
        if tail <= 0:
            # ``lines[-0:]`` is ``lines[0:]`` which returns everything;
            # we want "no lines". Match the docstring contract.
            return []
        return lines[-tail:]

    def reap(self, run_id: str, *, generation: UUID) -> int | None:
        """Explicitly drop a finished retained handle and return its exit code.

        Idempotent: returns ``None`` if no matching generation exists or the
        process is still running. A stale generation cannot reap a replacement.
        """
        with self._lock:
            handle = self._handles.get(run_id)
            if handle is None or handle.generation != generation:
                return None
            rc = handle.process.poll()
            if rc is None:
                return None
            del self._handles[run_id]
        return rc

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tee_loop(self, handle: LocalRunHandle) -> None:
        """Read stdout line by line, append to deque + write to disk.

        Runs on a daemon thread per handle. Exits when ``stdout`` is
        closed by the subprocess (i.e. it has exited).
        """
        stdout: IO[bytes] | None = handle.process.stdout
        if stdout is None:  # pragma: no cover - never happens with PIPE
            return
        try:
            with handle.stdout_log_path.open(
                "ab", buffering=0
            ) as disk_log:
                for raw in iter(stdout.readline, b""):
                    line = raw.decode("utf-8", errors="replace")
                    with handle.buffer_lock:
                        handle.buffer.append(line)
                    try:
                        disk_log.write(raw)
                    except OSError:  # pragma: no cover - disk full etc.
                        logger.exception(
                            "tee disk write failed for run_id=%s",
                            handle.run_id,
                        )
        except Exception:  # pragma: no cover - belt-and-braces
            logger.exception(
                "tee loop crashed for run_id=%s", handle.run_id
            )
        finally:
            try:
                stdout.close()
            except OSError:
                pass

    def _observe_exit(
        self,
        handle: LocalRunHandle,
        on_exit: ExitCallback | None,
    ) -> None:
        """Wait for terminal process evidence and notify the owner once."""
        returncode = handle.process.wait()
        if handle.tee_thread is not None:
            handle.tee_thread.join(timeout=_TEE_DRAIN_GRACE_SECONDS)
        handle.finished_at = time.monotonic()
        if on_exit is not None:
            try:
                on_exit(handle, returncode)
            except Exception:
                logger.exception(
                    "local exit callback failed for run_id=%s generation=%s",
                    handle.run_id,
                    handle.generation,
                )
        with self._lock:
            self._evict_finished_handles_locked(
                protected_run_id=handle.run_id
            )

    @staticmethod
    def _terminate_and_reap(
        handle: LocalRunHandle,
        *,
        grace_seconds: float,
    ) -> None:
        """Best-effort terminate and synchronously reap one child."""
        process = handle.process
        if process.poll() is None:
            try:
                process.terminate()
            except (ProcessLookupError, OSError):
                pass
        try:
            process.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                logger.error(
                    "process %s did not exit after observer-start cleanup",
                    handle.run_id,
                )

    def _evict_finished_handles_locked(
        self,
        *,
        protected_run_id: str,
    ) -> None:
        """Evict oldest finished handles until the retention bound holds."""
        excess = len(self._handles) - self._retention_limit
        if excess <= 0:
            return
        candidates = sorted(
            (
                handle
                for run_id, handle in self._handles.items()
                if run_id != protected_run_id
                and handle is not None
                and handle.process.poll() is not None
            ),
            key=lambda item: (
                item.finished_at
                if item.finished_at is not None
                else item.started_at
            ),
        )
        for handle in candidates[:excess]:
            if self._handles.get(handle.run_id) is handle:
                self._handles.pop(handle.run_id, None)

    def _snapshot_handles_for_shutdown(self) -> list[LocalRunHandle]:
        """Return current handles for process-exit cleanup only.

        This private adapter is intentionally unscoped because process exit
        must terminate every child generation. Interactive lifecycle callers
        must use the generation-fenced public methods.
        """
        with self._lock:
            return [
                handle
                for handle in self._handles.values()
                if handle is not None
            ]

    @classmethod
    def _atexit_cleanup_all(cls) -> None:
        """SIGTERM every live subprocess across every runner instance.

        Snapshots both the instances list and each runner's retained handles
        once, before either loop. Otherwise a run registered between the
        first (terminate) loop and the second (wait) loop would land on
        the SIGKILL fast-path without first receiving a SIGTERM.
        """
        with cls._atexit_lock:
            instances = list(cls._instances)
        runner_snapshots: dict[LocalRunner, list[LocalRunHandle]] = {
            runner: runner._snapshot_handles_for_shutdown()
            for runner in instances
        }
        for handles in runner_snapshots.values():
            for handle in handles:
                if handle.process.poll() is not None:
                    continue
                try:
                    handle.process.terminate()
                except (ProcessLookupError, OSError):
                    pass
        # Best-effort short wait so terminating cleanly beats SIGKILL.
        deadline = time.monotonic() + 2.0
        for handles in runner_snapshots.values():
            for handle in handles:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    handle.process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    try:
                        handle.process.kill()
                    except (ProcessLookupError, OSError):
                        pass


# Pacify type-checker on Iterable import (kept for symmetry with future
# typing tweaks).
_ = (Iterable, os, signal)
