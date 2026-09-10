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
    each. Status comes from :func:`~phenotypic.sdk_.resolve_run_state`
    (spec §4.2 demoted ``manifest.json`` from evidence); mode and the
    scheduler-id hint come from the CLI's own ``job_metadata.json``.

    ``"unknown"`` means *this directory holds no run of ours*, and nothing
    else. A run the verdict knows is unfinished reads ``"incomplete"`` — the
    two were one badge until O-4, which is what made a run killed by OOM
    indistinguishable from a foreign folder.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    cast,
)
from uuid import UUID, uuid4

from phenotypic._cli._cli_gui_lifecycle import (
    local_manifest_completion_problem,
)
from phenotypic.sdk_ import (
    BundleLayout,
    JobMetadataKey,
    atomic_write_json,
    gui_launch_owner_path,
    job_metadata_path,
    manifest_json_path,
    resolve_processing_state_path,
    resolve_run_state,
    run_completion_marker_path,
)
from phenotypic.sdk_._file_locking import exclusive_path_lock

from phenotypic.gui.shell._classifier import classify
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = [
    "RunMode",
    "RunStatus",
    "RunRecord",
    "RunRegistry",
    "run_status_is_nonterminal",
]

# Mode and status tags typed as Literal aliases. We keep them as ``str``
# supersets (via Literal) so the records survive ``json.dumps`` for any
# future persistence step while gaining static narrowability.
RunMode = Literal["local", "slurm", "validate", "unknown"]
#: ``incomplete`` is the O-4 addition, and it is deliberately the same word
#: :data:`~phenotypic.sdk_.Completion` uses. A synonym here would be a second
#: vocabulary for one state, and a translation layer between two vocabularies
#: is the drift this whole change exists to remove.
#:
#: **It is NOT terminal**, and that is the trap. A run that did not finish can
#: be resumed by re-running the same command, so it is not an outcome —
#: :data:`_TERMINAL_STATUSES` below excludes it, and
#: :func:`run_status_is_nonterminal` therefore returns ``True`` for it. Two
#: consumers read that off a persisted owner record
#: (``results_viewer/_output_root.py``, ``_qc_tab/review/_rebuild.py``), and
#: :meth:`RunRegistry._release_dead_owner_locked` coerces any nonterminal
#: verdict to ``failed`` before persisting precisely so a repaired record can
#: never carry it.
RunStatus = Literal[
    "queued",
    "submitting",
    "running",
    "reconciling",
    "cancelling",
    "complete",
    "incomplete",
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
        "incomplete",
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


def _system_boot_time() -> float | None:
    """Return this host's boot time in epoch seconds, or ``None``.

    ``None`` on any failure, which makes every caller fall through to the
    pid probe rather than declaring an owner dead on missing evidence.
    """
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a hard dependency
        return None
    try:
        return float(psutil.boot_time())
    except (OSError, RuntimeError):  # pragma: no cover - defensive
        return None


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


def _output_holds_a_run(output_dir: Path) -> bool:
    """Return whether any run has ever written durable state here.

    **The one question :class:`~phenotypic.sdk_.RunState` cannot answer**, and
    the reason both readers below need it. ``resolve_run_state`` reports
    ``completion == "incomplete"`` for a directory holding no run *and* for a
    run that is genuinely unfinished — the same verdict for "this is not ours"
    and "this one died". Separating them is what lets the claim gate allow a
    launch into a fresh directory (:meth:`RunRegistry._foreign_run_conflict`)
    and what lets Recent Runs say *unfinished* instead of *unknown*
    (:meth:`RunRegistry._rehydrated_status`).

    **Fires** when the output carries a `processing_state.json` (in either the
    `.phenotypic/` or the legacy root spelling — the resolver handles both) or
    a staged controller's `staged_orchestration.json`. Both are checked
    because a controller writes its record *before* its first stage writes any
    processing state, so a tree in that window holds a run and has no state
    file.

    **Why not `RunState`'s own fields.** Two look like they would work and
    neither may be used:

    * ``identity.processing_generation == ""`` is the `_UNIDENTIFIED`
      sentinel, private to ``_run_state`` — and a *real* pre-P2 tree whose
      ``config`` block predates the field reads empty too, so this would
      classify an actual run as "not a run", in the direction that lets a
      second launch write over it.
    * ``diagnostics.accepted == 0`` is forbidden outright: spec §9 says a
      predicate reaching into ``diagnostics`` is visibly wrong in review, and
      :class:`~phenotypic.sdk_.RunDiagnostics` says nothing branches on those
      counts.

    Args:
        output_dir: Any directory, including one this package never wrote.

    Returns:
        Whether durable run state exists here. Never raises.
    """
    from phenotypic._cli._cli_staged_orchestration import (
        orchestration_state_path,
    )

    try:
        return bool(
            resolve_processing_state_path(output_dir).exists()
            or orchestration_state_path(output_dir).exists()
        )
    except OSError:
        # An unreadable parent must not raise out of a boot-time scan. "No
        # run here" is the degrade that lets discovery continue; the claim
        # gate's own owner-record check still stands in front of it.
        return False


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
        status: Current status — one of :data:`RunStatus`. ``"submitting"``
            is the transient state for SLURM runs between sbatch dispatch and
            the first chunk's sentinel update; ``"incomplete"`` is a run the
            verdict knows did not finish (O-4), and is **not** terminal —
            see :data:`RunStatus` for why that matters; ``"unknown"`` means
            this directory holds no run of ours.
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
        from phenotypic._cli._cli_completion import (
            _all_accepted_images_succeeded,
        )

        # P6 Task 0: NOT `resolve_run_state(...).completion`. This site asks
        # *"have the accepted images succeeded?"* -- its `is False` message
        # says "marker evidence is incomplete", and its `is True` branch then
        # reads the run proof **itself**, separately, below. `.completion`
        # already requires that proof, which would make the branch below dead
        # and make `is False` report the wrong cause whenever the images had
        # succeeded but nothing had published yet.
        #
        # Task 4 owns this file; converted here because P6 Task 0's deletion
        # is not local to its own task. No depth decision is owed after all:
        # this call is not `resolve_run_state`.
        marker_complete = _all_accepted_images_succeeded(record.output_dir)
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
                mode, slurm_job_id = self._submission_identity(output_dir)
                status, detail = self._rehydrated_status(output_dir)
                record = RunRecord(
                    run_id=run_id,
                    generation=None,
                    mode=mode,
                    output_dir=output_dir,
                    rel_path=rel,
                    status=status,
                    slurm_job_id=slurm_job_id,
                    status_detail=detail,
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
                    # O-4's SECOND collapse site. This arm used to hard-code
                    # `"unknown"`, so a GUI-launched run -- the common case on
                    # this cluster -- lost the verdict entirely and fixing
                    # `_rehydrated_status` alone would have left most of the
                    # symptom standing.
                    status, detail = self._rehydrated_status(
                        record.output_dir
                    )
                    if status == "running":
                        # The ONLY liveness authority reachable here is the
                        # pid in the record being downgraded, so believing
                        # this arm would be reading back our own write --
                        # the same non-fence `RunIdentity.owner_generation`
                        # is (gui/CLAUDE.md, "What RunState cannot answer").
                        # Liveness is precisely what a restarted GUI cannot
                        # vouch for; completion it can still ask about.
                        status = "unknown"
                        detail = "its liveness cannot be re-established"
                    record.status = status
                    record.status_detail = (
                        "GUI restarted before local process exit was "
                        f"observed; {detail}"
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
    def _rehydrated_status(output_dir: Path) -> tuple[RunStatus, str]:
        """Return the boot status + detail for an output with no GUI owner.

        Replaces the manifest-count reader retired here (spec §11, §4.2).
        ``manifest.json`` records what a run *reported having done*; the
        verdict records what its artifacts still prove, and only the second
        survives a GUI restart intact.

        **The ``running`` arm fires** for an output one of spec §4.1's
        liveness authorities still claims -- an active SLURM lifecycle fence
        at or above the run's restart epoch, or a GUI owner record naming a
        live pid. The retired reader could not reach that arm at all: its own
        comment said progress counts cannot support a ``running`` claim after
        a restart, which was correct, and is why it answered ``"unknown"``
        for every run actually in flight.

        **``incomplete`` and ``unknown`` are different answers (O-4).** They
        used to be the same one: this method collapsed the verdict onto
        ``"unknown"`` and the row lost what ``resolve_run_state`` had just
        established. Three states reached one badge — a directory holding no
        run of ours, a run the system knows is unfinished, and a run killed by
        infrastructure — and on this cluster, where ``DefMemPerCPU`` is 1 GB
        and ``short`` caps at two hours, the middle and last ones are the
        common case rather than the exotic one.

        ``_output_holds_a_run`` is what separates the first from the other
        two; see it for why ``RunState``'s own fields cannot.

        **The infrastructure kill shares ``incomplete``, deliberately.**
        Nothing on disk distinguishes an OOM kill from a ``kill -9`` or a
        reboot — a dead pid and no proof, in every case — and the SLURM
        evidence that would (``sacct``'s ``OUT_OF_MEMORY`` / ``TIMEOUT``)
        lives in the observer, which DEFERRED D-1 keeps out of this path. A
        fourth status nothing could ever set is worse than three that are
        honest. What it would take to earn one: a writer that records a
        terminal-kill fact to disk.

        **Cost.** One shallow resolution per discovered output, on the boot
        walk :meth:`_discover_output_dirs` already flags as synchronous. A
        tree with a warm ``verification_cache.json`` re-stats rather than
        re-hashes; a tree holding no run returns from the gate above without
        resolving at all. The tree that pays a full pass is a current-build
        run whose cache has not been written yet, once.

        Args:
            output_dir: A discovered CLI output root.

        Returns:
            ``(status, status_detail)`` for the rehydrated record.
        """
        if not _output_holds_a_run(output_dir):
            return (
                "unknown",
                "no run state under this directory, so it has no status to "
                "report",
            )
        completion = resolve_run_state(output_dir, depth="shallow").completion
        if completion == "complete":
            return (
                "complete",
                "historical output has no GUI launch generation",
            )
        if completion == "failed":
            return (
                "failed",
                "historical output has no GUI launch generation",
            )
        if completion == "active":
            return (
                "running",
                "a liveness record claims work for this output, but no GUI "
                "launch generation owns it",
            )
        return (
            "incomplete",
            "this run did not finish, and no liveness record claims it; "
            "re-running the same command resumes it",
        )

    @staticmethod
    def _submission_identity(output_dir: Path) -> tuple[RunMode, str | None]:
        """Return the recorded execution mode and a scheduler-id hint.

        Reads ``job_metadata.json`` -- the record the CLI writes at
        submission, and the one the run console's own SLURM reader already
        uses (``run_console/_slurm.py:306``) -- rather than ``manifest.json``,
        which spec §4.2 demotes from evidence. Both carry these two fields;
        only one of them is the record that owns them.

        **Returns ``("unknown", None)``** for a directory with no readable
        submission record: a tree this package never wrote, one written
        before the record existed, a truncated write, or a JSON document that
        is not an object. Deliberately not ``"local"`` --
        :func:`~phenotypic.sdk_.resolve_execution_mode` coerces a missing
        record to ``"local"``, which here would label every foreign folder in
        the sandbox a local run of ours.

        Args:
            output_dir: A discovered CLI output root.

        Returns:
            ``(mode, primary scheduler id or None)``.
        """
        try:
            payload = json.loads(
                job_metadata_path(output_dir).read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return ("unknown", None)
        if not isinstance(payload, dict):
            return ("unknown", None)
        recorded = payload.get(JobMetadataKey.EXECUTION_MODE)
        mode: RunMode = (
            "slurm"
            if recorded == "slurm"
            else "local"
            if recorded == "local"
            else "unknown"
        )
        return (mode, RunRegistry._first_scheduler_id(payload))

    @staticmethod
    def _first_scheduler_id(
        job_metadata: Mapping[str, object],
    ) -> str | None:
        """Return one scheduler job id from a submission record, or ``None``.

        A **hint**, exactly as the retired manifest reader's was: the first
        recorded id with its array-task suffix dropped (``45678901_0`` →
        ``45678901``). The generation-fenced enumeration lives in
        ``run_console/_slurm.py`` and is not duplicated here.

        Returns ``None`` for every shape that is not a recorded id: a local
        run (neither map present), an id map that is absent, empty or not a
        mapping, and a first entry that is neither a string nor a mapping
        carrying a string ``job_id``.

        Args:
            job_metadata: A parsed ``job_metadata.json`` object.

        Returns:
            A bare job id, or ``None``.
        """
        for key in (
            JobMetadataKey.CHUNK_JOB_IDS,
            JobMetadataKey.SLURM_JOB_IDS,
        ):
            raw = job_metadata.get(key)
            if not isinstance(raw, dict) or not raw:
                continue
            value: object = next(iter(raw.values()))
            if isinstance(value, dict):
                value = value.get("job_id")
            if isinstance(value, str) and value:
                return value.split("_")[0]
        return None

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
                dead = self._dead_owner_reason(persisted)
                if dead is None:
                    raise RuntimeError(
                        "output already has a durable nonterminal launch "
                        f"generation: {persisted.generation}"
                    )
                self._release_dead_owner_locked(persisted, reason=dead)

        foreign_conflict = self._foreign_run_conflict(output_dir)
        if foreign_conflict is not None:
            raise RuntimeError(foreign_conflict)

    @staticmethod
    def _dead_owner_reason(record: RunRecord) -> str | None:
        """Return why a nonterminal owner is provably gone, or ``None``.

        DEFERRED D-2 / audit S7 **[verified against the tree]**: nothing in
        this codebase ever deletes or repairs ``gui_launch_owner.json``.
        :meth:`remove` and :meth:`clear` drop the in-memory record and leave
        the file; :meth:`rehydrate_from_sandbox`'s downgrade is
        ``persist=False``; and no ``unlink`` of that path exists anywhere in
        ``src/``. So a SIGKILLed GUI leaves ``status: "running"`` on disk
        permanently and :meth:`_assert_output_claimable_locked` refuses the
        output forever, with no UI affordance to clear it.

        **The liveness probe is imported, not rewritten.** Spec §4.1 makes
        this record one of three liveness authorities, and P1 Task 5 already
        taught the Q2 ladder to believe it only while the pid it names is
        alive (``_run_state._process_is_alive``). A second probe here could
        disagree with the ladder about the same pid -- two authorities for
        one fact, which is the defect class this whole change exists to
        remove. The import is function-local so that patching the ladder's
        probe visibly moves this verdict too.

        ``None`` -- refuse the claim -- for everything that is not *proof* of
        death:

        * a SLURM launch, whose liveness belongs to the scheduler and whose
          ``pid`` is legitimately ``None``. These are the same modes
          :meth:`rehydrate_from_sandbox` already excludes from its downgrade.
        * a record carrying no ``pid`` and no pre-boot ``started_at``. A GUI
          killed between :meth:`allocate` and :meth:`update_pid` is a real
          state and is indistinguishable from a live owner within one boot.
        * a live pid.

        A pid recycled **within one boot** is a real but bounded risk, left
        undefended on purpose: ``started_at`` records when the *registry
        record* was created, which precedes the subprocess, so comparing it
        against the process's own creation time does not discriminate a
        recycled pid from the legitimate one. It would only look like it did.
        """
        from phenotypic.sdk_._run_state import _process_is_alive

        if record.mode not in {"local", "validate"}:
            return None
        boot_time = _system_boot_time()
        if boot_time is not None and record.started_at < boot_time:
            # Nothing survives a reboot. This is the only arm that can
            # retire a record carrying no pid at all.
            return "it predates the current system boot"
        pid = record.pid
        if pid is None:
            return None
        if _process_is_alive(pid):
            return None
        return f"its owning process {pid} is no longer running"

    def _release_dead_owner_locked(
        self,
        record: RunRecord,
        *,
        reason: str,
    ) -> None:
        """Persist a terminal downgrade for an owner whose process is gone.

        **The replacement status is read off the tree, not asserted.** A GUI
        killed after its CLI child finished owns a run that is *complete*,
        and writing ``failed`` over it would put a second wrong answer where
        the first one was. :meth:`_rehydrated_status` is this file's one
        producer of "what does this output's own evidence say", and the
        ladder underneath it already disbelieves the dead pid in the record
        being replaced, so asking it here is not circular.

        A verdict that is still nonterminal is written as ``failed``: an
        unfinished run nothing is working on resolves to ``"incomplete"``
        (``"unknown"`` before O-4), and both are nonterminal by
        :func:`run_status_is_nonterminal`. The record has to land terminal or
        the dead end survives its own repair — which is why this coerces
        rather than passing the verdict through.

        The caller holds ``exclusive_path_lock`` on the owner record, so this
        write cannot race another GUI's claim.
        """
        status, detail = self._rehydrated_status(record.output_dir)
        if status not in _TERMINAL_STATUSES:
            status = "failed"
            detail = "no terminal publication evidence survives it"
        repaired = replace(
            record,
            status=status,
            pid=None,
            terminal_at=datetime.now(timezone.utc),
            status_detail=(
                f"released a dead GUI launch generation ({reason}); {detail}"
            ),
            record_revision=record.record_revision + 1,
        )
        self._persist_record_locked(repaired)
        logger.warning(
            "Released a dead GUI launch generation for %s (%s); "
            "its evidence reads %s.",
            record.output_dir,
            reason,
            status,
        )

    @staticmethod
    def _foreign_run_conflict(output_dir: Path) -> str | None:
        """Return why a non-GUI run already here forbids claiming this output.

        One :func:`~phenotypic.sdk_.resolve_run_state` call replaces the three
        conflict predicates retired here (spec §11): the processing-state
        walk, its manifest-count publication check, and the
        staged-orchestration check. Their shared question -- *"did the run
        already in this directory finish successfully?"* -- is exactly
        ``completion``, and asking it once is what stops the GUI and the CLI
        answering it differently.

        **Fires when** the output holds a non-GUI run that has not finished:
        one a liveness authority still claims (``active``), one with a failed
        image (``failed``), or one whose accepted work never verified
        (``incomplete``) -- which includes the case
        ``_publication_evidence_conflict`` covered, where every image is done
        but no run proof was ever published over them. Returns ``None`` for a
        finished run, and for a directory holding no run at all.

        **The existence gate is not redundant with the verdict, and has to
        stay ahead of it.** ``resolve_run_state`` reports ``incomplete`` both
        for an empty directory and for a tree it cannot parse, and those are
        the two cases a launch must be *allowed* into; only the presence of
        durable non-GUI state separates them from an unfinished run. Deriving
        that from ``RunState.identity`` instead would fail in the dangerous
        direction: a pre-P2 ``processing_state.json`` carries no ``config``
        block, so it yields no identity, and the claim would then be granted
        over a run still using the directory. The staged-orchestration record
        is checked alongside it because a controller writes that record
        before its first stage writes any processing state.

        **Depth is ``shallow``, and that is a decision made here.** Spec §9's
        caller table has no row for this site: it is a launch-time gate taken
        once, not one of the two pollers. Shallow, because INV-VERDICT
        already forbids a cache entry from yielding a positive verdict on its
        own -- so the only tree on which shallow and deep disagree is one
        whose content changed without its ``size`` or ``mtime_ns`` moving --
        and because this runs with the exclusive ownership lock held and the
        Run button blocked, which is where an O(N) re-hash is least
        affordable.

        ``diagnostics`` is read for the message only. Nothing branches on it
        (spec §9).

        Args:
            output_dir: The output root the GUI is about to claim.

        Returns:
            A refusal message, or ``None`` when the output is claimable.
        """
        if not _output_holds_a_run(output_dir):
            return None

        state = resolve_run_state(output_dir, depth="shallow")
        counts = state.diagnostics
        if state.completion == "complete":
            return None
        if state.completion == "active":
            return (
                "output has a non-GUI run in flight; a liveness record still "
                "claims work for this output"
            )
        if state.completion == "failed":
            return (
                "output has failed non-GUI processing state with "
                f"{counts.failed} failed image(s) of {counts.accepted} "
                "accepted"
            )
        return (
            "output has incompatible non-GUI processing state: "
            f"{counts.verified} of {counts.accepted} accepted image(s) "
            "verified, and no successful run proof covers them"
        )

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
_ = Iterable
