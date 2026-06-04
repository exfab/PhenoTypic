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

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Literal

from phenotypic.tools_ import (
    DashboardManifestKey,
    DashboardManifestSlurmInfoKey,
    resolve_manifest_json_path,
)

from phenotypic.gui.shell._classifier import classify
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = [
    "RunMode",
    "RunStatus",
    "RunRecord",
    "RunRegistry",
]

# Mode and status tags typed as Literal aliases. We keep them as ``str``
# supersets (via Literal) so the records survive ``json.dumps`` for any
# future persistence step while gaining static narrowability.
RunMode = Literal["local", "slurm", "validate", "unknown"]
RunStatus = Literal["running", "submitting", "complete", "failed", "cancelled", "unknown"]


@dataclass
class RunRecord:
    """One row in the registry.

    Attributes:
        run_id: Stable identifier within this process. Local runs use the
            output-dir relative path; SLURM runs use ``slurm-<job_id>``.
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
        slurm_job_id: SLURM array job ID for SLURM runs (``None``
            otherwise). Read from
            ``progress/job_metadata.json::chunk_job_ids`` for rehydrated
            entries.
        started_at: Monotonic-style ``time.time()`` when the run was
            registered. Persisted on disk only as the manifest's
            ``start_time`` for SLURM rehydration.
        log_path: ``<output_dir>/.gui_log/stdout.log`` if the runner
            tee'd stdout there, else ``None``.
    """

    run_id: str
    mode: RunMode
    output_dir: Path
    rel_path: str
    status: RunStatus = "unknown"
    pid: int | None = None
    slurm_job_id: str | None = None
    started_at: float = field(default_factory=time.time)
    log_path: Path | None = None


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

    # ------------------------------------------------------------------
    # CRUD-ish API
    # ------------------------------------------------------------------

    def register(self, record: RunRecord) -> None:
        """Insert or replace ``record``. Idempotent on ``run_id``."""
        with self._lock:
            self._records[record.run_id] = record

    def get(self, run_id: str) -> RunRecord | None:
        """Return the record for ``run_id``, or ``None`` if missing."""
        with self._lock:
            return self._records.get(run_id)

    def list(self) -> list[RunRecord]:
        """Snapshot the records. Caller may iterate without holding the lock."""
        with self._lock:
            return list(self._records.values())

    def update_status(self, run_id: str, status: RunStatus) -> bool:
        """Update ``status`` for ``run_id``. Returns ``False`` if unknown."""
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return False
            record.status = status
            return True

    def update_pid(self, run_id: str, pid: int | None) -> bool:
        """Set ``pid`` (e.g. once Popen returns the subprocess handle)."""
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return False
            record.pid = pid
            return True

    def update_slurm_job_id(self, run_id: str, job_id: str | None) -> bool:
        """Set ``slurm_job_id`` (post-submit, after job_metadata.json exists)."""
        with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return False
            record.slurm_job_id = job_id
            return True

    def remove(self, run_id: str) -> bool:
        """Drop ``run_id`` from the registry."""
        with self._lock:
            return self._records.pop(run_id, None) is not None

    def clear(self) -> None:
        """Drop every record. Used by tests."""
        with self._lock:
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
                rel = str(output_dir.relative_to(sandbox.root))
            except ValueError:
                continue
            run_id = rel  # rel-path-as-id keeps it stable across boots
            if self.get(run_id) is not None:
                continue
            mode, status, slurm_job_id = self._read_status_from_manifest(
                output_dir
            )
            record = RunRecord(
                run_id=run_id,
                mode=mode,
                output_dir=output_dir,
                rel_path=rel,
                status=status,
                slurm_job_id=slurm_job_id,
            )
            self.register(record)
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
        stack: list[tuple[Path, int]] = [(sandbox.root, 0)]
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
                if not child.is_dir():
                    continue
                caps = classify(child)
                if caps.is_cli_output or caps.is_process_only_output:
                    yield child
                # Recurse regardless — a CLI output may itself contain
                # nested ones in unusual sandboxes (unlikely but cheap).
                stack.append((child, depth + 1))

    @staticmethod
    def _read_status_from_manifest(
        output_dir: Path,
    ) -> tuple[RunMode, RunStatus, str | None]:
        """Best-effort read of mode + status + SLURM job id from manifest.

        Returns ``("unknown", "unknown", None)`` on any failure.
        """
        import json

        manifest_path = resolve_manifest_json_path(output_dir)
        if not manifest_path.is_file():
            return ("unknown", "unknown", None)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ("unknown", "unknown", None)

        execution_mode = manifest.get(DashboardManifestKey.EXECUTION_MODE, "unknown")
        is_complete = bool(manifest.get(DashboardManifestKey.IS_COMPLETE))
        failed = int(manifest.get(DashboardManifestKey.FAILED, 0) or 0)
        completed = int(manifest.get(DashboardManifestKey.COMPLETED, 0) or 0)
        total = int(manifest.get(DashboardManifestKey.TOTAL_IMAGES, 0) or 0)

        if is_complete:
            status: RunStatus = "complete" if failed == 0 else "failed"
        elif total > 0 and (completed + failed) >= total:
            status = "complete" if failed == 0 else "failed"
        else:
            status = "running"

        mode: RunMode
        if execution_mode == "slurm":
            mode = "slurm"
        elif execution_mode == "local":
            mode = "local"
        else:
            mode = "unknown"

        slurm_info = manifest.get(DashboardManifestKey.SLURM_INFO) or {}
        chunk_job_ids = slurm_info.get(DashboardManifestSlurmInfoKey.CHUNK_JOB_IDS) or {}
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


# Re-export Iterable so type-checker sees it; import lifted to module
# scope to avoid the ``del`` shenanigans elsewhere.
_ = Iterable
