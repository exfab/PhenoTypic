"""Bounded process-local jobs for asynchronous Results/Analysis binding."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from phenotypic.gui._config import THREAD_NAME_PREFIX
from phenotypic.gui.results_viewer._discovery_contracts import (
    OutputDiscoveryCancellation,
    OutputDiscoveryCancelledError,
    OutputDiscoveryProgress,
)
from phenotypic.gui.shell._binding import BindingSupersededError

logger = logging.getLogger(__name__)

__all__ = [
    "ResultsBindJobContext",
    "ResultsBindJobFailure",
    "ResultsBindJobManager",
    "ResultsBindJobSnapshot",
    "ResultsBindJobSubmission",
]

ResultsBindJobStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "superseded",
]
ResultsBindJobPhase = Literal[
    "queued",
    "classifying",
    "inventory",
    "measurements",
    "indexing",
    "verifying",
    "building_results",
    "building_analysis",
    "publishing",
    "complete",
    "failed",
    "cancelled",
    "superseded",
]
ResultsBindErrorKind = Literal["invalid", "stale", "unavailable"]

_TERMINAL_STATUSES = frozenset(
    {"succeeded", "failed", "cancelled", "superseded"}
)


class ResultsBindJobFailure(RuntimeError):
    """Classify an expected binding failure for status presentation.

    Args:
        kind: Stable machine-readable failure category.
        message: Reader-facing failure detail.
    """

    def __init__(self, kind: ResultsBindErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class ResultsBindJobSnapshot:
    """Immutable, JSON-ready view of one binding job."""

    job_id: str
    target: Path
    ticket: int
    status: ResultsBindJobStatus
    phase: ResultsBindJobPhase
    detail: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    attempt: int
    completed: int | None
    total: int | None
    cache_hit: bool
    error_kind: ResultsBindErrorKind | None
    error: str | None
    result: Mapping[str, Any] | None

    @property
    def terminal(self) -> bool:
        """Return whether no further state transition will be published."""
        return self.status in _TERMINAL_STATUSES

    def as_dict(self) -> dict[str, Any]:
        """Return the stable HTTP payload for this job."""
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "target": str(self.target),
            "ticket": self.ticket,
            "status": self.status,
            "phase": self.phase,
            "detail": self.detail,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": (
                self.started_at.isoformat()
                if self.started_at is not None
                else None
            ),
            "finished_at": (
                self.finished_at.isoformat()
                if self.finished_at is not None
                else None
            ),
            "attempt": self.attempt,
            "completed": self.completed,
            "total": self.total,
            "cache_hit": self.cache_hit,
            "terminal": self.terminal,
            "error_kind": self.error_kind,
            "error": self.error,
        }
        if self.result is not None:
            payload["result"] = dict(self.result)
        return payload


@dataclass(frozen=True)
class ResultsBindJobSubmission:
    """Result of submitting or deduplicating one request."""

    job: ResultsBindJobSnapshot
    deduplicated: bool


@dataclass
class _ResultsBindJob:
    job_id: str
    target: Path
    ticket: int
    cancellation: OutputDiscoveryCancellation
    status: ResultsBindJobStatus = "queued"
    phase: ResultsBindJobPhase = "queued"
    detail: str = "Waiting for a Results binding worker."
    created_at: datetime = field(default_factory=lambda: _utc_now())
    updated_at: datetime = field(default_factory=lambda: _utc_now())
    started_at: datetime | None = None
    finished_at: datetime | None = None
    attempt: int = 1
    completed: int | None = None
    total: int | None = None
    cache_hit: bool = False
    error_kind: ResultsBindErrorKind | None = None
    error: str | None = None
    result: Mapping[str, Any] | None = None
    worker_active: bool = False


class ResultsBindJobContext:
    """Cancellation, progress, and CAS handle passed to a binding worker."""

    def __init__(
        self,
        manager: "ResultsBindJobManager",
        job_id: str,
        target: Path,
        ticket: int,
        cancellation: OutputDiscoveryCancellation,
    ) -> None:
        self._manager = manager
        self.job_id = job_id
        self.target = target
        self.ticket = ticket
        self.cancellation = cancellation

    def report_discovery(self, update: OutputDiscoveryProgress) -> None:
        """Publish one O1 discovery progress update."""
        self._manager._report_discovery(self.job_id, update)

    def set_phase(
        self,
        phase: Literal[
            "building_results",
            "building_analysis",
            "publishing",
        ],
        detail: str,
    ) -> None:
        """Publish a candidate-construction or publication phase."""
        self._manager._set_phase(self.job_id, phase, detail)

    def require_active(self) -> None:
        """Raise if cancellation or a newer request fenced this job."""
        self._manager._require_active(self.job_id)


class ResultsBindJobManager:
    """Run Results binding in a bounded worker set with latest-request wins.

    The manager admits at most ``max_workers`` running jobs plus one pending
    job. A newer distinct request cooperatively cancels all older work and
    replaces the single pending slot. Duplicate active requests reuse their
    existing job and ticket.

    Args:
        execute: Worker callback. It must call ``context.require_active()``
            immediately before any publication side effect.
        issue_ticket: Allocates the monotonic publication CAS ticket.
        max_workers: Fixed daemon worker count.
        max_terminal_jobs: Maximum completed job records retained for polling.
    """

    def __init__(
        self,
        execute: Callable[
            [ResultsBindJobContext],
            Mapping[str, Any],
        ],
        *,
        issue_ticket: Callable[[], int],
        max_workers: int = 2,
        max_terminal_jobs: int = 32,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if max_terminal_jobs < 1:
            raise ValueError("max_terminal_jobs must be at least 1")
        self._execute = execute
        self._issue_ticket = issue_ticket
        self._max_terminal_jobs = max_terminal_jobs
        self._condition = threading.Condition(threading.RLock())
        self._jobs: OrderedDict[str, _ResultsBindJob] = OrderedDict()
        self._pending_job_id: str | None = None
        self._stopping = False
        self._workers_started = False
        self._workers = tuple(
            threading.Thread(
                target=self._worker_loop,
                name=f"{THREAD_NAME_PREFIX}-results-bind-{index + 1}",
                daemon=True,
            )
            for index in range(max_workers)
        )
    def submit(self, target: Path) -> ResultsBindJobSubmission:
        """Submit a target, deduplicating identical active work."""
        normalized = Path(target)
        with self._condition:
            if self._stopping:
                raise RuntimeError("Results binding manager is shutting down")
            for job in reversed(self._jobs.values()):
                if (
                    job.target == normalized
                    and job.status in {"queued", "running"}
                ):
                    return ResultsBindJobSubmission(
                        job=self._snapshot(job),
                        deduplicated=True,
                    )

            now = _utc_now()
            for job in self._jobs.values():
                if job.status not in {"queued", "running"}:
                    continue
                job.cancellation.cancel()
                job.status = "superseded"
                job.phase = "superseded"
                job.detail = "Superseded by a newer Results binding request."
                job.updated_at = now
                job.finished_at = now
                if self._pending_job_id == job.job_id:
                    self._pending_job_id = None

            job_id = uuid4().hex
            job = _ResultsBindJob(
                job_id=job_id,
                target=normalized,
                ticket=self._issue_ticket(),
                cancellation=OutputDiscoveryCancellation(),
            )
            self._jobs[job_id] = job
            self._pending_job_id = job_id
            if not self._workers_started:
                for worker in self._workers:
                    worker.start()
                self._workers_started = True
            self._prune_locked()
            self._condition.notify_all()
            return ResultsBindJobSubmission(
                job=self._snapshot(job),
                deduplicated=False,
            )

    def get(self, job_id: str) -> ResultsBindJobSnapshot | None:
        """Return one job snapshot, or ``None`` when unknown/expired."""
        with self._condition:
            job = self._jobs.get(job_id)
            return None if job is None else self._snapshot(job)

    def cancel(self, job_id: str) -> ResultsBindJobSnapshot | None:
        """Cancel one job, invalidate its CAS ticket, and return its snapshot."""
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in {"queued", "running"}:
                now = _utc_now()
                job.cancellation.cancel()
                job.status = "cancelled"
                job.phase = "cancelled"
                job.detail = "Results binding cancelled."
                job.updated_at = now
                job.finished_at = now
                if self._pending_job_id == job_id:
                    self._pending_job_id = None
                # Linearize DELETE against publication. Once cancel() returns,
                # commit_if_latest() cannot accept this job's ticket even if
                # the worker had passed its cooperative cancellation check.
                self._issue_ticket()
                self._condition.notify_all()
            return self._snapshot(job)

    def shutdown(self, *, timeout_seconds: float = 2.0) -> None:
        """Cancel outstanding work and stop worker threads.

        Production workers are daemons and live for the process lifetime.
        This method exists so focused tests can release them deterministically.
        """
        with self._condition:
            if self._stopping:
                return
            self._stopping = True
            now = _utc_now()
            invalidated_publication = False
            for job in self._jobs.values():
                if job.status not in {"queued", "running"}:
                    continue
                job.cancellation.cancel()
                invalidated_publication = True
                job.status = "cancelled"
                job.phase = "cancelled"
                job.detail = "Results binding manager shut down."
                job.updated_at = now
                job.finished_at = now
            if invalidated_publication:
                self._issue_ticket()
            self._pending_job_id = None
            self._condition.notify_all()
        if self._workers_started:
            deadline = time.monotonic() + timeout_seconds
            for worker in self._workers:
                worker.join(timeout=max(0.0, deadline - time.monotonic()))

    @property
    def tracked_job_count(self) -> int:
        """Return the number of active and retained terminal job records."""
        with self._condition:
            return len(self._jobs)

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending_job_id is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                job_id = self._pending_job_id
                self._pending_job_id = None
                if job_id is None:
                    continue
                job = self._jobs.get(job_id)
                if job is None or job.status != "queued":
                    continue
                now = _utc_now()
                job.worker_active = True
                job.status = "running"
                job.detail = "Starting Results binding."
                job.started_at = now
                job.updated_at = now
                context = ResultsBindJobContext(
                    self,
                    job.job_id,
                    job.target,
                    job.ticket,
                    job.cancellation,
                )

            try:
                result = self._execute(context)
            except BindingSupersededError:
                self._finish_if_running(
                    job_id,
                    status="superseded",
                    phase="superseded",
                    detail="Superseded by a newer Results binding request.",
                )
            except OutputDiscoveryCancelledError:
                self._finish_if_running(
                    job_id,
                    status="cancelled",
                    phase="cancelled",
                    detail="Results binding cancelled.",
                )
            except ResultsBindJobFailure as exc:
                self._finish_if_running(
                    job_id,
                    status="failed",
                    phase="failed",
                    detail="Results binding failed.",
                    error_kind=exc.kind,
                    error=str(exc),
                )
            except Exception as exc:  # noqa: BLE001 - isolate worker failures
                logger.exception(
                    "unexpected Results binding failure for %s",
                    context.target,
                )
                self._finish_if_running(
                    job_id,
                    status="failed",
                    phase="failed",
                    detail="Results binding failed.",
                    error_kind="unavailable",
                    error=str(exc),
                )
            else:
                self._finish_if_running(
                    job_id,
                    status="succeeded",
                    phase="complete",
                    detail="Results and Analysis binding published.",
                    result=result,
                )
            finally:
                with self._condition:
                    current = self._jobs.get(job_id)
                    if current is not None:
                        current.worker_active = False
                    self._prune_locked()

    def _report_discovery(
        self,
        job_id: str,
        update: OutputDiscoveryProgress,
    ) -> None:
        with self._condition:
            job = self._require_running_locked(job_id)
            job.phase = update.phase
            job.detail = update.detail
            job.attempt = update.attempt
            job.completed = update.completed
            job.total = update.total
            job.cache_hit = update.cache_hit
            job.updated_at = _utc_now()

    def _set_phase(
        self,
        job_id: str,
        phase: ResultsBindJobPhase,
        detail: str,
    ) -> None:
        with self._condition:
            job = self._require_running_locked(job_id)
            job.phase = phase
            job.detail = detail
            job.completed = None
            job.total = None
            job.updated_at = _utc_now()

    def _require_active(self, job_id: str) -> None:
        with self._condition:
            self._require_running_locked(job_id)

    def _require_running_locked(self, job_id: str) -> _ResultsBindJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise BindingSupersededError(
                f"Results binding job {job_id} is no longer retained."
            )
        if job.status == "superseded":
            raise BindingSupersededError(
                f"Results binding job {job_id} was superseded."
            )
        if job.status == "cancelled" or job.cancellation.cancelled:
            raise OutputDiscoveryCancelledError(
                f"Results binding job {job_id} was cancelled."
            )
        if job.status != "running":
            raise BindingSupersededError(
                f"Results binding job {job_id} is no longer active."
            )
        return job

    def _finish_if_running(
        self,
        job_id: str,
        *,
        status: ResultsBindJobStatus,
        phase: ResultsBindJobPhase,
        detail: str,
        error_kind: ResultsBindErrorKind | None = None,
        error: str | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None or job.status != "running":
                return
            now = _utc_now()
            job.status = status
            job.phase = phase
            job.detail = detail
            job.updated_at = now
            job.finished_at = now
            job.error_kind = error_kind
            job.error = error
            job.result = result

    def _prune_locked(self) -> None:
        terminal_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in _TERMINAL_STATUSES and not job.worker_active
        ]
        excess = len(terminal_ids) - self._max_terminal_jobs
        for job_id in terminal_ids[: max(0, excess)]:
            self._jobs.pop(job_id, None)

    @staticmethod
    def _snapshot(job: _ResultsBindJob) -> ResultsBindJobSnapshot:
        return ResultsBindJobSnapshot(
            job_id=job.job_id,
            target=job.target,
            ticket=job.ticket,
            status=job.status,
            phase=job.phase,
            detail=job.detail,
            created_at=job.created_at,
            updated_at=job.updated_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            attempt=job.attempt,
            completed=job.completed,
            total=job.total,
            cache_hit=job.cache_hit,
            error_kind=job.error_kind,
            error=job.error,
            result=job.result,
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)
