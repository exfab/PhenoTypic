"""Unit tests for bounded asynchronous Results binding jobs."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from phenotypic.gui.results_viewer._discovery_contracts import (
    OutputDiscoveryProgress,
)
from phenotypic.gui.shell._binding_jobs import ResultsBindJobManager


def _wait_for_job(
    manager: ResultsBindJobManager,
    job_id: str,
    *,
    status: str,
    timeout: float = 3.0,
) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.get(job_id)
        if snapshot is not None and snapshot.status == status:
            return snapshot
        threading.Event().wait(0.01)
    raise AssertionError(f"job {job_id} did not reach {status}")


def test_success_reports_discovery_and_publication_phases(
    tmp_path: Path,
) -> None:
    """A worker publishes O1 progress before returning its binding result."""
    release = threading.Event()
    tickets = iter(range(1, 10))

    def _execute(context: Any) -> dict[str, Any]:
        context.report_discovery(
            OutputDiscoveryProgress(
                phase="inventory",
                detail="Scanning 2 processing files.",
                completed=1,
                total=2,
            )
        )
        release.wait(3.0)
        context.set_phase("building_results", "Building Results.")
        context.set_phase("building_analysis", "Building Analysis.")
        context.set_phase("publishing", "Publishing.")
        context.require_active()
        return {"binding_generation": "generation-1"}

    manager = ResultsBindJobManager(
        _execute,
        issue_ticket=lambda: next(tickets),
        max_workers=1,
    )
    try:
        submission = manager.submit(tmp_path / "output")
        _wait_for_job(manager, submission.job.job_id, status="running")
        deadline = time.monotonic() + 3.0
        progress = manager.get(submission.job.job_id)
        while (
            progress is not None
            and progress.phase != "inventory"
            and time.monotonic() < deadline
        ):
            threading.Event().wait(0.01)
            progress = manager.get(submission.job.job_id)
        assert progress is not None
        assert progress.phase == "inventory"
        assert (progress.completed, progress.total) == (1, 2)
        release.set()
        complete = _wait_for_job(
            manager,
            submission.job.job_id,
            status="succeeded",
        )
        assert complete.phase == "complete"
        assert complete.result == {"binding_generation": "generation-1"}
    finally:
        release.set()
        manager.shutdown()


def test_same_active_request_reuses_job_and_ticket(tmp_path: Path) -> None:
    """A duplicate proxy/browser retry does not enqueue duplicate discovery."""
    entered = threading.Event()
    release = threading.Event()
    calls: list[Path] = []
    tickets: list[int] = []

    def _issue_ticket() -> int:
        ticket = len(tickets) + 1
        tickets.append(ticket)
        return ticket

    def _execute(context: Any) -> dict[str, Any]:
        calls.append(context.target)
        entered.set()
        release.wait(3.0)
        context.require_active()
        return {}

    manager = ResultsBindJobManager(
        _execute,
        issue_ticket=_issue_ticket,
        max_workers=1,
    )
    try:
        first = manager.submit(tmp_path / "output")
        assert entered.wait(3.0)
        duplicate = manager.submit(tmp_path / "output")
        assert duplicate.deduplicated is True
        assert duplicate.job.job_id == first.job.job_id
        assert duplicate.job.ticket == first.job.ticket
        assert tickets == [1]
        release.set()
        _wait_for_job(manager, first.job.job_id, status="succeeded")
        assert calls == [tmp_path / "output"]
    finally:
        release.set()
        manager.shutdown()


def test_newer_request_supersedes_and_fences_running_publication(
    tmp_path: Path,
) -> None:
    """The old worker cannot publish after a newer ticket is issued."""
    first_entered = threading.Event()
    release_first = threading.Event()
    published: list[Path] = []
    next_ticket = 0

    def _issue_ticket() -> int:
        nonlocal next_ticket
        next_ticket += 1
        return next_ticket

    def _execute(context: Any) -> dict[str, Any]:
        if context.target.name == "old":
            first_entered.set()
            release_first.wait(3.0)
        context.require_active()
        published.append(context.target)
        return {}

    manager = ResultsBindJobManager(
        _execute,
        issue_ticket=_issue_ticket,
        max_workers=2,
    )
    try:
        old = manager.submit(tmp_path / "old")
        assert first_entered.wait(3.0)
        new = manager.submit(tmp_path / "new")
        assert manager.get(old.job.job_id).status == "superseded"  # type: ignore[union-attr]
        _wait_for_job(manager, new.job.job_id, status="succeeded")
        release_first.set()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and published != [tmp_path / "new"]:
            threading.Event().wait(0.01)
        assert published == [tmp_path / "new"]
    finally:
        release_first.set()
        manager.shutdown()


def test_cancelled_running_job_cannot_publish(tmp_path: Path) -> None:
    """DELETE-style cancellation terminalizes status and fences the worker."""
    entered = threading.Event()
    release = threading.Event()
    published: list[Path] = []

    def _execute(context: Any) -> dict[str, Any]:
        entered.set()
        release.wait(3.0)
        context.require_active()
        published.append(context.target)
        return {}

    manager = ResultsBindJobManager(
        _execute,
        issue_ticket=lambda: 1,
        max_workers=1,
    )
    try:
        submission = manager.submit(tmp_path / "output")
        assert entered.wait(3.0)
        cancelled = manager.cancel(submission.job.job_id)
        assert cancelled is not None
        assert cancelled.status == "cancelled"
        release.set()
        threading.Event().wait(0.05)
        assert published == []
        assert manager.get(submission.job.job_id).status == "cancelled"  # type: ignore[union-attr]
    finally:
        release.set()
        manager.shutdown()


def test_pending_slot_and_terminal_history_are_bounded(tmp_path: Path) -> None:
    """Repeated supersession retains one pending request and bounded history."""
    entered = threading.Event()
    release = threading.Event()

    def _execute(context: Any) -> dict[str, Any]:
        if context.target.name == "blocked":
            entered.set()
            release.wait(3.0)
        context.require_active()
        return {}

    ticket = 0

    def _issue_ticket() -> int:
        nonlocal ticket
        ticket += 1
        return ticket

    manager = ResultsBindJobManager(
        _execute,
        issue_ticket=_issue_ticket,
        max_workers=1,
        max_terminal_jobs=2,
    )
    try:
        blocked = manager.submit(tmp_path / "blocked")
        assert entered.wait(3.0)
        newest = None
        for index in range(8):
            newest = manager.submit(tmp_path / f"new-{index}")
        assert newest is not None
        assert manager.get(blocked.job.job_id).status == "superseded"  # type: ignore[union-attr]
        # The active worker is retained plus at most two terminal records and
        # one pending record. Superseded pending jobs are pruned immediately.
        assert manager.tracked_job_count <= 4
        release.set()
        _wait_for_job(manager, newest.job.job_id, status="succeeded")
        deadline = time.monotonic() + 3.0
        while manager.tracked_job_count > 2 and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        assert manager.tracked_job_count <= 2
    finally:
        release.set()
        manager.shutdown()
