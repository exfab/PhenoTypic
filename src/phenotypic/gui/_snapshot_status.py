"""Shared status presentation for refreshable output snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING

from phenotypic.gui.results_viewer._output_consistency import (
    inspect_output_consistency,
)

if TYPE_CHECKING:
    from phenotypic.gui.results_viewer._output_root import OutputRoot

__all__ = ["snapshot_refresh_status"]


def snapshot_refresh_status(
    output_root: "OutputRoot",
    *,
    refresh_supported: bool,
) -> tuple[str, str, bool]:
    """Return the snapshot label, color, and Refresh disabled state."""
    active_now = output_root.active_run_is_currently_running()
    if not refresh_supported:
        if active_now:
            return (
                "Active run detected · restart app after it finishes",
                "warning",
                True,
            )
        if output_root.snapshot.active_run:
            return "Run finished · restart standalone app", "info", True
        completion_status = _completion_evidence_status(
            output_root,
            refresh_supported=False,
        )
        if completion_status is not None:
            return completion_status
        current = (
            output_root.snapshot_is_current()
            and output_root.refresh_state_is_current()
        )
        if current:
            return "Current · restart app to refresh", "success", True
        return "Changed on disk · restart standalone app", "danger", True
    if active_now:
        if output_root.snapshot.active_run:
            return "Active run snapshot", "warning", True
        return "Active run detected · refresh snapshot", "warning", False
    if output_root.snapshot.active_run:
        return "Run finished · refresh snapshot", "info", False
    completion_status = _completion_evidence_status(
        output_root,
        refresh_supported=True,
    )
    if completion_status is not None:
        return completion_status
    current = (
        output_root.snapshot_is_current()
        and output_root.refresh_state_is_current()
    )
    if current:
        return "Current", "success", False
    return "Changed on disk", "danger", False


def _completion_evidence_status(
    output_root: "OutputRoot",
    *,
    refresh_supported: bool,
) -> tuple[str, str, bool] | None:
    """Return a non-current status for unsafe or changed completion evidence."""
    try:
        fresh = inspect_output_consistency(output_root.layout)
    except OSError:
        return (
            "Read-only · completion evidence unavailable",
            "danger",
            not refresh_supported,
        )
    if fresh.is_read_only:
        return (
            f"Read-only · {fresh.state} completion evidence",
            "danger",
            not refresh_supported,
        )
    if (
        fresh.evidence_fingerprint
        != output_root.consistency.evidence_fingerprint
    ):
        action = (
            "refresh snapshot"
            if refresh_supported
            else "restart standalone app"
        )
        return (
            f"Completion evidence changed · {action}",
            "danger",
            not refresh_supported,
        )
    return None
