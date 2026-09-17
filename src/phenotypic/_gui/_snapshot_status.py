"""Shared status presentation for refreshable output snapshots.

**The badge answers two questions, not one** (spec §11, CAN-18). "Is the run
finished?" comes from :func:`~phenotypic.sdk_.resolve_run_state`; "does the
snapshot this session is holding still match disk?" comes from
``OutputRoot.snapshot_is_current``. Collapsing them loses a real state: a
re-finalize over an unchanged inventory rewrites the deliverables while
``completion`` stays ``complete``, so a badge driven by ``completion`` alone
reads "Current" over a stale mirror.

What this module no longer does is re-derive completion from
``inspect_output_consistency`` and re-hash the seven consumed-state
deliverables (``measurements.parquet``, ``measurements.csv``,
``pipeline.json``, ``curation_labels.parquet``, ``custom_categories.json``,
``qc.duckdb``, ``review_state.json``) on every 5-10 s poll. Dropping that
second read also fixes audit S2: the GUI's own curation write is no longer
reported back to the user as external drift, because the currency axis is
``snapshot_is_current``, which deliberately excludes GUI-owned mutable
state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from phenotypic.sdk_ import resolve_run_state

if TYPE_CHECKING:
    from phenotypic._gui.results_viewer._output_root import OutputRoot
    from phenotypic.sdk_._run_state import Completion

__all__ = ["snapshot_refresh_status"]

#: The unfinished half of the badge: ``completion`` -> (label, colour).
#:
#: ``complete`` and ``active`` are deliberately absent, for opposite reasons.
#: ``complete`` is the only completion whose badge still depends on the second
#: axis, so it is answered by the currency check below. ``active`` is answered
#: by the liveness branch above, which owns the three different labels an
#: active run gets depending on whether Refresh is wired and whether the
#: binding was captured mid-run.
#:
#: ``test_every_completion_literal_has_a_badge`` binds this to
#: :data:`~phenotypic.sdk_._run_state.Completion`, so a fifth literal is a
#: failing test rather than a ``KeyError`` on a poll.
_UNFINISHED_BADGE: Mapping[str, tuple[str, str]] = {
    "incomplete": ("Run incomplete", "warning"),
    "failed": ("Run failed", "danger"),
}


def snapshot_refresh_status(
    output_root: "OutputRoot",
    *,
    refresh_supported: bool,
) -> tuple[str, str, bool]:
    """Return the snapshot label, color, and Refresh disabled state.

    Args:
        output_root: The bound output handle backing this session.
        refresh_supported: Whether an in-place Refresh is wired. ``False`` in
            the standalone apps, where the only remedy is a restart, so every
            branch reports a restart action and leaves the button disabled.

    Returns:
        ``(label, bootstrap colour, Refresh disabled)``.
    """
    completion = _run_completion(output_root)
    if output_root.active_run_is_currently_running() or completion == "active":
        if not refresh_supported:
            return (
                "Active run detected · restart app after it finishes",
                "warning",
                True,
            )
        if output_root.snapshot.active_run:
            return "Active run snapshot", "warning", True
        return "Active run detected · refresh snapshot", "warning", False
    if output_root.snapshot.active_run:
        if not refresh_supported:
            return "Run finished · restart standalone app", "info", True
        return "Run finished · refresh snapshot", "info", False
    unfinished = _UNFINISHED_BADGE.get(completion or "")
    if unfinished is not None:
        label, color = unfinished
        return (
            f"{label} · {_remedy(refresh_supported)}",
            color,
            not refresh_supported,
        )
    if output_root.snapshot_is_current():
        if not refresh_supported:
            return "Current · restart app to refresh", "success", True
        return "Current", "success", False
    if not refresh_supported:
        return "Changed on disk · restart standalone app", "danger", True
    return "Changed on disk", "danger", False


def _run_completion(output_root: "OutputRoot") -> "Completion | None":
    """Return the run's completion, or ``None`` for a standalone bundle.

    ``None`` fires on exactly one input: a portable ``deliverables/`` bundle,
    where :class:`~phenotypic.sdk_.BundleLayout` resolves ``output_root`` to
    ``None`` because there is no run directory above the deliverables and so
    no machine state to read. Passing that ``None`` to ``resolve_run_state``
    would raise ``TypeError`` out of a status poll; reporting it as
    ``incomplete`` would be worse, permanently badging every bundle as an
    unfinished run. A bundle has no completion question, so the badge skips
    to the currency question -- which is what the retired
    ``inspect_output_consistency`` did for a bundle too.

    ``depth="shallow"`` re-stats the verification cache rather than re-reading
    artifact bytes. The first poll of a session is cold and pays for a deep
    pass; every subsequent poll is served from the in-process cache.
    """
    root = output_root.layout.output_root
    if root is None:
        return None
    return resolve_run_state(root, depth="shallow").completion


def _remedy(refresh_supported: bool) -> str:
    """Return the action clause for a session that can or cannot refresh."""
    if refresh_supported:
        return "refresh snapshot"
    return "restart standalone app"
