"""Recent Runs scanner — sandbox-driven, classifier-typed.

The Run console's "Recent Runs" side panel is rehydrated on every
sandbox refresh by walking the sandbox tree, classifying each directory,
and surfacing those with ``has_dashboard`` (the iframe link) or
``is_cli_output`` (a complete run that may have no dashboard left
behind).

This is a thin wrapper over :class:`RunRegistry.rehydrate_from_sandbox`
that returns view-friendly rows (relative paths, status icons, last-
modified timestamps) without leaking the registry implementation into
the UI layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from phenotypic.gui.shell._classifier import classify
from phenotypic.gui.shell._runs_registry import RunMode, RunRegistry, RunStatus
from phenotypic.gui.shell._sandbox import SandboxRoot

logger = logging.getLogger(__name__)

__all__ = ["RecentRunRow", "scan_recent_runs"]


@dataclass(frozen=True)
class RecentRunRow:
    """One row in the Run console's Recent Runs panel.

    Attributes:
        rel_path: Path of the output dir relative to sandbox root. Used
            both as a display label and as the iframe-src component
            (``/runs/<rel>/dashboard.html``).
        status: ``"running"``, ``"complete"``, ``"failed"``, ``"cancelled"``,
            or ``"unknown"``. Matches :data:`RunStatus`.
        has_dashboard: Whether ``<rel>/dashboard.html`` exists. The UI
            disables the "Open dashboard" link when ``False``.
        last_modified_seconds: Unix epoch (seconds) of the directory's
            most recent modification. Lets the UI sort by recency.
        mode: ``"local"`` / ``"slurm"`` / ``"unknown"`` from the manifest.
    """

    rel_path: str
    status: RunStatus
    has_dashboard: bool
    last_modified_seconds: float
    mode: RunMode


def scan_recent_runs(
    sandbox: SandboxRoot,
    *,
    registry: RunRegistry | None = None,
    max_depth: int = 3,
) -> list[RecentRunRow]:
    """Scan the sandbox + return recent-run rows sorted by recency.

    Args:
        sandbox: Sandbox root.
        registry: Optional :class:`RunRegistry` to update with newly
            discovered output dirs. If supplied, the registry is
            rehydrated as a side effect (cheap; idempotent on existing
            ``run_id`` keys).
        max_depth: How many levels below the root to scan. Default 3
            matches the spec's ``--scan-depth``.

    Returns:
        List of :class:`RecentRunRow`, newest first. Empty if no output
        directories are found.
    """
    if registry is not None:
        registry.rehydrate_from_sandbox(sandbox, max_depth=max_depth)
        records = registry.list()
    else:
        # Standalone path (no registry stash): build records lazily.
        tmp_registry = RunRegistry()
        tmp_registry.rehydrate_from_sandbox(sandbox, max_depth=max_depth)
        records = tmp_registry.list()

    rows: list[RecentRunRow] = []
    for record in records:
        try:
            stat_info = record.output_dir.stat()
        except OSError:
            continue
        caps = classify(record.output_dir)
        rows.append(
            RecentRunRow(
                rel_path=record.rel_path,
                status=record.status,
                has_dashboard=caps.has_dashboard,
                last_modified_seconds=stat_info.st_mtime,
                mode=record.mode,
            )
        )
    rows.sort(key=lambda r: r.last_modified_seconds, reverse=True)
    return rows


# Re-export Path for type-hint clarity in callers that need to construct
# absolute paths from the rel_path.
_ = Path
