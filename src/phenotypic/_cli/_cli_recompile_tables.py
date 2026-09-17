"""Republish one image's record after a table replacement.

**The per-store recompile rewrite this module was named for is gone** (user
ruling, 2026-09-11). ``recompile_embedded_measurement_table(s)`` read each
store's table, projected it onto the descriptor baseline, re-joined
``deliverables/metadata.csv``, wrote it back and republished the marker. That
made sense while the embedded table *was* the joined artifact. P4 moved the
join to finalization, and P7 Task 4 projects every table onto its descriptor
at read (``project_embedded_measurement_table``), so the rewrite changed
nothing the master or the mirror could see -- while creating the
inverted-store refusals, the mixed-Parquet-generation window, and the
table-transition recovery path. ``--mode recompile`` is now aggregate +
finalize only and writes no store byte.

What survives is the one piece that was never about recompile:
:func:`_republish_table_marker`, which ``--mode measure`` calls from
``_cli_process_single`` after ``replace_image_tables`` re-promotes a store, so
no store write outlives the publication that certifies it. The module keeps
its name because that is where its one consumer imports it from.
"""

from __future__ import annotations

import json
from pathlib import Path

from phenotypic.sdk_ import CommitGuard

from ._cli_completion import publish_image_success


def _marker_artifacts(output_dir: Path, marker: dict) -> dict[str, Path]:
    """Resolve the existing marker's artifacts below its output root."""
    raw = marker.get("artifacts")
    if not isinstance(raw, dict):
        raise ValueError("Image completion marker has no artifact mapping")
    artifacts: dict[str, Path] = {}
    output_root = Path(output_dir).resolve()
    for name, descriptor in raw.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError("Image completion marker has invalid artifacts")
        relative = descriptor.get("path")
        if not isinstance(relative, str):
            raise ValueError("Image completion marker artifact has no path")
        resolved = (output_root / relative).resolve()
        resolved.relative_to(output_root)
        artifacts[name] = resolved
    return artifacts


def _republish_table_marker(
    output_dir: Path,
    marker_path: Path,
    *,
    commit_guard: CommitGuard | None,
    lifecycle_epoch: str | None = None,
) -> None:
    """Rehash all existing artifacts and publish the marker last."""
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    publish_image_success(
        output_dir,
        work_id=str(marker["work_id"]),
        dataset=str(marker["dataset"]),
        relative_image_path=str(marker["relative_image_path"]),
        image_stem=str(marker["image_stem"]),
        mode=str(marker["mode"]),
        attempt_id=str(marker["attempt_id"]),
        lifecycle_epoch=(
            lifecycle_epoch
            if lifecycle_epoch is not None
            else str(marker["lifecycle_epoch"])
        ),
        artifacts=_marker_artifacts(output_dir, marker),
        commit_guard=commit_guard,
    )
