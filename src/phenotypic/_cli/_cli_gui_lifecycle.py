"""Private GUI-to-local-CLI completion publication contract."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from uuid import UUID

from phenotypic.sdk_ import (
    DashboardManifestKey,
    atomic_write_json,
    manifest_json_path,
    run_completion_marker_path,
)
from phenotypic.sdk_._io_constants import GUI_RECORD_GENERATION_ENV_VAR


def gui_record_generation_from_environment(
    environment: Mapping[str, str] | None = None,
) -> str | None:
    """Return the canonical private GUI generation token, when present."""
    raw = (os.environ if environment is None else environment).get(
        GUI_RECORD_GENERATION_ENV_VAR
    )
    if raw is None:
        return None
    try:
        return str(UUID(raw))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{GUI_RECORD_GENERATION_ENV_VAR} is not a valid UUID"
        ) from exc


def publish_local_gui_completion(output_dir: Path) -> bool:
    """Publish exact GUI generation evidence for a coherent local manifest.

    Ordinary non-GUI CLI invocations do not carry the private environment
    token and remain unchanged. GUI-launched local runs publish only after
    their caller has completed the rest of local finalization.

    Args:
        output_dir: Canonical local run output root.

    Returns:
        ``True`` when a GUI marker was published; ``False`` when the process
        was not launched by the GUI.

    Raises:
        RuntimeError: If a GUI generation is present but the canonical local
            manifest is missing, unreadable, incomplete, or failed.
    """
    generation = gui_record_generation_from_environment()
    if generation is None:
        return False

    path = manifest_json_path(output_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("manifest is not an object")
    except (
        FileNotFoundError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
    ) as exc:
        raise RuntimeError(
            "Cannot publish GUI local completion without a readable "
            f"canonical manifest at {path}"
        ) from exc

    completed = payload.get(DashboardManifestKey.COMPLETED)
    failed = payload.get(DashboardManifestKey.FAILED)
    total = payload.get(DashboardManifestKey.TOTAL_IMAGES)
    counts_are_ints = all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (completed, failed, total)
    )
    if (
        payload.get(DashboardManifestKey.EXECUTION_MODE) != "local"
        or payload.get(DashboardManifestKey.IS_COMPLETE) is not True
        or not counts_are_ints
        or failed != 0
        or completed != total
    ):
        raise RuntimeError(
            "Cannot publish GUI local completion for an incomplete, failed, "
            "or non-local canonical manifest"
        )

    atomic_write_json(
        run_completion_marker_path(output_dir),
        {
            "schema_version": 1,
            "generation": generation,
            "mode": "local",
            "status": "complete",
            "finalizer_succeeded": True,
            "completed_at": datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            ),
        },
    )
    return True


__all__ = [
    "gui_record_generation_from_environment",
    "publish_local_gui_completion",
]
