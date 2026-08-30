"""Private GUI-to-local-CLI completion publication contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Mapping
from uuid import UUID

from phenotypic.sdk_ import (
    DashboardManifestKey,
    manifest_json_path,
)
from phenotypic.sdk_._io_constants import GUI_RECORD_GENERATION_ENV_VAR

LocalManifestCompletionProblem = Literal[
    "non_local",
    "wrong_generation",
    "incomplete",
]


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


def local_manifest_completion_problem(
    payload: Mapping[str, object],
    generation: str,
) -> LocalManifestCompletionProblem | None:
    """Classify why a manifest cannot complete one local GUI generation."""
    if payload.get(DashboardManifestKey.EXECUTION_MODE) != "local":
        return "non_local"
    if payload.get(DashboardManifestKey.GUI_RECORD_GENERATION) != generation:
        return "wrong_generation"

    completed = payload.get(DashboardManifestKey.COMPLETED)
    failed = payload.get(DashboardManifestKey.FAILED)
    total = payload.get(DashboardManifestKey.TOTAL_IMAGES)
    counts_are_ints = all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (completed, failed, total)
    )
    if (
        payload.get(DashboardManifestKey.IS_COMPLETE) is not True
        or not counts_are_ints
        or failed != 0
        or completed != total
    ):
        return "incomplete"
    return None


def publish_local_gui_completion(output_dir: Path) -> bool:
    """Publish exact GUI generation evidence for a coherent local run.

    Ordinary non-GUI CLI invocations do not carry the private environment
    token and remain unchanged. GUI-launched local runs publish only after
    their caller has completed the rest of local finalization.

    Args:
        output_dir: Canonical local run output root.

    Returns:
        ``True`` when a GUI marker was published; ``False`` when the process
        was not launched by the GUI.

    Raises:
        RuntimeError: If current marker evidence is incomplete, or a legacy
            run's canonical manifest is missing, unreadable, or incomplete.
    """
    generation = gui_record_generation_from_environment()

    from ._cli_completion import current_run_is_complete

    marker_complete = current_run_is_complete(output_dir)
    if marker_complete is False:
        raise RuntimeError(
            "Cannot publish GUI local completion while current image outcomes "
            "remain incomplete"
        )
    if marker_complete is None and generation is None:
        return False
    if marker_complete is None:
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
                "Cannot publish legacy GUI local completion without a readable "
                f"canonical manifest at {path}"
            ) from exc

        # ``generation`` is ``str | None`` on the way in; the guard above
        # has already raised for an unreadable manifest, and the predicate
        # requires a concrete generation to compare against.
        if generation is None:
            raise RuntimeError(
                "Cannot publish legacy GUI local completion without a "
                "processing generation"
            )
        if local_manifest_completion_problem(payload, generation) is not None:
            raise RuntimeError(
                "Cannot publish legacy GUI local completion for a "
                "stale-generation, incomplete, failed, or non-local manifest"
            )

    from ._cli_completion import publish_run_completion_evidence

    publish_run_completion_evidence(
        output_dir,
        execution_epoch="local",
        gui_record_generation=generation,
    )
    return True


__all__ = [
    "gui_record_generation_from_environment",
    "local_manifest_completion_problem",
    "publish_local_gui_completion",
]
