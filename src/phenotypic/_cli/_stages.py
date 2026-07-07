"""Closed staged-GPU stage tags used in CLI event logs."""
from __future__ import annotations

from typing import Literal

StageTag = Literal["stage1", "stage2", "stage3"]

STAGE_PREPROCESS: StageTag = "stage1"
STAGE_GPU_DETECT: StageTag = "stage2"
STAGE_MEASURE: StageTag = "stage3"
STAGED_TERMINAL_STAGE: StageTag = STAGE_MEASURE

VALID_STAGE_TAGS = frozenset(
    {STAGE_PREPROCESS, STAGE_GPU_DETECT, STAGE_MEASURE}
)


def validate_stage_tag(stage: str | None) -> StageTag | None:
    """Return a valid staged-GPU tag, preserving legacy ``None`` rows.

    Args:
        stage: Event-log stage tag or ``None`` for legacy non-staged rows.

    Returns:
        The validated stage tag, or ``None``.

    Raises:
        ValueError: If a non-empty stage tag is not a known staged-GPU tag.
    """
    if stage is None:
        return None
    if stage in VALID_STAGE_TAGS:
        return stage  # type: ignore[return-value]
    raise ValueError(
        f"Invalid stage tag: '{stage}' "
        f"(expected one of {sorted(VALID_STAGE_TAGS)})"
    )
