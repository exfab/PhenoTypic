"""Closed staged-GPU stage tags used in CLI event logs."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from ._cli_embed_stage import EMBED_STAGE_CONFIG_KEY

StageTag = Literal["stage1", "stage2", "stage3", "stage4"]

STAGE_PREPROCESS: StageTag = "stage1"
STAGE_GPU_DETECT: StageTag = "stage2"
STAGE_MEASURE: StageTag = "stage3"
#: The GPU embedding stage (#282): every ``MeasureDinoEmbedding`` of a run
#: whose ``--embed-stage`` resolved ``gpu``, after the store is final.
STAGE_EMBED: StageTag = "stage4"
#: The terminal stage of a run that embeds inline (every pre-#282 run).
STAGED_TERMINAL_STAGE: StageTag = STAGE_MEASURE

VALID_STAGE_TAGS = frozenset(
    {STAGE_PREPROCESS, STAGE_GPU_DETECT, STAGE_MEASURE, STAGE_EMBED}
)

def run_terminal_stage(state_config: Mapping[str, object] | None) -> StageTag:
    """The stage whose completion is an image's overall completion, for one run.

    A run that embeds on the GPU (recorded ``embed_stage == "gpu"``) defers
    its embeddings and its per-image record to Stage 4, so a Stage-3
    completion is progress, not completion (#282 plan review I8). Every other
    run, including one whose recorded configuration predates ``embed_stage``
    (which reads ``"inline"``), finishes at Stage 3.

    Args:
        state_config: ``processing_state.json``'s ``config``, or a run's
            ``job_metadata.json``, whose ``JobMetadataKey.EMBED_STAGE`` is
            the same key; ``None`` when there is none.

    Returns:
        :data:`STAGE_EMBED` or :data:`STAGED_TERMINAL_STAGE`.
    """
    if isinstance(state_config, Mapping) and state_config.get(EMBED_STAGE_CONFIG_KEY) == "gpu":
        return STAGE_EMBED
    return STAGED_TERMINAL_STAGE


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
