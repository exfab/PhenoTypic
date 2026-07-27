"""Pure Monitor logic for run switching, live-view state, and cancel prompts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict
from uuid import UUID

LiveViewKind = Literal["local-log", "slurm-fleet", "slurm-detached"]

_CANCELLABLE = {"running", "submitting"}


class RunReceipt(TypedDict):
    """JSON-safe identity for one exact Tune launch generation."""

    run_id: str
    generation: str


def run_receipt(record: Any) -> RunReceipt | None:
    """Return an exact JSON-safe receipt for a generation-bearing record."""
    generation = getattr(record, "generation", None)
    run_id = getattr(record, "run_id", None)
    if not isinstance(run_id, str) or not isinstance(generation, UUID):
        return None
    return {"run_id": run_id, "generation": str(generation)}


def parse_run_receipt(payload: object) -> tuple[str, UUID] | None:
    """Parse one exact Tune receipt without consulting the registry."""
    if not isinstance(payload, dict):
        return None
    run_id = payload.get("run_id")
    raw_generation = payload.get("generation")
    if not isinstance(run_id, str):
        return None
    try:
        generation = UUID(str(raw_generation))
    except (TypeError, ValueError):
        return None
    return run_id, generation


@dataclass(frozen=True)
class SwitcherItem:
    """One run-switcher row or pill."""

    run_id: str
    generation: str | None
    mode: str
    status: str
    active: bool
    killable: bool


def run_switcher_items(
    records: list[Any],
    *,
    active_receipt: object,
) -> list[SwitcherItem]:
    """Build switcher items matched against one exact active generation."""
    active_identity = parse_run_receipt(active_receipt)
    return [
        SwitcherItem(
            run_id=str(record.run_id),
            generation=(
                str(record.generation)
                if isinstance(getattr(record, "generation", None), UUID)
                else None
            ),
            mode=str(record.mode),
            status=str(record.status),
            active=(
                active_identity is not None
                and record.run_id == active_identity[0]
                and record.generation == active_identity[1]
            ),
            killable=record.mode == "local" and record.status in _CANCELLABLE,
        )
        for record in records
    ]


def live_view_kind(mode: str, *, store_reachable: bool) -> LiveViewKind:
    """Return the live Monitor view kind for a run mode."""
    if mode == "slurm":
        return "slurm-fleet" if store_reachable else "slurm-detached"
    return "local-log"


def cancel_prompt(name: str, mode: str) -> str:
    """Return the Local cancellation confirmation text."""
    if mode != "local":
        raise ValueError("SLURM cancellation is not supported in v1")
    return (
        f"Send SIGTERM to {name}? Trials already recorded in the journal are "
        "kept, and the study can be resumed."
    )


__all__ = [
    "LiveViewKind",
    "RunReceipt",
    "SwitcherItem",
    "cancel_prompt",
    "live_view_kind",
    "parse_run_receipt",
    "run_receipt",
    "run_switcher_items",
]
