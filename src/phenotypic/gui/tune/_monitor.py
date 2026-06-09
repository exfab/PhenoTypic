"""Pure Monitor logic for run switching, live-view state, and cancel prompts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

LiveViewKind = Literal["local-log", "slurm-fleet", "slurm-detached"]

_CANCELLABLE = {"running", "submitting"}


@dataclass(frozen=True)
class SwitcherItem:
    """One run-switcher row or pill."""

    run_id: str
    mode: str
    status: str
    active: bool
    killable: bool


def run_switcher_items(records: list[Any], *, active_id: str | None) -> list[SwitcherItem]:
    """Build run-switcher view models from registry records."""
    return [
        SwitcherItem(
            run_id=str(record.run_id),
            mode=str(record.mode),
            status=str(record.status),
            active=record.run_id == active_id,
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
    "SwitcherItem",
    "cancel_prompt",
    "live_view_kind",
    "run_switcher_items",
]
