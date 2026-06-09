"""Pure model for the tune hamburger nav: Setup / Run / Monitor."""
from __future__ import annotations

from typing import Literal

Destination = Literal["setup", "run", "monitor"]

DESTINATIONS: tuple[Destination, ...] = ("setup", "run", "monitor")

DESTINATION_LABELS: dict[Destination, str] = {
    "setup": "Setup",
    "run": "Run",
    "monitor": "Monitor",
}


def destination_button_id(name: Destination) -> str:
    """Return the static ID for a destination button."""
    return f"tune-dest-{name}"


def destination_view_id(name: Destination) -> str:
    """Return the static ID for a destination view container."""
    return f"tune-destview-{name}"


def active_destination(
    trigger_id: str | None,
    *,
    pipeline_path: str | None = None,
) -> Destination:
    """Map a clicked destination button to its destination name."""
    if trigger_id:
        for name in DESTINATIONS:
            if trigger_id == destination_button_id(name):
                if destination_button_disabled(name, pipeline_path=pipeline_path):
                    return "setup"
                return name
    return "setup"


def destination_button_disabled(
    name: Destination,
    *,
    pipeline_path: str | None,
) -> bool:
    """Return whether a destination button should be inert."""
    return name == "run" and not pipeline_path


def destination_button_class(name: Destination, active: Destination | None) -> str:
    """Return CSS classes for a destination button."""
    classes = ["tune-dest"]
    if name == active:
        classes.append("tune-dest-active")
    return " ".join(classes)


def destination_view_class(name: Destination, active: Destination | None) -> str:
    """Return CSS classes for a destination view container."""
    classes = ["tune-view"]
    if name != active:
        classes.append("tune-view-hidden")
    return " ".join(classes)
