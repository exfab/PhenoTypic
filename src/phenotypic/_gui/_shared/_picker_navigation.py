"""Dropdown previous/next helpers shared by GUI image pickers."""

from __future__ import annotations

from typing import Any, Literal

PickerDirection = Literal["previous", "next"]


def enabled_picker_values(options: list[dict[str, Any]] | None) -> list[str]:
    """Return enabled dropdown option values in display order.

    Args:
        options: Dash dropdown options. Options with truthy ``disabled``
            are skipped.

    Returns:
        Enabled option values coerced to strings.
    """
    values: list[str] = []
    for option in options or []:
        if not isinstance(option, dict) or option.get("disabled"):
            continue
        value = option.get("value")
        if value is not None:
            values.append(str(value))
    return values


def step_picker_value(
    current: str | None,
    options: list[dict[str, Any]] | None,
    direction: PickerDirection,
) -> str | None:
    """Step a dropdown value through enabled options.

    Args:
        current: Current dropdown value, or ``None``.
        options: Dash dropdown options in display order.
        direction: ``"previous"`` or ``"next"``.

    Returns:
        The stepped dropdown value, clamped at the first or last enabled
        option. If ``current`` is missing from the enabled values,
        ``"next"`` selects the first enabled option and ``"previous"``
        selects the last enabled option.
    """
    values = enabled_picker_values(options)
    if not values:
        return None
    if current not in values:
        return values[-1] if direction == "previous" else values[0]
    index = values.index(current)
    if direction == "previous":
        return values[max(0, index - 1)]
    return values[min(len(values) - 1, index + 1)]


def offset_picker_value(
    current: str | None,
    options: list[dict[str, Any]] | None,
    delta: int,
) -> str | None:
    """Move a dropdown value by a signed number of enabled options.

    Args:
        current: Current dropdown value, or ``None``.
        options: Dash dropdown options in display order.
        delta: Signed number of positions to move. The result is clamped to
            the first or last enabled option.

    Returns:
        The offset dropdown value. If ``current`` is absent, a non-negative
        delta starts at the first option and a negative delta starts at the
        last option. Returns ``None`` when there are no enabled options.
    """
    values = enabled_picker_values(options)
    if not values:
        return None
    if current not in values:
        return values[-1] if delta < 0 else values[0]
    index = values.index(current)
    target = min(len(values) - 1, max(0, index + delta))
    return values[target]


def picker_position(
    current: str | None,
    options: list[dict[str, Any]] | None,
) -> tuple[int, int]:
    """Return the one-based current position and enabled-option total.

    Args:
        current: Current dropdown value, or ``None``.
        options: Dash dropdown options in display order.

    Returns:
        ``(position, total)``. Position is zero when the current value is not
        among the enabled options.
    """
    values = enabled_picker_values(options)
    if current not in values:
        return 0, len(values)
    return values.index(current) + 1, len(values)


def picker_button_disabled_states(
    current: str | None,
    options: list[dict[str, Any]] | None,
) -> tuple[bool, bool]:
    """Return ``(previous_disabled, next_disabled)`` for a picker.

    Args:
        current: Current dropdown value, or ``None``.
        options: Dash dropdown options in display order.

    Returns:
        Pair of booleans for the previous and next buttons.
    """
    values = enabled_picker_values(options)
    if not values:
        return True, True
    if current not in values:
        return False, False
    index = values.index(current)
    return index <= 0, index >= len(values) - 1
