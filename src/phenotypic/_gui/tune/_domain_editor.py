"""Pure parse / summarize / feasibility helpers for Setup domain editing."""
from __future__ import annotations

from typing import Any

from phenotypic.tune._search_space import (
    Categorical,
    Domain,
    Fixed,
    FloatRange,
    IntRange,
    SearchSpace,
)


def domain_from_editor(
    *,
    mode: str,
    low: float | int | None,
    high: float | int | None,
    step: float | int | None,
    log: bool,
    choices: list[Any] | None,
    is_int: bool,
) -> Domain:
    """Build a search-space domain from raw editor field values."""
    if mode == "choices":
        if not choices:
            raise ValueError("Choices mode requires at least one value")
        return Categorical(choices=tuple(choices))
    if mode != "range":
        raise ValueError(f"unknown domain editor mode {mode!r}")
    if low is None or high is None:
        raise ValueError("Range mode requires low and high")
    if is_int:
        step_value = 1
        if step is not None:
            step_float = float(step)
            step_value = int(step_float)
            if step_float != step_value:
                raise ValueError("Integer ranges require an integer step")
            if step_value <= 0:
                raise ValueError("Integer ranges require a positive step")
        return IntRange(
            low=int(low),
            high=int(high),
            step=step_value,
            log=log,
        )
    return FloatRange(
        low=float(low),
        high=float(high),
        step=float(step) if step else None,
        log=log,
    )


def domain_summary(domain: Domain) -> str:
    """Return compact chip text for a domain."""
    if isinstance(domain, Categorical):
        return "{" + ", ".join(str(choice) for choice in domain.choices) + "}"
    if isinstance(domain, IntRange):
        text = f"{domain.low}-{domain.high} · step {domain.step}"
        return text + (" · by-magnitude" if domain.log else "")
    if isinstance(domain, FloatRange):
        suffix = f"step {domain.step}" if domain.step is not None else "float"
        text = f"{domain.low}-{domain.high} · {suffix}"
        return text + (" · by-magnitude" if domain.log else "")
    if isinstance(domain, Fixed):
        return f"fixed {domain.value}"
    return str(domain)


def grid_feasibility(space: SearchSpace) -> tuple[bool, str]:
    """Return whether every knob can be enumerated by grid search."""
    for knob in space.knobs:
        domain = knob.domain
        if isinstance(domain, FloatRange) and domain.step is None:
            return (
                False,
                f"Grid unavailable: {knob.key} is a continuous float. "
                "Add a step, pin it, or use Optuna.",
            )
    return True, "All active knobs are enumerable."
