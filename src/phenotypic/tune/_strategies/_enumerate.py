"""Conditional Cartesian-product enumeration of a SearchSpace (grid search).

Respects ``conditional_on`` — a child knob is included only when its parent
presence knob takes the matching value, so an absent op collapses to a single
combination (the legacy ``Presence`` semantics, master §9).
"""
from __future__ import annotations

import itertools
from typing import Any

from .._search_space import Categorical, Domain, Fixed, IntRange, Knob, SearchSpace


def grid_values(domain: Domain) -> list[Any]:
    """The discrete grid values for a domain. ``FloatRange`` is not enumerable."""
    if isinstance(domain, Categorical):
        return list(domain.choices)
    if isinstance(domain, IntRange):
        return list(range(domain.low, domain.high + 1, domain.step))
    if isinstance(domain, Fixed):
        return [domain.value]
    raise ValueError(
        "GridStrategy cannot enumerate a continuous FloatRange; use "
        "Categorical / IntRange, or a non-grid strategy."
    )


def _is_active(knob: Knob, chosen: dict[str, Any]) -> bool:
    if knob.conditional_on is None:
        return True
    return all(chosen.get(pkey) == pval for pkey, pval in knob.conditional_on)


def enumerate_grid(space: SearchSpace) -> list[dict[str, Any]]:
    """All param dicts in the conditional Cartesian product.

    Unconditional knobs (including presence ``__enabled__`` knobs) form the
    outer product; each conditional knob is only assigned when its parent
    value is present in the combination.
    """
    roots = [k for k in space.knobs if k.conditional_on is None]
    conditionals = [k for k in space.knobs if k.conditional_on is not None]

    combos: list[dict[str, Any]] = []
    root_values = [grid_values(k.domain) for k in roots]
    for root_combo in itertools.product(*root_values):
        base = {k.key: v for k, v in zip(roots, root_combo)}
        active = [k for k in conditionals if _is_active(k, base)]
        if not active:
            combos.append(dict(base))
            continue
        cond_values = [grid_values(k.domain) for k in active]
        for cond_combo in itertools.product(*cond_values):
            full = dict(base)
            full.update({k.key: v for k, v in zip(active, cond_combo)})
            combos.append(full)
    return combos
