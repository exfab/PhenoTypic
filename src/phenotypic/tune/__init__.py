"""Parameter-tuning engine — public API (in progress).

This package is built up phase-by-phase; each phase appends its public symbols
to ``__all__`` below. Phase 1a ships the hand-authorable **search space**: the
discriminated-union domains plus the ``Knob`` / ``SearchSpace`` containers.

Hand-author a search space and inspect it:

    >>> from phenotypic.tune import SearchSpace, Knob, FloatRange, Categorical
    >>> space = SearchSpace(knobs=(
    ...     Knob(key="0.sigma", domain=FloatRange(low=0.5, high=8.0)),
    ...     Knob(key="1.ignore_zeros", domain=Categorical(choices=(True, False))),
    ... ))
    >>> space.keys()
    ['0.sigma', '1.ignore_zeros']
    >>> space.domain("0.sigma").high
    8.0
"""
from __future__ import annotations

# --- Phase 1a: search space (domains + Knob/SearchSpace) ----------------------
from ._search_space import (
    Categorical,
    Domain,
    Fixed,
    FloatRange,
    IntRange,
    Knob,
    SearchSpace,
)

__all__ = [
    # Phase 1a: search space
    "Categorical",
    "IntRange",
    "FloatRange",
    "Fixed",
    "Domain",
    "Knob",
    "SearchSpace",
]
