"""RandomStrategy — seeded random sampling over the SearchSpace."""
from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

from .._search_space import (
    Categorical,
    Domain,
    Fixed,
    FloatRange,
    IntRange,
    SearchSpace,
)
from ._pruning import NoOpChannel, PruningChannel


class RandomStrategy:
    """Samples ``n_trials`` random configurations under a fixed seed.

    Respects ``conditional_on``: a child knob is sampled only when its parent
    presence value was sampled to match. Relies on a parent presence knob
    appearing **before** its conditional children in ``space.knobs`` (true for
    inferred and hand-authored spaces — the ``__enabled__`` knob is emitted
    first); a topological pass is unnecessary at the depth-cap of 1.

    Args:
        space: The search space to sample over.
        n_trials: The number of configurations to draw before exhaustion.
        seed: The RNG seed; fixing it makes the sampled sequence deterministic.
    """

    def __init__(self, space: SearchSpace, *, n_trials: int, seed: int = 0) -> None:
        self._space = space
        self._n = n_trials
        self._rng = random.Random(seed)
        self._count = 0

    def _sample_domain(self, domain: Domain) -> Any:
        if isinstance(domain, Categorical):
            return self._rng.choice(list(domain.choices))
        if isinstance(domain, IntRange):
            return self._rng.choice(domain.values())
        if isinstance(domain, FloatRange):
            if domain.step is not None:
                return self._rng.choice(domain.values())
            if domain.log:
                lo, hi = math.log(domain.low), math.log(domain.high)
                return math.exp(self._rng.uniform(lo, hi))
            return self._rng.uniform(domain.low, domain.high)
        if isinstance(domain, Fixed):
            return domain.value
        raise TypeError(f"unsupported domain {type(domain).__name__}")

    def suggest(self) -> tuple[Mapping[str, Any], PruningChannel]:
        chosen: dict[str, Any] = {}
        # Sample knobs in order; a conditional knob is sampled only if active.
        for knob in self._space.knobs:
            if not knob.is_active(chosen):
                continue
            chosen[knob.key] = self._sample_domain(knob.domain)
        self._count += 1
        return chosen, NoOpChannel()

    def register_result(
        self, params: Mapping[str, Any], result: Any, *, pruned: bool = False
    ) -> None:
        return None  # random does not learn

    def is_exhausted(self) -> bool:
        return self._count >= self._n
