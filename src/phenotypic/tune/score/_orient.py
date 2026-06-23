"""Cost orientation — map a scorer's natural per-term value to bounded cost.

A *cost* is in ``[0, 1]`` where ``0`` is perfect and ``1`` is worst (lower is
better; the optimizer minimizes). Scorers declare a :class:`Sense` and emit
their natural per-term values; :func:`to_cost` orients them. This is the one
place orientation happens, replacing the per-scorer hand-rolled flips.
"""
from __future__ import annotations

import math
from enum import Enum


class Sense(str, Enum):
    """Direction of a scorer's natural per-term values.

    ``LOWER_BETTER`` — a larger value is *worse* (a loss/divergence); maps to a
    QC check's ``_HIGHER_IS_BAD=True``. ``HIGHER_BETTER`` — a larger value is
    *better* (Dice, IoU, ICC, solidity).
    """

    LOWER_BETTER = "lower_better"
    HIGHER_BETTER = "higher_better"


def clamp01(value: float) -> float:
    """Clamp ``value`` into ``[0, 1]``.

    Used on the robust-aggregated cost: ``median + λ·IQR`` can reach ``~1+λ``
    (B1), so the term/child cost must be clamped to keep the ``0 ≤ cost ≤ 1``
    invariant the composite relies on.

    Args:
        value: Any float.

    Returns:
        ``0.0`` if ``value < 0``, ``1.0`` if ``value > 1``, else ``float(value)``.
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def to_cost(value: float, *, sense: Sense, anchor: float | None = None) -> float:
    """Map a scorer's natural per-term value to cost in ``[0, 1]``.

    Args:
        value: The scorer's natural per-term value.
        sense: Whether larger values are better or worse.
        anchor: ``None`` when ``value`` is already bounded in ``[0, 1]``;
            a positive float (the half-cost scale, e.g. a check's
            ``fail_threshold``) when ``value`` is an unbounded magnitude.

    Returns:
        The cost in ``[0, 1]`` (``0`` perfect, ``1`` worst).

    Examples:
        >>> to_cost(0.3, sense=Sense.LOWER_BETTER)
        0.3
        >>> to_cost(0.3, sense=Sense.HIGHER_BETTER)
        0.7
        >>> round(to_cost(0.1, sense=Sense.LOWER_BETTER, anchor=0.1), 3)
        0.5
        >>> to_cost(float("inf"), sense=Sense.LOWER_BETTER, anchor=0.1)
        1.0
    """
    if anchor is None:
        return value if sense is Sense.LOWER_BETTER else 1.0 - value
    if not math.isfinite(value):
        return 1.0
    goodness = math.exp(-math.log(2.0) * value / anchor)
    return (1.0 - goodness) if sense is Sense.LOWER_BETTER else goodness
