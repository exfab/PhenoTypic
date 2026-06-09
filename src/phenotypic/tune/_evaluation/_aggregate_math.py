"""Shared numpy-only aggregate-math primitives for the evaluation layer.

The two robust-aggregation building blocks reused across the per-trial robust
evaluator (``_evaluator.py``) and the held-out generalization pass
(``_generalization.py``): the median/IQR quartile reduction and the
eps-floored relative ratio. Both the calibration-stability gap and the
generalization-drop gate floor their denominator with the **same** ``_GAP_EPS``,
so the constant lives here once rather than being re-declared per module.

optuna-free (numpy only) — the lazy-import lock keeps this boundary importable
without optuna.
"""
from __future__ import annotations

from typing import Final, Sequence

import numpy as np

#: Denominator floor for every relative ratio in the evaluation layer — guards a
#: near-zero median (per-trial dispersion) or calibration score (generalization
#: drop) so the ratio stays finite. One shared convention across both call sites.
_GAP_EPS: Final[float] = 1e-12


def _median_iqr(values: Sequence[float]) -> tuple[float, float]:
    """The median and inter-quartile range (``q75 - q25``) of ``values``.

    The quartile reduction shared by the robust term aggregate
    (``median - λ·IQR``) and the per-trial dispersion gap. Uses
    ``np.percentile(arr, [75, 25])`` with the default linear interpolation.

    Args:
        values: The per-image scores (higher = better).

    Returns:
        A ``(median, iqr)`` pair, both ``float``. For a single value the IQR is
        ``0.0``.
    """
    arr = np.asarray(values, dtype=float)
    median = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    return median, float(q75 - q25)


def _relative(numerator: float, denominator: float) -> float:
    """The eps-floored relative ratio ``numerator / max(|denominator|, eps)``.

    Args:
        numerator: The signed difference (an IQR, or a calibration→held-out drop).
        denominator: The reference magnitude (a median, or a calibration score);
            floored to :data:`_GAP_EPS` in absolute value to stay finite.

    Returns:
        ``numerator / max(abs(denominator), _GAP_EPS)``.
    """
    return numerator / max(abs(denominator), _GAP_EPS)
