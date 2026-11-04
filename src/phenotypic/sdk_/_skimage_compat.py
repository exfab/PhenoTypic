"""Bridges over scikit-image API changes.

``scikit-image`` is declared without a lower bound and the lock pins an older
release than a fresh PyPI install resolves, so the same call has to behave
identically on both. scikit-image is imported inside each function to keep
this module free to import from light entry points.
"""

from __future__ import annotations

import functools
import inspect
import math

import numpy as np


@functools.cache
def _remove_small_objects_takes_max_size() -> bool:
    """Whether ``remove_small_objects`` has the 0.26 ``max_size`` parameter."""
    from skimage.morphology import remove_small_objects

    return "max_size" in inspect.signature(remove_small_objects).parameters


def _max_size_for(min_size: float) -> int:
    """The ``max_size`` that removes exactly the components ``< min_size``.

    Component sizes are integers, so ``size < min_size`` is
    ``size <= ceil(min_size) - 1``. For an integer that is ``min_size - 1``;
    for ``2.5`` it is ``2``, where ``min_size - 1`` would give ``1.5`` and
    ``floor`` would give ``1``. Floored at 0, where both parameters remove
    nothing.
    """
    return max(math.ceil(min_size) - 1, 0)


def remove_objects_smaller_than(
    ar: np.ndarray, min_size: float, *, connectivity: int = 1
) -> np.ndarray:
    """Zero every connected component with fewer than ``min_size`` pixels.

    The ``skimage.morphology.remove_small_objects(ar, min_size=...)`` rule,
    without the ``FutureWarning`` scikit-image 0.26 raises for ``min_size``.
    0.26 replaced it with ``max_size``, which removes objects of ``max_size``
    pixels **or fewer**, so 0.26+ is called with
    ``max_size=ceil(min_size) - 1`` (see :func:`_max_size_for`). Older
    releases get ``min_size`` unchanged. Call this rather than
    ``remove_small_objects`` directly anywhere in ``src/``;
    ``tests/unit/test_remove_small_objects_guard.py`` enforces it.

    Args:
        ar: Boolean mask (components are labelled first, using
            ``connectivity``) or non-negative integer label image.
        min_size: Smallest component size, in pixels, that is kept. Any
            finite real number; a fractional value keeps components of
            ``ceil(min_size)`` pixels and up, as ``<`` would.
        connectivity: Neighbourhood used to label a boolean ``ar``.

    Returns:
        A copy of ``ar`` with the small components set to 0.
    """
    from skimage.morphology import remove_small_objects

    if _remove_small_objects_takes_max_size():
        # The locked 0.25.2 signature has no max_size; 0.26+ does.
        return remove_small_objects(  # type: ignore[call-arg]
            ar, max_size=_max_size_for(min_size), connectivity=connectivity
        )
    return remove_small_objects(
        ar, min_size=min_size, connectivity=connectivity
    )
