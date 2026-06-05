"""Pareto-front + knee-point math over multi-objective ``Trial`` records.

The shared, store-agnostic core behind every backend's ``pareto_front`` /
``knee_point`` (plan §0a+§0b). It reads each trial's ``objectives`` sidecar (a
``{objective_name: value}`` dict, higher-is-better — robust-eval §5) and:

* :func:`pareto_front_of` selects the **non-dominated** trials — those no other
  trial beats on *every* objective. Failed and objective-less trials are
  skipped; the objective-name order is taken from the first multi-objective
  trial so the per-axis comparison is stable across the front.
* :func:`knee_point_of` returns the front trial at **maximum perpendicular
  distance to the chord** between the two extreme objective points (the
  max-curvature elbow). It is the canonical "best compromise" pick for a
  multi-objective study (plan §0b; supervised-scorers' Pareto-front knee).

Pure functions (no I/O, no store coupling) so a journal, an Optuna study, or any
future backend can delegate here and stay consistent.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle with _study_store
    from .._study_store import Trial


def _objective_keys(trials: list["Trial"]) -> list[str]:
    """The stable objective-name order, taken from the first scored trial.

    Every multi-objective trial in one study shares the same objective names
    (the scorer's, e.g. ``CompositeScorer``'s child handles), so the first
    scored trial fixes the axis order for the whole front.

    Args:
        trials: The candidate trials (some may carry no ``objectives``).

    Returns:
        The ordered objective names, or ``[]`` when no trial is scored.
    """
    for trial in trials:
        if trial.objectives:
            return list(trial.objectives.keys())
    return []


def _vector(trial: "Trial", keys: list[str]) -> list[float]:
    """The trial's objective vector in ``keys`` order (``0.0`` for any missing)."""
    objectives = trial.objectives or {}
    return [float(objectives.get(key, 0.0)) for key in keys]


def _dominates(lhs: list[float], rhs: list[float]) -> bool:
    """Whether ``lhs`` Pareto-dominates ``rhs`` (higher-is-better objectives).

    ``lhs`` dominates ``rhs`` iff it is **at least as good on every** objective
    and **strictly better on at least one**. Equal vectors do not dominate each
    other, so identical points both survive a pairwise test (the front keeps one
    representative via the "no *other* point dominates me" selection).

    Args:
        lhs: The candidate dominating vector.
        rhs: The candidate dominated vector.

    Returns:
        ``True`` when ``lhs`` dominates ``rhs``.
    """
    no_worse = all(left >= right for left, right in zip(lhs, rhs))
    strictly_better = any(left > right for left, right in zip(lhs, rhs))
    return no_worse and strictly_better


def pareto_front_of(trials: list["Trial"]) -> list["Trial"]:
    """The non-dominated trials by their ``objectives`` sidecar (plan §0a).

    Failed trials and trials without an ``objectives`` dict are ignored. A trial
    is on the front when **no other scored trial dominates it**. Duplicate
    objective vectors are deduplicated by their vector so an exact tie keeps a
    single representative (otherwise every member of a tie cluster would sit on
    the front).

    Args:
        trials: All recorded trials (the store's journal).

    Returns:
        The non-dominated trials in journaling order; ``[]`` when no trial
        carries objectives (the single-objective back-compat path).
    """
    keys = _objective_keys(trials)
    if not keys:
        return []
    scored = [t for t in trials if t.objectives and not t.failed]
    vectors = {id(t): _vector(t, keys) for t in scored}

    front: list[Trial] = []
    seen_vectors: set[tuple[float, ...]] = set()
    for candidate in scored:
        cand_vec = vectors[id(candidate)]
        if any(
            _dominates(vectors[id(other)], cand_vec)
            for other in scored
            if other is not candidate
        ):
            continue
        key = tuple(cand_vec)
        if key in seen_vectors:  # an earlier tie already represents this point
            continue
        seen_vectors.add(key)
        front.append(candidate)
    return front


def knee_point_of(front: list["Trial"]) -> "Trial | None":
    """The front trial at max perpendicular distance to the extremes' chord.

    Builds the chord between the two **extreme** front points (the lexicographic
    min and max objective vectors), then returns the front trial whose objective
    vector is farthest (perpendicular distance) from that chord — the
    max-curvature elbow, the canonical multi-objective compromise pick (plan
    §0b). A degenerate front (zero/one point, or all points coincident so the
    chord has zero length) returns the first member (or ``None`` when empty).

    Args:
        front: The Pareto front (e.g. from :func:`pareto_front_of`).

    Returns:
        The knee trial, or ``None`` for an empty front.
    """
    if not front:
        return None
    if len(front) == 1:
        return front[0]

    keys = _objective_keys(front)
    if not keys:
        return front[0]
    vectors = [_vector(t, keys) for t in front]

    # The chord endpoints: the lexicographically smallest and largest vectors.
    lo = min(vectors)
    hi = max(vectors)
    chord = [h - low for h, low in zip(hi, lo)]
    chord_len = math.sqrt(sum(component * component for component in chord))
    if chord_len == 0.0:  # all points coincide — no chord to project onto
        return front[0]

    best_trial = front[0]
    best_distance = -1.0
    for trial, vector in zip(front, vectors):
        distance = _perpendicular_distance(vector, lo, chord, chord_len)
        if distance > best_distance:
            best_distance = distance
            best_trial = trial
    return best_trial


def _perpendicular_distance(
    point: list[float], anchor: list[float], chord: list[float], chord_len: float
) -> float:
    """The perpendicular distance from ``point`` to the line ``anchor + t·chord``.

    Generalizes the 2-D point-line distance to ``n`` objectives:
    ``|v|² − (v·û)²`` where ``v = point − anchor`` and ``û`` is the unit chord.
    Clamped at ``0`` before the square root to absorb floating-point negatives.

    Args:
        point: The objective vector being projected.
        anchor: One chord endpoint (the line's base point).
        chord: The chord direction vector (``hi − lo``).
        chord_len: The chord's Euclidean length (precomputed, non-zero).

    Returns:
        The perpendicular distance from ``point`` to the chord line.
    """
    offset = [p - a for p, a in zip(point, anchor)]
    offset_sq = sum(component * component for component in offset)
    projection = sum(o * c for o, c in zip(offset, chord)) / chord_len
    residual = offset_sq - projection * projection
    return math.sqrt(max(residual, 0.0))
