"""Pareto-front + knee-point math over multi-objective ``Trial`` records.

The shared, store-agnostic core behind every backend's ``pareto_front`` /
``knee_point`` (plan §0a+§0b). It reads each trial's ``objectives`` sidecar (a
``{objective_name: value}`` dict, cost, lower-is-better — robust-eval §5 / cost
convention) and:

* :func:`pareto_front_of` selects the **finite non-dominated COMPLETE** trials.
  Production callers supply the scorer-declared objective axes, and a candidate
  must contain exactly that full finite vector. First-sidecar inference remains
  only for direct legacy callers without authoritative axes.
* :func:`knee_point_of` returns the front trial at **maximum perpendicular
  distance to the chord** between the two extreme objective points (the
  max-curvature elbow). It is the canonical "best compromise" pick for a
  multi-objective study (plan §0b; supervised-scorers' Pareto-front knee).

Pure functions (no I/O, no store coupling) so a journal, an Optuna study, or any
future backend can delegate here and stay consistent.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING
from .._multi_objective import validate_objective_axes


if TYPE_CHECKING:  # avoid a runtime import cycle with _study_store
    from .._study_store import Trial


def _objective_keys(
    trials: list["Trial"], objective_axes: Sequence[str] | None = None
) -> list[str]:
    """Return scorer-supplied or legacy-inferred objective names.

    Production publication supplies the scorer's axes. The first scored trial
    is consulted only for direct/legacy callers without a scorer contract.

    Args:
        trials: The candidate trials (some may carry no ``objectives``).
        objective_axes: The scorer-required axes in stable order, or ``None``
            for legacy sidecar inference.

    Returns:
        The ordered objective names, or ``[]`` when no trial is scored.
    """
    if objective_axes is not None:
        return list(validate_objective_axes(objective_axes))
    for trial in trials:
        if trial.objectives:
            return list(trial.objectives.keys())
    return []


def _vector(
    trial: "Trial", keys: list[str], *, require_all: bool = False
) -> list[float]:
    """Return one objective vector in ``keys`` order.

    Direct callers always supply an authoritative multi-objective axis contract.
    ``require_all`` controls only whether missing values raise or use the
    worst-cost fallback.
    """
    validated_keys = validate_objective_axes(keys)
    objectives = trial.objectives or {}
    if require_all:
        return [float(objectives[key]) for key in validated_keys]
    return [float(objectives.get(key, 1.0)) for key in validated_keys]


def _legacy_vector(trial: "Trial", keys: list[str]) -> list[float]:
    """Build a vector only for the supported no-axes inference contract."""
    validated_keys = validate_objective_axes(keys, multi_objective=False)
    objectives = trial.objectives or {}
    return [float(objectives.get(key, 1.0)) for key in validated_keys]


def _dominates(lhs: list[float], rhs: list[float]) -> bool:
    """Whether ``lhs`` Pareto-dominates ``rhs`` (cost objectives, lower-is-better).

    ``lhs`` dominates ``rhs`` iff it is **at least as good (no higher cost) on
    every** objective and **strictly better (lower cost) on at least one**. Equal
    vectors do not dominate each other, so identical points both survive a
    pairwise test (the front keeps one representative via the "no *other* point
    dominates me" selection).

    Args:
        lhs: The candidate dominating vector.
        rhs: The candidate dominated vector.

    Returns:
        ``True`` when ``lhs`` dominates ``rhs``.
    """
    no_worse = all(left <= right for left, right in zip(lhs, rhs))
    strictly_better = any(left < right for left, right in zip(lhs, rhs))
    return no_worse and strictly_better


def pareto_front_of(
    trials: list["Trial"],
    *,
    objective_axes: Sequence[str] | None = None,
) -> list["Trial"]:
    """Return finite non-dominated COMPLETE trials (plan §0a).

    Failed, pruned, non-finite, and objective-less trials are ignored. When
    ``objective_axes`` is supplied, each candidate must carry exactly that
    complete vector; missing coordinates are never synthesized. Direct legacy
    calls without axes preserve first-sidecar inference.

    Args:
        trials: All recorded trials (the store's journal).
        objective_axes: The authoritative scorer-required axes, or ``None``
            for legacy sidecar inference.

    Returns:
        The non-dominated trials in journaling order; ``[]`` when no eligible
        full objective vector exists.
    """
    authoritative = objective_axes is not None
    keys = _objective_keys(trials, objective_axes)
    if not keys:
        return []
    required = set(keys)
    scored = [
        trial
        for trial in trials
        if trial.objectives
        and not trial.failed
        and not trial.pruned
        and (not authoritative or set(trial.objectives) == required)
    ]
    vectors = {
        id(trial): (
            _vector(trial, keys, require_all=True)
            if authoritative
            else _legacy_vector(trial, keys)
        )
        for trial in scored
    }
    scored = [
        trial
        for trial in scored
        if all(math.isfinite(value) for value in vectors[id(trial)])
    ]

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


def knee_point_of(
    front: list["Trial"],
    *,
    objective_axes: Sequence[str] | None = None,
) -> "Trial | None":
    """The front trial at max perpendicular distance to the extremes' chord.

    Builds the chord between the two **extreme** front points (the lexicographic
    min and max objective vectors), then returns the front trial whose objective
    vector is farthest (perpendicular distance) from that chord — the
    max-curvature elbow, the canonical multi-objective compromise pick (plan
    §0b). A degenerate front (zero/one point, or all points coincident so the
    chord has zero length) returns the first member (or ``None`` when empty).

    **Two-objective exactness, n-objective heuristic.** For two objectives the
    lexicographic min/max *are* the two front-spanning extremes, so the chord is
    the true endpoint-to-endpoint line and the knee is exact — the case Phase-4
    multi-objective runs use. For ≥3 objectives the lexicographic min/max are not
    generally the spanning extremes, so the chord (and hence the "knee") is a
    reasonable but heuristic compromise pick, not a provably-optimal one. (A
    true n-D knee would project onto the hyperplane through all extreme points;
    deferred until a ≥3-objective scorer ships.)

    The chord/projection geometry is direction-agnostic — the elbow is the same
    front point whether axes are goodness or cost — so this is unchanged under
    the cost cutover.

    Args:
        front: The Pareto front (e.g. from :func:`pareto_front_of`).
        objective_axes: The authoritative scorer-required axis order, or
            ``None`` for legacy first-sidecar inference.

    Returns:
        The knee trial, or ``None`` for an empty front.
    """
    authoritative = objective_axes is not None
    keys = _objective_keys(front, objective_axes)
    if not front:
        return None
    if len(front) == 1:
        return front[0]
    if not keys:
        return front[0]
    vectors = [
        _vector(trial, keys, require_all=True)
        if authoritative
        else _legacy_vector(trial, keys)
        for trial in front
    ]

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
