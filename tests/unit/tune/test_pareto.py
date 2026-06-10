"""4.6 — ``pareto_front()`` + ``knee_point()`` on the StudyStore (plan §0a+§0b).

The multi-objective sidecar (``Trial.objectives``) drives a true Pareto front:
``pareto_front()`` returns the non-dominated trials (by their ``objectives`` dict,
ignoring failed/objective-less trials), and ``knee_point(front)`` returns the
front trial at **maximum perpendicular distance to the chord** between the two
extreme objective points (the max-curvature elbow, plan §0b). A single-objective
store (no trial carries ``objectives``) returns an empty front while scalar
``best()`` still works (the back-compat lock). The methods live on the
``StudyStore`` Protocol and both concrete backends (``JournalStudyStore`` +
``OptunaStudyStore``).
"""
from __future__ import annotations

from phenotypic.tune._study._protocol import StudyStore as StudyStoreProtocol
from phenotypic.tune._study_store import JournalStudyStore, Trial


def _trial(n: int, *, objectives=None, score=None, failed=False) -> Trial:
    """A trial whose scalar ``score`` defaults to the mean of its objectives."""
    if score is None:
        score = (
            sum(objectives.values()) / len(objectives) if objectives else 0.0
        )
    return Trial(
        number=n,
        params={"a": n},
        score=score,
        terms={},
        n_images=2,
        objectives=objectives,
        failed=failed,
    )


# ---------------------------------------------------------------------------
# pareto_front()
# ---------------------------------------------------------------------------


def test_pareto_front_excludes_dominated():
    """A hand-built cost-objective set → the known non-dominated subset.

    Cost coordinates (lower is better). Points (cost_seg, cost_qc): A=(0.1,0.8),
    B=(0.5,0.5), C=(0.8,0.1) are mutually non-dominated (each wins an axis by
    being lower); D=(0.6,0.6) is dominated by B (0.5,0.5 ≤ on both, strictly on
    both). The front is {A, B, C}.
    """
    store = JournalStudyStore()
    store.append(_trial(0, objectives={"Dice": 0.1, "IoU": 0.8}))  # A
    store.append(_trial(1, objectives={"Dice": 0.5, "IoU": 0.5}))  # B
    store.append(_trial(2, objectives={"Dice": 0.8, "IoU": 0.1}))  # C
    store.append(_trial(3, objectives={"Dice": 0.6, "IoU": 0.6}))  # D (dominated)

    front_numbers = {t.number for t in store.pareto_front()}
    assert front_numbers == {0, 1, 2}


def test_pareto_front_excludes_dominated_under_cost():
    """Cost coordinates (lower is better): the dominated point is the HIGH-cost one.

    Points (cost_seg, cost_qc): A=(0.1,0.8), B=(0.5,0.5), C=(0.8,0.1) are mutually
    non-dominated (each wins an axis by being lower). D=(0.6,0.6) is dominated by
    B (0.5,0.5 ≤ on both, strictly on both). The front is {A, B, C}.
    """
    store = JournalStudyStore()
    store.append(_trial(0, objectives={"seg": 0.1, "qc": 0.8}, score=0.45))  # A
    store.append(_trial(1, objectives={"seg": 0.5, "qc": 0.5}, score=0.50))  # B
    store.append(_trial(2, objectives={"seg": 0.8, "qc": 0.1}, score=0.45))  # C
    store.append(_trial(3, objectives={"seg": 0.6, "qc": 0.6}, score=0.60))  # D dominated
    front_numbers = {t.number for t in store.pareto_front()}
    assert front_numbers == {0, 1, 2}


def test_lower_cost_vector_dominates_higher_cost_vector():
    """A strictly-lower-cost trial dominates a strictly-higher-cost one (cost B1)."""
    from phenotypic.tune._study._pareto import _dominates

    assert _dominates([0.2, 0.3], [0.5, 0.6]) is True   # lower on both → dominates
    assert _dominates([0.5, 0.6], [0.2, 0.3]) is False  # higher on both → dominated
    assert _dominates([0.2, 0.6], [0.2, 0.3]) is False  # ties one, worse on other
    assert _dominates([0.2, 0.3], [0.2, 0.3]) is False  # equal vectors never dominate


def test_vector_missing_axis_fills_worst_cost():
    """A trial missing an axis is filled with 1.0 (worst cost), not 0.0 (best)."""
    from phenotypic.tune._study._pareto import _vector

    partial = _trial(0, objectives={"seg": 0.2}, score=0.2)
    assert _vector(partial, ["seg", "qc"]) == [0.2, 1.0]


def test_pareto_front_ignores_failed_trials():
    """A failed trial is never on the front even if its costs look strong (low)."""
    store = JournalStudyStore()
    store.append(_trial(0, objectives={"Dice": 0.1, "IoU": 0.1}, failed=True))
    store.append(_trial(1, objectives={"Dice": 0.5, "IoU": 0.5}))
    front = store.pareto_front()
    assert {t.number for t in front} == {1}


def test_pareto_front_ignores_objectiveless_trials():
    """Trials without an ``objectives`` sidecar are excluded from the front."""
    store = JournalStudyStore()
    store.append(_trial(0, objectives={"Dice": 0.8, "IoU": 0.3}))
    store.append(_trial(1, score=0.9))  # scalar-only, no objectives
    front = store.pareto_front()
    assert {t.number for t in front} == {0}


def test_single_objective_store_has_empty_front_and_best_still_works():
    """No trial carries ``objectives`` → empty front; scalar ``best()`` intact."""
    store = JournalStudyStore()
    store.append(_trial(0, score=0.3))
    store.append(_trial(1, score=0.9))
    store.append(_trial(2, score=0.5))
    assert store.pareto_front() == []
    best = store.best()
    # Cost convention (minimize): the lowest-cost trial wins.
    assert best is not None and best.number == 0 and best.score == 0.3


def test_pareto_front_duplicate_objectives_keeps_one_representative():
    """Two trials with identical objectives → only one survives (mutual non-dom).

    Identical points do not dominate each other (no strict improvement), so a
    naive strict-or-equal test must not drop both; the front retains exactly one.
    """
    store = JournalStudyStore()
    store.append(_trial(0, objectives={"Dice": 0.5, "IoU": 0.5}))
    store.append(_trial(1, objectives={"Dice": 0.5, "IoU": 0.5}))
    front = store.pareto_front()
    assert len(front) == 1


# ---------------------------------------------------------------------------
# knee_point()
# ---------------------------------------------------------------------------


def test_knee_point_is_max_distance_to_chord():
    """The knee is the front point at max perpendicular distance to the chord.

    Cost coordinates (lower is better). With extremes A=(0.0,1.0) and
    C=(1.0,0.0), the chord is the line ``x+y=1``. B=(0.1,0.1) is the convex elbow
    toward the origin (the low-cost corner, distance ``0.8/√2`` below the chord);
    the near-chord point E=(0.5,0.5) sits *on* the chord (distance 0). The knee is
    B. The chord/projection geometry is direction-agnostic, so the elbow is the
    same front point whether axes are goodness or cost.
    """
    store = JournalStudyStore()
    store.append(_trial(0, objectives={"Dice": 0.0, "IoU": 1.0}))  # A extreme
    store.append(_trial(1, objectives={"Dice": 0.1, "IoU": 0.1}))  # B elbow
    store.append(_trial(2, objectives={"Dice": 1.0, "IoU": 0.0}))  # C extreme
    store.append(_trial(3, objectives={"Dice": 0.5, "IoU": 0.5}))  # E on chord
    front = store.pareto_front()
    knee = store.knee_point(front)
    assert knee is not None and knee.number == 1


def test_knee_point_none_for_empty_front():
    store = JournalStudyStore()
    assert store.knee_point([]) is None


def test_knee_point_single_member_front_returns_that_member():
    """A degenerate one-point front has no chord; the knee is that point."""
    store = JournalStudyStore()
    t = _trial(0, objectives={"Dice": 0.7, "IoU": 0.4})
    store.append(t)
    front = store.pareto_front()
    knee = store.knee_point(front)
    assert knee is not None and knee.number == 0


def test_knee_point_two_member_front_picks_a_front_member():
    """A two-point front is all-chord (distance 0); the knee is a front member."""
    store = JournalStudyStore()
    store.append(_trial(0, objectives={"Dice": 0.8, "IoU": 0.2}))
    store.append(_trial(1, objectives={"Dice": 0.2, "IoU": 0.8}))
    front = store.pareto_front()
    knee = store.knee_point(front)
    assert knee is not None and knee.number in {0, 1}


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_journal_satisfies_pareto_protocol_methods():
    """The concrete journal exposes the Protocol's pareto methods structurally."""
    store: StudyStoreProtocol = JournalStudyStore()
    store.append(_trial(0, objectives={"Dice": 0.6, "IoU": 0.6}))
    front = store.pareto_front()
    assert [t.number for t in front] == [0]
    assert store.knee_point(front) is not None
