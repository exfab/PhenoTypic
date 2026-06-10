"""Pure study-read helpers — running-best curve, gap badge, shortlist, MO flag.

All functions are pure over a ``StudyStore`` (a :class:`JournalStudyStore` built
from a handful of :class:`Trial` records) or a :class:`TuneRunRoot`; none import
``optuna`` or touch disk.
"""
from __future__ import annotations

from pathlib import Path

from phenotypic.tune._study_store import JournalStudyStore, Trial

from phenotypic.gui.tune import TuneRunRoot
from phenotypic.gui.tune._study_read import (
    gap_badge,
    is_multi_objective,
    running_best,
    shortlist,
)


def _trial(number: int, score: float, *, gap: float | None = None) -> Trial:
    """A minimal single-objective ``Trial`` with a given score and gap."""
    return Trial(
        number=number,
        params={"x": number},
        score=score,
        terms={"primary": score},
        n_images=4,
        gap=gap,
    )


def _store() -> JournalStudyStore:
    """A journal of costs (lower is better); trial 2 is the gap-flagged one.

    Costs descend toward the best (lowest); the running best is a cumulative
    MIN curve. Trial 1 (cost 0.40) is the lowest-cost trial, and trial 2's gap
    (0.25 > 0.15) flags it as unstable.
    """
    return JournalStudyStore([
        _trial(0, 0.70, gap=0.05),
        _trial(1, 0.40, gap=0.10),  # lowest cost = best
        _trial(2, 0.60, gap=0.25),  # gap-flagged (unstable)
        _trial(3, 0.50, gap=0.08),
        _trial(4, 0.45, gap=0.12),
    ])


def test_running_best_is_monotone_non_increasing():
    """The running best is the cumulative min of the trial costs, in order."""
    store = _store()
    curve = running_best(store.trials)
    assert curve == [0.70, 0.40, 0.40, 0.40, 0.40]
    assert all(b <= a for a, b in zip(curve, curve[1:]))


def test_running_best_is_monotone_non_increasing_under_cost():
    """The running best is the cumulative MIN of the trial costs, in order."""
    trials = JournalStudyStore([
        _trial(0, 0.70),
        _trial(1, 0.50),
        _trial(2, 0.60),
        _trial(3, 0.30),
        _trial(4, 0.40),
    ]).trials
    curve = running_best(trials)
    assert curve == [0.70, 0.50, 0.50, 0.30, 0.30]
    assert all(b <= a for a, b in zip(curve, curve[1:]))  # non-increasing


def test_gap_badge_flags_high_dispersion_winner():
    """A winner whose gap exceeds 0.15 is flagged; a stable winner is not.

    Gap semantics did not flip with the cost cutover (it is a non-negative
    relative dispersion); a gap of 0.25 > 0.15 still flags.
    """
    flagged_label, flagged = gap_badge(JournalStudyStore([
        _trial(0, 0.1, gap=0.25),
    ]))
    assert flagged is True
    assert isinstance(flagged_label, str) and flagged_label

    _, stable = gap_badge(JournalStudyStore([
        _trial(0, 0.1, gap=0.05),
    ]))
    assert stable is False


def test_shortlist_includes_top_scorers_and_gap_flagged_deduped():
    """Top-k ∪ gap-flagged, de-duped by number, cost-asc, length bounded."""
    store = _store()
    picks = shortlist(store, k=2)
    numbers = [t.number for t in picks]

    # Top-2 by cost (lowest) are trials 1 (0.40) and 4 (0.45).
    assert 1 in numbers
    assert 4 in numbers
    # Trial 2 is gap-flagged (0.25 > 0.15) → included even though not top-2.
    assert 2 in numbers
    # De-duped by trial number.
    assert len(numbers) == len(set(numbers))
    # Cost-ascending order (lowest cost first).
    scores = [t.score for t in picks]
    assert scores == sorted(scores)
    # Bounded: at most k plus the extras (pareto + gap-flagged).
    assert len(picks) <= 2 + len(store.trials)


def test_shortlist_top_k_is_lowest_cost():
    """Top-k by cost = the LOWEST-cost trials (best), not the highest."""
    store = JournalStudyStore([
        _trial(0, 0.30), _trial(1, 0.50), _trial(2, 0.40),
        _trial(3, 0.70), _trial(4, 0.60),
    ])
    picks = shortlist(store, k=2)
    numbers = [t.number for t in picks]
    assert 0 in numbers and 2 in numbers      # lowest-cost two are 0 (0.30), 2 (0.40)
    # Score-ascending order (lowest cost first).
    scores = [t.score for t in picks]
    assert scores == sorted(scores)


def test_is_multi_objective_reads_directions():
    """A root with ≥2 directions is multi-objective; a None-directions one isn't."""
    base = dict(
        path=Path("/x"),
        trials_path=None,
        storage_url=None,
        study_name="tune",
        images_dir=None,
        best_pipeline_path=Path("/x/best.json"),
    )
    mo = TuneRunRoot(directions=["minimize", "minimize"], **base)
    so = TuneRunRoot(directions=None, **base)
    assert is_multi_objective(mo) is True
    assert is_multi_objective(so) is False
