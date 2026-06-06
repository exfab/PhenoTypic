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
    """A journal whose scores are non-monotone and whose trial 2 is gap-flagged."""
    return JournalStudyStore([
        _trial(0, 0.30, gap=0.05),
        _trial(1, 0.50, gap=0.10),
        _trial(2, 0.40, gap=0.25),  # higher score than 0/1's running max? no — flagged
        _trial(3, 0.70, gap=0.08),
        _trial(4, 0.60, gap=0.12),
    ])


def test_running_best_is_monotone_non_decreasing():
    """The running best is the cumulative max of the trial scores, in order."""
    store = _store()
    curve = running_best(store.trials)
    assert curve == [0.30, 0.50, 0.50, 0.70, 0.70]
    assert all(b >= a for a, b in zip(curve, curve[1:]))


def test_gap_badge_flags_high_dispersion_winner():
    """A winner whose gap exceeds 0.15 is flagged; a stable winner is not."""
    flagged_label, flagged = gap_badge(JournalStudyStore([
        _trial(0, 0.9, gap=0.25),
    ]))
    assert flagged is True
    assert isinstance(flagged_label, str) and flagged_label

    _, stable = gap_badge(JournalStudyStore([
        _trial(0, 0.9, gap=0.05),
    ]))
    assert stable is False


def test_shortlist_includes_top_scorers_and_gap_flagged_deduped():
    """Top-k ∪ gap-flagged, de-duped by number, score-desc, length bounded."""
    store = _store()
    picks = shortlist(store, k=2)
    numbers = [t.number for t in picks]

    # Top-2 by score are trials 3 (0.70) and 4 (0.60).
    assert 3 in numbers
    assert 4 in numbers
    # Trial 2 is gap-flagged (0.25 > 0.15) → included even though not top-2.
    assert 2 in numbers
    # De-duped by trial number.
    assert len(numbers) == len(set(numbers))
    # Score-descending order.
    scores = [t.score for t in picks]
    assert scores == sorted(scores, reverse=True)
    # Bounded: at most k plus the extras (pareto + gap-flagged).
    assert len(picks) <= 2 + len(store.trials)


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
    mo = TuneRunRoot(directions=["maximize", "maximize"], **base)
    so = TuneRunRoot(directions=None, **base)
    assert is_multi_objective(mo) is True
    assert is_multi_objective(so) is False
