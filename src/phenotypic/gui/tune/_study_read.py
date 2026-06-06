"""Pure study-read helpers — running-best curve, gap badge, shortlist, MO flag.

The read-only math the ``/tune/`` GUI co-pilot draws its Monitor view from. Every
function is **pure** over a ``StudyStore`` (the engine's trial-backend Protocol —
``trials`` / ``best`` / ``pareto_front``; in practice a
:class:`~phenotypic.tune._study_store.JournalStudyStore`) or a
:class:`~phenotypic.gui.tune.TuneRunRoot`. None touch disk, none re-optimize, and
none import ``optuna`` — the module stays in the package's optuna-free import
surface (the engine reads a study only through the Protocol).

* :func:`running_best` — the monotone cumulative-best curve over a trial list.
* :func:`gap_badge` — the winner's instability badge (label + ``is_flagged``),
  flagged when the best trial's relative across-plate dispersion exceeds
  :data:`GAP_FLAG_THRESHOLD`.
* :func:`shortlist` — the *k* candidates worth a human look: top-*k* by score,
  the Pareto front, and every gap-flagged trial, de-duped and score-sorted.
* :func:`is_multi_objective` — whether a run is a Pareto (≥2-axis) run, read off
  the :class:`TuneRunRoot`'s ``directions``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:  # keep the module import-light + optuna-free
    from phenotypic.tune._study_store import Trial

    from ._run_root import TuneRunRoot


#: The relative across-plate dispersion above which a winner is "unstable".
#: A trial's ``gap`` is its primary term's relative dispersion across calibration
#: plates (``_study_store.Trial.gap``); a best trial whose gap clears this is a
#: cheap overfit / instability flag worth surfacing before trusting the winner.
#: 0.15 ⇒ "the primary score swings by more than ~15% plate-to-plate".
GAP_FLAG_THRESHOLD: float = 0.15

#: The badge text for a stable winner (gap at or below the threshold, or absent).
_GAP_LABEL_STABLE: str = "stable"

#: The badge text for an unstable winner (gap above the threshold).
_GAP_LABEL_UNSTABLE: str = "unstable"


class _ReadableStore(Protocol):
    """The read-only slice of ``StudyStore`` these helpers need.

    A structural subset of :class:`~phenotypic.tune._study._protocol.StudyStore`
    (and its concrete ``JournalStudyStore``): the journaled ``trials``, the scalar
    ``best``, and the multi-objective ``pareto_front``. Declared locally so the
    helpers stay decoupled from the engine's full backend contract (and so the
    module never imports the store concretely at runtime).
    """

    @property
    def trials(self) -> list["Trial"]:
        """The journaled trials in order (a copy)."""
        ...

    def best(self) -> Optional["Trial"]:
        """The non-failed trial with the highest score, or ``None``."""
        ...

    def pareto_front(self) -> list["Trial"]:
        """The non-dominated trials by their ``objectives`` sidecar; ``[]`` if SO."""
        ...


def running_best(trials: list["Trial"]) -> list[float]:
    """The monotone non-decreasing cumulative-best score curve over ``trials``.

    The classic optimization-progress trace: position *i* is the best score seen
    in ``trials[: i + 1]`` (higher is better — robust-eval §5), so the curve never
    decreases. Reads each trial's scalar ``score`` in journaling order; failed
    trials are kept (they scored the failure floor, which simply never advances
    the running best).

    Args:
        trials: The journaled trials in order (e.g. ``store.trials``).

    Returns:
        One running-best value per trial, same length and order as ``trials``;
        ``[]`` for an empty journal.
    """
    curve: list[float] = []
    best_so_far: float | None = None
    for trial in trials:
        best_so_far = (
            trial.score if best_so_far is None else max(best_so_far, trial.score)
        )
        curve.append(best_so_far)
    return curve


def gap_badge(store: _ReadableStore) -> tuple[str, bool]:
    """The winner's stability badge — ``(label, is_flagged)``.

    Reads the best (highest-score, non-failed) trial's ``gap`` — its primary
    term's relative across-plate dispersion — and flags it when it exceeds
    :data:`GAP_FLAG_THRESHOLD`. A flagged winner is a cheap "this best may be
    unstable / overfit to a lucky plate split" signal the Monitor surfaces before
    the user trusts the tuned pipeline.

    Args:
        store: The study to read (``best()`` must be available).

    Returns:
        ``(label, is_flagged)``: a human-readable badge label and the boolean
        flag. An empty study, a winner with no ``gap`` signal, or a gap at/below
        the threshold all yield ``(_GAP_LABEL_STABLE, False)``.
    """
    best = store.best()
    gap = None if best is None else best.gap
    if gap is not None and gap > GAP_FLAG_THRESHOLD:
        return _GAP_LABEL_UNSTABLE, True
    return _GAP_LABEL_STABLE, False


def _is_gap_flagged(trial: "Trial") -> bool:
    """Whether ``trial``'s ``gap`` exceeds :data:`GAP_FLAG_THRESHOLD`."""
    return trial.gap is not None and trial.gap > GAP_FLAG_THRESHOLD


def shortlist(store: _ReadableStore, k: int = 5) -> list["Trial"]:
    """The candidates worth a human look — top-*k* ∪ Pareto ∪ gap-flagged.

    The union of three signals, de-duplicated by trial ``number`` and returned
    score-descending:

    * the top-*k* trials by scalar ``score`` (the obvious winners);
    * the Pareto front (``store.pareto_front()`` — non-empty only for a
      multi-objective study, where a high scalar score can still miss a
      front-defining trade-off);
    * every gap-flagged trial (``gap`` > :data:`GAP_FLAG_THRESHOLD`) — surfaced
      even when it is not top-*k*, so an unstable-but-high candidate is never
      silently dropped.

    Failed trials are excluded from the top-*k* ranking (they scored the failure
    floor); a failed trial reachable only via the front or the gap flag is still
    excluded here, matching ``best``/``pareto_front`` semantics.

    Args:
        store: The study to read.
        k: How many top scorers to seed the shortlist with (default 5). The
            result may exceed ``k`` by the extra Pareto / gap-flagged trials.

    Returns:
        The de-duplicated shortlist, score-descending; ``[]`` for an empty study.
    """
    valid = [t for t in store.trials if not t.failed]
    top_k = sorted(valid, key=lambda t: t.score, reverse=True)[:k]
    gap_flagged = [t for t in valid if _is_gap_flagged(t)]

    picked: dict[int, "Trial"] = {}
    for trial in (*top_k, *store.pareto_front(), *gap_flagged):
        picked.setdefault(trial.number, trial)
    return sorted(picked.values(), key=lambda t: t.score, reverse=True)


def is_multi_objective(root: "TuneRunRoot") -> bool:
    """Whether ``root`` describes a multi-objective (Pareto) tuning run.

    Read off the :class:`TuneRunRoot`'s ``directions``: a run is multi-objective
    when it has two or more Optuna objective axes. A single-objective run carries
    ``None`` (or, defensively, a single-axis list), so the test is strictly
    "more than one direction".

    Args:
        root: The validated tune output handle.

    Returns:
        ``True`` when ``root.directions`` has length > 1; ``False`` otherwise
        (including ``directions is None``).
    """
    directions = root.directions
    return directions is not None and len(directions) > 1
