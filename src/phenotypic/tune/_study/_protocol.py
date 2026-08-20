"""The ``StudyStore`` seam — what the engine needs from a trial backend.

Phase 1 ships one concrete backend (``JournalStudyStore``, a parquet-journalled
list). Phase 2 adds an Optuna ``RDBStorage``-backed store that resumes **in
place** (the database already holds the sampler state), so the engine must not
replay the deterministic strategy past recorded trials for it. Typing the
engine's ``store`` against this Protocol keeps both backends swappable
(optuna-integration.md §7).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:  # avoid a runtime import cycle with _study_store
    from .._study_store import Trial


@runtime_checkable
class StudyStore(Protocol):
    """The trial-backend contract the engine drives.

    A backend accumulates :class:`Trial` records, reports the ``best`` and a
    ``completed_count`` (non-failed trials), and declares whether it resumes **in
    place** — i.e. whether re-opening it already restores the sampler state so
    the engine should *not* fast-forward the strategy with ``suggest()`` replays.
    """

    def append(self, trial: "Trial") -> None:
        """Record one completed trial."""
        ...

    @property
    def trials(self) -> list["Trial"]:
        """Every recorded trial in order (a copy), in-flight ones included."""
        ...

    def terminal_trials(self) -> list["Trial"]:
        """The recorded trials that will never change again.

        A backend that can hold in-flight rows (the Optuna store: a worker's
        ``RUNNING`` trial, which a Slurm-killed worker leaves behind forever)
        must exclude them here. Everything that ranks a winner or counts real
        progress reads this, not :attr:`trials`; the journal, whose rows are
        appended only once a candidate has resolved, returns the same list.
        """
        ...

    def __len__(self) -> int:
        """The number of recorded trials."""
        ...

    def best(self) -> Optional["Trial"]:
        """The terminal, non-failed trial with the lowest cost, or ``None``."""
        ...

    def is_resumable_in_place(self) -> bool:
        """Whether resume restores sampler state without a ``suggest()`` replay.

        ``False`` for the deterministic journal (the engine replays the strategy
        past the recorded trials to fast-forward it); ``True`` for a backend
        whose own storage reconstructs the sampler (e.g. Optuna ``RDBStorage``).
        """
        ...

    def completed_count(self) -> int:
        """The number of budget-consuming trials.

        Pruned trials count as completed — a pruned trial is a real, if partial,
        evaluation (optuna-integration.md §8). Failed and in-flight trials do
        not, which makes this the same quantity ``OptunaStrategy.is_exhausted``
        compares against ``n_trials``.
        """
        ...

    def param_importances(self) -> Optional[dict[str, float]]:
        """Native (fANOVA) parameter importances, or ``None`` when unsupported.

        A backend that owns a richer importance model (e.g. an Optuna study →
        ``optuna.importance.get_param_importances`` fANOVA, which attributes
        interaction variance to each parameter) returns the ranked dict here.
        ``None`` signals "no native model" — the screening layer then falls back
        to its own RandomForest + permutation estimate (screening-importance.md
        §1). The journal returns ``None`` (no model); the Optuna store returns
        ``None`` too whenever its trials carry no native sampler dimensions (the
        ``append`` path stores params off-band), so the fallback still fires.
        """
        ...

    def pareto_front(self) -> list["Trial"]:
        """The non-dominated trials by their ``objectives`` sidecar (plan §0a).

        A multi-objective trial carries an ``objectives`` dict (set when the
        scorer's ``finalize`` returns a dict — e.g. a
        ``CompositeScorer(multi_objective=True)``); a trial is on the front when
        no other scored trial beats it on *every* objective. Failed trials and
        single-objective (``objectives is None``) trials are excluded, so a
        single-objective study returns ``[]`` here while scalar :meth:`best`
        still works (the multi-objective back-compat lock).
        """
        ...

    def knee_point(self, front: list["Trial"]) -> Optional["Trial"]:
        """The front trial at max perpendicular distance to the extremes' chord.

        The canonical multi-objective compromise pick (plan §0b): the
        max-curvature elbow of the Pareto ``front``, found by maximizing each
        front point's perpendicular distance to the chord between the two extreme
        objective points. ``None`` for an empty front.
        """
        ...
