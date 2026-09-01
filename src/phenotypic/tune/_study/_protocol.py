"""The ``StudyStore`` seam — what the engine needs from a trial backend.

Phase 1 ships one concrete backend (``JournalStudyStore``, a parquet-journalled
list). Phase 2 adds an Optuna ``RDBStorage``-backed store that resumes **in
place** (the database already holds the sampler state), so the engine must not
replay the deterministic strategy past recorded trials for it. Typing the
engine's ``store`` against this Protocol keeps both backends swappable
(optuna-integration.md §7).
"""
from __future__ import annotations

from collections.abc import Sequence
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
        """The journaled trials in order (a copy)."""
        ...

    def __len__(self) -> int:
        """The number of recorded trials."""
        ...

    def terminal_trials(self) -> list["Trial"]:
        """The subset of trials in a terminal state."""
        ...

    def best(self) -> Optional["Trial"]:
        """The finite COMPLETE trial with the lowest cost, or ``None``."""
        ...

    def is_resumable_in_place(self) -> bool:
        """Whether resume restores sampler state without a ``suggest()`` replay.

        ``False`` for the deterministic journal (the engine replays the strategy
        past the recorded trials to fast-forward it); ``True`` for a backend
        whose own storage reconstructs the sampler (e.g. Optuna ``RDBStorage``).
        """
        ...

    def completed_count(self) -> int:
        """The number of completed (non-failed) trials.

        Pruned trials count as completed — a pruned trial is a real, if partial,
        evaluation (optuna-integration.md §8).
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

    def pareto_front(
        self, objective_axes: Sequence[str] | None = None
    ) -> list["Trial"]:
        """Return finite non-dominated COMPLETE trials (plan §0a).

        When ``objective_axes`` is supplied, only exact full finite vectors on
        those scorer-required axes are eligible. ``None`` preserves legacy
        sidecar inference for direct callers. Failed, PRUNED, non-finite, and
        single-objective trials remain excluded.
        """
        ...

    def knee_point(
        self,
        front: list["Trial"],
        objective_axes: Sequence[str] | None = None,
    ) -> Optional["Trial"]:
        """Return the maximum-distance knee on the supplied objective axes.

        The canonical multi-objective compromise pick (plan §0b) uses the same
        scorer-required axis order as front selection. ``None`` preserves
        legacy first-sidecar inference; an empty front returns ``None``.
        """
        ...
