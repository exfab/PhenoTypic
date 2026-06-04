"""The orchestrator — the ask-and-tell loop over a strategy + evaluator.

Drives ``suggest → evaluate → register_result`` until the strategy exhausts or
the budget caps, journaling every trial. Resumes by fast-forwarding a
deterministic strategy past the trials already in the store.
"""
from __future__ import annotations

from typing import Optional

from phenotypic import ImagePipeline

from ._evaluation import build_pipeline
from ._spec import TuningSpec
from ._study_store import StudyStore, Trial


class TuningEngine:
    """Runs a ``TuningSpec`` over a calibration image set, journaling to a store."""

    def __init__(self, spec: TuningSpec, store: Optional[StudyStore] = None) -> None:
        """Initialize the engine.

        Args:
            spec: The tuning recipe (base pipeline + space + scorer + strategy +
                budget).
            store: An optional pre-populated journal (resume); a fresh
                :class:`StudyStore` is created when omitted.
        """
        self._spec = spec
        self._store = store if store is not None else StudyStore()

    @property
    def store(self) -> StudyStore:
        """The trial journal this engine appends to."""
        return self._store

    def best_pipeline(self) -> Optional[ImagePipeline]:
        """Build the winning ``ImagePipeline`` from the best trial (or ``None``)."""
        best = self._store.best()
        if best is None:
            return None
        return build_pipeline(self._spec.pipeline, best.params)

    def optimize(self, images: list) -> Optional[Trial]:
        """Run the loop; return the best trial.

        Args:
            images: The calibration images (non-empty).

        Returns:
            The best :class:`Trial`, or ``None`` if none succeeded.
        """
        spec = self._spec
        strategy = spec.strategy.build(spec.search_space, self._store)

        # Resume: replay the deterministic strategy past recorded trials.
        completed = len(self._store)
        for _ in range(completed):
            if strategy.is_exhausted():
                break
            strategy.suggest()

        # Seed both budget counters from the store so resume is symmetric:
        # n_trials counts all recorded trials, max_failures counts recorded
        # failures (else the failure safety-valve resets to 0 on every resume).
        failures = sum(1 for t in self._store.trials if t.failed)
        number = completed
        budget = spec.budget
        while not strategy.is_exhausted():
            if budget.n_trials is not None and number >= budget.n_trials:
                break
            # Checked at the top (not only after a failing trial) so a run
            # resumed at/over the failure cap stops immediately.
            if budget.max_failures is not None and failures >= budget.max_failures:
                break
            params, _channel = strategy.suggest()
            params = dict(params)
            result = spec.evaluator.evaluate(
                spec.pipeline, spec.scorer, params, images
            )
            self._store.append(
                Trial(
                    number=number,
                    params=params,
                    score=result.score,
                    terms=result.terms,
                    n_images=result.n_images,
                    failed=result.failed,  # explicit flag from the Evaluator
                )
            )
            strategy.register_result(params, result)
            number += 1
            if result.failed:
                failures += 1

        return self._store.best()
