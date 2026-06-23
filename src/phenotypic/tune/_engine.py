"""The orchestrator — the ask-and-tell loop over a strategy + evaluator.

Drives ``suggest → evaluate → register_result`` until the strategy exhausts or
the budget caps, journaling every trial. Resumes by fast-forwarding a
deterministic strategy past the trials already in the store.
"""
from __future__ import annotations

from typing import Optional

from phenotypic import ImagePipeline

from ._evaluation import build_pipeline
from ._multi_objective import objective_directions
from .score import CompositeScorer
from ._spec import TuningSpec
from ._study._protocol import StudyStore
from ._study_store import JournalStudyStore, Trial


class TuningEngine:
    """Runs a ``TuningSpec`` over a calibration image set, journaling to a store."""

    def __init__(self, spec: TuningSpec, store: Optional[StudyStore] = None) -> None:
        """Initialize the engine.

        Args:
            spec: The tuning recipe (base pipeline + space + scorer + strategy +
                budget).
            store: An optional pre-populated store (resume); any backend
                satisfying the :class:`StudyStore` Protocol. A fresh
                :class:`JournalStudyStore` is created when omitted.
        """
        self._spec = spec
        self._store: StudyStore = store if store is not None else JournalStudyStore()

    @property
    def store(self) -> StudyStore:
        """The trial store this engine appends to."""
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
        # Multi-objective is inferred from the scorer (plan §0b): a dict-returning
        # scorer yields per-objective ``directions`` the Optuna backend turns into
        # an NSGA-II Pareto study; ``None`` keeps the scalar single-objective path.
        directions = objective_directions(spec.scorer)

        # Pin the study-global active set for the augmented Tchebycheff composite
        # (§6.3): the children available study-wide form the fixed roster for both
        # the Tchebycheff max numerator and the normalizer, so the normalizer is a
        # study-global constant and per-image abstention stays a robust-aggregate
        # matter (not a max-composition one). Non-composite / non-Tchebycheff
        # scorers ignore this.
        #
        # Meta-validation ordering caveat (SF3): ReferenceFreeScorer.availability()
        # is False until meta_validate() runs; the engine does not invoke
        # meta_validate today (it is referenced only as a guard message in the CLI
        # path), so a ReferenceFreeScorer child is correctly dropped from the
        # roster here and the engine degrades to its fallback — matching today's
        # behavior. If a later phase wires meta_validate into the engine, move this
        # pin to immediately after it.
        scorer = spec.scorer
        if isinstance(scorer, CompositeScorer):
            active = tuple(
                handle
                for handle, child in zip(scorer.objective_names(), scorer.scorers)
                if child.availability()
            )
            scorer.set_active_set(active)

        strategy = spec.strategy.build(
            spec.search_space, self._store, directions=directions
        )

        # Resume: a deterministic journal is fast-forwarded by replaying the
        # strategy past the recorded trials; an in-place-resumable backend (e.g.
        # an Optuna RDB) already restores the sampler state, so it skips replay.
        # The same flag decides who writes each trial below (see the loop).
        completed = len(self._store)
        resumable = self._store.is_resumable_in_place()
        if not resumable:
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
            params, channel = strategy.suggest()
            params = dict(params)
            result = spec.evaluator.evaluate(
                spec.pipeline, spec.scorer, params, images, channel=channel
            )
            # Who writes the trial depends on the backend. On an in-place-resumable
            # store (an Optuna RDB) the strategy's register_result IS the writer:
            # its native ask/tell trial carries the sampler distributions AND our
            # off-model fields (via set_trial_user_attrs) into the ONE shared
            # study, so appending here too would double-record it. Only the
            # deterministic journal needs the engine to append a Trial record.
            if not resumable:
                self._store.append(
                    Trial(
                        number=number,
                        params=params,
                        score=result.score,
                        terms=result.terms,
                        # The multi-objective sidecar (plan §0a): carried verbatim
                        # from the Evaluator so the Pareto front / knee can read it.
                        n_images=result.n_images,
                        objectives=result.objectives,
                        # The robust-eval per-trial signals (plan 4.5p1): the
                        # calibration-dispersion gap + the under-detection
                        # suspicious flag, carried from the Evaluator so the
                        # journal and the data-poor generalization fallback (which
                        # reads the winner's gap) see real values, not nulls.
                        gap=result.gap,
                        suspicious=result.suspicious,
                        failed=result.failed,  # explicit flag from the Evaluator
                        pruned=result.pruned,  # early-stopped via the channel
                    )
                )
            # Tell the strategy how the trial ended; a pruned trial flows back as
            # pruned=True (Optuna marks it TrialState.PRUNED). On the Optuna path
            # this is the sole write of the shared study (params + user_attrs).
            strategy.register_result(params, result, pruned=result.pruned)
            # Budget counts completed + pruned (a pruned trial is a real, if
            # partial, evaluation). Only true failures feed the failure cap.
            number += 1
            if result.failed:
                failures += 1

        return self._store.best()
